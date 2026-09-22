"""Opt-in single-process recovery training. Every checkpoint is unqualified.

The native, workers=0 DataLoader is deliberately NOT Accelerate-wrapped: it
needs no sharding/device placement and avoids hidden lookahead/epoch state.
Its yielded position is not acknowledged until optimizer/LR/EMA work completes.
Model/optimizer are prepared by Accelerate; native codecs restore afterwards.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import random
import sys
import uuid

import torch
from torch.utils.data import DataLoader, Sampler

from scripts import klein_checkpoint as ck
from scripts import klein_recovery_metadata as md


RECOVERY_SOURCES = md.SOURCE_FILES + ("scripts/klein_model_resolver.py", "scripts/klein_recovery_training.py")


def seed_recovery(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _SuffixSampler(Sampler):
    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class AcknowledgedData:
    def __init__(self, dataset, seed, collate_fn):
        self.dataset = dataset
        self.order = torch.Generator()
        self.loader_rng = torch.Generator()
        for name, generator in (("data-order", self.order), ("loader", self.loader_rng)):
            domain = f"klein-recovery-v1:{name}:{seed}".encode()
            generator.manual_seed(int.from_bytes(hashlib.sha256(domain).digest()[:8], "little"))
        self.permutation = torch.empty(0, dtype=torch.int64)
        self.epoch = self.cursor = 0
        self.sampler = _SuffixSampler()
        self.loader = DataLoader(dataset, batch_size=1, sampler=self.sampler, num_workers=0,
                                 drop_last=True, collate_fn=collate_fn, generator=self.loader_rng)
        self.iterator = None
        self.pending = False

    def initialize_iterator(self, *, resumed=False):
        if self.cursor == 0:
            self.permutation = torch.randperm(len(self.dataset), generator=self.order)
        self.sampler.indices = self.permutation[self.cursor:].tolist()
        saved = self.loader_rng.get_state()
        self.iterator = iter(self.loader)
        if resumed and self.cursor:
            # Recreating a mid-epoch iterator must not consume another base seed.
            self.loader_rng.set_state(saved)

    def next(self):
        if self.pending:
            raise RuntimeError("Previous batch is not acknowledged")
        if self.iterator is None:
            self.initialize_iterator()
        batch = next(self.iterator)
        self.pending = True
        return batch

    def acknowledge(self):
        if not self.pending:
            raise RuntimeError("No batch to acknowledge")
        self.pending = False
        self.cursor += 1
        if self.cursor == len(self.dataset):
            self.epoch += 1
            self.cursor = 0
            self.iterator = None

    def capture(self, progress, fingerprint):
        if self.pending:
            raise RuntimeError("Cannot checkpoint an unacknowledged batch")
        return ck.capture_data_order_state(progress=progress, dataset_fingerprint=fingerprint,
                    permutation=self.permutation, order_generator=self.order, loader_generator=self.loader_rng)

    def restore(self, state, progress, fingerprint):
        self.epoch, self.cursor, self.permutation = ck.restore_data_order_state(
            state, self.order, self.loader_rng, progress=progress, dataset_size=len(self.dataset),
            dataset_fingerprint=fingerprint)
        self.initialize_iterator(resumed=True)


def preallocation_check(args, dataset, files, checkpoint):
    """Check available identities/settings before pipeline allocation; objects later."""
    from scripts.train_klein_standalone import BUCKET_SIZES
    fingerprints = {"model": md.fingerprint_model(args.model_path, files),
                    "dataset": md.fingerprint_dataset(dataset),
                    "preprocessing": md.fingerprint_preprocessing(dataset, bucket_sizes=BUCKET_SIZES),
                    "source_code": md.fingerprint_source_code(Path(__file__).resolve().parents[1], RECOVERY_SOURCES)}
    if checkpoint is None:
        return fingerprints
    m = ck._plain(checkpoint.manifest)
    if m["schema_version"] != 2:
        raise ck.CheckpointCompatibilityError("Recovery trainer requires v2; v1 lacks execution metadata. No automatic migration.")
    saved = m["metadata"]["document"]
    if saved["fingerprints"] != fingerprints:
        raise ck.CheckpointCompatibilityError("Model, dataset, preprocessing or source-code fingerprint differs")
    c = saved["configuration"]
    for key, actual in {"total_optimizer_steps": args.steps, "warmup_steps": min(args.warmup_steps, args.steps // 10),
                        "seed": args.seed, "max_grad_norm": args.max_grad_norm,
                        "gradient_checkpointing": args.gradient_checkpointing,
                        "optimizer": args.optimizer, "ema_enabled": args.use_ema,
                        "ema_decay": args.ema_decay if args.use_ema else None}.items():
        if type(c[key]) is not type(actual) or c[key] != actual:
            raise ck.CheckpointCompatibilityError(f"Recovery setting differs before allocation: {key}")
    if c["optimizer_options"][0]["lr"] != args.lr or (args.optimizer == "adamw" and
            c["optimizer_options"][0]["weight_decay"] != args.weight_decay):
        raise ck.CheckpointCompatibilityError("Effective initial optimizer LR/weight decay differs")
    e = saved["environment"]
    for key, actual in {"python": platform.python_version(), "platform": sys.platform,
                        "machine": platform.machine(), "cuda_runtime": torch.version.cuda}.items():
        if e[key] != actual:
            raise ck.CheckpointCompatibilityError(f"Recovery environment differs: {key}")
    for package, version in e["dependencies"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if package == "torch":
            actual = str(torch.__version__)
        if version != actual:
            raise ck.CheckpointCompatibilityError(f"Recovery dependency differs: {package}")
    return fingerprints


class RecoverySession:
    """Post-prepare state owner. CPU fixture execution does not qualify CUDA."""
    def __init__(self, model, optimizer, scheduler, ema, accelerator, data, metadata):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.ema, self.accelerator, self.data = ema, accelerator, data
        md.validate_metadata_compatibility(metadata, metadata)
        self.metadata = metadata
        self.configuration = ck.v2_configuration(metadata)
        self.run_id = uuid.uuid4().hex
        self.parent = None
        self.attempts = self.completed = self.skipped = self.cosine = 0
        self.loss_sum = 0.0
        self.log_steps = 0

    def progress(self):
        return {"attempted_optimizer_steps": self.attempts, "completed_optimizer_steps": self.completed,
                "skipped_optimizer_steps": self.skipped, "loop_iterations": self.attempts,
                "epoch": self.data.epoch, "next_batch_index": self.data.cursor, "accumulation_step": 0}

    def acknowledge(self, *, skipped, evidence, loss):
        if type(skipped) is not bool or (not skipped and not evidence["sufficient"]):
            raise RuntimeError("Optimizer returned without sufficient completed-step evidence")
        # Same after-attempt LR policy as legacy training; EMA includes skips.
        warmup = self.configuration["warmup_steps"]
        if self.attempts < warmup:
            factor = (self.attempts + 1) / warmup
            for group in self.optimizer.param_groups:
                group["lr"] = self.configuration["optimizer_options"]["lr"] * factor
        elif not skipped:
            self.scheduler.step()
            self.cosine += 1
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)
        self.data.acknowledge()
        self.attempts += 1
        self.skipped += int(skipped)
        self.completed += int(not skipped)
        self.loss_sum += loss
        self.log_steps += 1

    def manifest(self):
        return {"format": ck.FORMAT, "schema_version": 2, "checkpoint_id": uuid.uuid4().hex,
                "run_id": self.run_id, "parent_checkpoint_id": self.parent,
                "created_at": datetime.now(timezone.utc).isoformat(), "qualification": "unqualified",
                "metadata": self.metadata, "configuration": self.configuration,
                "environment": self.metadata["document"]["environment"], "progress": self.progress(),
                "objects": {"model_class": "diffusers." + type(self.model).__name__,
                            "model_tensors": ck.tensor_inventory(self.model),
                            "parameter_groups": ck._group_names(self.model, self.optimizer),
                            "optimizer_class": {"adamw": "torch.optim.AdamW", "adafactor": "transformers.Adafactor"}[ck._optimizer_kind(self.optimizer)],
                            "scheduler_class": "torch.optim.lr_scheduler.CosineAnnealingLR"}, "payloads": []}

    def save(self, destination, *, fault_injector=None):
        if self.data.pending or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Recovery save requires an acknowledged, zero-gradient boundary")
        self.accelerator.wait_for_everyone()
        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)
        manifest = self.manifest()
        trainer = {"progress": self.progress(), "global_step": self.attempts,
                   "ema_updates": self.attempts if self.ema else 0, "cosine_updates": self.cosine,
                   "group_lrs": [g["lr"] for g in self.optimizer.param_groups],
                   "accelerator_step": self.accelerator.step, "loss_accumulator": self.loss_sum,
                   "log_steps": self.log_steps}
        states = {"trainer": trainer,
                  "optimizer": ck.capture_optimizer_state(self.model, self.optimizer, completed_steps=self.completed),
                  "scheduler": ck.capture_scheduler_state(self.scheduler, self.cosine),
                  "data_order": self.data.capture(self.progress(), self.configuration["dataset_fingerprint"]),
                  "rng": ck.capture_rng_state(cuda_devices=1)}
        result = ck.publish_checkpoint(destination, manifest=manifest, states=states, model=self.model,
                                       optimizer=self.optimizer, scheduler=self.scheduler, ema=self.ema,
                                       max_shard_bytes=512 * 1024**2, fault_injector=fault_injector)
        self.parent = manifest["checkpoint_id"]
        return result

    def restore(self, checkpoint):
        m = ck._plain(checkpoint.manifest)
        if m["schema_version"] != 2:
            raise ck.CheckpointCompatibilityError("Only v2 recovery integration is supported")
        md.validate_metadata_compatibility(m["metadata"], self.metadata)
        states = ck.read_recovery_states(checkpoint, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler)
        ck.restore_model_state(checkpoint.root / "model", self.model, expected_inventory=m["objects"]["model_tensors"])
        t = states["trainer"]
        ck.restore_optimizer_state(self.model, self.optimizer, states["optimizer"],
                                   completed_steps=m["progress"]["completed_optimizer_steps"], group_lrs=t["group_lrs"])
        ck.restore_scheduler_state(self.scheduler, states["scheduler"], configuration=self.configuration, trainer_state=t)
        if self.ema is not None:
            ck.restore_ema_state(checkpoint.root / "ema", self.ema, self.model, decay=self.ema.decay)
        self.attempts = t["global_step"]
        self.completed = m["progress"]["completed_optimizer_steps"]
        self.skipped = m["progress"]["skipped_optimizer_steps"]
        self.cosine, self.loss_sum, self.log_steps = t["cosine_updates"], t["loss_accumulator"], t["log_steps"]
        self.accelerator.step = t["accelerator_step"]
        self.run_id, self.parent = m["run_id"], m["checkpoint_id"]
        self.data.restore(states["data_order"], m["progress"], self.configuration["dataset_fingerprint"])
        # All loading, iterator creation and wrapper construction precede this.
        ck.restore_rng_state(states["rng"], cuda_devices=1)


def staged_batch(pipe, batch, device, trainer, *, trace=None):
    """Offload transformer during each uncached encoding; optimizer stays resident.

    This is intentionally slower than caching. Tensor identities are verified;
    no master weights or second model are created. Optimizer states on CUDA
    still consume memory during encoding. Total CPU RAM is not bounded.
    """
    ids = [id(p) for p in pipe.transformer.parameters()]
    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    try:
        pipe.vae.to(device)
        with torch.no_grad():
            if trace is None:
                latents = trainer.encode_images_klein(pipe.vae, batch["pixel_values"], device, torch.bfloat16)
            else:
                latents = trainer.encode_images_klein(pipe.vae, batch["pixel_values"], device, torch.bfloat16, trace=trace)
                trace.emit("vae_normalized", tensors=[("latents", latents)], rng=True)
    finally:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
    try:
        pipe.text_encoder.to(device)
        with torch.no_grad():
            if trace is not None:
                trace.emit("text_encoding_begin", rng=True)
            embeds = pipe._get_qwen3_prompt_embeds(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                        prompt=batch["captions"], device=device, max_sequence_length=256, hidden_states_layers=(9, 18, 27))
            if trace is not None:
                trace.emit("text_encoding_end", tensors=[("embeddings", embeds)], rng=True)
    finally:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    pipe.transformer.to(device)
    if ids != [id(p) for p in pipe.transformer.parameters()]:
        raise RuntimeError("Device staging replaced Parameter objects; refusing recovery training")
    return latents, embeds


def flow_loss(model, latents, embeds, pipe, trainer, *, trace=None):
    device = latents.device
    if trace is None:
        times = (torch.sigmoid(torch.randn(latents.shape[0], device=device)) * 1000).long().clamp(0, 999)
    else:
        trace.emit("flow_begin", rng=True)
        timestep_draw = torch.randn(latents.shape[0], device=device)
        times = (torch.sigmoid(timestep_draw) * 1000).long().clamp(0, 999)
        trace.emit("timestep", tensors=[("draw", timestep_draw), ("timesteps", times)], rng=True)
    sigmas = trainer.get_sigmas(times, n_dim=4, dtype=latents.dtype)
    noise = torch.randn_like(latents)
    noisy = (1 - sigmas) * latents + sigmas * noise
    patched = trainer.patchify(noisy)
    if trace is None:
        prediction = model(hidden_states=trainer.pack_latents(patched), timestep=times.float() / 1000,
                           guidance=None, encoder_hidden_states=embeds,
                           txt_ids=pipe._prepare_text_ids(embeds).to(device),
                           img_ids=pipe._prepare_latent_ids(patched).to(device), return_dict=False)[0]
    else:
        packed = trainer.pack_latents(patched)
        timestep = times.float() / 1000
        txt_ids = pipe._prepare_text_ids(embeds).to(device)
        img_ids = pipe._prepare_latent_ids(patched).to(device)
        trace.emit("flow_inputs", tensors=[("noise", noise), ("sigmas", sigmas),
                   ("noisy_latents", noisy), ("packed", packed), ("timestep", timestep),
                   ("text_ids", txt_ids), ("image_ids", img_ids)], rng=True)
        prediction = model(hidden_states=packed, timestep=timestep, guidance=None,
                           encoder_hidden_states=embeds, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    prediction = trainer.unpack_latents(prediction, noisy.shape[2] // 2, noisy.shape[3] // 2)
    prediction = trainer.unpatchify(prediction, channels=latents.shape[1])
    loss = torch.nn.functional.mse_loss(prediction.float(), (noise - latents).float())
    if trace is not None:
        trace.emit("forward", tensors=[("prediction", prediction), ("loss", loss)], rng=True)
    return loss


def train_recovery(args):
    from scripts.train_klein_standalone import BUCKET_SIZES
    from scripts import train_klein_standalone as trainer
    from scripts.klein_model_resolver import resolve_model_files, load_selected_pipeline
    from accelerate import Accelerator
    # Dataset.__getitem__ imports transforms lazily. Import before seeding and
    # before restoring RNG so a fresh resume cannot add first-import RNG draws.
    import torchvision.transforms  # noqa: F401
    # Validate controls even for direct Python callers, before allocations.
    validate_controls(args)
    checkpoint = ck.validate_checkpoint(args.recovery_resume) if args.recovery_resume else None
    dataset = trainer.ImageTextDataset(args.data_dir, args.target_size, fixed_size=True)
    if not len(dataset):
        raise ck.CheckpointValidationError("Recovery dataset is empty")
    files = resolve_model_files(args.model_path)
    model_index = ck._read_json(Path(args.model_path) / "model_index.json")
    if "is_distilled" not in model_index and args.recovery_model_variant != "base-9b":
        raise ck.CheckpointValidationError("Missing is_distilled requires --recovery_model_variant base-9b as a user declaration")
    trainer.preflight_model_config(model_index,
        ck._read_json(Path(args.model_path) / "transformer/config.json"),
        smoke_model_variant=args.recovery_model_variant, model_path=args.model_path)
    fingerprints = preallocation_check(args, dataset, files, checkpoint)
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    trainer.preflight_smoke_runtime(accelerator)
    plugin = getattr(accelerator.state, "dynamo_plugin", None)
    backend = getattr(plugin, "backend", "NO")
    if getattr(backend, "value", backend) != "NO":
        raise ck.CheckpointValidationError("Recovery does not support compiled/Dynamo execution")
    if torch.cuda.device_count() != 1:
        raise ck.CheckpointValidationError("Expose exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    seed_recovery(args.seed)
    trace = None
    if getattr(args, "determinism_trace", None):
        from scripts.klein_determinism import DeterminismTrace, model_tensors, optimizer_observation
        trace = DeterminismTrace(args.determinism_trace, getattr(args, "determinism_trace_steps", None) or 2)
        trace.emit("seeded", details={"seed": args.seed}, rng=True)
    pipe = load_selected_pipeline(args.model_path, files, torch.bfloat16)
    if trace is not None:
        trace.emit("loaded_cpu", tensors=[(component + "/" + name, value)
                   for component in ("transformer", "vae", "text_encoder")
                   for name, value in model_tensors(getattr(pipe, component))], full_cpu=True, rng=True)
    # Verify immutable input selection across loading before any training.
    if fingerprints["model"] != md.fingerprint_model(args.model_path, files):
        raise ck.CheckpointValidationError("Model files changed while loading")
    trainer.preflight_9b_architecture(pipe.transformer.config)
    trainer.preflight_components(pipe, args.target_size)
    transformer = pipe.transformer.requires_grad_(True)
    transformer.train()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        native_optimizer = Adafactor(transformer.parameters(), lr=args.lr, relative_step=False,
                                     scale_parameter=False, warmup_init=False)
    else:
        native_optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay, foreach=False)
    warmup = min(args.warmup_steps, args.steps // 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(native_optimizer, T_max=args.steps - warmup, eta_min=args.lr * 0.1)
    data = AcknowledgedData(dataset, args.seed, trainer.collate_fn)
    # EMA created before metadata; moved with the transformer after preparation.
    ema = trainer.EMAModel(transformer, args.ema_decay) if args.use_ema else None
    metadata = md.build_recovery_configuration(args=args, dataset=dataset, dataloader=data.loader,
        model=transformer, optimizer=native_optimizer, scheduler=scheduler, accelerator=accelerator,
        resolved_model_files=files, bucket_sizes=BUCKET_SIZES, ema=ema)
    metadata["document"]["fingerprints"]["source_code"] = fingerprints["source_code"]
    metadata["sha256"] = md.canonical_sha256(metadata["document"])
    if trace is not None:
        from scripts import klein_determinism
        trace.emit("configuration", details={"metadata": metadata,
                   "diagnostic_source_sha256": hashlib.sha256(Path(klein_determinism.__file__).read_bytes()).hexdigest(),
                   "gpu_name": torch.cuda.get_device_name(accelerator.device),
                   "gpu_capability": list(torch.cuda.get_device_capability(accelerator.device)),
                   "save_every": args.save_every, "stop_after": args.recovery_stop_after,
                   "optimizer": optimizer_observation(transformer, native_optimizer)}, rng=True, data=data)
    if checkpoint:
        md.validate_metadata_compatibility(ck._plain(checkpoint.manifest)["metadata"], metadata)
    # No Accelerate scheduler/loader wrappers: single-process native scheduler
    # explicitly follows the wrapper's skip policy, and loader stays on CPU.
    prepared_model, prepared_optimizer = accelerator.prepare(transformer, native_optimizer)
    transformer = accelerator.unwrap_model(prepared_model)
    if trace is not None:
        trace.emit("prepared", tensors=model_tensors(transformer), probe_only=True, rng=True, data=data)
    if ema:
        ema.shadow = {name: value.to(accelerator.device) for name, value in ema.shadow.items()}
    session = RecoverySession(transformer, native_optimizer, scheduler, ema, accelerator, data, metadata)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if checkpoint:
        session.restore(checkpoint)
    else:
        data.initialize_iterator()
    if trace is not None:
        trace.emit("iterator_ready", rng=True, data=data)
    limit = args.recovery_stop_after or args.steps
    if session.attempts > limit:
        raise ck.CheckpointCompatibilityError("Checkpoint is beyond recovery_stop_after; refusing a misleading stopped-run result")
    last_saved = None
    while session.attempts < limit:
        observe = trace if trace is not None and session.attempts < trace.steps else None
        if observe is not None:
            observe.attempt = session.attempts
            observe.emit("batch_begin", rng=True, data=data)
        batch = data.next()
        if observe is None:
            latents, embeds = staged_batch(pipe, batch, accelerator.device, trainer)
        else:
            observe.emit("batch", tensors=[("pixels", batch["pixel_values"])], full_cpu=True,
                details={"samples": [Path(p).relative_to(dataset.data_dir).as_posix() for p in batch["paths"]],
                         "caption_sha256": [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in batch["captions"]]},
                rng=True, data=data)
            latents, embeds = staged_batch(pipe, batch, accelerator.device, trainer, trace=observe)
        trainer.parameter_coverage(transformer, native_optimizer)
        with accelerator.accumulate(prepared_model):
            with accelerator.autocast():
                if observe is None:
                    loss = flow_loss(prepared_model, latents, embeds, pipe, trainer)
                else:
                    loss = flow_loss(prepared_model, latents, embeds, pipe, trainer, trace=observe)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite recovery loss; no checkpoint boundary")
            accelerator.backward(loss)
            if observe is not None:
                observe.emit("gradients_before_clip", tensors=model_tensors(transformer, gradients=True), probe_only=True, rng=True)
            accelerator.clip_grad_norm_(prepared_model.parameters(), args.max_grad_norm)
            if observe is not None:
                observe.emit("gradients_after_clip", tensors=model_tensors(transformer, gradients=True), probe_only=True,
                             details={"lr_used": [g["lr"] for g in native_optimizer.param_groups]})
            before = trainer.optimizer_step_snapshot(transformer, native_optimizer, args.optimizer)
            prepared_optimizer.step()
            skipped = accelerator.optimizer_step_was_skipped
            evidence = trainer.verify_optimizer_step(before,
                trainer.optimizer_step_snapshot(transformer, native_optimizer, args.optimizer), skipped)
            if observe is not None:
                observe.emit("optimizer_result", tensors=model_tensors(transformer), probe_only=True,
                    details={"skipped": skipped, "evidence": evidence,
                             "optimizer": optimizer_observation(transformer, native_optimizer)}, rng=True)
            session.acknowledge(skipped=skipped, evidence=evidence, loss=float(loss.item()))
            if observe is not None:
                observe.emit("acknowledged", tensors=list(ema.shadow.items()) if ema is not None else (),
                    probe_only=True, details={"progress": session.progress(),
                    "scheduler": scheduler.state_dict(), "lr_next": [g["lr"] for g in native_optimizer.param_groups]},
                    rng=True, data=data)
        del loss, latents, embeds, batch
        if session.attempts % args.log_every == 0:
            print(f"Recovery attempt {session.attempts}: completed={session.completed}, skipped={session.skipped}, lr={native_optimizer.param_groups[0]['lr']}")
        if session.attempts % args.save_every == 0 or session.attempts == limit:
            destination = Path(args.output_dir) / f"checkpoint-{session.attempts}"
            if observe is not None:
                observe.emit("before_save", rng=True, data=data)
            session.save(destination)  # Any failure is fatal; no fallback export.
            if observe is not None:
                observe.emit("after_save", rng=True, data=data)
            last_saved = destination
            print(f"Published unqualified recovery checkpoint: {destination}")
    accelerator.wait_for_everyone()
    print(f"Recovery run stopped at attempt {session.attempts}; checkpoint={last_saved or args.recovery_resume}; exact recovery remains unqualified")
    if trace is not None:
        trace.finish()


def validate_controls(args):
    if getattr(args, "determinism_trace", None) and (args.recovery_resume or args.smoke_test):
        raise ck.CheckpointValidationError("Determinism trace supports only fresh, uninterrupted recovery runs")
    trace_steps = getattr(args, "determinism_trace_steps", None)
    if trace_steps is not None and (not getattr(args, "determinism_trace", None) or not 1 <= trace_steps <= 4):
        raise ck.CheckpointValidationError("Trace steps require --determinism_trace and must be in [1,4]")
    required = {"batch_size": 1, "grad_accum": 1, "num_workers": 0,
                "use_cached_latents": False, "smoke_test": False, "wandb": False}
    for key, value in required.items():
        if getattr(args, key) != value:
            raise ck.CheckpointValidationError(f"--recovery requires {key}={value}")
    if args.resume_from or args.log_dir or args.sample_prompts:
        raise ck.CheckpointValidationError("Recovery rejects legacy resume, trackers and sampling; use --sample_prompts with no values")
    if args.optimizer not in ("adamw", "adafactor"):
        raise ck.CheckpointValidationError("Recovery requires explicit AdamW or Adafactor")
    if not args.model_path or args.target_size is None or args.target_size < 16 or args.target_size % 16:
        raise ck.CheckpointValidationError("Recovery requires an explicit local model and target size divisible by 16")
    if any(getattr(args, k) <= 0 for k in ("steps", "save_every", "log_every")):
        raise ck.CheckpointValidationError("steps/save_every/log_every must be positive")
    if not 0 <= args.seed < 2**64 or args.warmup_steps < 0:
        raise ck.CheckpointValidationError("Invalid recovery seed or warmup")
    if args.recovery_stop_after is not None and not 1 <= args.recovery_stop_after <= args.steps:
        raise ck.CheckpointValidationError("recovery_stop_after is an absolute attempt in [1, steps]; steps remains the schedule horizon")
    for key in ("lr", "max_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ck.CheckpointValidationError(f"Recovery {key} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ck.CheckpointValidationError("Recovery weight_decay must be finite and nonnegative")
    if args.use_ema and (not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1):
        raise ck.CheckpointValidationError("Recovery EMA decay must be in [0,1)")
    for name in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
        if os.environ.get(name, "").lower() in ("1", "true", "yes"):
            raise ck.CheckpointValidationError(f"Recovery rejects {name}")
