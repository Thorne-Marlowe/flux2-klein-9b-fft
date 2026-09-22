"""CPU vertical-slice tests. Flux class and CUDA RNG count are explicit fixtures.

Subprocesses use real tiny BF16 models, native optimizers, codecs and publisher.
Only CUDA RNG count is adapted to zero; these tests do not certify CUDA/Flux.
"""
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import klein_checkpoint as ck
from scripts import klein_recovery_metadata as md
from scripts import klein_recovery_training as recovery
from scripts import train_klein_standalone as trainer
from scripts.klein_model_resolver import resolve_model_files, selected_model_view
import test_recovery_metadata as fixtures


@contextmanager
def cpu_rng_fixture():
    capture, validate, restore = ck.capture_rng_state, ck.validate_rng_state, ck.restore_rng_state
    with patch.object(ck, "capture_rng_state", side_effect=lambda **kw: capture(cuda_devices=0)), \
         patch.object(ck, "validate_rng_state", side_effect=lambda state, **kw: validate(state, cuda_devices=0)), \
         patch.object(ck, "restore_rng_state", side_effect=lambda state, **kw: restore(state, cuda_devices=0)):
        yield


class Flux2Transformer2DModel(torch.nn.Linear):
    """Tiny class-name fixture; not a Diffusers model."""
    def __init__(self):
        super().__init__(3, 2, dtype=torch.bfloat16)
        self.register_buffer("counter", torch.tensor(0))
        self.is_gradient_checkpointing = True
        self.attn_processors = {"attention.processor": fixtures.Processor()}


def make_session(fixture, kind="adamw", real_accelerate=False, image_data=False):
    if image_data:
        # Match the production import-before-seeding/restoration sequence.
        import torchvision.transforms  # noqa: F401
        from PIL import Image
        for index in range(3):
            image = Image.new("RGB", (23 + index, 19 + index))
            image.putdata([((x * 11 + index * 41) % 256,
                            (x * 7 + index * 53) % 256,
                            (x * 3 + index * 67) % 256)
                           for x in range(image.width * image.height)])
            image.save(fixture.data_root / f"{index}.png")
            (fixture.data_root / f"{index}.txt").write_text(f"  real caption {index}\n", encoding="utf-8")
        fixture.dataset = trainer.ImageTextDataset(fixture.data_root, target_size=16, fixed_size=True)
    recovery.seed_recovery(42)
    model = Flux2Transformer2DModel()
    if kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, foreach=False)
    else:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(model.parameters(), lr=3e-5, relative_step=False, scale_parameter=False, warmup_init=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=3e-5 * 0.1)
    args = copy.copy(fixture.args)
    args.seed, args.optimizer, args.use_ema = 42, kind, True
    data = recovery.AcknowledgedData(fixture.dataset if image_data else list(range(2)), 42,
                                     trainer.collate_fn if image_data else lambda batch: batch[0])
    # Fingerprint the real image-caption fixture; execution uses integer inputs.
    ema = trainer.EMAModel(model, args.ema_decay)
    metadata = md.build_recovery_configuration(args=args, dataset=fixture.dataset,
        dataloader=data.loader if image_data else fixture.loader,
        model=model, optimizer=optimizer, scheduler=scheduler, accelerator=fixture.accelerator,
        resolved_model_files=fixture.files, bucket_sizes=fixture.buckets, ema=ema)
    metadata["document"]["fingerprints"]["source_code"] = md.fingerprint_source_code(
        Path(__file__).resolve().parents[1], recovery.RECOVERY_SOURCES)
    metadata["sha256"] = md.canonical_sha256(metadata["document"])
    accelerator = SimpleNamespace(device=torch.device("cpu"), step=0, wait_for_everyone=lambda: None)
    prepared = optimizer
    prepared_model = model
    if real_accelerate:
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1)
        prepared_model, prepared = accelerator.prepare(model, optimizer)
    session = recovery.RecoverySession(model, optimizer, scheduler, ema, accelerator, data, metadata)
    return session, prepared_model, prepared


def observe_image_iterators(data):
    """Assert native base-seed/permutation draw counts without altering the RNGs.

    Torch's workers=0 iterator draws one int64 base seed from loader.generator.
    Resume cancels that draw only for a mid-epoch iterator reconstruction.
    Reference generators are independent copies, never the live generators.
    """
    events = []
    original = data.initialize_iterator

    def initialize(*, resumed=False):
        before_rng = ck.capture_rng_state(cuda_devices=1)
        loader_before = data.loader_rng.get_state().clone()
        order_before = data.order.get_state().clone()
        expected_loader = torch.Generator().set_state(loader_before)
        expected_order = torch.Generator().set_state(order_before)
        permutation = data.permutation.clone()
        if data.cursor == 0:
            permutation = torch.randperm(len(data.dataset), generator=expected_order)
        if not (resumed and data.cursor):
            torch.empty((), dtype=torch.int64).random_(generator=expected_loader)
        original(resumed=resumed)
        assert torch.equal(data.loader_rng.get_state(), expected_loader.get_state()), "Wrong loader seed draw count"
        assert torch.equal(data.order.get_state(), expected_order.get_state()), "Wrong permutation draw count"
        assert torch.equal(data.permutation, permutation), "Wrong epoch permutation"
        assert ck._equal_state(before_rng, ck.capture_rng_state(cuda_devices=1)), "Iterator consumed global RNG"
        events.append({"epoch": data.epoch, "cursor": data.cursor, "resumed": resumed,
                       "loader_before": loader_before, "loader_after": data.loader_rng.get_state().clone(),
                       "order_before": order_before, "order_after": data.order.get_state().clone()})

    data.initialize_iterator = initialize
    return events


def advance_images(session, end):
    """Actual image/caption/preprocessing/loader path, with a tiny CPU loss."""
    import random
    import numpy as np
    trace = []
    while session.attempts < end:
        batch = session.data.next()
        a = session.attempts
        rng_before = ck.capture_rng_state(cuda_devices=1)
        noise = torch.randn(1, 3, dtype=torch.bfloat16)
        timestep = torch.sigmoid(torch.randn(1))
        python_draw, numpy_draw = random.random(), float(np.random.random())
        values = batch["pixel_values"].mean(dim=(2, 3)).to(torch.bfloat16) + noise
        target = python_draw + numpy_draw + timestep.item()
        used_lrs = [g["lr"] for g in session.optimizer.param_groups]
        loss = (session.model(values).float() - target).square().mean()
        loss.backward()
        before = trainer.optimizer_step_snapshot(session.model, session.optimizer, ck._optimizer_kind(session.optimizer))
        skipped = a in {1, 5}
        if not skipped:
            session.optimizer.step()
        evidence = trainer.verify_optimizer_step(before,
            trainer.optimizer_step_snapshot(session.model, session.optimizer, ck._optimizer_kind(session.optimizer)), skipped)
        session.accelerator.step += 1
        session.acknowledge(skipped=skipped, evidence=evidence, loss=loss.item())
        trace.append({"attempt": a, "samples": [Path(path).name for path in batch["paths"]],
                      "captions": batch["captions"], "pixels": batch["pixel_values"].clone(),
                      "noise": noise, "timestep": timestep, "python_draw": python_draw, "numpy_draw": numpy_draw,
                      "inputs": values, "rng_before": rng_before, "rng_after": ck.capture_rng_state(cuda_devices=1),
                      "data": session.data.capture(session.progress(), session.configuration["dataset_fingerprint"]),
                      "progress": session.progress(), "used_lrs": used_lrs, "loss": loss.item(),
                      "scheduler": copy.deepcopy(session.scheduler.state_dict())})
    return trace


def run_image_worker(root, mode, kind, boundary):
    fixture = fixtures.RecoveryMetadataTests()
    fixture.setUp()
    try:
        torch.set_num_threads(1)
        session, _, _ = make_session(fixture, kind, image_data=True)
        assert type(session.data.dataset) is trainer.ImageTextDataset
        assert type(session.data.loader) is torch.utils.data.DataLoader
        root = Path(root)
        with cpu_rng_fixture():
            events = observe_image_iterators(session.data)
            if mode == "resume":
                checkpoint = ck.validate_checkpoint(root / "boundary")
                saved_rng = ck.read_state_payload(checkpoint.root / "rng.pt", checkpoint.manifest["checkpoint_id"], cuda_devices=1)
                session.restore(checkpoint)
                assert ck._equal_state(saved_rng, ck.capture_rng_state(cuda_devices=1)), "Global RNG not restored last"
            else:
                session.data.initialize_iterator()
            trace = advance_images(session, boundary if mode == "save" else 20)
            if mode == "save":
                before_save = ck.capture_rng_state(cuda_devices=1)
                session.save(root / "boundary")
                assert ck._equal_state(before_save, ck.capture_rng_state(cuda_devices=1)), "Publication consumed global RNG"
            torch.save({"trace": trace, "iterator_events": events,
                        "weights": session.model.state_dict(), "ema": session.ema.shadow,
                        "optimizer": session.optimizer.state_dict(), "scheduler": session.scheduler.state_dict(),
                        "progress": session.progress(), "rng": ck.capture_rng_state(cuda_devices=1),
                        "data": session.data.capture(session.progress(), session.configuration["dataset_fingerprint"]),
                        "cosine": session.cosine, "accelerator_step": session.accelerator.step,
                        "loss_sum": session.loss_sum, "log_steps": session.log_steps}, root / (mode + ".pt"))
    finally:
        fixture.doCleanups()


def advance(session, end, prepared=None):
    trace = []
    while session.attempts < end:
        index = session.data.next()
        a = session.attempts
        # Global draws and independent permutation both affect subsequent weights.
        import random
        import numpy as np
        values = torch.randn(1, 3, dtype=torch.bfloat16) + index
        target = random.random() + float(np.random.random())
        lr = [g["lr"] for g in session.optimizer.param_groups]
        loss = (session.model(values).float() - target).square().mean()
        loss.backward()
        before = trainer.optimizer_step_snapshot(session.model, session.optimizer, ck._optimizer_kind(session.optimizer))
        skipped = a in {1, 5}  # Simulated authoritative skip, not CUDA evidence.
        if not skipped:
            (prepared or session.optimizer).step()
        evidence = trainer.verify_optimizer_step(before,
            trainer.optimizer_step_snapshot(session.model, session.optimizer, ck._optimizer_kind(session.optimizer)), skipped)
        session.accelerator.step += 1
        session.acknowledge(skipped=skipped, evidence=evidence, loss=float(loss.item()))
        trace.append({"attempt": a, "index": index, "lr": lr, "loss": loss.item(),
                      "groups": [g["lr"] for g in session.optimizer.param_groups],
                      "scheduler": copy.deepcopy(session.scheduler.state_dict())})
    return trace


def run_worker(root, mode, kind, boundary):
    fixture = fixtures.RecoveryMetadataTests()
    fixture.setUp()
    try:
        session, _, _ = make_session(fixture, kind)
        root = Path(root)
        with cpu_rng_fixture():
            if mode == "resume":
                checkpoint = ck.validate_checkpoint(root / "boundary")
                ids = [id(p) for p in session.model.parameters()]
                session.restore(checkpoint)
                assert ids == [id(p) for p in session.model.parameters()]
            else:
                session.data.initialize_iterator()
            trace = advance(session, boundary if mode == "save" else 20)
            if mode == "save":
                session.save(root / "boundary")
            torch.save({"trace": trace, "weights": session.model.state_dict(), "ema": session.ema.shadow,
                        "optimizer": session.optimizer.state_dict(), "scheduler": session.scheduler.state_dict(),
                        "progress": session.progress(), "rng": ck.capture_rng_state(cuda_devices=1)}, root / (mode + ".pt"))
    finally:
        fixture.doCleanups()


class RecoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RecoveryMetadataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def image_continuity(self, boundary, kind):
        root = self.root / f"images-{kind}-{boundary}"
        root.mkdir()
        for mode in ("full", "save", "resume"):
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--image-worker",
                       str(root), mode, kind, str(boundary)]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        full, before, after = [torch.load(root / f"{mode}.pt", weights_only=True)
                               for mode in ("full", "save", "resume")]
        joined = before["trace"] + after["trace"]
        self.assertEqual(len(joined), 20)
        for expected, actual in zip(full["trace"], joined):
            self.assertTrue(ck._equal_state(expected, actual), f"Attempt {expected['attempt']} differs")
            count = actual["attempt"] + 1
            self.assertEqual((actual["progress"]["epoch"], actual["progress"]["next_batch_index"]), divmod(count, 3))
            name = actual["samples"][0]
            self.assertEqual(actual["captions"], [f"real caption {Path(name).stem}"])
        for start in range(0, 18, 3):
            self.assertEqual({row["samples"][0] for row in joined[start:start + 3]}, {"0.png", "1.png", "2.png"})
        for key in ("weights", "ema", "optimizer", "scheduler", "progress", "rng", "data",
                    "cosine", "accelerator_step", "loss_sum", "log_steps"):
            self.assertTrue(ck._equal_state(full[key], after[key]), key)
        # Compare the resume event against the saved generator states and cursor.
        # At a mid-epoch boundary neither generator may advance. At an epoch
        # boundary both must advance exactly once (checked by the worker oracle).
        first = after["iterator_events"][0]
        self.assertTrue(first["resumed"])
        self.assertEqual((first["epoch"], first["cursor"]), divmod(boundary, 3))
        for name, state_key in (("loader", "loader_generator_state"), ("order", "order_generator_state")):
            self.assertTrue(torch.equal(first[name + "_before"], before["data"][state_key]))
            self.assertEqual(torch.equal(first[name + "_before"], first[name + "_after"]), boundary % 3 != 0)
        self.assertEqual(ck.validate_checkpoint(root / "boundary").manifest["qualification"], "unqualified")

    def test_actual_image_dataset_mid_epoch_fresh_process(self):
        self.image_continuity(1, "adamw")

    def test_actual_image_dataset_epoch_boundary_fresh_process(self):
        self.image_continuity(3, "adafactor")

    def test_actual_image_dataset_multiple_epochs_fresh_process(self):
        self.image_continuity(7, "adamw")

    def test_fresh_process_continuity_adamw_and_adafactor(self):
        for kind, boundary in (("adamw", 1), ("adafactor", 6)):
            root = self.root / kind
            root.mkdir()
            for mode in ("full", "save", "resume"):
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(root), mode, kind, str(boundary)]
                result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            full, before, after = [torch.load(root / f"{mode}.pt", weights_only=True) for mode in ("full", "save", "resume")]
            self.assertEqual(full["trace"], before["trace"] + after["trace"])
            for key in ("weights", "ema", "optimizer", "scheduler", "progress", "rng"):
                self.assertTrue(ck._equal_state(full[key], after[key]), key)
            self.assertEqual(ck.validate_checkpoint(root / "boundary").manifest["schema_version"], 2)

    def test_publication_failure_is_fatal_and_previous_checkpoint_survives(self):
        session, _, _ = make_session(self.fixture)
        session.data.initialize_iterator()
        with cpu_rng_fixture():
            advance(session, 1)
            session.save(self.root / "first")
            first = ck.validate_checkpoint(self.root / "first")
            advance(session, 2)
            def fail(stage):
                if stage == "rename":
                    raise OSError("injected rename failure")
            with self.assertRaises(ck.CheckpointPublicationError):
                session.save(self.root / "second", fault_injector=fail)
            self.assertFalse((self.root / "second").exists())
            self.assertEqual(first.manifest, ck.validate_checkpoint(self.root / "first").manifest)

    def test_corruption_configuration_mismatch_and_v1_rejected(self):
        session, _, _ = make_session(self.fixture)
        session.data.initialize_iterator()
        with cpu_rng_fixture():
            advance(session, 1)
            session.save(self.root / "saved")
            checkpoint = ck.validate_checkpoint(self.root / "saved")
            changed, _, _ = make_session(self.fixture)
            changed.metadata["document"]["configuration"]["max_grad_norm"] = 2.0
            changed.metadata["sha256"] = md.canonical_sha256(changed.metadata["document"])
            with self.assertRaises(ck.CheckpointCompatibilityError):
                changed.restore(checkpoint)
            (self.root / "saved/optimizer.pt").write_bytes(b"corrupted")
            with self.assertRaises(ck.CheckpointValidationError):
                session.restore(checkpoint)
        from test_checkpoint_recovery import CheckpointContractTests
        old = CheckpointContractTests()
        old.setUp()
        self.addCleanup(old.doCleanups)
        with self.assertRaisesRegex(ck.CheckpointCompatibilityError, "v2"):
            session.restore(ck.validate_checkpoint(old.root))

    def test_noop_step_and_unacknowledged_save_rejected(self):
        session, _, _ = make_session(self.fixture)
        session.data.next()
        with self.assertRaisesRegex(RuntimeError, "acknowledged"):
            session.save(self.root / "bad")
        with self.assertRaisesRegex(RuntimeError, "evidence"):
            session.acknowledge(skipped=False, evidence={"sufficient": False}, loss=1.0)
        self.assertEqual(session.attempts, 0)

    def test_data_initial_mid_epoch_and_epoch_boundary(self):
        for boundary in (0, 1, 2, 5):
            data = recovery.AcknowledgedData(list(range(2)), 7, lambda b: b[0])
            for _ in range(boundary):
                data.next()
                data.acknowledge()
            progress = {"epoch": data.epoch, "next_batch_index": data.cursor,
                        "loop_iterations": boundary, "attempted_optimizer_steps": boundary}
            state = data.capture(progress, "a" * 64)
            restored = recovery.AcknowledgedData(list(range(2)), 999, lambda b: b[0])
            restored.restore(state, progress, "a" * 64)
            for _ in range(6):
                self.assertEqual(data.next(), restored.next())
                data.acknowledge()
                restored.acknowledge()
            self.assertTrue(torch.equal(data.order.get_state(), restored.order.get_state()))
            self.assertTrue(torch.equal(data.loader_rng.get_state(), restored.loader_rng.get_state()))

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "Accelerate is not installed; prepared CPU path untested")
    def test_real_prepared_accelerate_roundtrip(self):
        session, _, prepared = make_session(self.fixture, "adafactor", real_accelerate=True)
        session.data.initialize_iterator()
        with cpu_rng_fixture():
            advance(session, 4, prepared)
            session.save(self.root / "prepared")
            expected = advance(session, 8, prepared)
            weights = copy.deepcopy(session.model.state_dict())
            restored, _, prepared2 = make_session(self.fixture, "adafactor", real_accelerate=True)
            restored.restore(ck.validate_checkpoint(self.root / "prepared"))
            self.assertEqual(expected, advance(restored, 8, prepared2))
            self.assertTrue(ck._equal_state(weights, restored.model.state_dict()))


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "model"
        index = {"_class_name": "Flux2KleinPipeline", "transformer": ["diffusers", "Flux2Transformer2DModel"],
                 "vae": ["diffusers", "AutoencoderKLFlux2"], "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                 "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"], "tokenizer": ["transformers", "Qwen2Tokenizer"]}
        self.write("model_index.json", index)
        for component, stem in (("transformer", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")):
            self.write(component + "/config.json", {})
            self.write(component + "/" + stem + ".safetensors", None)
        for path in ("scheduler/scheduler_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
            self.write(path, {})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture weights" if value is None else json.dumps(value).encode())

    def test_selection_and_private_view_exact_inventory(self):
        self.write("README.md", None)
        self.write("tokenizer/vocab.json", {})
        self.write("tokenizer/merges.txt", None)
        files = resolve_model_files(self.root)
        self.assertNotIn("tokenizer/vocab.json", files)
        with selected_model_view(self.root, files) as view:
            self.assertEqual(sorted(p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file()), list(files))
            self.assertEqual(md.fingerprint_model(view, files), md.fingerprint_model(self.root, files))
        self.assertFalse(view.exists())

    def test_missing_ambiguous_custom_and_unknown_dependencies(self):
        for path, value in (("transformer/other.safetensors", None),
                            ("transformer/diffusion_pytorch_model.safetensors.index.json", {"weight_map": {"a": "x.safetensors"}}),
                            ("tokenizer/chat_templates", None)):
            self.write(path, value)
            with self.assertRaises(ck.CheckpointValidationError):
                resolve_model_files(self.root)
            (self.root / path).unlink()
        self.write("text_encoder/config.json", {"auto_map": {"model": "evil.py"}})
        with self.assertRaises(ck.CheckpointValidationError):
            resolve_model_files(self.root)
        self.write("text_encoder/config.json", {})
        (self.root / "tokenizer/tokenizer.json").unlink()
        with self.assertRaises(ck.CheckpointValidationError):
            resolve_model_files(self.root)

    def test_cli_recovery_is_opt_in_and_separate(self):
        base = ["--output_dir", "out", "--data_dir", "data"]
        args = trainer.parse_args(base)
        self.assertFalse(args.recovery)
        with self.assertRaises(SystemExit):
            trainer.parse_args(base + ["--recovery_resume", "saved"])
        args = trainer.parse_args(base + ["--recovery", "--model_path", "local", "--target_size", "256",
            "--optimizer", "adafactor", "--batch_size", "1", "--grad_accum", "1", "--num_workers", "0", "--sample_prompts"])
        recovery.validate_controls(args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--image-worker":
        run_image_worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        unittest.main()
