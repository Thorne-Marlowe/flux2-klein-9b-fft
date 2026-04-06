"""Klein2 extension for ai-toolkit: Flux2 Klein-base 4B/9B full fine-tuning."""
import os
from typing import TYPE_CHECKING

import torch
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype
from diffusers import (
    Flux2Transformer2DModel,
    Flux2KleinPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from einops import rearrange

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True
}


class Klein2(BaseModel):
    arch = "klein2"

    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['Flux2Transformer2DModel']

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Klein2 model")

        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.name_or_path_original

        # Load full pipeline to get all components
        self.print_and_status_update("Loading pipeline")
        pipe = Flux2KleinPipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
        )

        transformer = pipe.transformer
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

        # Check if we have a saved transformer checkpoint to load instead
        transformer_path = os.path.join(model_path, 'transformer')
        if os.path.exists(transformer_path) and model_path != base_model_path:
            self.print_and_status_update(f"Loading transformer checkpoint from {transformer_path}")
            transformer = Flux2Transformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=dtype,
            )

        # Move to devices
        self.print_and_status_update("Moving to device")

        if self.model_config.quantize:
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update("Quantizing transformer")
            transformer.to(self.quantize_device, dtype=dtype)
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)

        flush()

        # Text encoder (Qwen3) - freeze
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text encoder (Qwen3)")
            text_encoder.to(self.quantize_device, dtype=dtype)
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            text_encoder.to(self.device_torch)
        else:
            text_encoder.to(self.device_torch, dtype=dtype)

        text_encoder.requires_grad_(False)
        text_encoder.eval()
        flush()

        # VAE (Flux2 VAE with BatchNorm)
        vae.to(self.device_torch, dtype=dtype)
        vae.requires_grad_(False)
        vae.eval()

        self.noise_scheduler = Klein2.get_train_scheduler()

        # Store references
        self.vae = vae
        self.text_encoder = [text_encoder]  # list for compatibility
        self.tokenizer = [tokenizer]
        self.model = transformer
        self.pipeline = pipe
        # Store pipeline methods for encoding
        self._get_qwen3_prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds
        self._prepare_latent_ids = Flux2KleinPipeline._prepare_latent_ids
        self._prepare_text_ids = Flux2KleinPipeline._prepare_text_ids
        self._pack_latents = Flux2KleinPipeline._pack_latents
        self._patchify_latents = Flux2KleinPipeline._patchify_latents

        self.print_and_status_update("Klein2 Model Loaded")

    def get_generation_pipeline(self):
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_config.name_or_path_original, subfolder="scheduler"
        )
        pipeline = Flux2KleinPipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.model),
        )
        return pipeline

    def generate_single_image(
        self,
        pipeline: Flux2KleinPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        bypass_guidance_embedding: bool,
        **kwargs
    ):
        with torch.no_grad():
            bs, c, h, w = latent_model_input.shape

            # Klein uses _patchify then _pack
            # Patchify: (B, C, H, W) -> (B, C*4, H/2, W/2)
            latents_patched = self._patchify_latents(latent_model_input)

            # Prepare 4D latent IDs (T, H, W, L)
            latent_ids = self._prepare_latent_ids(latents_patched).to(self.device_torch)

            # Pack: (B, C*4, H/2, W/2) -> (B, H/2*W/2, C*4)
            latent_packed = self._pack_latents(latents_patched)

            # Text IDs
            txt_ids = self._prepare_text_ids(text_embeddings.text_embeds).to(self.device_torch)

        cast_dtype = self.unet.dtype

        noise_pred = self.unet(
            hidden_states=latent_packed.to(self.device_torch, cast_dtype),
            timestep=timestep / 1000,  # Klein uses [0, 1] range
            guidance=None,  # Klein doesn't use guidance during training
            encoder_hidden_states=text_embeddings.text_embeds.to(self.device_torch, cast_dtype),
            txt_ids=txt_ids.to(cast_dtype),
            img_ids=latent_ids.to(cast_dtype),
            return_dict=False,
            **kwargs,
        )[0]

        if isinstance(noise_pred, QTensor):
            noise_pred = noise_pred.dequantize()

        # Unpack: (B, H/2*W/2, C*4) -> (B, C*4, H/2, W/2) -> unpatchify -> (B, C, H, W)
        h_half = latent_model_input.shape[2] // 2
        w_half = latent_model_input.shape[3] // 2
        noise_pred = noise_pred.reshape(bs, h_half, w_half, -1).permute(0, 3, 1, 2)

        # Unpatchify: (B, C*4, H/2, W/2) -> (B, C, H, W)
        c_latent = self.vae.config.latent_channels
        noise_pred = noise_pred.reshape(bs, c_latent, 2, 2, h_half, w_half)
        noise_pred = noise_pred.permute(0, 1, 4, 2, 5, 3)
        noise_pred = noise_pred.reshape(bs, c_latent, h_half * 2, w_half * 2)

        return noise_pred

    @torch.no_grad()
    def encode_images(self, images, device=None, dtype=None):
        """Override base encode_images for Klein2 VAE (BatchNorm + patchify, no shift_factor)."""
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.torch_dtype

        if not isinstance(images, list):
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            image_list = [img for img in images]
        else:
            image_list = images

        images = torch.stack(image_list).to(device, dtype=dtype)
        latents = self.vae.encode(images).latent_dist.sample()

        # Klein2 VAE: patchify then BatchNorm normalize
        latents = self._patchify_latents(latents)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std

        # Unpatchify back to (B, C, H, W) for the training framework
        # patchify: (B, 32, H/8, W/8) -> (B, 128, H/16, W/16)
        # unpatchify: (B, 128, H/16, W/16) -> (B, 32, H/8, W/8)
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // (2 * 2), height * 2, width * 2)

        return latents.to(device, dtype=dtype)

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        prompt_embeds = self._get_qwen3_prompt_embeds(
            text_encoder=self.text_encoder[0],
            tokenizer=self.tokenizer[0],
            prompt=[prompt] if isinstance(prompt, str) else prompt,
            device=self.device_torch,
            max_sequence_length=256,
            hidden_states_layers=(9, 18, 27),
        )
        text_ids = self._prepare_text_ids(prompt_embeds).to(self.device_torch)

        pe = PromptEmbeds(prompt_embeds)
        pe.pooled_embeds = None  # Klein doesn't use pooled embeds
        return pe

    def get_model_has_grad(self):
        return self.model.proj_out.weight.requires_grad

    def get_te_has_grad(self):
        # Qwen3 TE - check first layer
        return False  # TE is always frozen for Klein

    def save_model(self, output_path, meta, save_dtype):
        transformer: Flux2Transformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )
        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        return (noise - batch.latents).detach()

    def condition_noisy_latents(self, latents: torch.Tensor, batch: 'DataLoaderBatchDTO'):
        # Klein doesn't have inpainting/control - just return latents as-is
        return latents
