import inspect
import os
from contextlib import ExitStack
from typing import Type, Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as T
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.adapter import T2IAdapter
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_tcd import TCDScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin as Scheduler
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPVisionModelWithProjection

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, invocation_output, BaseInvocationOutput
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.fields import (
    ConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
    OutputField,
    Field,
)
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import ModelIdentifierField, UNetField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.lora.lora_patcher import LoRAPatcher
from invokeai.backend.model_manager import BaseModelType, ModelVariantType
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.stable_diffusion import PipelineIntermediateState
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext, DenoiseInputs
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    IPAdapterConditioningInfo,
    IPAdapterData,
    Range,
    SDXLConditioningInfo,
    TextConditioningData,
    TextConditioningRegions,
)
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.controlnet import ControlNetExt
from invokeai.backend.stable_diffusion.extensions.freeu import FreeUExt
from invokeai.backend.stable_diffusion.extensions.inpaint import InpaintExt
from invokeai.backend.stable_diffusion.extensions.inpaint_model import InpaintModelExt
from invokeai.backend.stable_diffusion.extensions.lora import LoRAExt
from invokeai.backend.stable_diffusion.extensions.preview import PreviewExt
from invokeai.backend.stable_diffusion.extensions.rescale_cfg import RescaleCFGExt
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.extensions.t2i_adapter import T2IAdapterExt
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.hotfixes import ControlNetModel
from invokeai.backend.util.mask import to_standard_float_mask
from invokeai.backend.util.silence_warnings import SilenceWarnings


# imports purely for local implementation
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.denoise_latents import get_scheduler
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase
from invokeai.backend.util.logging import info, warning, error
from pydantic import BaseModel

SD12X_EXTENSIONS = {}

def base_guidance_extension(name: str):
    """Register a guidance extension class object under a string reference"""
    def decorator(cls: Type[ExtensionBase]):
        if name in SD12X_EXTENSIONS:
            raise ValueError(f"Extension {name} already registered")
        info(f"Registered extension {cls.__name__} as {name}")
        SD12X_EXTENSIONS[name] = cls
        return cls
    return decorator

class GuidanceField(BaseModel):
    """Guidance information for extensions in the denoising process."""
    guidance_name: str = Field(description="The name of the guidance extension class")
    priority: int = Field(default=100, description="Execution order for multiple guidance. Lower numbers go first.")
    extension_kwargs: dict[str, Any] = Field(default={}, description="Keyword arguments for the guidance extension")

@invocation_output("guidance_module_output")
class GuidanceDataOutput(BaseInvocationOutput):
    guidance_data_output: GuidanceField | None = OutputField(
        title="Guidance Module",
        description="Information to alter the denoising process"
    )

@invocation(
    "exposed_denoise_latents",
    title="Exposed Denoise Latents",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="2.0.0",
)
class ExposedDenoiseLatentsInvocation(DenoiseLatentsInvocation):
    """includes all of the inputs and methods of the parent class"""

    guidance_module: Optional[GuidanceField] = InputField(
        default=None,
        description="Guidance information for extensions in the denoising process.",
        input=Input.Connection,
        ui_order=10,
    )

    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        ext_manager = ExtensionsManager(is_canceled=context.util.is_canceled)

        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_torch_dtype()

        seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
        _, _, latent_height, latent_width = latents.shape

        conditioning_data = self.get_conditioning_data(
            context=context,
            positive_conditioning_field=self.positive_conditioning,
            negative_conditioning_field=self.negative_conditioning,
            cfg_scale=self.cfg_scale,
            steps=self.steps,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
            # TODO: old backend, remove
            cfg_rescale_multiplier=self.cfg_rescale_multiplier,
        )

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
            seed=seed,
        )

        timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
            scheduler,
            seed=seed,
            device=device,
            steps=self.steps,
            denoising_start=self.denoising_start,
            denoising_end=self.denoising_end,
        )

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        ### preview
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        ext_manager.add_extension(PreviewExt(step_callback))

        ### cfg rescale
        if self.cfg_rescale_multiplier > 0:
            ext_manager.add_extension(RescaleCFGExt(self.cfg_rescale_multiplier))

        ### freeu
        if self.unet.freeu_config:
            ext_manager.add_extension(FreeUExt(self.unet.freeu_config))

        ### lora
        if self.unet.loras:
            for lora_field in self.unet.loras:
                ext_manager.add_extension(
                    LoRAExt(
                        node_context=context,
                        model_id=lora_field.lora,
                        weight=lora_field.weight,
                    )
                )
        ### seamless
        if self.unet.seamless_axes:
            ext_manager.add_extension(SeamlessExt(self.unet.seamless_axes))

        ### inpaint
        mask, masked_latents, is_gradient_mask = self.prep_inpaint_mask(context, latents)
        # NOTE: We used to identify inpainting models by inpecting the shape of the loaded UNet model weights. Now we
        # use the ModelVariantType config. During testing, there was a report of a user with models that had an
        # incorrect ModelVariantType value. Re-installing the model fixed the issue. If this issue turns out to be
        # prevalent, we will have to revisit how we initialize the inpainting extensions.
        if unet_config.variant == ModelVariantType.Inpaint:
            ext_manager.add_extension(InpaintModelExt(mask, masked_latents, is_gradient_mask))
        elif mask is not None:
            ext_manager.add_extension(InpaintExt(mask, is_gradient_mask))

        # Initialize context for modular denoise
        latents = latents.to(device=device, dtype=dtype)
        if noise is not None:
            noise = noise.to(device=device, dtype=dtype)
        denoise_ctx = DenoiseContext(
            inputs=DenoiseInputs(
                orig_latents=latents,
                timesteps=timesteps,
                init_timestep=init_timestep,
                noise=noise,
                seed=seed,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                attention_processor_cls=CustomAttnProcessor2_0,
            ),
            unet=None,
            scheduler=scheduler,
        )

        # context for loading additional models
        with ExitStack() as exit_stack:
            # later should be smth like:
            # for extension_field in self.extensions:
            #    ext = extension_field.to_extension(exit_stack, context, ext_manager)
            #    ext_manager.add_extension(ext)
            self.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
            self.parse_t2i_adapter_field(exit_stack, context, self.t2i_adapter, ext_manager)

            # ext: t2i/ip adapter
            ext_manager.run_callback(ExtensionCallbackType.SETUP, denoise_ctx)

            unet_info = context.models.load(self.unet.unet)
            assert isinstance(unet_info.model, UNet2DConditionModel)
            with (
                unet_info.model_on_device() as (cached_weights, unet),
                ModelPatcher.patch_unet_attention_processor(unet, denoise_ctx.inputs.attention_processor_cls),
                # ext: controlnet
                ext_manager.patch_extensions(denoise_ctx),
                # ext: freeu, seamless, ip adapter, lora
                ext_manager.patch_unet(unet, cached_weights),
            ):
                sd_backend = StableDiffusionBackend(unet, scheduler)
                denoise_ctx.unet = unet
                result_latents = sd_backend.latents_from_embeddings(denoise_ctx, ext_manager)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.detach().to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)