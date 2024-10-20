from invokeai.invocation_api import InputField, InvocationContext, BaseInvocation, invocation

from .exposed_denoise_latents import base_guidance_extension, GuidanceField, GuidanceDataOutput
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback

from .APG_util import MomentumBuffer, normalized_guidance

@base_guidance_extension("APG_Base")
class APG_Base_Ext(ExtensionBase):
    def __init__(self, guidance_scale: float, eta: float, norm_threshold: float, momentum: float):
        super().__init__()
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.momentum_buffer = MomentumBuffer(momentum)

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def rescale_noise_pred(self, ctx: DenoiseContext):
        ctx.noise_pred = normalized_guidance(
            ctx.positive_noise_pred,
            ctx.negative_noise_pred,
            self.guidance_scale,
            self.momentum_buffer,
            self.eta,
            self.norm_threshold,
        )


@invocation(
    "APG_Base",
    title="APG Base Implementation Extension",
    tags=["APG", "Extension"],
    category="extension",
    version="1.0.0",
)
class APGBaseExtensionInvocation(BaseInvocation):
    """
    APG Base Implementation Extension
    """
    guidance_scale: float = InputField(default=9.0, gt=0, description="Guidance scale", title="Guidance Scale", ui_order=0)
    eta: float = InputField(default=1.0, description="Eta", title="Eta", ui_order=1)
    norm_threshold: float = InputField(default=15, ge=0, description="Norm threshold", title="Norm Threshold", ui_order=2)
    momentum: float = InputField(default=-0.05, description="Momentum", title="Momentum", ui_order=3)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = dict(
            guidance_scale=self.guidance_scale,
            eta=self.eta,
            norm_threshold=self.norm_threshold,
            momentum=self.momentum,
        )
        guidance = GuidanceField(
            guidance_name="APG_Base",
            #priority=1000,
            extension_kwargs=kwargs,
        )
        return GuidanceDataOutput(guidance_data_output=guidance)