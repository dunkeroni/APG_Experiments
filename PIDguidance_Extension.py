from invokeai.invocation_api import InputField, InvocationContext, BaseInvocation, invocation

from .exposed_denoise_latents import base_guidance_extension, GuidanceField, GuidanceDataOutput
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.util.logging import info, warning, error
import torch

from .APG_util import MomentumBuffer, normalized_guidance, project

@base_guidance_extension("PID_Guidance")
class PID_Guidance_Ext(ExtensionBase):
    def __init__(self, Kp: float, Ki: float, Kd: float, Perpendicular: float):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Perpendicular = Perpendicular
        self.prior_latent = None
        self.diff_integral = None

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def rescale_noise_pred(self, ctx: DenoiseContext):
        diff = ctx.positive_noise_pred - ctx.negative_noise_pred
        diff_parallel, diff_orthogonal = project(diff, ctx.latent_model_input) # diff relative to current latent image

        #derivative tensor
        if self.prior_latent is not None:
            diff_prior = (ctx.latent_model_input - self.prior_latent)*self.Kd
            diff_prior_parallel, diff_prior_orthogonal = project(diff_prior, ctx.latent_model_input)
        else:
            diff_prior_parallel = torch.zeros_like(diff_parallel)
        self.prior_latent = ctx.latent_model_input
        
        #integral tensor
        if self.diff_integral is None:
            self.diff_integral = torch.zeros_like(diff_parallel)
        self.diff_integral += diff_parallel*self.Ki

        #PID control
        ctx.noise_pred = ctx.negative_noise_pred + (diff_parallel*self.Kp + self.diff_integral - diff_prior_parallel + self.Perpendicular*diff_orthogonal)

@invocation(
    "PID_Guidance_Extension",
    title="PID Guidance Extension",
    tags=["PID", "Extension"],
    category="extension",
    version="1.0.0",
)
class PIDGuidanceExtensionInvocation(BaseInvocation):
    """
    PID Guidance Extension
    """
    Kp: float = InputField(default=7, description="Proportional Gain", title="Kp", ui_order=1)
    Ki: float = InputField(default=0.0, description="Integral Gain", title="Ki", ui_order=2)
    Kd: float = InputField(default=0.0, description="Derivative Gain", title="Kd", ui_order=3)
    Perpendicular: float = InputField(default=7, description="Perpendicular Gain", title="Perpendicular", ui_order=4)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = dict(
            Kp=self.Kp,
            Ki=self.Ki,
            Kd=self.Kd,
            Perpendicular=self.Perpendicular,
        )
        guidance = GuidanceField(
            guidance_name="PID_Guidance",
            extension_kwargs=kwargs,
        )
        return GuidanceDataOutput(guidance_data_output=guidance)