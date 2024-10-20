from invokeai.invocation_api import InputField, InvocationContext, BaseInvocation, invocation

from .exposed_denoise_latents import base_guidance_extension, GuidanceField, GuidanceDataOutput
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.util.logging import info, warning, error
import torch

from .APG_util import MomentumBuffer, normalized_guidance, project

@base_guidance_extension("PerpNormCFG_Guidance")
class PID_Guidance_Ext(ExtensionBase):
    def __init__(self, limit:float):
        super().__init__()
        self.limit = limit

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def rescale_noise_pred(self, ctx: DenoiseContext):
        pos_parallel, pos_orthogonal = project(ctx.positive_noise_pred, ctx.latent_model_input) # diff relative to current latent image

        #limit perpendicular component if necessary
        par_norm = pos_parallel.norm()
        orth_norm = pos_orthogonal.norm()
        if orth_norm.item() > self.limit*par_norm.item():
            info(f"Perpendicular component norm: {orth_norm.item()}")
            info(f"Parallel component norm: {par_norm.item()}")
            pos_orthogonal = pos_orthogonal*self.limit*par_norm.item()/orth_norm.item()

        #Apply CFG
        ctx.noise_pred = ctx.negative_noise_pred + ((pos_parallel + pos_orthogonal) - ctx.negative_noise_pred) * ctx.inputs.conditioning_data.guidance_scale

@invocation(
    "PerpNormCFG_Guidance_Extension",
    title="Perpendicular CFG Normalization Extension",
    tags=["PerpNorm", "CFG", "Extension"],
    category="extension",
    version="1.0.0",
)
class PNCFGExtensionInvocation(BaseInvocation):
    """
    Perpendicular CFG Normalization Extension
    """
    limit: float = InputField(default=1, description="Proportional Limit of Perpendicular CFG Component", title="Limit", ui_order=1)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = dict(
            limit=self.limit,
        )
        guidance = GuidanceField(
            guidance_name="PerpNormCFG_Guidance",
            extension_kwargs=kwargs,
        )
        return GuidanceDataOutput(guidance_data_output=guidance)