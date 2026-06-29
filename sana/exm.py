"""ComfyUI glue for the Sana skeleton — vendored from ComfyUI_ExtraModels (Apache-2.0).

SanaLatent is the 32-channel DC-AE latent format; EXM_Sana tells ComfyUI not to build a UNet
(we attach our own diffusion_model) and to drive it as a FLOW model under the native KSampler.
"""
import comfy.supported_models_base
import comfy.model_base
import comfy.conds
from comfy.latent_formats import LatentFormat


class SanaLatent(LatentFormat):
    latent_channels = 32

    def __init__(self):
        self.scale_factor = 0.41407


class EXM_Sana(comfy.supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}
    latent_format = SanaLatent

    def __init__(self, model_conf):
        self.model_target = model_conf.get("target")
        self.unet_config = model_conf.get("unet_config", {})
        self.sampling_settings = model_conf.get("sampling_settings", {})
        self.latent_format = self.latent_format()
        self.unet_config["disable_unet_model_creation"] = True

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.FLOW


class EXM_Sana_Model(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cn_hint = kwargs.get("cn_hint", None)
        if cn_hint is not None:
            out["cn_hint"] = comfy.conds.CONDRegular(cn_hint)
        return out
