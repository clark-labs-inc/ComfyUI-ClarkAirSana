"""ComfyUI-ClarkAirSana — load the GemLite INT2 (ternary ~1.58-bit) Sana 1.6B transformer as a
native ComfyUI MODEL, driven by the standard KSampler.

Builds ComfyUI_ExtraModels' SanaMS skeleton, then replaces its attention + GLU-FFN trunk with
GemLite INT2 kernels (gemlite_inject). Pair it with ExtraModels' GemmaLoader/GemmaTextEncode +
ExtraVAELoader (dcae-f32c32-sana) + EmptySanaLatentImage. CUDA + gemlite required.
"""
import importlib
import os
import sys

import torch
from safetensors.torch import load_file

import folder_paths
import comfy.model_base
import comfy.model_management
import comfy.model_patcher

from .gemlite_inject import inject_gemlite


def _extra_models():
    """Import ComfyUI_ExtraModels' Sana pieces, ensuring custom_nodes is importable."""
    cn = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../custom_nodes
    if cn not in sys.path:
        sys.path.insert(0, cn)
    loader = importlib.import_module("ComfyUI_ExtraModels.Sana.loader")
    conf = importlib.import_module("ComfyUI_ExtraModels.Sana.conf")
    sms = importlib.import_module("ComfyUI_ExtraModels.Sana.models.sana_multi_scale")
    return loader.EXM_Sana, loader.EXM_Sana_Model, conf.sana_conf, sms.SanaMS


# Models live in ComfyUI/models/clark_air_sana/
folder_paths.add_model_folder_path(
    "clark_air_sana", os.path.join(folder_paths.models_dir, "clark_air_sana")
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


class ClarkAirSanaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pack_name": (folder_paths.get_filename_list("clark_air_sana"),),
                "model_variant": (["SanaMS_1600M_P1_D20"], {"default": "SanaMS_1600M_P1_D20"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "ClarkAir/Sana"
    TITLE = "Clark Air Sana Loader (GemLite INT2)"

    def load(self, pack_name, model_variant, dtype):
        EXM_Sana, EXM_Sana_Model, sana_conf, SanaMS = _extra_models()
        torch_dtype = _DTYPES[dtype]
        pack = load_file(folder_paths.get_full_path("clark_air_sana", pack_name))

        conf = EXM_Sana(sana_conf[model_variant])
        load_device = comfy.model_management.get_torch_device()
        model = EXM_Sana_Model(conf, model_type=comfy.model_base.ModelType.FLOW, device=load_device)
        model.diffusion_model = SanaMS(**conf.unet_config)

        inject_gemlite(model.diffusion_model, pack, torch_dtype, load_device,
                       num_layers=conf.unet_config.get("depth", 20))
        model.diffusion_model.dtype = torch_dtype
        model.diffusion_model.eval()

        # GemLite packed buffers must stay on the GPU — pin the trunk resident by making the
        # offload device the load device (the ~0.5 GB trunk fits; Gemma/VAE still offload).
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=load_device
        )
        return (patcher,)


class ClarkAirSanaEmptyLatent:
    """A 32-channel (DC-AE) empty latent for Sana. Self-contained so the workflow does not
    depend on ExtraModels' EmptySanaLatentImage, which breaks on current ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "ClarkAir/Sana"
    TITLE = "Clark Air Sana Empty Latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros(
            [batch_size, 32, height // 32, width // 32],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "ClarkAirSanaLoader": ClarkAirSanaLoader,
    "ClarkAirSanaEmptyLatent": ClarkAirSanaEmptyLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClarkAirSanaLoader": "Clark Air Sana Loader (GemLite INT2)",
    "ClarkAirSanaEmptyLatent": "Clark Air Sana Empty Latent",
}
