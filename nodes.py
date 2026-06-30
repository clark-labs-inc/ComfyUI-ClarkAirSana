"""ComfyUI-ClarkAirSana — self-contained nodes for Clark Air Sana 1.6B (ternary ~1.58-bit,
GemLite INT2). No external custom-node dependency: the Sana skeleton, the Gemma text encoder,
and the DC-AE VAE are all provided here. CUDA + gemlite required.

The Sana model code under `sana/` and the ComfyUI glue (`sana/exm.py`) are vendored from
ComfyUI_ExtraModels (Apache-2.0) which adapts NVlabs/Sana — see NOTICE.
"""
import os

import torch

import folder_paths
import comfy.model_base
import comfy.model_management
import comfy.model_patcher

from .gemlite_inject import inject_gemlite
from .sana.exm import EXM_Sana, EXM_Sana_Model
from .sana.conf import sana_conf
from .sana.sana_multi_scale import SanaMS

folder_paths.add_model_folder_path(
    "clark_air_sana", os.path.join(folder_paths.models_dir, "clark_air_sana")
)

_TE_DTYPE = {"BF16": torch.bfloat16, "FP16": torch.float16, "FP32": torch.float32}
_MODEL_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _from_pretrained(cls, name, **kwargs):
    """Load from the local HF cache first — fully offline, no network HEAD check — and only
    reach the network if the files aren't cached yet. Without this, `from_pretrained` validates
    the cache against huggingface.co every run, which hangs when offline."""
    try:
        return cls.from_pretrained(name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(name, **kwargs)


# --------------------------------------------------------------------------- transformer
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
        from safetensors.torch import load_file

        torch_dtype = _MODEL_DTYPE[dtype]
        pack = load_file(folder_paths.get_full_path("clark_air_sana", pack_name))
        conf = EXM_Sana(sana_conf[model_variant])
        load_device = comfy.model_management.get_torch_device()
        model = EXM_Sana_Model(conf, model_type=comfy.model_base.ModelType.FLOW, device=load_device)
        model.diffusion_model = SanaMS(**conf.unet_config)
        inject_gemlite(model.diffusion_model, pack, torch_dtype, load_device,
                       num_layers=conf.unet_config.get("depth", 20))
        model.diffusion_model.dtype = torch_dtype
        model.diffusion_model.eval()
        # GemLite packed buffers must stay on the GPU — pin the trunk resident.
        return (comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=load_device),)


# --------------------------------------------------------------------------- latent
class ClarkAirSanaEmptyLatent:
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


# --------------------------------------------------------------------------- gemma text encoder
class ClarkAirGemmaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["unsloth/gemma-2-2b-it-bnb-4bit", "Efficient-Large-Model/gemma-2-2b-it"],),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["BF16", "FP16", "FP32"], {"default": "BF16"}),
            }
        }

    RETURN_TYPES = ("CLARKAIR_GEMMA",)
    FUNCTION = "load"
    CATEGORY = "ClarkAir/Sana"
    TITLE = "Clark Air Gemma Loader"

    def load(self, model_name, device, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        td = _TE_DTYPE[dtype]
        tokenizer = _from_pretrained(AutoTokenizer, model_name)
        tokenizer.padding_side = "right"
        model = _from_pretrained(AutoModelForCausalLM, model_name, torch_dtype=td)
        text_encoder = model.get_decoder()
        if device != "cpu":
            text_encoder = text_encoder.to(device)
        return ({"tokenizer": tokenizer, "text_encoder": text_encoder, "model": model},)


class ClarkAirGemmaEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True}), "gemma": ("CLARKAIR_GEMMA",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ClarkAir/Sana"
    TITLE = "Clark Air Gemma Encode"

    def encode(self, text, gemma):
        tokenizer = gemma["tokenizer"]
        text_encoder = gemma["text_encoder"]
        with torch.no_grad():
            tokens = tokenizer(text, max_length=300, padding="max_length", truncation=True,
                               return_tensors="pt").to(text_encoder.device)
            cond = text_encoder(tokens.input_ids, tokens.attention_mask)[0]
            cond = cond * tokens.attention_mask.unsqueeze(-1)
        return ([[cond, {}]],)


# --------------------------------------------------------------------------- DC-AE VAE
class ClarkAirVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (["mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                              "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"],),
            }
        }

    RETURN_TYPES = ("CLARKAIR_VAE",)
    FUNCTION = "load"
    CATEGORY = "ClarkAir/Sana"
    TITLE = "Clark Air DC-AE VAE Loader"

    def load(self, vae_name):
        from diffusers import AutoencoderDC

        device = comfy.model_management.get_torch_device()
        vae = _from_pretrained(AutoencoderDC, vae_name, torch_dtype=torch.bfloat16).to(device).eval()
        return (vae,)


class ClarkAirVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("CLARKAIR_VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ClarkAir/Sana"
    TITLE = "Clark Air DC-AE VAE Decode"

    def decode(self, samples, vae):
        # ComfyUI's SanaLatent already divides the KSampler output by scale_factor, so decode
        # the latent directly (matching ExtraModels' EXVAE); dividing again over-softens.
        device = next(vae.parameters()).device
        lat = samples["samples"].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            img = vae.decode(lat, return_dict=False)[0]
        img = ((img.float() + 1.0) / 2.0).clamp(0, 1)
        return (img.permute(0, 2, 3, 1).contiguous().cpu(),)


NODE_CLASS_MAPPINGS = {
    "ClarkAirSanaLoader": ClarkAirSanaLoader,
    "ClarkAirSanaEmptyLatent": ClarkAirSanaEmptyLatent,
    "ClarkAirGemmaLoader": ClarkAirGemmaLoader,
    "ClarkAirGemmaEncode": ClarkAirGemmaEncode,
    "ClarkAirVAELoader": ClarkAirVAELoader,
    "ClarkAirVAEDecode": ClarkAirVAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClarkAirSanaLoader": "Clark Air Sana Loader (GemLite INT2)",
    "ClarkAirSanaEmptyLatent": "Clark Air Sana Empty Latent",
    "ClarkAirGemmaLoader": "Clark Air Gemma Loader",
    "ClarkAirGemmaEncode": "Clark Air Gemma Encode",
    "ClarkAirVAELoader": "Clark Air DC-AE VAE Loader",
    "ClarkAirVAEDecode": "Clark Air DC-AE VAE Decode",
}
