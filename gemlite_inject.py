"""Inject the GemLite INT2 (ternary) trunk into an ExtraModels SanaMS skeleton.

Self-contained (no clark-quantize dependency): unpacks the reference-keyed
`clark_air_sana_gemlite_comfy.safetensors` and replaces the 140 trunk Linear/Conv2d modules
with GemLite-backed wrappers, leaving islands (norms/embeds/modulation/depthwise conv) bf16.
CUDA + gemlite required.
"""
import numpy as np
import torch
import torch.nn as nn


# 7 quantized modules per block (reference layout), matching build_comfy_pack.
_BLOCK_QUANT = [
    "attn.qkv", "attn.proj",
    "cross_attn.q_linear", "cross_attn.kv_linear", "cross_attn.proj",
    "mlp.inverted_conv.conv", "mlp.point_conv.conv",
]


def reference_quant_modules(num_layers):
    return [f"blocks.{n}.{s}" for n in range(num_layers) for s in _BLOCK_QUANT]


def unpack2b(packed, out, inn):  # uint8 [out, in/4] -> ternary int8 [out, in]
    b = packed.astype(np.uint8); lev = np.empty((out, inn), np.int8)
    lev[:, 0::4] = b & 3; lev[:, 1::4] = (b >> 2) & 3; lev[:, 2::4] = (b >> 4) & 3; lev[:, 3::4] = (b >> 6) & 3
    return lev.astype(np.int8) - 1


class _GemliteLin(nn.Module):
    def __init__(self, gl): super().__init__(); self.gl = gl
    def forward(self, x):
        s = x.shape
        return self.gl(x.reshape(-1, s[-1]).half()).reshape(*s[:-1], -1).to(x.dtype)


class _GemliteConv1x1(nn.Module):  # 1x1 conv as a channel matmul
    def __init__(self, gl): super().__init__(); self.gl = gl
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.gl(x.permute(0, 2, 3, 1).reshape(-1, c).half())
        return y.reshape(b, h, w, -1).permute(0, 3, 1, 2).to(x.dtype)


def inject_gemlite(model, pack, dtype, device, num_layers=20):
    """Load islands into `model` and swap its trunk modules for GemLite kernels (in place)."""
    from gemlite import GemLiteLinear, DType
    if not torch.cuda.is_available():
        raise RuntimeError("ClarkAirSana needs a CUDA GPU — GemLite INT2 kernels are CUDA-only.")

    # 1) islands: every non-"::" tensor (FP8 weights upcast to the model dtype).
    islands = {k: v.to(dtype) for k, v in pack.items() if "::" not in k}
    missing, unexpected = model.load_state_dict(islands, strict=False)
    # `missing` is expected to be the quantized weights (filled next) + optional buffers.
    model.to(device=device, dtype=dtype)

    # 2) trunk: build GemLite from codes/scales/bias and replace the module.
    for ref in reference_quant_modules(num_layers):
        out_, inn_ = [int(x) for x in pack[ref + "::shape"].tolist()]
        code = unpack2b(pack[ref + "::codes2b"].numpy(), out_, inn_)
        scale = pack[ref + "::scale"].float()
        ng = scale.shape[1]; g = inn_ // ng
        wq = torch.from_numpy((code + 1).astype(np.uint8)).to(device)
        bias = pack.get(ref + ".bias")
        gl = GemLiteLinear(W_nbits=2, group_size=g, in_features=inn_, out_features=out_,
                           input_dtype=DType.FP16, output_dtype=DType.FP16)
        gl.pack(wq, scale.to(device), torch.ones(out_, ng, device=device),
                bias=(bias.to(device).half() if bias is not None else None), fma_mode=False)
        parent = model.get_submodule(ref.rsplit(".", 1)[0])
        leaf = ref.rsplit(".", 1)[1]
        is_conv = isinstance(getattr(parent, leaf), nn.Conv2d)
        parent.add_module(leaf, _GemliteConv1x1(gl) if is_conv else _GemliteLin(gl))
    return model
