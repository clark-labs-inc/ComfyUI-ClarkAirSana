# ComfyUI-ClarkAirSana

Run the **Clark Air Sana 1.6B** transformer — ternary (~1.58-bit) weights on real **GemLite INT2**
CUDA kernels — natively in ComfyUI under the standard **KSampler**. The `ClarkAirSanaLoader` node
builds ComfyUI_ExtraModels' `SanaMS` skeleton and swaps its attention + GLU-FFN trunk for GemLite
layers (~0.5 GB), leaving the small "island" layers in bf16.

## Quick start

1. In **ComfyUI-Manager**, install **Clark Air Sana** and **ComfyUI ExtraModels**, then restart ComfyUI.
2. **Download the model** → put it in `ComfyUI/models/clark_air_sana/`:
   **[clark_air_sana_gemlite_comfy.safetensors](https://huggingface.co/clark-labs/clark-air-sana-1.6b-gemlite-2bit/resolve/main/clark_air_sana_gemlite_comfy.safetensors)** (495 MB).
3. **⬇ Download the workflow:** **[example_workflow.json](https://github.com/clark-labs-inc/ComfyUI-ClarkAirSana/raw/main/example_workflow.json)** → drag it onto the ComfyUI canvas → press **Queue**.

(Gemma + the DC-AE VAE download themselves on the first run.) If you installed an older version and
see a `dtype 'bfloat16' not in [...]` error, update the node in ComfyUI-Manager and re-open the
workflow — or just set the **GemmaLoader** and **ExtraVAELoader** `dtype` dropdowns to **BF16**.

## Requirements

- **NVIDIA CUDA GPU** — GemLite INT2 is CUDA-only.
- [`ComfyUI_ExtraModels`](https://github.com/lawrence-cj/ComfyUI_ExtraModels) (provides Sana,
  the `GemmaLoader`/`GemmaTextEncode` text encoder, and the DC-AE VAE loader). Pin a known-good
  commit.
- `pip install -r requirements.txt` (`gemlite`, `safetensors`, `numpy`).

## Install

**Easiest — ComfyUI-Manager:** open the Manager, search **Clark Air Sana**, and click Install.
Also install **ComfyUI ExtraModels** (it provides the Gemma text encoder and DC-AE VAE). Restart
ComfyUI.

**Or manually** (clone into `custom_nodes`, then install each pack's `requirements.txt`):

```
cd ComfyUI/custom_nodes
git clone https://github.com/lawrence-cj/ComfyUI_ExtraModels
git clone https://github.com/clark-labs-inc/ComfyUI-ClarkAirSana
pip install -r ComfyUI-ClarkAirSana/requirements.txt
pip install -r ComfyUI_ExtraModels/requirements.txt
```

**Standalone / portable ComfyUI (Windows):** there is no `pip` on PATH — call the bundled Python.
From the ComfyUI root (the folder that contains `python_embeded`):

```
.\python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\ComfyUI-ClarkAirSana\requirements.txt
.\python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\ComfyUI_ExtraModels\requirements.txt
```

(Installing both packs through **ComfyUI-Manager** does all of this for you — recommended for the
portable build.)

**Get the model:** make a folder `ComfyUI/models/clark_air_sana/` and download this one file into it:

> [clark_air_sana_gemlite_comfy.safetensors](https://huggingface.co/clark-labs/clark-air-sana-1.6b-gemlite-2bit/resolve/main/clark_air_sana_gemlite_comfy.safetensors) (495 MB)

The Gemma text encoder and DC-AE VAE download by themselves the first time you run the workflow.

**Footprint (~3.2 GB total).** The example workflow defaults to the small components:

| Component | Download | Note |
|---|---|---|
| transformer (this repo) | **495 MB** | ternary, GemLite INT2 + FP8 islands |
| Gemma text encoder | **~2.1 GB** | `unsloth/gemma-2-2b-it-bnb-4bit` (4-bit). Swap to `Efficient-Large-Model/gemma-2-2b-it` for fp16 (9.8 GB) if you prefer. |
| DC-AE VAE | 1.2 GB (fp32) | loaded in bf16 at runtime; drop a bf16 copy (624 MB) at the VAE path to halve the download. |

## Workflow

Download **[`example_workflow.json`](example_workflow.json)** and drag it onto the ComfyUI canvas
(or Workflow → Open), then press **Queue**. That is the whole graph, ready to run:

```
GemmaLoader ─┬─ GemmaTextEncode (positive) ─┐
             └─ GemmaTextEncode (negative) ─┤
ClarkAirSanaLoader ──────── MODEL ──────────┼─ KSampler ─ VAEDecode ─ SaveImage
ExtraVAELoader (dcae-f32c32-sana) ── VAE ────┘     │
ClarkAirSanaEmptyLatent ──────── LATENT ───────────┘
```

KSampler: **`euler` / `normal`, 20 steps, cfg 4.5, 512×512** (verified end-to-end through
ComfyUI's KSampler). The trunk runs on real GemLite INT2 kernels. `example_workflow_api.json` is
the same graph in API format for scripting.

![demo](demo.png)

## Notes / limitations

- **Keep the transformer GPU-resident.** GemLite holds packed codes as buffers; ComfyUI's lowvram
  weight-streaming can't move them. The loader pins the trunk on the GPU (`offload_device =
  load_device`); the ~0.5 GB trunk fits easily, and Gemma/VAE still offload normally. Avoid
  `--lowvram` for the transformer.
- The null-caption bank (`y_embedder.y_embedding`) isn't in the diffusers weights; it's unused at
  inference with a real negative prompt (the skeleton's value is left in place).
- Trunk math is identical to the standalone GemLite render; only scheduler conventions differ.
