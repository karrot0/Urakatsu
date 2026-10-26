# Runtimes

Urakatsu always runs the newest release of every runtime it uses.

| Runtime | How Urakatsu gets it | Used by |
| --- | --- | --- |
| PyTorch | pip: the newest CUDA build the NVIDIA driver supports, AMD's ROCm build, Intel's XPU build, or the CPU build | RF-DETR, manga-ocr, Baberu, Hayai, LaMa, AOT |
| ONNX Runtime | pip: `onnxruntime-gpu[cuda,cudnn]` for NVIDIA (CUDA and cuDNN from the `nvidia-*` wheels, or PyTorch's when its CUDA major matches), `onnxruntime-directml` for any other GPU on Windows | ONNX providers |
| llama.cpp | the newest official release, run as `llama-server` | PaddleOCR-VL (GGUF), local translation |
| stable-diffusion.cpp | the newest official release, run as `sd-server` | FLUX.2 Klein, RORem, catalog diffusion models |
| Bun | the newest official release, installed the first time a source system needs it | Mana sources |
| Suwayomi Server | the newest official release (it bundles its own Java), installed on first use | Tachiyomi and Mihon sources |
| Chromium (headless) | an installed Chrome, Edge or Brave, else the newest stable Chrome for Testing, installed the first time a site needs it | the Mana built-in browser |

`URAKATSU_BROWSER` picks the browser for Mana; `managed` forces Chrome for Testing.

## GPUs

Urakatsu finds every GPU without loading a GPU runtime (`urakatsu/runtimes/hardware.py` and
`gpus.py`): NVIDIA GPUs through `nvidia-smi`, the others through DXGI and the OpenCL loader on
Windows, sysfs and the KFD topology on Linux. As in Koharu, it picks one GPU in this order:
Metal or a discrete NVIDIA GPU, a discrete AMD GPU on ROCm, an integrated NVIDIA GPU, an
integrated AMD GPU on ROCm, then any GPU on Vulkan (discrete first). `urakatsu devices` lists
them; `device` in `pipeline.toml` overrides the choice (`cuda:1`, `rocm`, `vulkan`, `cpu`).

| GPU | PyTorch | llama.cpp, stable-diffusion.cpp | ONNX Runtime |
| --- | --- | --- | --- |
| NVIDIA | CUDA | CUDA | CUDA |
| AMD, ROCm-supported ISA | ROCm (AMD's wheels) | ROCm | DirectML (Windows) |
| AMD, other | CPU | Vulkan | DirectML (Windows) |
| Intel Arc | XPU (Windows) | Vulkan | DirectML (Windows) |
| Other, with Vulkan | CPU | Vulkan | DirectML (Windows) |
| Apple silicon | MPS | Metal | CoreML |

The ROCm ISAs are the ones the official llama.cpp and stable-diffusion.cpp builds compile for
(`urakatsu/runtimes/rocm.py`): RDNA 1 to 4 (`gfx1010` to `gfx1201`) on Windows and Linux, plus
the Instinct data center GPUs on Linux. Their Windows builds bundle the HIP runtime but load
hipBLAS and rocBLAS from the library path, which AMD's PyTorch wheels install
(`rocm[libraries]`); without those libraries llama.cpp and stable-diffusion.cpp use their
Vulkan builds. On an integrated GPU, models are rated against the shared memory.

## Native runtimes

llama.cpp and stable-diffusion.cpp are not Python packages: their bindings bundle snapshots
months behind upstream. Urakatsu instead installs each project's own release binaries into
`URAKATSU_HOME/runtimes`, verifies every file against GitHub's SHA-256 digest, keeps the
previous version, and talks to each model's server over localhost HTTP.

```bash
urakatsu runtimes          # installed versions
urakatsu runtimes check    # compare with the newest releases (exit code 1 if behind)
urakatsu runtimes update   # install the newest releases now
```

`urakatsu serve` also checks for new releases at startup and every six hours. With
`URAKATSU_OFFLINE=1` nothing is downloaded and the installed versions are used.

## Pinning a release

Latest is the default, not a requirement. Settings, then Runtimes, lists recent releases and
can pin llama.cpp or stable-diffusion.cpp to one of them, or roll back to the kept previous
install. A pinned runtime is never updated automatically; models it cannot load still appear
on the Models page, with a warning naming the release that can.
