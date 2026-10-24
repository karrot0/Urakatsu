"""Hardware fit estimates: will this model (and which quantization) run on this machine?

Estimates are deliberately simple and conservative, in the spirit of Hugging Face's
hardware-compatibility badges and LM Studio's offload hints:

- GGUF LLMs: weights + KV cache for the context the provider uses + compute overhead.
  The KV cache comes from the GGUF header (layers, KV heads, key/value width; only
  full-attention layers for hybrid models).
- Diffusion models: the sum of the components (diffusion model, text encoder, VAE) plus
  activations; stable-diffusion.cpp can keep weights in RAM and stream the active one to
  the GPU (``--offload-to-cpu``), which is the ``offload`` level.
- PyTorch / ONNX models: weights + a fixed runtime overhead; there is no partial offload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from urakatsu.runtimes.device import Device
from urakatsu.runtimes.hardware import SystemProfile, rank

GIB = 1024**3
VRAM_RESERVE = int(0.6 * GIB)  # desktop compositor, driver, other apps
RAM_SHARE = 0.85  # leave headroom for the OS and Urakatsu itself
LLM_OVERHEAD = int(0.5 * GIB)
DIFFUSION_ACTIVATIONS = int(1.5 * GIB)  # about 1024x1024 latents and VAE decode
TORCH_OVERHEAD = int(1.0 * GIB)

LEVELS = ("gpu", "partial", "offload", "cpu", "too_large")


@dataclass
class Fit:
    level: str  # gpu | partial | offload | cpu | too_large
    required: int  # bytes the model needs in total
    weights: int
    context: int = 0  # bytes of KV cache included in `required`
    gpu_share: float = 0.0  # fraction of the model that fits on the GPU (partial)
    notes: list[str] = field(default_factory=list)

    @property
    def runs(self) -> bool:
        return self.level != "too_large"


def usable_vram(devices: list[Device], system: SystemProfile) -> int:
    accelerators = [d for d in devices if d.is_accelerator]
    if system.unified_memory:
        return int(system.ram_total * 0.70)
    if not accelerators:
        return 0
    # The GPU the pipeline picks: a discrete GPU over an integrated one with more shared memory.
    best = min(accelerators, key=rank)
    return max(0, int(best.memory_total * 0.95) - VRAM_RESERVE)


def _cpu_notes(system: SystemProfile) -> list[str]:
    if system.cpu_features and "avx2" not in system.cpu_features and "neon" not in system.cpu_features:
        return ["This CPU lacks AVX2, so CPU inference will be very slow."]
    return []


def fit_llm(weights: int, kv: int, devices: list[Device], system: SystemProfile) -> Fit:
    required = weights + kv + max(LLM_OVERHEAD, int(weights * 0.08))
    vram = usable_vram(devices, system)
    ram = int(system.ram_available * RAM_SHARE)
    if vram and required <= vram:
        return Fit("gpu", required, weights, kv, 1.0)
    if vram and required <= vram + ram:
        share = vram / required
        return Fit("partial", required, weights, kv, share, [f"About {share:.0%} of the model fits on the GPU; the rest runs on the CPU."] + _cpu_notes(system))
    if required <= ram:
        notes = ["No GPU memory is available for it, so it runs on the CPU."] if vram else []
        return Fit("cpu", required, weights, kv, 0.0, notes + _cpu_notes(system))
    return Fit("too_large", required, weights, kv, 0.0, ["Needs more memory than this machine has free."])


def fit_diffusion(components: list[int], devices: list[Device], system: SystemProfile) -> Fit:
    """stable-diffusion.cpp runs the text encoder, the diffusion model and the VAE one after
    another, memory-maps their weights and fits each onto the GPU as it runs (``--auto-fit``):
    the largest component must fit in VRAM, and all of them in installed RAM (mapped weights
    are paged in from disk, so free RAM is not the limit)."""
    total = sum(components)
    largest = max(components, default=0)
    vram = usable_vram(devices, system)
    ram = int(system.ram_total * RAM_SHARE)
    if vram and total + DIFFUSION_ACTIVATIONS <= vram:
        return Fit("gpu", total + DIFFUSION_ACTIVATIONS, total, 0, 1.0)
    if vram and largest + DIFFUSION_ACTIVATIONS <= vram and total <= ram:
        return Fit("offload", total + DIFFUSION_ACTIVATIONS, total, 0, (largest + DIFFUSION_ACTIVATIONS) / (total + DIFFUSION_ACTIVATIONS), ["Weights stay in RAM and move to the GPU as each part runs (slower)."])
    if total + DIFFUSION_ACTIVATIONS <= ram:
        return Fit("cpu", total + DIFFUSION_ACTIVATIONS, total, 0, 0.0, ["Runs on the CPU; diffusion on a CPU takes minutes per page."] + _cpu_notes(system))
    return Fit("too_large", total + DIFFUSION_ACTIVATIONS, total, 0, 0.0, ["Needs more memory than this machine has free."])


def fit_dense(weights: int, devices: list[Device], system: SystemProfile) -> Fit:
    required = weights + TORCH_OVERHEAD
    vram = usable_vram(devices, system)
    if vram and required <= vram:
        return Fit("gpu", required, weights, 0, 1.0)
    if required <= int(system.ram_available * RAM_SHARE):
        return Fit("cpu", required, weights, 0, 0.0, ["Too large for the GPU, so it runs on the CPU."] if vram else [])
    return Fit("too_large", required, weights, 0, 0.0, ["Needs more memory than this machine has free."])


# ---- GGUF specifics -------------------------------------------------------------------------------

_QUANT = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q[1-8](?:_[A-Z0-9]+)*|BF16|F16|F32|FP16|FP8|MXFP4(?:_MOE)?|NVFP4|TQ[12]_0))(?=(?:-\d{5}-of-\d{5})?\.gguf$)"
)
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def quantization_of(filename: str) -> Optional[str]:
    match = _QUANT.search(filename.rsplit("/", 1)[-1])
    return match.group(1).upper().replace("FP16", "F16") if match else None


def is_projector(filename: str) -> bool:
    name = filename.rsplit("/", 1)[-1].lower()
    return name.startswith("mmproj") or "-mmproj" in name or "_mmproj" in name


def is_draft(filename: str) -> bool:
    """Speculative-decoding companions (MTP heads, draft models) shipped next to the main model."""
    lower = filename.lower()
    name = lower.rsplit("/", 1)[-1]
    return "mtp/" in lower or name.startswith(("mtp-", "mtp_")) or "-draft" in name or name.startswith("draft")


def group_quantizations(files: list[tuple[str, int]]) -> dict[str, list[tuple[str, int]]]:
    """Quantization label -> its files (several when sharded), ignoring projectors and draft models."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for path, size in files:
        if not path.lower().endswith(".gguf") or is_projector(path) or is_draft(path):
            continue
        label = quantization_of(path) or path.rsplit("/", 1)[-1].removesuffix(".gguf")
        groups.setdefault(label, []).append((path, size))
    return {label: sorted(items) for label, items in groups.items()}


def first_shard(files: list[tuple[str, int]]) -> str:
    for path, _ in files:
        match = _SHARD.search(path)
        if match is None or match.group(1) == "00001":
            return path
    return files[0][0]


def kv_cache_bytes(header: dict[str, Any], context: int) -> int:
    """KV cache size in bytes (f16) for ``context`` tokens, from a GGUF header."""
    arch = header.get("general.architecture")
    if not arch:
        return 0

    def get(name: str, default: Any = None) -> Any:
        return header.get(f"{arch}.{name}", default)

    layers = int(get("block_count", 0) or 0)
    heads = get("attention.head_count", 0) or 0
    kv_heads = get("attention.head_count_kv", heads) or heads
    embedding = int(get("embedding_length", 0) or 0)
    if isinstance(heads, list):
        heads = max(heads)
    per_layer_kv = kv_heads if isinstance(kv_heads, list) else [kv_heads] * layers
    head_dim = embedding // heads if heads else 0
    key = int(get("attention.key_length", head_dim) or head_dim)
    value = int(get("attention.value_length", key) or key)
    interval = int(get("full_attention_interval", 1) or 1)
    window = int(get("attention.sliding_window", 0) or 0)
    total = 0
    for index, kv in enumerate(per_layer_kv[:layers]):
        if not kv:
            continue
        if interval > 1 and (index + 1) % interval != 0:
            continue  # linear-attention / recurrent layer of a hybrid model: no growing KV cache
        tokens = min(context, window) if window and arch.startswith("gemma") and (index + 1) % 6 != 0 else context
        total += tokens * int(kv) * (key + value) * 2
    return total


HEADER_KEYS = {
    "general.architecture",
    "general.parameter_count",
    "*.block_count",
    "*.attention.head_count",
    "*.attention.head_count_kv",
    "*.attention.key_length",
    "*.attention.value_length",
    "*.embedding_length",
    "*.full_attention_interval",
    "*.attention.sliding_window",
    "*.context_length",
}


def header_keys(arch: str) -> set[str]:
    return {k.replace("*", arch) for k in HEADER_KEYS}


UNQUANTIZED = re.compile(r"(?i)^(b?f16|f32|fp16|fp32)$")


def best_quantization(fits: dict[str, Fit], sizes: dict[str, int]) -> Optional[str]:
    """Largest quantization that runs fully on the GPU; otherwise the largest partial offload that
    keeps at least three quarters on the GPU (else the one with the most on the GPU); else CPU.
    Unquantized weights (BF16 / F16 / F32) are only picked when no quantization fits as well:
    above Q8_0 the quality gain is negligible and the memory and speed cost is not."""
    quantized = {q: f for q, f in fits.items() if not UNQUANTIZED.search(q)}
    for level in ("gpu", "offload"):
        candidates = [q for q, f in quantized.items() if f.level == level] or [q for q, f in fits.items() if f.level == level]
        if candidates:
            return max(candidates, key=lambda q: sizes.get(q, 0))
    partial = [q for q, f in fits.items() if f.level == "partial"]
    if partial:
        mostly = [q for q in partial if fits[q].gpu_share >= 0.75]
        return max(mostly, key=lambda q: sizes.get(q, 0)) if mostly else max(partial, key=lambda q: fits[q].gpu_share)
    cpu = [q for q, f in fits.items() if f.level == "cpu"]
    return min(cpu, key=lambda q: sizes.get(q, 0)) if cpu else None
