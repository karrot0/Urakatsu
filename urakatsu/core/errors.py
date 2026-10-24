"""Error taxonomy shared by runtimes, providers, and the pipeline."""

from __future__ import annotations


class UrakatsuError(Exception):
    """Base class for every error the package raises on purpose."""


class ProviderUnavailable(UrakatsuError):
    """The provider's optional dependencies or runtime are missing."""


class ProjectBusy(UrakatsuError):
    """The project has a running job, so its structure cannot change."""


class ModelLoadError(UrakatsuError):
    """Weights could not be resolved or the network could not be built."""


class ProcessingError(UrakatsuError):
    """A stage ran but failed on its input."""


class OutOfMemory(ProcessingError):
    """Accelerator memory pressure; the pipeline unloads other stages and retries once."""


OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "not enough memory",
    "failed to allocate",
)


def is_out_of_memory(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OutOfMemory):
            return True
        message = str(current).lower()
        if any(marker in message for marker in OOM_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False
