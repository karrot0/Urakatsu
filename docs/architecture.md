# Architecture

How Urakatsu is put together: the code layout, how the pipeline runs, the service and its
API, the web UI, and how to add a provider.

## Code layout

```
urakatsu/
├── core/        stages and operations, result types, TOML config, provider registry
├── runtimes/    GPU discovery (NVIDIA, AMD, Intel, Vulkan), model store, Torch / ONNX
│                runtimes, and the llama.cpp / stable-diffusion.cpp release manager and
│                server processes
├── providers/   one package per stage; each provider is a class registered with @register
│   ├── detection/   koharu-layout-rfdetr-seg-2xl (Torch)
│   ├── ocr/         paddleocr-vl-1.6 (llama.cpp or Torch), manga-ocr, baberu-ocr, hayai-ocr
│   ├── inpainting/  lama, aot (Torch); flux2-klein, rorem-mixed, any catalog diffusion model
│   │                (stable-diffusion.cpp)
│   └── translation/ local GGUF models (llama.cpp) and hosted APIs (Claude, OpenAI, DeepL, ...)
├── scene/       scene model: entities, components, relations, undo
├── pipeline/    project store (Project, then Chapters, then Pages), scheduler, stage processors
├── renderer/    typesetting and layout (HarfBuzz shaping, Skia drawing)
├── catalog/     the Hugging Face model catalog (see models.md)
├── sources/     source systems: Suwayomi Server and Mana (see sources.md)
├── desktop/     the desktop app window and its first-start GPU setup (see app.md)
└── service/     proto contract, gRPC servicer, gRPC-web bridge, JSON API, CLI
web/             the workspace UI (Vite, React, Tailwind and Base UI)
packaging/       the installer and launcher (Rust; Windows, macOS, Linux) and its build script
tests/           offline tests, plus real-weight tests (KOHARU_ROOT, URAKATSU_TEST_HOME)
```

## The pipeline

Urakatsu keeps Koharu's stage semantics:

- A stage runs on one page. OCR needs detection, translation needs OCR, inpainting needs
  detection, and rendering needs translation and inpainting. Pages enter in project order,
  and a page's later stages become eligible as soon as their prerequisites finish.
- At most one job per stage runs at a time, and accelerated stages share one admission lane.
  On out-of-memory the other models are unloaded and the job retries once; any other failure
  unloads the failing stage.
- Stopping is cooperative and discards the uncommitted result. Each committed stage saves the
  project.
- A user edit (`origin = user`) is never overwritten by a later generated result.
- Detection, inpainting and rendering are skipped when their output already exists, unless
  the request sets `force`.

Projects are organized as Project, then Chapters, then Pages (`urakatsu/pipeline/projects.py`).

Automations (`docs/automations.md`) run a different order on top of the same stages: every
page through one stage, unload, then the next stage (`urakatsu/pipeline/batch.py`).

## Service modes

| Command | What runs |
| --- | --- |
| `urakatsu serve` | The UI, the gRPC-web bridge and the JSON API on `--port` (8090), plus native gRPC on `--grpc-port` (50051), in one process |
| `urakatsu listen` | API mode: the JSON API and native gRPC without the UI; binds `0.0.0.0` by default |
| `urakatsu run` | One-shot CLI: `urakatsu run page.png --out out/` |
| `urakatsu desktop` | The workspace in its own window (needs the `desktop` extra) |

`urakatsu providers`, `urakatsu devices` and `urakatsu config` print the registered
providers, the hardware and the active configuration.

## API

The JSON API lives under `/api/v1`, with OpenAPI docs at `/api/docs`. Highlights:

| Route | Purpose |
| --- | --- |
| `GET /api/v1/capabilities` | Device, runtimes, providers and their availability |
| `GET`, `PUT`, `PATCH /api/v1/config` | The pipeline configuration |
| `POST /api/v1/projects`, `POST /api/v1/projects/{id}/pages` (multipart) | Projects and pages |
| `POST /api/v1/projects/{id}/run`, then `GET /api/v1/jobs/{job}/events` | Run the pipeline, with progress as server-sent events |
| `PATCH /api/v1/projects/{id}/pages/{page}/regions/{region}` | Edit a block's text or typography |
| `POST /api/v1/process?operation=full` | Upload one image, get the rendered page back |

The gRPC contract is `urakatsu/service/proto/urakatsu/v1/urakatsu.proto`. The browser uses the
same contract over gRPC-web (`POST /urakatsu.v1.Urakatsu/<Method>`), served by an in-process
bridge, so no Envoy proxy is needed. Regenerate the stubs with `urakatsu codegen` (Python) and
`npm run generate` in `web/` (TypeScript).

## The web UI

`web/` is a Vite and React workspace styled with Tailwind and Base UI:

- a chapter sidebar with each chapter's pages;
- a zoomable canvas for editing text blocks (select, move, resize, draw, nudge with the arrow
  keys), with Source, Layers and Rendered views;
- a right panel with Properties (type and layers), Activity and the Agent.

Themes are light, dark or system, with a choice of accent and neutral tone. `npm run dev`
serves the UI with hot reload and forwards API calls to the Python service
(`URAKATSU_API`, default `http://127.0.0.1:8090`).

## Writing a provider

Every model is a swappable provider behind a stage. A provider is a class registered with
`@register`:

```python
from urakatsu.core import register
from urakatsu.providers.base import OcrProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile


class MySettings(ProviderSettings):
    beam_size: int = 3


@register
class MyOcr(OcrProvider):
    id = "my-ocr"
    name = "My OCR"
    runtime = "onnx"                     # or "torch", or None
    requires = ("onnxruntime",)          # importable modules
    weights = (HuggingFaceFile("org/repo", "model.onnx", revision="main"),)
    Settings = MySettings

    def _load(self):
        (path,) = self.resolve_weights()
        self.session = self.runtime_instance().session(path)

    def _recognize(self, crops):
        ...  # return one OcrResult per crop
```

Register it from your own package with an entry point:

```toml
[project.entry-points."urakatsu.providers"]
my-ocr = "my_package.ocr:MyOcr"
```

The same pattern applies to `DetectionProvider`, `InpaintingProvider`, `RenderingProvider` and
`TranslationProvider`.
