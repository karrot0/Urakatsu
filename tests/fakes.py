"""Test doubles: providers and stage processors that build scenes without weights."""

from __future__ import annotations

import io
from typing import Optional

from PIL import Image, ImageDraw

from urakatsu.core.registry import Registry
from urakatsu.core.stage import Stage
from urakatsu.pipeline.stages.base import StageInput, StageProcessor, generation
from urakatsu.providers.base import DetectionProvider, InpaintingProvider, OcrProvider, TranslationProvider
from urakatsu.scene import (
    At,
    AssetInput,
    Authored,
    Geometry,
    RasterLayer,
    RasterLayerKind,
    RegionKinds,
    Relations,
    SourceText,
    TextLayout,
    TextLayoutKind,
    Translation,
)
from urakatsu.scene import components as c


def page_png(width: int = 400, height: int = 300) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 200, 160), outline="black", width=3)
    draw.rectangle((90, 70, 150, 130), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeDetector(DetectionProvider):
    id = "fake-detector"
    name = "Fake detector"
    runtimes = ("torch", "onnx")


class FakeOcr(OcrProvider):
    id = "fake-ocr"
    name = "Fake OCR"

    def _recognize(self, crops):
        return ["テスト" for _ in crops]


class FakeTranslator(TranslationProvider):
    id = "fake-translator"
    name = "Fake translator"

    async def translate(self, request) -> list[str]:
        return [f"EN:{segment}" for segment in request]


class FakeInpainter(InpaintingProvider):
    id = "fake-inpainter"
    name = "Fake inpainter"


class FailingTranslator(FakeTranslator):
    id = "failing-translator"
    calls = 0
    oom_once = False

    async def translate(self, request) -> list[str]:
        type(self).calls += 1
        if self.oom_once and type(self).calls == 1:
            raise RuntimeError("CUDA out of memory while translating")
        if not self.oom_once:
            raise RuntimeError("model exploded")
        return [f"EN:{segment}" for segment in request]


def fake_registry() -> Registry:
    registry = Registry()
    registry._discovered = True
    for cls in (FakeDetector, FakeOcr, FakeTranslator, FakeInpainter, FailingTranslator):
        registry.add(cls)
    return registry


class FakeDetectionStage(StageProcessor):
    """Writes one text region, content, and layer per page."""

    def skip(self, input: StageInput, provider) -> bool:
        return not input.force and self.done(input)

    def done(self, input: StageInput) -> bool:
        return any(input.scene.region_kind(e) == RegionKinds.TEXT for e in input.scene.descendants(input.page))

    async def process(self, input: StageInput, provider):
        edit = input.scene.edit_as(generation("dev.koharu.pipeline.detection", provider.id))
        region = edit.add_analysis_region(input.page, At.END, RegionKinds.TEXT, Geometry.rectangle(90, 70, 60, 60))
        content = edit.add_text_content(input.page)
        layer = edit.add_text_layer(input.page, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
        edit.relate(Relations.RECOGNIZED_FROM, content, region)
        edit.relate(Relations.FITS_TO, layer, region)
        return edit.finish()


class FakeOcrStage(StageProcessor):
    async def process(self, input: StageInput, provider):
        edit = input.scene.edit_as(generation("dev.koharu.pipeline.ocr", provider.id))
        for layer in input.scene.text_layers(input.page):
            content = input.scene.layer_content(layer)
            text = (await provider.recognize([Image.new("RGB", (4, 4))]))[0]
            edit.set(content, SourceText(text=Authored.user(text)))
        return edit.finish()


class FakeTranslationStage(StageProcessor):
    async def process(self, input: StageInput, provider):
        contents = [input.scene.layer_content(layer) for layer in input.scene.text_layers(input.page)]
        sources = [input.scene.component(content, SourceText).text.value for content in contents]
        translated = await provider.translate(sources)
        edit = input.scene.edit_as(generation("dev.koharu.pipeline.translation", provider.id))
        for content, text in zip(contents, translated):
            edit.observe(content, SourceText)
            edit.set(content, Translation(text=Authored.user(text), language="en-US"))
        return edit.finish()


class FakeInpaintingStage(StageProcessor):
    def skip(self, input: StageInput, provider) -> bool:
        return not input.force and self.done(input)

    def done(self, input: StageInput) -> bool:
        return bool(input.scene.raster_layers(input.page, RasterLayerKind.CLEANUP))

    async def process(self, input: StageInput, provider):
        edit = input.scene.edit_as(generation("dev.koharu.pipeline.inpainting", provider.id))
        edit.observe_assets(input.page)
        layer = edit.add_entity(input.page, At.START)
        edit.set(layer, RasterLayer(name="Cleanup", kind=RasterLayerKind.CLEANUP))
        edit.set_asset(layer, "source", AssetInput(data=page_png(), media_type="image/png"))
        return edit.finish()


def fake_processors() -> dict[Stage, StageProcessor]:
    return {
        Stage.DETECTION: FakeDetectionStage(),
        Stage.OCR: FakeOcrStage(),
        Stage.TRANSLATION: FakeTranslationStage(),
        Stage.INPAINTING: FakeInpaintingStage(),
    }


def fake_config(translator: str = "fake-translator", runtimes: Optional[dict[str, str]] = None):
    from urakatsu.core.config import PipelineConfig

    config = PipelineConfig(device="cpu")
    config.detection.model = "fake-detector"
    config.ocr.model = "fake-ocr"
    config.translation.model.provider = translator
    config.inpainting.model = "fake-inpainter"
    config.runtimes = runtimes or {}
    return config


_ = c  # re-export convenience for tests
