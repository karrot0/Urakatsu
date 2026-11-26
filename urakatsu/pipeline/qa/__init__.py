"""QA: proofreading and layout checks of a translated page (not in Koharu).

``review_page`` runs both halves on one page of a snapshot and returns the patch the QA stage
commits (``pipeline/stages/qa.py``), with what it found and cost. Benchmarks can call it, or each
half on its own: ``layout.check_layout`` (no model) and ``proofread.proofread`` (one model call,
plus the rounds in which the model asks for context; with the meaning check, a reading call and
short calls per line, ``proofread``'s section notes).

What is written (on each text content, ``QaReview``; on the page, ``QaPage``):

- Layout findings, and the wording checks that need no model (``checks``: source text left in,
  glossary renderings, full-width forms), are recomputed on every pass; they cost nothing. A
  correction the model wrote for one of those findings is kept with it while the line is unchanged.
- Every model issue passes ``proofread.vet`` before it is recorded.
- A block is sent to the model only when its fingerprint (source, translation, language, the
  length its layout allows) differs from the one its review recorded, or on a forced run (Redo).
  An unchanged block keeps its earlier proofreading issues. Knowledge changes alone do not send
  a block again; Redo does.
- ``flag`` only records. ``fix`` also writes the correction of a major or critical issue as the
  block's translation, where that translation is generated: one the user wrote is only flagged.
  That includes a correction recorded on an earlier pass, so turning ``fix`` on after ``flag``
  needs no second reading by the model. The new translation stays the translation stage's (``Edit.set_as``), so re-translating
  replaces it like any generated translation; the review keeps the text it replaced (``original``)
  for as long as the block's translation is the one QA wrote (``written``).
- A block that no longer has anything to review (its translation cleared, no lettering laid out)
  loses its review, so no finding or correction outlives the text it was about.
  Contrary to the first design (a producer id of QA's own), this keeps the scene's ownership rule
  working: a value only its producer may replace would be one translation's Redo can no longer
  replace.
- Typography and geometry are never changed, generated or not: layout findings are flagged, and
  the wording half answers the ones wording can fix (overflow, small text) by shortening, which
  the renderer's auto-fit then lays out. Resizing or moving a block would fight the auto-fit and
  the balloon flows other blocks share, and a size that stands out is often deliberate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from urakatsu.core.config import QaConfig, TranslationConfig, TypesettingConfig
from urakatsu.providers.translation.language import Language
from urakatsu.renderer.renderer import Renderer
from urakatsu.scene import Authored, Generation, Knowledge, Patch, QaIssue, QaPage, QaReview, Snapshot, SourceText, Translation
from urakatsu.scene.components import is_user

from ..stages.sound_effects import is_sound_effect
from . import checks
from . import proofread as proofread_module
from .layout import KINDS as LAYOUT_KINDS, BlockLayout, check_layout, renderer_for
from .proofread import Line, Proofread, proofread

PRODUCER = "dev.urakatsu.pipeline.qa"
TRANSLATION_PRODUCER = "dev.koharu.pipeline.translation"  # stages/translation.py: fixes stay its own
FIXED_SEVERITIES = frozenset({"warning", "error"})  # major and critical


@dataclass
class PageReview:
    """``review_page``'s result."""

    patch: Patch
    summary: QaPage
    reviews: dict[str, QaReview] = field(default_factory=dict)
    layout: dict[str, BlockLayout] = field(default_factory=dict)
    proofread: Optional[Proofread] = None


def targets(scene: Snapshot, page: str, entities: Optional[set[str]] = None) -> list[tuple[str, str, str, str]]:
    """``(layer, content, source, translation)`` of the page's translated blocks, in layer order;
    sound effects left out, and only ``entities`` (text layers) when given."""
    found = []
    for layer in scene.text_layers(page):
        if entities is not None and layer not in entities:
            continue
        content = scene.layer_content(layer)
        source = scene.component(content, SourceText)
        translation = scene.component(content, Translation)
        if source is None or translation is None or not source.text.value.strip() or not translation.text.value.strip():
            continue
        if is_sound_effect(scene, content):
            continue
        found.append((layer, content, source.text.value, translation.text.value))
    return found


def can_proofread(provider) -> bool:
    can = getattr(provider, "can_complete", None)
    return bool(can and can())


def meaning_check(provider, config: QaConfig) -> bool:
    """Whether proofreading runs the meaning check: the setting, else the provider's default."""
    if config.meaning_check is not None:
        return config.meaning_check
    return bool(getattr(provider, "meaning_check", False))


def has_work(scene: Snapshot, page: str, entities: Optional[set[str]] = None) -> bool:
    """Whether the page has translated text to check, or a review of text that is gone to drop
    (cheap: no layout)."""
    for layer in scene.text_layers(page):
        if entities is not None and layer not in entities:
            continue
        content = scene.layer_content(layer)
        translation = scene.component(content, Translation)
        if (translation is not None and translation.text.value.strip()) or scene.has(content, QaReview):
            return True
    return False


async def review_page(
    scene: Snapshot,
    page: str,
    provider,
    config: QaConfig,
    *,
    entities: Optional[set[str]] = None,
    typesetting: Optional[TypesettingConfig] = None,
    renderer: Optional[Renderer] = None,
    force: bool = False,
) -> PageReview:
    """Checks ``page``'s layout and proofreads what changed since its last review with
    ``provider`` (a translation provider; one that cannot ``complete`` gets the layout half only).
    The provider is loaded here, and only when the model has something to read."""
    renderer = renderer or renderer_for(typesetting)
    # A third-party translation provider may not carry the translation settings.
    translation_config = getattr(provider.settings, "translation", None) or TranslationConfig()
    target = Language.coerce(translation_config.target_language)
    language = target.tag
    layout = await asyncio.to_thread(check_layout, scene, page, renderer)

    blocks = targets(scene, page, entities)
    stored = scene.project_component(Knowledge)
    entries = stored.entries if stored is not None and stored.enabled else []
    found = {content: checks.check_line(source, translation, entries, language) for _, content, source, translation in blocks}
    required = {content: checks.requirements(source, entries, language) for _, content, source, _ in blocks}
    style = checks.PageStyle(language=language, capitals=checks.capitals(translation for *_, translation in blocks))
    lines = [
        _line(n, content, source, translation, layout.get(content), required[content], found[content])
        for n, (_, content, source, translation) in enumerate(blocks, 1)
    ]
    previous = {line.content: scene.component(line.content, QaReview) for line in lines}
    send = [line for line in lines if force or previous[line.content] is None or previous[line.content].fingerprint != line.fingerprint(language)]

    reason = ""
    result: Optional[Proofread] = None
    if send and not can_proofread(provider):
        reason = f"{provider.name or provider.id} only translates, so it cannot proofread; only the layout was checked"
    elif send:
        await provider.load()
        budget = min(config.context_tokens, getattr(provider, "context_budget", config.context_tokens))
        result = await proofread(
            provider,
            scene,
            page,
            send,
            target=target,
            instructions=translation_config.instructions,
            budget=budget,
            fetch_rounds=config.fetch_rounds,
            max_tokens=config.max_tokens,
            meaning_check=meaning_check(provider, config),
        )
        reason = result.error
        if not result.error:
            proofread_module.vet(result, send, lines, found, required, style)

    generated = Generation(producer=PRODUCER, model=provider.id)
    edit = scene.edit_as(generated)
    reviewed = {line.content for line in send} if result is not None and not result.error else set()
    reviews: dict[str, QaReview] = {}
    fixes: dict[str, str] = {}
    for line in lines:
        old = previous[line.content]
        if line.content in reviewed:
            model_issues = (result.issues if result else {}).get(line.content, [])
            fingerprint: Optional[str] = line.fingerprint(language)
        elif old is not None and old.fingerprint == line.fingerprint(language):
            # The program's findings are recomputed below; one a fix answered stays, applied.
            model_issues = [issue.model_copy() for issue in old.issues if issue.kind not in LAYOUT_KINDS and (issue.kind not in checks.KINDS or issue.applied)]
            fingerprint = old.fingerprint
            # A finding the model completed on an earlier pass keeps its correction.
            for finding in found[line.content]:
                earlier = next((i for i in old.issues if i.kind == finding.kind and i.message == finding.message and i.suggestion), None)
                if finding.suggestion is None and earlier is not None:
                    finding.suggestion = earlier.suggestion
        else:
            model_issues, fingerprint = [], None
        language_issues = list(found[line.content]) + model_issues
        review = QaReview(issues=list(layout[line.content].issues if line.content in layout else []) + language_issues, fingerprint=fingerprint, **_past(old, line.translation))
        if config.mode == "fix":
            # Also a correction an earlier pass only recorded (flag mode, then fix): it is written
            # without asking the model again.
            fix = _fix(language_issues)
            translation = scene.component(line.content, Translation)
            if fix is not None and translation is not None and fix != translation.text.value and not is_user(translation.text.origin):
                fixes[line.content] = fix
                review.original = review.original or translation.text.value
                review.written = fix
                for issue in language_issues:
                    issue.applied = issue.suggestion == fix
        reviews[line.content] = review
    # Blocks the model does not read (sound effects) still get their layout findings.
    scoped = {scene.layer_content(layer) for layer in entities} if entities is not None else None
    for content, block in layout.items():
        if content in reviews or (scoped is not None and content not in scoped):
            continue
        old = scene.component(content, QaReview)
        previous[content] = old
        kept = [issue for issue in old.issues if issue.kind not in LAYOUT_KINDS and issue.kind not in checks.KINDS] if old else []
        translation = scene.component(content, Translation)
        reviews[content] = QaReview(issues=list(block.issues) + kept, fingerprint=old.fingerprint if old else None, **_past(old, translation.text.value if translation else ""))

    if fixes:
        written = Generation(producer=TRANSLATION_PRODUCER, model=provider.id)
        for content, text in fixes.items():
            translation = scene.component(content, Translation)
            edit.set_as(content, Translation(text=Authored.generated(text, written), language=translation.language or language), written)
        # The layout of the new wording, and the fingerprint the next pass compares against.
        layout = await asyncio.to_thread(check_layout, edit.snapshot(), page, renderer)
        for n, (_, content, source, _) in enumerate(blocks, 1):
            if content in fixes:
                review = reviews[content]
                # The program's findings on the new wording; the ones the fix answered stay, applied.
                kept = [issue for issue in review.issues if issue.kind not in LAYOUT_KINDS and (issue.kind not in checks.KINDS or issue.applied)]
                fresh = [issue for issue in checks.check_line(source, fixes[content], entries, language) if not any(k.kind == issue.kind for k in kept)]
                review.issues = list(layout[content].issues if content in layout else []) + fresh + kept
                review.fingerprint = _line(n, content, source, fixes[content], layout.get(content)).fingerprint(language)

    for content, review in reviews.items():
        if previous[content] is None or previous[content].model_dump(exclude={"origin"}) != review.model_dump(exclude={"origin"}):
            edit.set(content, review)
    # The run's blocks with nothing left to review drop theirs; the page's other blocks keep theirs
    # and still count.
    elsewhere = 0
    for layer in scene.text_layers(page):
        content = scene.layer_content(layer)
        stale = scene.component(content, QaReview) if content not in reviews else None
        if stale is None:
            continue
        if scoped is None or content in scoped:
            edit.remove(content, QaReview)
        else:
            elsewhere += len(stale.issues)

    old_summary = scene.component(page, QaPage)
    summary = QaPage(
        model=provider.id,
        proofread=result is not None and not result.error,
        reason=reason,
        sent=len(send) if result is not None else 0,
        unchanged=len(lines) - len(send),
        issues=sum(len(review.issues) for review in reviews.values()) + elsewhere,
        fixed=len(fixes),
        calls=result.calls if result else 0,
        prompt_tokens=result.prompt_tokens if result else 0,
        completion_tokens=result.completion_tokens if result else 0,
        tokens_estimated=bool(result and result.estimated),
        total_prompt_tokens=(old_summary.total_prompt_tokens if old_summary else 0) + (result.prompt_tokens if result else 0),
        total_completion_tokens=(old_summary.total_completion_tokens if old_summary else 0) + (result.completion_tokens if result else 0),
    )
    if old_summary is None or old_summary.model_dump(exclude={"origin"}) != summary.model_dump(exclude={"origin"}):
        edit.set(page, summary)
    return PageReview(patch=edit.finish(), summary=summary, reviews=reviews, layout=layout, proofread=result)


def _line(n: int, content: str, source: str, translation: str, block: Optional[BlockLayout], required=(), found=()) -> Line:
    return Line(
        n=n,
        content=content,
        source=source,
        translation=translation,
        max_chars=block.max_chars if block is not None else None,
        layout=block.wording_hint() if block is not None else "",
        use={requirement.source: requirement.rendering for requirement in required},
        found=[f"{issue.kind}: {issue.message}" for issue in found],
        open=[issue.kind for issue in found if issue.suggestion is None],
    )


def _past(old: Optional[QaReview], translation: str) -> dict[str, Optional[str]]:
    """``original`` and ``written`` of a block's new review: its earlier review's, while
    ``translation`` is still what QA wrote."""
    if old is None or old.written is None or old.written != translation:
        return {"original": None, "written": None}
    return {"original": old.original, "written": old.written}


def _fix(issues: list[QaIssue]) -> Optional[str]:
    """The correction to write: the most severe major or critical issue's, if it has one."""
    rank = {"error": 2, "warning": 1}
    candidates = [issue for issue in issues if issue.severity in FIXED_SEVERITIES and issue.suggestion]
    if not candidates:
        return None
    return max(candidates, key=lambda issue: rank[issue.severity]).suggestion


__all__ = ["PRODUCER", "PageReview", "can_proofread", "check_layout", "has_work", "proofread", "review_page", "targets"]
