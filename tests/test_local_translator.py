"""The local translator's chat template environment."""

from __future__ import annotations

from datetime import datetime

import jinja2
import pytest

from urakatsu.providers.translation.local import LocalTranslator

MESSAGES = [{"role": "system", "content": "Translate."}, {"role": "user", "content": "こんにちは"}]

# The head of tokenizer.chat_template from TheDrummer_Cydonia-24B-v4.3-GGUF.
MISTRAL_TEMPLATE = """{%- set today = strftime_now("%Y-%m-%d") %}
{%- set default_system_message = "The current date is " + today + "." %}
{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = default_system_message %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}
{%- for message in loop_messages %}
    {{- '[INST]' + message['content'] + '[/INST]' }}
{%- endfor %}"""


def render(template: str, messages: list[dict] = MESSAGES) -> str:
    translator = LocalTranslator()
    translator._template = template
    translator._bos = "<s>"
    translator._eos = "</s>"
    return translator._render(messages, enable_thinking=False)


def test_loop_controls_work_when_nothing_else_imported_the_extension(monkeypatch):
    # ``import jinja2`` alone does not set ``jinja2.ext``; another import used to hide this.
    monkeypatch.delattr(jinja2, "ext", raising=False)
    template = "{% for m in messages %}{% if loop.first %}{% continue %}{% endif %}{{ m.content }}{% break %}{% endfor %}"
    assert render(template) == "こんにちは"


def test_mistral_template_renders():
    assert render(MISTRAL_TEMPLATE) == "<s>[SYSTEM_PROMPT]Translate.[/SYSTEM_PROMPT][INST]こんにちは[/INST]"


def test_strftime_now_is_the_current_local_time():
    today = render(MISTRAL_TEMPLATE, [{"role": "user", "content": "hi"}])
    assert f"The current date is {datetime.now():%Y-%m-%d}." in today


def test_raise_exception_reaches_the_caller():
    with pytest.raises(ValueError, match="no tools"):
        render("{{ raise_exception('no tools') }}")


def test_knowledge_is_dropped_until_the_prompt_fits_the_window():
    from PIL import Image

    from urakatsu.providers.translation import local
    from urakatsu.providers.translation.backend import TranslationContext, TranslationRequest, TranslationTerm

    class Server:
        """Counts a character as a token; the room is set so only part of the context fits."""

        def strip_bos(self, text: str) -> str:
            return text

        def count_tokens(self, text: str) -> int:
            return len(text)

    translator = LocalTranslator()
    translator._template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    translator._server = Server()
    lines = [TranslationContext(source=f"line {i}", translation=f"Line {i}") for i in range(200)]
    request = TranslationRequest(
        segments=["こんにちは"],
        target_language="en-US",
        context=list(lines),
        glossary=[TranslationTerm(source="アリス", translation="Alice", note="A knight.")],
        image=Image.new("RGB", (4, 4)),
    )
    room = local.N_CTX - 1000 - local.FIT_MARGIN - local.IMAGE_RESERVE
    system, payload = translator._fit(request, enable_thinking=False, max_tokens=1000)
    assert len(system) + len(payload) <= room
    # The oldest lines go first; the newest context and the glossary stay.
    assert 0 < len(request.context) < len(lines) and request.context[-1] == lines[-1]
    assert request.glossary[0].note == "A knight."
