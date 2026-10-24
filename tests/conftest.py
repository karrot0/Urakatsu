import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch):
    """Keep every test away from the user's ~/.urakatsu."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("URAKATSU_HOME", str(home))
    monkeypatch.setenv("URAKATSU_OFFLINE", "1")
    yield home


@pytest.fixture
def synthetic_page_png() -> bytes:
    """A white page with two speech balloons containing dark text and some line art."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(image)
    # Panel border and a character silhouette (should not be detected as text).
    draw.rectangle((20, 20, 780, 1080), outline="black", width=4)
    draw.ellipse((300, 500, 520, 900), fill="black")
    draw.polygon([(150, 1050), (650, 1050), (560, 700), (240, 700)], fill="#222222")
    # Balloons
    draw.ellipse((60, 60, 420, 300), fill="white", outline="black", width=4)
    draw.ellipse((430, 320, 760, 560), fill="white", outline="black", width=4)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default(size=30)
    draw.multiline_text((110, 120), "HELLO THERE\nHOW ARE YOU\nTODAY?", fill="black", font=font, spacing=8)
    draw.multiline_text((470, 380), "I AM FINE\nTHANK YOU", fill="black", font=font, spacing=8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
