"""Builds Urakatsu's icon files from the app icon, docs/images/icon.png.

The icon is designed in Figma ("Urakatsu icon": a manga speech balloon with a "U" on an ink
tile, with a screentone in the accent) and exported there at 1024 px. Writes
urakatsu/desktop/urakatsu.ico and urakatsu.png (the app window on Windows and Linux) and, for the launcher,
packaging/launcher/urakatsu.ico (the Windows exe and its shortcuts), urakatsu.icns (the macOS
app bundle) and urakatsu.png (the Linux applications menu). Run after exporting a new icon;
the files are checked in.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "images" / "icon.png"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def main() -> None:
    icon = Image.open(SOURCE).convert("RGBA")
    images = [icon.resize((size, size), Image.LANCZOS) for size in SIZES]
    for target in (ROOT / "urakatsu" / "desktop" / "urakatsu.ico", ROOT / "packaging" / "launcher" / "urakatsu.ico"):
        target.parent.mkdir(parents=True, exist_ok=True)
        images[-1].save(target, format="ICO", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
        print("wrote", target)
    icon.resize((256, 256), Image.LANCZOS).save(ROOT / "urakatsu" / "desktop" / "urakatsu.png", format="PNG", optimize=True)
    launcher = ROOT / "packaging" / "launcher"
    icon.resize((1024, 1024), Image.LANCZOS).save(launcher / "urakatsu.icns", format="ICNS")
    icon.resize((256, 256), Image.LANCZOS).save(launcher / "urakatsu.png", format="PNG", optimize=True)
    print("wrote", launcher / "urakatsu.icns", "and", launcher / "urakatsu.png")


if __name__ == "__main__":
    main()
