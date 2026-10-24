"""Builds the Urakatsu installer for the system it runs on, into dist/.

    python packaging/build.py [--skip-ui] [--wheel PATH]

1. Builds the web UI (web/dist).
2. Builds the urakatsu wheel with the UI inside (urakatsu/service/ui, removed again afterwards).
3. Builds the Rust launcher (packaging/launcher), which embeds the wheel.
4. Packages it:
   - Windows: dist/Urakatsu-Setup-<version>.exe
   - macOS: dist/Urakatsu-<version>-macos-<arch>.dmg, holding Urakatsu.app (ad-hoc signed)
   - Linux: dist/Urakatsu-<version>-linux-<arch>.tar.gz, holding the Urakatsu binary

The result is one small binary that installs itself per user and sets up its Python
environment with uv on first start; see packaging/launcher/src/main.rs.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
UI_IN_PACKAGE = ROOT / "urakatsu" / "service" / "ui"
BUILD_INFO = ROOT / "urakatsu" / "_build.py"
LAUNCHER = ROOT / "packaging" / "launcher"
BUILD = ROOT / "build" / "desktop"
DIST = ROOT / "dist"


def run(*command: str, cwd: Path = ROOT, env: dict | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env, shell=sys.platform == "win32" and command[0] == "npm")


def arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)


def build_info() -> str:
    """``urakatsu/_build.py``: the update channel and commit this build is (see urakatsu/updates.py).

    The channel comes from URAKATSU_CHANNEL (release.yml passes stable or nightly), else "dev".
    """
    channel = os.environ.get("URAKATSU_CHANNEL") or "dev"
    commit = os.environ.get("GITHUB_SHA") or ""
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            commit = ""
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        '"""Written by packaging/build.py: which build this is, for update checks."""',
        "",
        f"CHANNEL = {channel!r}",
        f"COMMIT = {commit!r}",
        f"BUILT = {built!r}",
    ]
    return "\n".join(lines) + "\n"


def build_wheel(skip_ui: bool) -> Path:
    if not skip_ui:
        run("npm", "run", "build", cwd=WEB)
    if not (WEB / "dist" / "index.html").exists():
        raise SystemExit("web/dist is missing; build the UI first")
    wheels = BUILD / "wheel"
    shutil.rmtree(wheels, ignore_errors=True)
    shutil.rmtree(UI_IN_PACKAGE, ignore_errors=True)
    shutil.copytree(WEB / "dist", UI_IN_PACKAGE)
    BUILD_INFO.write_text(build_info(), encoding="utf-8")
    try:
        run("uv", "build", "--wheel", "--out-dir", str(wheels))
    finally:
        shutil.rmtree(UI_IN_PACKAGE, ignore_errors=True)
        BUILD_INFO.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "build" / "lib", ignore_errors=True)
    (wheel,) = wheels.glob("urakatsu-*.whl")
    return wheel


def build_launcher(wheel: Path) -> Path:
    run("cargo", "build", "--release", "--locked", cwd=LAUNCHER, env={**os.environ, "URAKATSU_WHEEL": str(wheel)})
    name = "urakatsu-launcher.exe" if sys.platform == "win32" else "urakatsu-launcher"
    return LAUNCHER / "target" / "release" / name


def package_windows(binary: Path, version: str) -> Path:
    target = DIST / f"Urakatsu-Setup-{version}.exe"
    shutil.copy2(binary, target)
    return target


def package_macos(binary: Path, version: str) -> Path:
    """Urakatsu.app (the bundle setup copies to ~/Applications) on a disk image."""
    stage = BUILD / "dmg"
    shutil.rmtree(stage, ignore_errors=True)
    app = stage / "Urakatsu.app" / "Contents"
    (app / "MacOS").mkdir(parents=True)
    (app / "Resources").mkdir()
    shutil.copy2(binary, app / "MacOS" / "Urakatsu")
    shutil.copy2(LAUNCHER / "urakatsu.icns", app / "Resources" / "urakatsu.icns")
    plist = (LAUNCHER / "Info.plist").read_text(encoding="utf-8").replace("{version}", version)
    (app / "Info.plist").write_text(plist, encoding="utf-8")
    run("codesign", "--force", "--deep", "--sign", "-", str(stage / "Urakatsu.app"))
    target = DIST / f"Urakatsu-{version}-macos-{arch()}.dmg"
    target.unlink(missing_ok=True)
    run("hdiutil", "create", "-volname", "Urakatsu", "-srcfolder", str(stage), "-ov", "-format", "UDZO", str(target))
    return target


def package_linux(binary: Path, version: str) -> Path:
    """The binary in a tarball, which keeps it executable (a plain download would not)."""
    target = DIST / f"Urakatsu-{version}-linux-{arch()}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        info = archive.gettarinfo(str(binary), arcname="Urakatsu")
        info.mode = 0o755
        with binary.open("rb") as handle:
            archive.addfile(info, handle)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-ui", action="store_true", help="reuse the existing web/dist")
    parser.add_argument("--wheel", type=Path, help="embed this wheel instead of building one")
    args = parser.parse_args()

    wheel = args.wheel.resolve() if args.wheel else build_wheel(args.skip_ui)
    version = wheel.name.split("-")[1]
    binary = build_launcher(wheel)
    DIST.mkdir(exist_ok=True)
    if sys.platform == "win32":
        target = package_windows(binary, version)
    elif sys.platform == "darwin":
        target = package_macos(binary, version)
    else:
        target = package_linux(binary, version)
    print(f"built {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
