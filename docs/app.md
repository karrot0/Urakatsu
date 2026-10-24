# The desktop app

One small Rust program (`packaging/launcher`) is the installer and the launcher in one, on
Windows, macOS and Linux. Its window shows a page styled like the app in the system web view
(WebView2, WKWebView or WebKitGTK).

| System | Download | Installs to | Opens from |
| --- | --- | --- | --- |
| Windows | `Urakatsu-Setup-<version>.exe` | `%LOCALAPPDATA%\Programs\Urakatsu` | the Start menu, and a desktop shortcut if you want one |
| macOS (Apple silicon) | `Urakatsu-<version>-macos-arm64.dmg` | `~/Applications/Urakatsu.app` | the Applications folder, Launchpad and Spotlight |
| Linux (x86-64) | `Urakatsu-<version>-linux-x86_64.tar.gz` | `~/.local/share/urakatsu` | the applications menu, and a desktop entry if you want one |

## What setup does

- **Installs per user**, without administrator rights. On Windows it adds an "Installed apps"
  entry; on Linux the menu entry has an Uninstall action.
- **Finds earlier installs.** An installed version turns setup into Update (or Reinstall).
  Data folders other installs left (the saved data folder, `~/.urakatsu`, and `Urakatsu` or
  `.urakatsu` at the top of each local drive or mounted volume) are listed with their projects
  and models, ready to reuse.
- **Lets you choose the data folder** and the **models folder** (written to the data folder's
  `storage.json`, as Settings, then Pipeline, then Storage does). Setup shows how many models a
  folder already holds, and Urakatsu uses them as they are. Windows saves the data folder as
  the user's `URAKATSU_HOME`; macOS and Linux save it in `~/.config/urakatsu/home`, which
  `urakatsu serve` reads too.

The macOS app is ad-hoc signed but not notarized, so macOS asks once before its first start
(System Settings, then Privacy & Security, then Open Anyway). On Linux the window needs
WebKitGTK 4.1 (`libwebkit2gtk-4.1-0`), which desktop installs of current distributions have.

## The app environment

- The environment lives in `URAKATSU_HOME/app`: the newest uv (SHA-256 checked), a uv-managed
  Python 3.14, and the Urakatsu wheel embedded in the launcher, with its dependencies. It is
  rebuilt only when that wheel changes, so normal starts are instant.
- The Python install stays in the install folder, because uv's version links need a filesystem
  that holds them (not exFAT or FAT). On an exFAT or FAT data drive the environment also goes
  next to the app, because unpacking thousands of files there is very slow.
- The first start opens the app window (`python -m urakatsu.desktop`, pywebview; its Qt backend
  on Linux) and installs the PyTorch build for the GPU, with progress in the window
  (`urakatsu/desktop/gpu_setup.py`): the newest CUDA build the NVIDIA driver supports, AMD's
  ROCm build with the device libraries for the GPU's ISA, Intel's XPU build for Arc graphics,
  or the CPU build (which runs on Apple silicon's GPU through MPS). ONNX Runtime comes with
  CUDA for NVIDIA and with DirectML for any other GPU on Windows.

## Command-line options

| Option | Effect |
| --- | --- |
| `--uninstall` | Removes the app, and optionally your projects and models (what "Installed apps" runs on Windows) |
| `--quiet` | No window: installs without asking, or uninstalls keeping your data |
| `--dump-state <file>` | Writes what setup detected (installs, data folders, models) as JSON, for diagnosing |

`python -m urakatsu.desktop --background` starts the app minimized, or does nothing when it is
already open; the wake task of a scheduled automation runs it ([Automations](automations.md)).

## Building it

```bash
python packaging/build.py
```

This builds the web UI, packs it into the Urakatsu wheel, compiles the launcher with the wheel
embedded, and packages it for the system it runs on into `dist/`: the setup exe on Windows,
`Urakatsu.app` on a disk image on macOS, the binary in a tarball on Linux. It needs Node.js and
Rust, and on Linux `libwebkit2gtk-4.1-dev` and `libgtk-3-dev`. GitHub builds all three on every
change ([CI and releases](releases.md)).

The per-system parts of the launcher (install location, shortcuts, registration, processes,
dialogs) are in `packaging/launcher/src/platform/`; everything else is shared.

The app icon is designed in Figma and exported to `docs/images/icon.png`;
`python packaging/make_icon.py` rebuilds the `.ico`, `.icns` and `.png` files from it.
