//! What differs between Windows, macOS and Linux: where the app is installed and how it is
//! registered with the desktop, the Python layout, processes, dialogs, the saved data folder,
//! and what the welcome screen says about the graphics card. `main.rs` and `discover.rs` stay
//! the same on every system.
//!
//! | | Windows | macOS | Linux |
//! | --- | --- | --- | --- |
//! | App | `%LOCALAPPDATA%\Programs\Urakatsu\Urakatsu.exe` | `~/Applications/Urakatsu.app` | `~/.local/share/urakatsu/urakatsu` |
//! | Menu | Start menu shortcut | the Applications folder | `urakatsu.desktop` entry |
//! | Registered | "Installed apps" entry | (the bundle) | `install.json` beside the app |
//! | Data folder saved in | the user's `URAKATSU_HOME` | `~/.config/urakatsu/home` | `~/.config/urakatsu/home` |

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use self::windows::*;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use self::unix::*;
