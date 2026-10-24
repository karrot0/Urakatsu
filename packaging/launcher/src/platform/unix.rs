//! macOS and Linux: a per-user install without administrator rights.
//!
//! - macOS: the app bundle goes to `~/Applications/Urakatsu.app`, where Launchpad and
//!   Spotlight find it. The bundle setup runs from (on the disk image) is copied whole, with
//!   its signature, and without the download's quarantine flag.
//! - Linux: the binary goes to `~/.local/share/urakatsu`, with an applications menu entry
//!   (`urakatsu.desktop`, with an Uninstall action) and optionally one on the desktop.
//!
//! Neither system keeps a user environment variable for apps started from the desktop, so
//! the data folder is saved in `~/.config/urakatsu/home`, which `urakatsu.core.config.home()`
//! reads too, and the launcher passes URAKATSU_HOME to the app it starts.

use std::env;
use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

use crate::{Context, Res};

#[cfg(target_os = "macos")]
pub const MENU: &str = "the Applications folder";
#[cfg(not(target_os = "macos"))]
pub const MENU: &str = "the applications menu";
/// macOS apps live in the Applications folder; a desktop alias is not the custom there.
pub const DESKTOP_SHORTCUT: bool = cfg!(not(target_os = "macos"));
/// macOS volumes ignore case by default; Linux filesystems do not.
pub const CASE_INSENSITIVE: bool = cfg!(target_os = "macos");

pub const UV_URL: &str = concat!(
    "https://github.com/astral-sh/uv/releases/latest/download/uv-",
    env!("URAKATSU_UV_TARGET"),
    ".tar.gz"
);

/// The applications menu icon (macOS takes the bundle's urakatsu.icns).
#[cfg(not(target_os = "macos"))]
const ICON_PNG: &[u8] = include_bytes!("../../urakatsu.png");

pub fn profile() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn xdg(variable: &str, fallback: &str) -> PathBuf {
    env::var_os(variable)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| profile().join(fallback))
}

#[cfg(target_os = "macos")]
pub fn install_dir() -> PathBuf {
    profile().join("Applications").join("Urakatsu.app")
}

#[cfg(not(target_os = "macos"))]
pub fn install_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("urakatsu")
}

#[cfg(target_os = "macos")]
pub fn installed_exe(install: &Path) -> PathBuf {
    install.join("Contents").join("MacOS").join("Urakatsu")
}

#[cfg(not(target_os = "macos"))]
pub fn installed_exe(install: &Path) -> PathBuf {
    install.join("urakatsu")
}

pub fn python(env: &Path) -> PathBuf {
    env.join("bin").join("python")
}

pub fn app_python(env: &Path) -> PathBuf {
    python(env)
}

pub fn uv_name() -> &'static str {
    "uv"
}

/// `uv` out of the release tarball (`uv-<target>/uv`).
pub fn uv_from_archive(archive: Vec<u8>) -> Res<Vec<u8>> {
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(archive.as_slice()));
    for entry in tar.entries().context(|| "open the uv archive".to_owned())? {
        let mut entry = entry.context(|| "read the uv archive".to_owned())?;
        let is_uv = entry
            .path()
            .ok()
            .and_then(|p| p.file_name().map(|n| n == "uv"))
            .unwrap_or(false);
        if is_uv {
            let mut bytes = Vec::new();
            entry
                .read_to_end(&mut bytes)
                .context(|| "read uv".to_owned())?;
            return Ok(bytes);
        }
    }
    Err("the uv archive has no uv".to_owned())
}

pub fn write_executable(path: &Path, bytes: &[u8]) -> Res<()> {
    fs::write(path, bytes).context(|| format!("write {}", path.display()))?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .context(|| format!("make {} executable", path.display()))
}

/// The filesystem of the volume holding `path` (or its nearest existing parent): "exfat", "msdos", ...
pub fn filesystem(path: &Path) -> Option<String> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent()?;
    }
    let c_path = CString::new(existing.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    filesystem_name(&stat)
}

#[cfg(target_os = "macos")]
fn filesystem_name(stat: &libc::statfs) -> Option<String> {
    let name: Vec<u8> = stat
        .f_fstypename
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8(name).ok()
}

#[cfg(not(target_os = "macos"))]
fn filesystem_name(stat: &libc::statfs) -> Option<String> {
    // Only the ones that matter here: the slow FAT family, and NTFS as a known-good answer.
    let name = match stat.f_type as i64 {
        0x2011_BAB0 => "exFAT",
        0x4D44 => "FAT",
        0x5346_544E | 0x6573_5546 => "NTFS",
        _ => return None,
    };
    Some(name.to_owned())
}

/// Child processes have no console to hide.
pub fn hidden(command: &mut Command) -> &mut Command {
    command
}

/// A helper the launcher may stop with `kill_tree` (uv): its own process group, so the
/// group's id is its pid and stopping the group stops what it started too.
pub fn background(command: &mut Command) -> &mut Command {
    command.process_group(0)
}

/// Its own process group: the app outlives the launcher.
pub fn detached(command: &mut Command) -> &mut Command {
    command.process_group(0)
}

/// Stops a process started with `background` and everything it started.
pub fn kill_tree(pid: u32) {
    unsafe {
        libc::kill(-(pid as libc::pid_t), libc::SIGTERM);
    }
}

pub fn alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

/// A last-resort error box, for failures before or without the setup window.
pub fn error_box(title: &str, text: &str) {
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title(title)
        .set_description(text)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

/// Shows a file or folder in Finder or the file manager.
pub fn reveal(path: &Path) {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    let target = if cfg!(target_os = "macos") {
        path.to_path_buf()
    } else {
        path.parent().map(Path::to_path_buf).unwrap_or_default() // xdg-open opens files with an app
    };
    let _ = Command::new(opener).arg(target).spawn();
}

/// Copies the running app into the install location and registers it. Returns the installed
/// executable.
#[cfg(target_os = "macos")]
pub fn install_app(install: &Path, version: &str, _desktop: bool) -> Res<PathBuf> {
    let me = env::current_exe().context(|| "find this installer".to_owned())?;
    // Contents/MacOS/Urakatsu inside Urakatsu.app: copy the bundle, signature and all.
    let bundle = me
        .ancestors()
        .nth(3)
        .filter(|b| b.extension().is_some_and(|e| e == "app"));
    let staged = install.with_extension("app.new");
    let _ = fs::remove_dir_all(&staged);
    match bundle {
        Some(bundle) => copy_tree(bundle, &staged)?,
        None => {
            // Not inside a bundle (a development build): make a minimal one around it.
            let exe = installed_exe(&staged);
            fs::create_dir_all(exe.parent().unwrap())
                .context(|| format!("create {}", staged.display()))?;
            fs::copy(&me, &exe).context(|| format!("copy to {}", exe.display()))?;
            let resources = staged.join("Contents").join("Resources");
            fs::create_dir_all(&resources).context(|| format!("create {}", resources.display()))?;
            fs::write(resources.join("urakatsu.icns"), include_bytes!("../../urakatsu.icns"))
                .context(|| "write the icon".to_owned())?;
            fs::write(staged.join("Contents").join("Info.plist"), info_plist(version))
                .context(|| "write Info.plist".to_owned())?;
        }
    }
    // Downloads are quarantined, and copies keep the flag: Gatekeeper would ask again.
    let _ = Command::new("xattr")
        .args(["-dr", "com.apple.quarantine"])
        .arg(&staged)
        .status();
    if install.exists() {
        fs::remove_dir_all(install)
            .context(|| "replace Urakatsu.app (close Urakatsu and try again)".to_owned())?;
    }
    if let Some(parent) = install.parent() {
        fs::create_dir_all(parent).context(|| format!("create {}", parent.display()))?;
    }
    fs::rename(&staged, install).context(|| format!("move to {}", install.display()))?;
    write_record(install, version)?;
    Ok(installed_exe(install))
}

/// The bundle's Info.plist, the same file the release build uses (packaging/build.py).
#[cfg(target_os = "macos")]
fn info_plist(version: &str) -> String {
    include_str!("../../Info.plist").replace("{version}", version)
}

#[cfg(target_os = "macos")]
fn copy_tree(from: &Path, to: &Path) -> Res<()> {
    fs::create_dir_all(to).context(|| format!("create {}", to.display()))?;
    for entry in fs::read_dir(from).context(|| format!("read {}", from.display()))? {
        let entry = entry.context(|| format!("read {}", from.display()))?;
        let target = to.join(entry.file_name());
        let kind = entry.file_type().context(|| format!("read {}", entry.path().display()))?;
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_symlink() {
            let link = fs::read_link(entry.path()).context(|| format!("read {}", entry.path().display()))?;
            std::os::unix::fs::symlink(link, &target).context(|| format!("write {}", target.display()))?;
        } else {
            fs::copy(entry.path(), &target).context(|| format!("copy to {}", target.display()))?;
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn install_app(install: &Path, version: &str, desktop: bool) -> Res<PathBuf> {
    fs::create_dir_all(install).context(|| format!("create {}", install.display()))?;
    let target = installed_exe(install);
    let staged = install.join("urakatsu.new");
    let me = env::current_exe().context(|| "find this installer".to_owned())?;
    fs::copy(&me, &staged).context(|| format!("copy to {}", staged.display()))?;
    fs::set_permissions(&staged, fs::Permissions::from_mode(0o755))
        .context(|| format!("make {} executable", staged.display()))?;
    // A running binary can be replaced on Linux: the old one lives on until it exits.
    fs::rename(&staged, &target).context(|| format!("replace {}", target.display()))?;
    let icon = install.join("urakatsu.png");
    fs::write(&icon, ICON_PNG).context(|| format!("write {}", icon.display()))?;

    let entry = desktop_entry(&target, &icon);
    for (path, always) in entry_paths() {
        if !always && !desktop {
            let _ = fs::remove_file(&path);
            continue;
        }
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&path, &entry).context(|| format!("write {}", path.display()))?;
        // Desktops only launch entries that are executable (and, on GNOME, trusted).
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o755));
        if !always {
            let _ = Command::new("gio")
                .args(["set", "-t", "string"])
                .arg(&path)
                .args(["metadata::trusted", "true"])
                .status();
        }
    }
    let _ = Command::new("update-desktop-database")
        .arg(applications_dir())
        .status();
    write_record(install, version)?;
    Ok(target)
}

#[cfg(not(target_os = "macos"))]
fn desktop_entry(exe: &Path, icon: &Path) -> String {
    let exe = exe.display().to_string().replace('"', "\\\"");
    format!(
        "[Desktop Entry]\nType=Application\nName=Urakatsu\nComment=Translate and typeset manga on your computer\nExec=\"{exe}\"\nIcon={}\nTerminal=false\nCategories=Graphics;Office;\nStartupWMClass=Urakatsu\nActions=Uninstall;\n\n[Desktop Action Uninstall]\nName=Uninstall Urakatsu\nExec=\"{exe}\" --uninstall\n",
        icon.display()
    )
}

#[cfg(not(target_os = "macos"))]
fn applications_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("applications")
}

/// (entry, always created): the applications menu entry, and the optional desktop one.
#[cfg(not(target_os = "macos"))]
fn entry_paths() -> Vec<(PathBuf, bool)> {
    let desktop = Command::new("xdg-user-dir")
        .arg("DESKTOP")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| p.is_dir() && p != &profile())
        .unwrap_or_else(|| profile().join("Desktop"));
    vec![
        (applications_dir().join("urakatsu.desktop"), true),
        (desktop.join("urakatsu.desktop"), false),
    ]
}

/// Records the installed version beside the app, as the Windows "Installed apps" entry does.
fn write_record(install: &Path, version: &str) -> Res<()> {
    let record = record_path(install);
    let value = json!({ "version": version, "location": install.display().to_string() });
    fs::write(&record, value.to_string()).context(|| format!("write {}", record.display()))
}

#[cfg(target_os = "macos")]
fn record_path(install: &Path) -> PathBuf {
    install.join("Contents").join("Resources").join("install.json")
}

#[cfg(not(target_os = "macos"))]
fn record_path(install: &Path) -> PathBuf {
    install.join("install.json")
}

/// Removes the menu entries and the app. A running binary can be deleted on macOS and Linux.
pub fn uninstall_app(install: &Path) {
    #[cfg(not(target_os = "macos"))]
    for (path, _) in entry_paths() {
        let _ = fs::remove_file(path);
    }
    let _ = fs::remove_dir_all(install);
}

/// The installed app: `{version, location}`.
pub fn installed() -> Option<Value> {
    let install = install_dir();
    if !installed_exe(&install).exists() {
        return None; // removed by hand: install as new
    }
    let text = fs::read_to_string(record_path(&install)).ok()?;
    serde_json::from_str(&text).ok()
}

fn home_record() -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join("urakatsu").join("home")
}

/// The data folder chosen in an earlier setup.
pub fn saved_home() -> Option<PathBuf> {
    let value = fs::read_to_string(home_record()).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| PathBuf::from(value))
}

pub fn save_home(home: &Path) -> Res<()> {
    let record = home_record();
    if let Some(parent) = record.parent() {
        fs::create_dir_all(parent).context(|| format!("create {}", parent.display()))?;
    }
    fs::write(&record, home.display().to_string()).context(|| format!("write {}", record.display()))
}

/// Mounted drives, where earlier installs may have put a data folder.
pub fn drive_roots() -> Vec<PathBuf> {
    let user = env::var("USER").unwrap_or_default();
    let parents: Vec<PathBuf> = if cfg!(target_os = "macos") {
        vec![PathBuf::from("/Volumes")]
    } else {
        vec![
            PathBuf::from("/mnt"),
            PathBuf::from("/media").join(&user),
            PathBuf::from("/run/media").join(&user),
        ]
    };
    let mut roots = Vec::new();
    for parent in parents {
        let Ok(entries) = fs::read_dir(&parent) else { continue };
        for entry in entries.flatten() {
            // Links are skipped: on macOS the system volume appears as one, to /.
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                roots.push(entry.path());
            }
        }
    }
    roots
}

/// None: the window asks the system theme itself.
pub fn dark_mode() -> Option<bool> {
    None
}

/// A graphics card `nvidia-smi` does not report: Apple silicon's GPU, or an AMD or Intel GPU
/// from sysfs (Linux names the vendor, not the model).
pub fn other_gpu() -> Option<(String, u64)> {
    if cfg!(target_os = "macos") {
        return cfg!(target_arch = "aarch64").then(|| ("Apple silicon GPU".to_owned(), 0));
    }
    let cards = fs::read_dir("/sys/class/drm").ok()?;
    let mut vendors = Vec::new();
    for card in cards.flatten() {
        let name = card.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let vendor = fs::read_to_string(card.path().join("device").join("vendor")).unwrap_or_default();
        vendors.push(vendor.trim().to_owned());
    }
    let name = if vendors.iter().any(|v| v == "0x1002") {
        "AMD graphics card"
    } else if vendors.iter().any(|v| v == "0x8086") {
        "Intel graphics"
    } else {
        return None;
    };
    Some((name.to_owned(), 0))
}
