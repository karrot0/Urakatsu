//! Windows: a per-user install with Start menu and desktop shortcuts and an "Installed apps"
//! entry, the data folder saved as the user's URAKATSU_HOME, and WebView2 for the window.

use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::Controls::{
    TASKDIALOGCONFIG, TD_ERROR_ICON, TDCBF_OK_BUTTON, TaskDialogIndirect,
};
use windows::Win32::UI::WindowsAndMessaging::{
    HWND_BROADCAST, SMTO_ABORTIFHUNG, SendMessageTimeoutW, WM_SETTINGCHANGE,
};
use windows::core::PCWSTR;
use winreg::RegKey;
use winreg::enums::HKEY_CURRENT_USER;

use crate::{Context, Res};

const UNINSTALL_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Urakatsu";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const DETACHED_PROCESS: u32 = 0x0000_0008;
const DRIVE_FIXED: u32 = 3;

pub const UV_URL: &str =
    "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";
/// Where the finished screen says the app can be opened from.
pub const MENU: &str = "the Start menu";
pub const DESKTOP_SHORTCUT: bool = true;
/// Windows paths ignore case.
pub const CASE_INSENSITIVE: bool = true;

pub fn install_dir() -> PathBuf {
    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| profile().join("AppData").join("Local"))
        .join("Programs")
        .join("Urakatsu")
}

pub fn installed_exe(install: &Path) -> PathBuf {
    install.join("Urakatsu.exe")
}

pub fn profile() -> PathBuf {
    env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\Public"))
}

pub fn python(env: &Path) -> PathBuf {
    env.join("Scripts").join("python.exe")
}

/// The interpreter that starts the app: `pythonw` opens no console window.
pub fn app_python(env: &Path) -> PathBuf {
    env.join("Scripts").join("pythonw.exe")
}

pub fn uv_name() -> &'static str {
    "uv.exe"
}

/// `uv.exe` out of the release zip.
pub fn uv_from_archive(archive: Vec<u8>) -> Res<Vec<u8>> {
    let mut zip =
        zip::ZipArchive::new(Cursor::new(archive)).context(|| "open the uv archive".to_owned())?;
    let name = zip
        .file_names()
        .filter_map(Result::ok)
        .find(|n| n.rsplit('/').next() == Some("uv.exe"))
        .map(|n| n.into_owned())
        .ok_or_else(|| "the uv archive has no uv.exe".to_owned())?;
    let mut bytes = Vec::new();
    zip.by_name(&name)
        .context(|| "read uv.exe".to_owned())?
        .read_to_end(&mut bytes)
        .context(|| "read uv.exe".to_owned())?;
    Ok(bytes)
}

pub fn write_executable(path: &Path, bytes: &[u8]) -> Res<()> {
    fs::write(path, bytes).context(|| format!("write {}", path.display()))
}

/// The filesystem of the volume holding `path` (or its nearest existing parent): "NTFS", "exFAT", ...
pub fn filesystem(path: &Path) -> Option<String> {
    use windows::Win32::Storage::FileSystem::{GetVolumeInformationW, GetVolumePathNameW};
    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent()?;
    }
    let wide_path: Vec<u16> = existing.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut root = [0u16; 261];
    let mut name = [0u16; 64];
    unsafe {
        GetVolumePathNameW(PCWSTR(wide_path.as_ptr()), &mut root).ok()?;
        GetVolumeInformationW(
            PCWSTR(root.as_ptr()),
            None,
            None,
            None,
            None,
            Some(&mut name),
        )
        .ok()?;
    }
    let len = name.iter().position(|c| *c == 0).unwrap_or(name.len());
    Some(String::from_utf16_lossy(&name[..len]))
}

/// No console window for a child process.
pub fn hidden(command: &mut Command) -> &mut Command {
    command.creation_flags(CREATE_NO_WINDOW)
}

/// A helper the launcher may stop with `kill_tree` (uv): no console window.
pub fn background(command: &mut Command) -> &mut Command {
    hidden(command)
}

/// The app outlives the launcher.
pub fn detached(command: &mut Command) -> &mut Command {
    command.creation_flags(DETACHED_PROCESS)
}

/// Stops a process and everything it started (uv and its build processes).
pub fn kill_tree(pid: u32) {
    let _ = hidden(Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"])).status();
}

pub fn alive(pid: u32) -> bool {
    use windows::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
    use windows::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };
    unsafe {
        let Ok(handle) = OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            false,
            pid,
        ) else {
            return false;
        };
        let running = WaitForSingleObject(handle, 0) == WAIT_TIMEOUT;
        let _ = CloseHandle(handle);
        running
    }
}

/// A last-resort error box, for failures before or without the setup window.
pub fn error_box(title: &str, text: &str) {
    let title: Vec<u16> = title.encode_utf16().chain(Some(0)).collect();
    let text: Vec<u16> = text.encode_utf16().chain(Some(0)).collect();
    let mut config = TASKDIALOGCONFIG {
        cbSize: size_of::<TASKDIALOGCONFIG>() as u32,
        dwCommonButtons: TDCBF_OK_BUTTON,
        pszWindowTitle: PCWSTR(title.as_ptr()),
        pszMainInstruction: PCWSTR(text.as_ptr()),
        ..Default::default()
    };
    config.Anonymous1.pszMainIcon = TD_ERROR_ICON;
    let _ = unsafe { TaskDialogIndirect(&config, None, None, None) };
}

/// Shows a file or folder in Explorer.
pub fn reveal(path: &Path) {
    let _ = Command::new("explorer").arg(path).spawn();
}

/// Copies this exe into the install folder, adds the shortcuts and the "Installed apps" entry.
/// Returns the installed exe.
pub fn install_app(install: &Path, version: &str, desktop: bool) -> Res<PathBuf> {
    fs::create_dir_all(install).context(|| format!("create {}", install.display()))?;
    let target = installed_exe(install);
    let staged = install.join("Urakatsu.exe.new");
    let me = env::current_exe().context(|| "find this installer".to_owned())?;
    fs::copy(&me, &staged).context(|| format!("copy to {}", staged.display()))?;
    fs::rename(&staged, &target)
        .context(|| "replace Urakatsu.exe (close Urakatsu and try again)".to_owned())?;

    for (link, always) in shortcut_paths() {
        if !always && !desktop {
            let _ = fs::remove_file(&link);
            continue;
        }
        if let Some(parent) = link.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let mut shortcut =
            mslnk::ShellLink::new(&target).context(|| "create a shortcut".to_owned())?;
        shortcut.set_working_dir(Some(install.display().to_string()));
        shortcut.set_name(Some("Urakatsu".to_owned()));
        shortcut
            .create_lnk(&link)
            .context(|| format!("write {}", link.display()))?;
    }

    let (key, _) = RegKey::predef(HKEY_CURRENT_USER)
        .create_subkey(UNINSTALL_KEY)
        .context(|| "register the app".to_owned())?;
    let exe = target.display().to_string();
    let values: [(&str, String); 7] = [
        ("DisplayName", "Urakatsu".to_owned()),
        ("DisplayVersion", version.to_owned()),
        ("Publisher", "Urakatsu".to_owned()),
        ("DisplayIcon", exe.clone()),
        ("InstallLocation", install.display().to_string()),
        ("UninstallString", format!("\"{exe}\" --uninstall")),
        (
            "QuietUninstallString",
            format!("\"{exe}\" --uninstall --quiet"),
        ),
    ];
    for (name, value) in values {
        key.set_value(name, &value)
            .context(|| format!("write {name}"))?;
    }
    let size = (fs::metadata(&target).map(|m| m.len()).unwrap_or(0) / 1024) as u32;
    for (name, value) in [
        ("NoModify", 1u32),
        ("NoRepair", 1u32),
        ("EstimatedSize", size),
    ] {
        key.set_value(name, &value)
            .context(|| format!("write {name}"))?;
    }
    Ok(target)
}

/// Removes the shortcuts and the "Installed apps" entry, then the install folder once this
/// process (which runs from it) has exited.
pub fn uninstall_app(install: &Path) {
    for (link, _) in shortcut_paths() {
        let _ = fs::remove_file(link);
    }
    let _ = RegKey::predef(HKEY_CURRENT_USER).delete_subkey_all(UNINSTALL_KEY);
    let folder = install.display().to_string();
    // raw_arg: cmd does not understand the C runtime's \" escaping that .arg() would add.
    let _ = hidden(Command::new("cmd").raw_arg(format!(
        "/C ping 127.0.0.1 -n 4 > nul & rmdir /S /Q \"{folder}\""
    )))
    .current_dir(env::temp_dir())
    .spawn();
}

/// (shortcut, always created): the Start menu entry, and the optional desktop shortcut.
fn shortcut_paths() -> Vec<(PathBuf, bool)> {
    let mut links = Vec::new();
    if let Some(appdata) = env::var_os("APPDATA") {
        links.push((
            PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs\Urakatsu.lnk"),
            true,
        ));
    }
    // The desktop may be redirected (OneDrive); Explorer records where.
    let folders = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey(r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders");
    let desktop: Option<String> = folders.ok().and_then(|key| key.get_value("Desktop").ok());
    links.push((
        desktop
            .map(PathBuf::from)
            .unwrap_or_else(|| profile().join("Desktop"))
            .join("Urakatsu.lnk"),
        false,
    ));
    links
}

/// The installed app from its "Installed apps" entry: `{version, location}`.
pub fn installed() -> Option<Value> {
    let key = RegKey::predef(HKEY_CURRENT_USER).open_subkey(UNINSTALL_KEY).ok()?;
    let location: String = key.get_value("InstallLocation").ok()?;
    if !installed_exe(Path::new(&location)).exists() {
        return None; // removed by hand: install as new
    }
    let version: String = key.get_value("DisplayVersion").unwrap_or_default();
    Some(json!({ "version": version, "location": location }))
}

/// URAKATSU_HOME as saved for the user, which a window started before the change does not see.
pub fn saved_home() -> Option<PathBuf> {
    let value: String = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey("Environment")
        .ok()?
        .get_value("URAKATSU_HOME")
        .ok()?;
    (!value.is_empty()).then(|| PathBuf::from(value))
}

/// Saves the data folder as the user's URAKATSU_HOME, which every part of Urakatsu reads (the
/// Python side included), and tells running programs the environment changed.
pub fn save_home(home: &Path) -> Res<()> {
    let environment = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey_with_flags("Environment", winreg::enums::KEY_SET_VALUE)
        .context(|| "open your environment settings".to_owned())?;
    environment
        .set_value("URAKATSU_HOME", &home.display().to_string())
        .context(|| "save the data folder".to_owned())?;
    let area: Vec<u16> = "Environment".encode_utf16().chain(Some(0)).collect();
    unsafe {
        SendMessageTimeoutW(
            HWND_BROADCAST,
            WM_SETTINGCHANGE,
            WPARAM(0),
            LPARAM(area.as_ptr() as isize),
            SMTO_ABORTIFHUNG,
            2000,
            None,
        );
    }
    Ok(())
}

/// The top of each local fixed drive, where earlier installs may have put a data folder.
pub fn drive_roots() -> Vec<PathBuf> {
    use windows::Win32::Storage::FileSystem::{GetDriveTypeW, GetLogicalDrives};
    let drives = unsafe { GetLogicalDrives() };
    let mut roots = Vec::new();
    for index in 0..26u32 {
        if drives & (1 << index) == 0 {
            continue;
        }
        let root = format!("{}:\\", (b'A' + index as u8) as char);
        let wide: Vec<u16> = root.encode_utf16().chain(Some(0)).collect();
        if unsafe { GetDriveTypeW(PCWSTR(wide.as_ptr())) } == DRIVE_FIXED {
            roots.push(PathBuf::from(root)); // network and removable drives can be slow or absent next time
        }
    }
    roots
}

/// Windows' app theme, so the window opens in the right colors before the page paints.
pub fn dark_mode() -> Option<bool> {
    RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        .and_then(|key| key.get_value::<u32, _>("AppsUseLightTheme"))
        .map(|light| light == 0)
        .ok()
}

/// The hardware adapter with the most memory of its own, skipping NVIDIA's (`nvidia-smi`
/// reports it) and Microsoft's software renderer: AMD and Intel run on ROCm, Vulkan or DirectML.
pub fn other_gpu() -> Option<(String, u64)> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
    };

    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
    let mut best: Option<(String, u64)> = None;
    for index in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
            || matches!(desc.VendorId, 0x10DE | 0x1414)
        {
            continue;
        }
        let end = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        let name = String::from_utf16_lossy(&desc.Description[..end])
            .trim()
            .to_owned();
        let memory = desc.DedicatedVideoMemory as u64;
        if best.as_ref().is_none_or(|(_, most)| memory > *most) {
            best = Some((name, memory));
        }
    }
    best
}
