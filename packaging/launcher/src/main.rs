//! Urakatsu's installer and launcher in one small binary, for Windows, macOS and Linux.
//!
//! - Run from anywhere but its install location, it shows the setup window and installs itself
//!   for the current user, with no administrator rights: on Windows in
//!   `%LOCALAPPDATA%\Programs\Urakatsu` with Start menu and desktop shortcuts and an
//!   "Installed apps" entry, on macOS as `~/Applications/Urakatsu.app`, on Linux in
//!   `~/.local/share/urakatsu` with an applications menu entry (see platform/).
//! - Run from the install location, it starts `python -m urakatsu.desktop` (`pythonw` on
//!   Windows) at once when the environment is current, and otherwise updates it first (with the
//!   same window). The environment lives in `URAKATSU_HOME/app`: the newest uv, a uv-managed
//!   Python, and the Urakatsu wheel embedded in this binary.
//! - Setup finds what earlier installs left (discover.rs): an installed version to update, data
//!   folders with projects and models to reuse, and a models folder can be chosen up front.
//! - `--uninstall` removes the app, and optionally your projects and models.
//! - `--quiet` skips the window: install without asking, uninstall keeping your data.
//! - `--dump-state <file>` writes what the setup window would show, for diagnosing detection.
//!
//! PyTorch is not installed here: the desktop app installs the build that matches the GPU
//! driver on first start, with progress in its own window.

#![windows_subsystem = "windows"]

mod discover;
mod platform;
mod ui;

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use ui::Ui;
const WHEEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/urakatsu.whl"));
const WHEEL_NAME: &str = env!("URAKATSU_WHEEL_NAME");
const VERSION: &str = env!("URAKATSU_VERSION");
/// The newest Python every dependency ships wheels for (onnxruntime-gpu stops at 3.14).
const PYTHON: &str = "3.14";
type Res<T> = Result<T, String>;

trait Context<T> {
    fn context(self, what: impl FnOnce() -> String) -> Res<T>;
}

impl<T, E: std::fmt::Display> Context<T> for Result<T, E> {
    fn context(self, what: impl FnOnce() -> String) -> Res<T> {
        self.map_err(|error| format!("{}: {error}", what()))
    }
}

// ---- paths ----------------------------------------------------------------------------------------

#[derive(Clone)]
struct Paths {
    install: PathBuf,
    /// uv-managed Pythons, inside the install folder (the uninstaller removes them with it).
    /// Not under URAKATSU_HOME: uv links Python versions with directory junctions, which
    /// exFAT and FAT drives (a common place for URAKATSU_HOME) cannot hold.
    pythons: PathBuf,
    home: PathBuf,
    /// Small records the app and launcher share (`desktop.json`): always `URAKATSU_HOMEpp`.
    state: PathBuf,
    /// The environment, uv and uv's cache. `URAKATSU_HOMEpp` on NTFS or ReFS; on other
    /// filesystems (exFAT, FAT) next to the app on the system drive, because unpacking
    /// PyTorch's thousands of files into one exFAT folder slows to a crawl.
    app: PathBuf,
    env: PathBuf,
    uv: PathBuf,
    log: PathBuf,
    /// A models folder picked in setup, saved to `storage.json` on install. None keeps the
    /// data folder's own choice (its `storage.json`, else `models\` inside it).
    models: Option<PathBuf>,
}

impl Paths {
    fn new() -> Self {
        // Same rule as urakatsu.core.config.home(): URAKATSU_HOME, else ~/.urakatsu. The saved
        // value counts too: a window opened before setup last changed it still has the old one.
        let home = env::var_os("URAKATSU_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(platform::saved_home)
            .unwrap_or_else(default_home);
        Self::with_home(home)
    }

    fn with_home(home: PathBuf) -> Self {
        let install = platform::install_dir();
        let app = if fast_filesystem(&home) {
            home.join("app")
        } else {
            install.join("app")
        };
        Paths {
            state: home.join("app"),
            pythons: install.join("python"),
            install,
            env: app.join("env"),
            uv: app.join("uv").join(platform::uv_name()),
            log: home.join("logs").join("launcher.log"),
            app,
            home,
            models: None,
        }
    }

    /// The models folder the app will use.
    fn models_dir(&self) -> PathBuf {
        self.models
            .clone()
            .unwrap_or_else(|| discover::models_dir_of(&self.home))
    }

    fn python(&self) -> PathBuf {
        platform::python(&self.env)
    }

    /// The interpreter that runs the app (`pythonw` on Windows: no console window).
    fn app_python(&self) -> PathBuf {
        platform::app_python(&self.env)
    }

    /// The installed launcher, which the shortcuts start.
    fn exe(&self) -> PathBuf {
        platform::installed_exe(&self.install)
    }

    fn marker(&self) -> PathBuf {
        self.env.join("urakatsu-launcher.json")
    }

    fn current(&self) -> bool {
        let sha = sha256_hex(WHEEL);
        self.app_python().exists()
            && fs::read_to_string(self.marker())
                .map(|m| m.contains(&sha))
                .unwrap_or(false)
    }

    /// Variables every child process gets. uv's Python and cache are Urakatsu's own, even when
    /// the user set global ones: the cache must sit on the environment's drive so uv can
    /// hardlink from it instead of copying gigabytes.
    fn child_env(&self) -> Vec<(String, String)> {
        vec![
            ("URAKATSU_HOME".to_owned(), self.home.display().to_string()),
            ("URAKATSU_UV".to_owned(), self.uv.display().to_string()),
            (
                "UV_PYTHON_INSTALL_DIR".to_owned(),
                self.pythons.display().to_string(),
            ),
            (
                "UV_CACHE_DIR".to_owned(),
                self.app.join("cache").display().to_string(),
            ),
            ("NO_COLOR".to_owned(), "1".to_owned()),
        ]
    }

    /// Why the environment is not in the data folder, for the welcome screen.
    fn storage_note(&self) -> Option<String> {
        if self.app.starts_with(&self.home) {
            return None;
        }
        // Windows names the drive (D:\); elsewhere a drive is a folder, so name the data folder.
        let place = if cfg!(windows) {
            self.home
                .components()
                .next()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            self.home.display().to_string()
        };
        let kind = platform::filesystem(&self.home).unwrap_or_else(|| "its filesystem".to_owned());
        Some(format!(
            "{place} uses {kind}, which is too slow for a Python environment, so it goes next to the app (about 6 GB). Models and projects stay in your data folder."
        ))
    }
}

/// The FAT family (exFAT on an external drive, say) handles big folders and hardlinks badly;
/// NTFS, ReFS, APFS, ext4 and unknown answers count as fine.
fn fast_filesystem(path: &Path) -> bool {
    platform::filesystem(path).is_none_or(|fs| {
        !matches!(
            fs.to_ascii_lowercase().as_str(),
            "exfat" | "fat" | "fat16" | "fat32" | "vfat" | "msdos"
        )
    })
}

fn profile() -> PathBuf {
    platform::profile()
}

fn default_home() -> PathBuf {
    profile().join(".urakatsu")
}

fn same_folder(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn log(paths: &Paths, message: &str) {
    if let Some(parent) = paths.log.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log)
    {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "[{now}] {message}");
    }
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---- entry ----------------------------------------------------------------------------------------

#[derive(Clone)]
enum Flow {
    Install { desktop: bool },
    Update,
}

fn main() {
    let paths = Paths::new();
    let args: Vec<String> = env::args().skip(1).collect();
    let quiet = args.iter().any(|a| a == "--quiet");
    let here = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_default();

    // Diagnostics: what the setup window would show, as JSON (`--dump-state <file>`).
    if let Some(at) = args.iter().position(|a| a == "--dump-state") {
        let state = welcome_state(&paths, true);
        if let Some(file) = args.get(at + 1) {
            let _ = fs::write(file, serde_json::to_string_pretty(&state).unwrap_or_default());
        }
        return;
    }

    if args.iter().any(|a| a == "--uninstall") {
        if app_running(&paths) {
            let message = "Urakatsu is open. Close its window, then uninstall again.";
            if quiet {
                log(&paths, message);
                std::process::exit(1); // scripted uninstalls get an exit code, not a dialog
            }
            return fatal(&paths, message);
        }
        if quiet {
            return uninstall_files(&paths, false);
        }
        let state = json!({ "screen": "uninstall", "home": paths.home.display().to_string() });
        return window(paths, state, None);
    }

    let installed_here = paths.exe().parent().is_some_and(|folder| same_folder(&here, folder));
    if installed_here {
        // An open app is never updated underneath; starting it again only focuses its window.
        if paths.current() || app_running(&paths) {
            match start_app(&paths) {
                Ok(()) => return,
                Err(error) => {
                    log(&paths, &format!("error: {error}"));
                    let state = error_state("Urakatsu could not start", &error, true);
                    return window(paths, state, Some(Flow::Update));
                }
            }
        }
        log(
            &paths,
            &format!("updating the app environment to {VERSION}"),
        );
        return window(
            paths,
            json!({ "screen": "progress", "title": "Updating Urakatsu", "lede": "Bringing its Python libraries up to date. This takes a minute or two." }),
            Some(Flow::Update),
        );
    }

    if quiet {
        let tracker = Tracker::silent();
        if let Err(error) = install_files(&paths, true, &tracker) {
            return fatal(&paths, &error);
        }
        let _ = Command::new(paths.exe()).current_dir(&paths.install).spawn();
        return;
    }
    let state = welcome_state(&paths, false);
    window(paths, state, None)
}

/// The welcome screen. Without `sizes` it is quick (counts only); the window shows that at
/// once and a background scan re-renders it with folder sizes.
fn welcome_state(paths: &Paths, sizes: bool) -> Value {
    let installed = discover::installed();
    let mode = match &installed {
        Some(found) if found["version"] == VERSION => "reinstall",
        Some(_) => "update",
        None => "install",
    };
    let models = paths.models_dir();
    json!({
        "screen": "welcome",
        "mode": mode,
        "version": VERSION,
        "installed": installed,
        "install": paths.install.display().to_string(),
        "home": paths.home.display().to_string(),
        "data": discover::scan_home(&paths.home, sizes),
        "models": discover::scan_models(&models, sizes),
        "models_default": discover::same_place(&models, &paths.home.join("models")),
        "others": discover::other_homes(&paths.home, &default_home(), sizes),
        "gpu": gpu_note(),
        "desktop_option": platform::DESKTOP_SHORTCUT,
        "storage": paths.storage_note(),
    })
}

/// Shows the welcome screen for `paths` at once, then again with folder sizes.
fn show_welcome(ui: &Ui, paths: &Paths) {
    ui.render(welcome_state(paths, false));
    show_sizes(ui, paths);
}

fn show_sizes(ui: &Ui, paths: &Paths) {
    *WELCOME.lock().unwrap() = Some(paths.home.display().to_string());
    *SIZES.lock().unwrap() = Some(paths.models_dir());
    let ui = ui.clone();
    let paths = paths.clone();
    thread::spawn(move || {
        let state = welcome_state(&paths, true);
        // Only if the user has not moved on (another folder, or setup started) meanwhile.
        if WELCOME.lock().unwrap().as_deref() == Some(&paths.home.display().to_string()) && SIZES.lock().unwrap().as_ref() == Some(&paths.models_dir()) {
            ui.render(state);
        }
    });
}

static WELCOME: Mutex<Option<String>> = Mutex::new(None);
static SIZES: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Runs the setup window. `start` begins a flow at once (updates and retries after errors).
fn window(paths: Paths, initial: Value, start: Option<Flow>) {
    let initial_screen = initial["screen"].as_str().unwrap_or_default().to_owned();
    let busy = Arc::new(AtomicBool::new(false));
    let last: Arc<Mutex<Option<Flow>>> = Arc::new(Mutex::new(start.clone()));
    let paths = Arc::new(Mutex::new(paths));
    let starting = start.filter(|_| initial["screen"] == "progress");
    let begin = {
        let busy = busy.clone();
        let paths = paths.clone();
        let last = last.clone();
        move |flow: Flow, ui: &Ui| {
            *WELCOME.lock().unwrap() = None; // a size scan still running must not redraw the welcome screen
            busy.store(true, Ordering::SeqCst);
            *last.lock().unwrap() = Some(flow.clone());
            let paths = paths.lock().unwrap().clone();
            let ui = ui.clone();
            let busy = busy.clone();
            thread::spawn(move || {
                run_flow(&paths, flow, &ui);
                busy.store(false, Ordering::SeqCst);
            });
        }
    };
    let mut started = false;
    let begin_first = begin.clone();
    ui::run(initial, move |command, ui, window| {
        if !started {
            started = true;
            if let Some(flow) = starting.clone() {
                begin_first(flow, ui);
            }
        }
        let current = paths.lock().unwrap().clone();
        match command.name.as_str() {
            "ready" => {
                if initial_screen == "welcome" {
                    show_sizes(ui, &current);
                }
            }
            "close" | "cancel" => {
                if busy.load(Ordering::SeqCst) {
                    cancel_child();
                    log(&current, "cancelled");
                }
                ui.exit();
            }
            "open-log" => platform::reveal(&current.log),
            "pick-home" => {
                let chosen = rfd::FileDialog::new()
                    .set_title("Choose where Urakatsu keeps its data")
                    .set_directory(&current.home)
                    .set_parent(window)
                    .pick_folder();
                if let Some(folder) = chosen {
                    let home = discover::home_for(folder);
                    let chosen = Paths::with_home(home);
                    show_welcome(ui, &chosen);
                    *paths.lock().unwrap() = chosen;
                }
            }
            "use-home" => {
                // A data folder an earlier install left: its projects, models and settings.
                if let Some(home) = command.data["path"].as_str().filter(|p| !p.is_empty()) {
                    let chosen = Paths::with_home(PathBuf::from(home));
                    show_welcome(ui, &chosen);
                    *paths.lock().unwrap() = chosen;
                }
            }
            "pick-models" => {
                let chosen = rfd::FileDialog::new()
                    .set_title("Choose where Urakatsu keeps its models")
                    .set_directory(discover::models_dir_of(&current.home))
                    .set_parent(window)
                    .pick_folder();
                if let Some(folder) = chosen {
                    let mut next = current.clone();
                    next.models = Some(discover::models_folder_in(folder));
                    show_welcome(ui, &next);
                    *paths.lock().unwrap() = next;
                }
            }
            "reset-models" => {
                let mut next = current.clone();
                next.models = Some(next.home.join("models"));
                show_welcome(ui, &next);
                *paths.lock().unwrap() = next;
            }
            "install" => {
                if app_running(&current) {
                    ui.render(json!({ "screen": "error", "title": "Urakatsu is open", "lede": "Close its window, then try again. Setup replaces the app it runs from.", "retry": true }));
                    *last.lock().unwrap() = Some(Flow::Install { desktop: command.data["desktop"].as_bool().unwrap_or(true) });
                    return;
                }
                let desktop = command.data["desktop"].as_bool().unwrap_or(true);
                begin(Flow::Install { desktop }, ui);
            }
            "retry" => {
                let flow = last.lock().unwrap().clone().unwrap_or(Flow::Update);
                begin(flow, ui);
            }
            "launch" => {
                let ui = ui.clone();
                thread::spawn(move || match start_app(&current) {
                    Ok(()) => ui.exit(),
                    Err(error) => ui.render(error_state("Urakatsu could not start", &error, false)),
                });
            }
            "uninstall" => {
                let everything = command.data["everything"].as_bool().unwrap_or(false);
                let ui = ui.clone();
                thread::spawn(move || {
                    let tracker = Tracker::new(
                        &ui,
                        "Uninstalling Urakatsu",
                        "",
                        &["Remove Urakatsu"],
                        &[1.0],
                    );
                    tracker.start(0, "");
                    uninstall_files(&current, everything);
                    tracker.finish(0);
                    ui.render(json!({ "screen": "done", "title": "Urakatsu was uninstalled", "lede": if everything { "Your projects and models were deleted too." } else { "Your projects and models are still in your data folder." }, "launch": false }));
                });
            }
            _ => {}
        }
    })
}

fn run_flow(paths: &Paths, flow: Flow, ui: &Ui) {
    let installing = matches!(flow, Flow::Install { .. });
    let (titles, weights): (Vec<&str>, Vec<f64>) = if installing {
        (
            vec![
                "Install the app",
                "Download uv",
                "Set up Python",
                "Install Urakatsu and its libraries",
            ],
            vec![3.0, 7.0, 15.0, 75.0],
        )
    } else {
        (
            vec![
                "Download uv",
                "Set up Python",
                "Install Urakatsu and its libraries",
            ],
            vec![7.0, 15.0, 78.0],
        )
    };
    let tracker = Tracker::new(
        ui,
        if installing {
            "Installing Urakatsu"
        } else {
            "Updating Urakatsu"
        },
        "About 1 GB of Python libraries. PyTorch for your graphics card follows on first start.",
        &titles,
        &weights,
    );
    tracker.hint(if installing {
        "Keep this window open until setup finishes."
    } else {
        "This window closes when Urakatsu opens."
    });
    let desktop_shortcut = matches!(flow, Flow::Install { desktop: true });
    let result = (|| {
        if let Flow::Install { desktop } = flow {
            install_files(paths, desktop, &tracker)?;
        }
        prepare(paths, &tracker, if installing { 1 } else { 0 })
    })();
    match result {
        Ok(()) if installing => ui.render(json!({
            "screen": "done",
            "title": if discover::installed().is_some_and(|i| i["version"] != VERSION) { "Urakatsu is updated" } else { "Urakatsu is installed" },
            "lede": format!("Open it now, or any time from {}.", platform::MENU),
            "launch": true,
            "rows": [
                { "title": "Shortcuts", "detail": shortcuts_note(desktop_shortcut) },
                { "title": "First start", "detail": "Sets up PyTorch for your graphics card (about 3 GB, once)." },
                { "title": "Data folder", "detail": paths.home.display().to_string(), "path": true },
                { "title": "Models folder", "detail": paths.models_dir().display().to_string(), "path": true },
            ],
        })),
        Ok(()) => match start_app(paths) {
            Ok(()) => ui.exit(),
            Err(error) => ui.render(error_state("Urakatsu could not start", &error, true)),
        },
        Err(error) if CANCELLED.load(Ordering::SeqCst) => log(paths, &format!("stopped: {error}")),
        Err(error) => {
            log(paths, &format!("error: {error}"));
            tracker.fail();
            let title = if installing { "Setup did not finish" } else { "The update did not finish" };
            ui.render(error_state(title, &error, true));
        }
    }
}

/// "Start menu and desktop", "Applications folder", ...
fn shortcuts_note(desktop: bool) -> String {
    let menu = platform::MENU.trim_start_matches("the ");
    let mut note = menu[..1].to_uppercase() + &menu[1..];
    if desktop && platform::DESKTOP_SHORTCUT {
        note.push_str(" and desktop");
    }
    note
}

fn error_state(title: &str, error: &str, retry: bool) -> Value {
    json!({
        "screen": "error",
        "title": title,
        "lede": "Nothing you made was changed. The log has the details.",
        "details": error,
        "retry": retry,
    })
}

/// A last-resort error box, for failures before or without the setup window.
fn fatal(paths: &Paths, error: &str) {
    log(paths, &format!("error: {error}"));
    platform::error_box(
        "Urakatsu",
        &format!("{error}\n\nDetails are in {}", paths.log.display()),
    );
}

/// One line about the graphics card for the welcome screen: NVIDIA's own report first, then the
/// system's (AMD and Intel run on ROCm, Vulkan or DirectML; Apple silicon on Metal).
fn gpu_note() -> String {
    match nvidia_gpu().or_else(platform::other_gpu) {
        Some((name, bytes)) => {
            let gb = if bytes >= 1 << 30 {
                format!(" ({:.0} GB)", bytes as f64 / (1u64 << 30) as f64)
            } else {
                String::new() // an integrated GPU: its own memory is a small carve-out
            };
            format!("{name}{gb} found. Urakatsu sets up GPU support for it on first start.")
        }
        None => "No graphics card found. Models will run on the CPU, which is slower.".to_owned(),
    }
}

fn nvidia_gpu() -> Option<(String, u64)> {
    let output = platform::hidden(Command::new("nvidia-smi").args([
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]))
    .output()
        .ok()
        .filter(|o| o.status.success())?;
    let text = String::from_utf8(output.stdout).ok()?;
    let (name, mib) = text.lines().next()?.rsplit_once(',')?;
    let mib = mib.trim().parse::<f64>().unwrap_or(0.0);
    Some((name.trim().to_owned(), (mib * 1024.0 * 1024.0) as u64))
}

// ---- progress -------------------------------------------------------------------------------------

/// The progress screen's state: steps with weights, the active step's share, the latest uv line.
struct Tracker {
    ui: Option<Ui>,
    title: String,
    hint: Mutex<String>,
    lede: String,
    titles: Vec<String>,
    weights: Vec<f64>,
    inner: Mutex<TrackerState>,
}

struct TrackerState {
    status: Vec<&'static str>,
    detail: Vec<String>,
    fraction: f64,
    line: String,
    sent: Option<Instant>,
}

impl Tracker {
    fn new(ui: &Ui, title: &str, lede: &str, titles: &[&str], weights: &[f64]) -> Self {
        let tracker = Tracker {
            ui: Some(ui.clone()),
            title: title.to_owned(),
            hint: Mutex::new(String::new()),
            lede: lede.to_owned(),
            titles: titles.iter().map(|t| t.to_string()).collect(),
            weights: weights.to_vec(),
            inner: Mutex::new(TrackerState {
                status: vec!["pending"; titles.len()],
                detail: vec![String::new(); titles.len()],
                fraction: 0.0,
                line: String::new(),
                sent: None,
            }),
        };
        tracker.push(true);
        tracker
    }

    fn silent() -> Self {
        Tracker {
            ui: None,
            title: String::new(),
            hint: Mutex::new(String::new()),
            lede: String::new(),
            titles: vec![String::new(); 8],
            weights: vec![1.0; 8],
            inner: Mutex::new(TrackerState {
                status: vec!["pending"; 8],
                detail: vec![String::new(); 8],
                fraction: 0.0,
                line: String::new(),
                sent: None,
            }),
        }
    }

    fn hint(&self, text: &str) {
        *self.hint.lock().unwrap() = text.to_owned();
        self.push(true);
    }

    fn start(&self, step: usize, detail: &str) {
        {
            let mut s = self.inner.lock().unwrap();
            s.status[step] = "active";
            s.detail[step] = detail.to_owned();
            s.fraction = 0.0;
        }
        self.push(true);
    }

    fn progress(&self, step: usize, fraction: f64, detail: &str) {
        {
            let mut s = self.inner.lock().unwrap();
            s.fraction = fraction.clamp(0.0, 1.0);
            s.detail[step] = detail.to_owned();
        }
        self.push(false);
    }

    fn line(&self, line: &str) {
        self.inner.lock().unwrap().line = line.trim().to_owned();
        self.push(false);
    }

    fn finish(&self, step: usize) {
        {
            let mut s = self.inner.lock().unwrap();
            s.status[step] = "done";
            s.detail[step] = String::new();
            s.fraction = 0.0;
        }
        self.push(true);
    }

    fn fail(&self) {
        let mut s = self.inner.lock().unwrap();
        if let Some(active) = s.status.iter().position(|st| *st == "active") {
            s.status[active] = "failed";
        }
    }

    /// Renders the screen; frequent updates (uv output) are limited to ten a second.
    fn push(&self, force: bool) {
        let Some(ui) = &self.ui else { return };
        let mut s = self.inner.lock().unwrap();
        if !force
            && s.sent
                .is_some_and(|t| t.elapsed() < Duration::from_millis(100))
        {
            return;
        }
        s.sent = Some(Instant::now());
        let total: f64 = self.weights.iter().sum();
        let mut done = 0.0;
        for (i, weight) in self.weights.iter().enumerate() {
            done += match s.status[i] {
                "done" => *weight,
                "active" => weight * s.fraction,
                _ => 0.0,
            };
        }
        let steps: Vec<Value> = self
            .titles
            .iter()
            .enumerate()
            .map(|(i, title)| json!({ "title": title, "status": s.status[i], "detail": s.detail[i] }))
            .collect();
        ui.render(json!({
            "screen": "progress",
            "title": self.title,
            "lede": self.lede,
            "steps": steps,
            "percent": (done / total * 100.0).min(100.0),
            "line": s.line,
            "hint": *self.hint.lock().unwrap(),
            "cancellable": true,
        }));
    }
}

static CHILD: AtomicU32 = AtomicU32::new(0);
static CANCELLED: AtomicBool = AtomicBool::new(false);

fn cancel_child() {
    CANCELLED.store(true, Ordering::SeqCst);
    let pid = CHILD.swap(0, Ordering::SeqCst);
    if pid != 0 {
        platform::kill_tree(pid);
    }
}

// ---- install --------------------------------------------------------------------------------------

/// Installs this binary as the app (shortcuts and registration included, see platform/), and
/// remembers a data folder and a models folder the user picked.
fn install_files(paths: &Paths, desktop: bool, tracker: &Tracker) -> Res<()> {
    tracker.start(0, "");
    platform::install_app(&paths.install, VERSION, desktop)?;
    remember_home(paths)?;
    remember_models(paths)?;
    log(
        paths,
        &format!("installed {VERSION} to {}", paths.install.display()),
    );
    tracker.finish(0);
    Ok(())
}

/// A data folder other than the default is saved for the user (URAKATSU_HOME on Windows,
/// `~/.config/urakatsu/home` elsewhere), which every part of Urakatsu reads, the Python side
/// included.
fn remember_home(paths: &Paths) -> Res<()> {
    let current = env::var_os("URAKATSU_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(default_home);
    if current == paths.home {
        return Ok(());
    }
    platform::save_home(&paths.home)?;
    log(
        paths,
        &format!("data folder set to {}", paths.home.display()),
    );
    Ok(())
}

/// A models folder picked in setup goes into the data folder's `storage.json`, the file
/// Settings > Pipeline > Storage writes (urakatsu/runtimes/models_folder.py): null means
/// `models\` inside the data folder.
fn remember_models(paths: &Paths) -> Res<()> {
    let Some(models) = &paths.models else {
        return Ok(());
    };
    let default = discover::same_place(models, &paths.home.join("models"));
    let settings = paths.home.join("storage.json");
    fs::create_dir_all(&paths.home).context(|| format!("create {}", paths.home.display()))?;
    let value = json!({ "models_dir": if default { Value::Null } else { json!(models.display().to_string()) } });
    fs::write(&settings, serde_json::to_string_pretty(&value).unwrap_or_default())
        .context(|| format!("write {}", settings.display()))?;
    if !default {
        fs::create_dir_all(models).context(|| format!("create {}", models.display()))?;
    }
    log(paths, &format!("models folder set to {}", models.display()));
    Ok(())
}

// ---- app environment ------------------------------------------------------------------------------

/// Brings `URAKATSU_HOME/app` up to date: newest uv, the Python, and this binary's wheel.
/// `first` is the tracker step of the uv download (the steps after it follow in order).
fn prepare(paths: &Paths, tracker: &Tracker, first: usize) -> Res<()> {
    let (uv_step, python_step, libraries_step) = (first, first + 1, first + 2);
    fs::create_dir_all(&paths.app).context(|| format!("create {}", paths.app.display()))?;

    tracker.start(uv_step, "");
    if let Err(error) = fetch_uv(paths, tracker, uv_step) {
        if !paths.uv.exists() {
            return Err(error);
        }
        log(paths, &format!("keeping the installed uv: {error}"));
    }
    tracker.finish(uv_step);

    tracker.start(python_step, &format!("Python {PYTHON}"));
    if !paths.python().exists() {
        let env_dir = paths.env.display().to_string();
        run_uv(
            paths,
            &[
                "venv",
                env_dir.as_str(),
                "--python",
                PYTHON,
                "--managed-python",
            ],
            |line| tracker.line(line),
        )?;
    }
    tracker.finish(python_step);

    tracker.start(libraries_step, "");
    let wheels = paths.app.join("wheels");
    fs::create_dir_all(&wheels).context(|| format!("create {}", wheels.display()))?;
    let wheel = wheels.join(WHEEL_NAME);
    fs::write(&wheel, WHEEL).context(|| format!("write {}", wheel.display()))?;
    let python = paths.python().display().to_string();
    let requirement = format!("{}[desktop]", wheel.display());
    let mut total = 0usize;
    let mut prepared = 0usize;
    run_uv(
        paths,
        &[
            "pip",
            "install",
            "--python",
            python.as_str(),
            requirement.as_str(),
            "--reinstall-package",
            "urakatsu",
        ],
        |line| {
            let text = line.trim();
            if let Some(count) = text
                .strip_prefix("Resolved ")
                .and_then(|t| t.split_whitespace().next())
                .and_then(|n| n.parse().ok())
            {
                total = count;
            } else if text.starts_with("Downloaded ") || text.starts_with("Built ") {
                prepared += 1;
            } else if text.starts_with("Prepared ") {
                prepared = total;
            }
            let (fraction, detail) = if total == 0 {
                (0.05, String::new())
            } else if prepared >= total {
                (0.75, format!("Installing {total} packages"))
            } else {
                (
                    0.05 + 0.7 * prepared as f64 / total as f64,
                    format!("{prepared} of {total} packages"),
                )
            };
            tracker.progress(libraries_step, fraction, &detail);
            tracker.line(text);
        },
    )?;
    let marker = json!({ "version": VERSION, "wheel_sha256": sha256_hex(WHEEL) });
    fs::write(paths.marker(), marker.to_string())
        .context(|| "write the environment marker".to_owned())?;
    tracker.finish(libraries_step);
    log(paths, &format!("environment ready for {VERSION}"));
    Ok(())
}

/// The newest uv release, checked against the SHA-256 file published beside it.
fn fetch_uv(paths: &Paths, tracker: &Tracker, step: usize) -> Res<()> {
    let archive = download(platform::UV_URL, |done, total| {
        let mb = |b: u64| b as f64 / 1_048_576.0;
        let detail = if total > 0 {
            format!("{:.1} of {:.1} MB", mb(done), mb(total))
        } else {
            format!("{:.1} MB", mb(done))
        };
        tracker.progress(
            step,
            if total > 0 {
                done as f64 / total as f64
            } else {
                0.5
            },
            &detail,
        );
    })?;
    let published = String::from_utf8(download(&format!("{}.sha256", platform::UV_URL), |_, _| {})?)
        .context(|| "read uv's checksum".to_owned())?;
    let expected = published
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    if expected.is_empty() || sha256_hex(&archive) != expected {
        return Err("the uv download failed its SHA-256 check".to_owned());
    }
    let bytes = platform::uv_from_archive(archive)?;
    let dir = paths.uv.parent().unwrap();
    fs::create_dir_all(dir).context(|| format!("create {}", dir.display()))?;
    let staged = dir.join(format!("{}.new", platform::uv_name()));
    platform::write_executable(&staged, &bytes)?;
    fs::rename(&staged, &paths.uv).context(|| "replace uv".to_owned())?;
    log(paths, "uv updated");
    Ok(())
}

fn download(url: &str, mut progress: impl FnMut(u64, u64)) -> Res<Vec<u8>> {
    let response = ureq::get(url)
        .call()
        .context(|| format!("download {url}"))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0u64);
    let mut reader = response.into_body().into_reader();
    let mut bytes = Vec::with_capacity(total as usize);
    let mut chunk = vec![0u8; 256 * 1024];
    loop {
        if CANCELLED.load(Ordering::SeqCst) {
            return Err("cancelled".to_owned());
        }
        let read = reader
            .read(&mut chunk)
            .context(|| format!("download {url}"))?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..read]);
        progress(bytes.len() as u64, total);
    }
    Ok(bytes)
}

/// Runs uv with `args`, passing each output line on; fails with the last lines of it.
fn run_uv(paths: &Paths, args: &[&str], mut on_line: impl FnMut(&str)) -> Res<()> {
    log(paths, &format!("uv {}", args.join(" ")));
    let mut child = platform::background(
        Command::new(&paths.uv)
            .args(args)
            .envs(paths.child_env())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
    )
    .spawn()
        .context(|| "run uv".to_owned())?;
    CHILD.store(child.id(), Ordering::SeqCst);
    let stdout = child.stdout.take().unwrap();
    let quiet_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = BufReader::new(stdout).read_to_string(&mut text);
        text
    });
    let mut tail: Vec<String> = Vec::new();
    for line in BufReader::new(child.stderr.take().unwrap())
        .lines()
        .map_while(Result::ok)
    {
        if line.trim().is_empty() {
            continue;
        }
        log(paths, &line);
        on_line(&line);
        tail.push(line);
        if tail.len() > 6 {
            tail.remove(0);
        }
    }
    let status = child.wait().context(|| "run uv".to_owned())?;
    CHILD.store(0, Ordering::SeqCst);
    let stdout = quiet_reader.join().unwrap_or_default();
    if !stdout.trim().is_empty() {
        log(paths, stdout.trim_end());
    }
    if CANCELLED.load(Ordering::SeqCst) {
        return Err("cancelled".to_owned());
    }
    if !status.success() {
        return Err(tail.join("\n"));
    }
    Ok(())
}

fn start_app(paths: &Paths) -> Res<()> {
    let errors = paths.home.join("logs").join("desktop-errors.log");
    let _ = fs::create_dir_all(errors.parent().unwrap());
    let stderr = File::create(&errors)
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());
    let mut child = platform::detached(
        Command::new(paths.app_python())
            .args(["-m", "urakatsu.desktop"])
            .envs(paths.child_env())
            .current_dir(&paths.home)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr),
    )
    .spawn()
        .context(|| "start the Urakatsu window".to_owned())?;
    // A broken environment exits at once; say so rather than vanish. A healthy start keeps running.
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(4) {
        if let Ok(Some(status)) = child.try_wait() {
            if !status.success() {
                let _ = fs::remove_file(paths.marker()); // rebuild the environment next time
                let detail = fs::read_to_string(&errors).unwrap_or_default();
                let lines: Vec<&str> = detail.lines().collect();
                return Err(format!(
                    "The app exited at startup ({status}).\n{}",
                    lines[lines.len().saturating_sub(6)..].join("\n")
                ));
            }
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    log(paths, "started the app");
    Ok(())
}

// ---- uninstall ------------------------------------------------------------------------------------

fn uninstall_files(paths: &Paths, everything: bool) {
    let mut doomed = vec!["app", "logs", "runtimes"];
    if everything {
        doomed.extend(["models", "projects", "cache", "pipeline.toml"]);
    }
    for name in doomed {
        remove(&paths.home.join(name));
    }
    if !paths.app.starts_with(&paths.home) {
        remove(&paths.app); // the environment kept on the system drive
    }
    let _ = fs::remove_dir(&paths.home); // only if nothing else is left in it
    let _ = fs::remove_dir_all(env::temp_dir().join("urakatsu-setup-webview"));
    // Shortcuts, registration, and the app itself (on Windows once this process has exited).
    platform::uninstall_app(&paths.install);
}

/// Whether the app is open: it records its process in `app\desktop.json` from startup on.
/// Deletes a file or folder, retrying for a few seconds: a just-closed app's WebView2 helper
/// processes and log handles let go of their files a moment after the window disappears.
fn remove(path: &Path) {
    for _ in 0..20 {
        let result = if path.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };
        if result.is_ok() || !path.exists() {
            return;
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn app_running(paths: &Paths) -> bool {
    let Ok(text) = fs::read_to_string(paths.state.join("desktop.json")) else {
        return false;
    };
    let Some(pid) = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v["pid"].as_u64())
    else {
        return false;
    };
    platform::alive(pid as u32)
}
