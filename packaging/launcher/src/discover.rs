//! What setup finds on this computer before installing: an installed Urakatsu, data folders
//! from earlier installs (projects, models, settings), and the models a folder already holds.
//!
//! A data folder (URAKATSU_HOME) holds `projects\`, `models\`, `pipeline.toml` and
//! `storage.json`; the last names a models folder chosen elsewhere (Settings > Pipeline >
//! Storage writes it too, see urakatsu/runtimes/models_folder.py). A models folder holds
//! `installed.json` (models added from the Models page) and the weights the app fetched
//! itself under `hf\<owner>\<repo>` and `github\<owner>\<repo>`.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::platform;
pub use crate::platform::installed;

/// True when `path` is a Urakatsu data folder: files only Urakatsu writes, not just a
/// `models` or `projects` folder (ComfyUI and others keep a `models` folder at a drive's top).
pub fn is_home(path: &Path) -> bool {
    path.join("pipeline.toml").is_file()
        || path.join("storage.json").is_file()
        || path.join("app").join("desktop.json").is_file()
        || holds_models(&path.join("models"))
        || subfolders(&path.join("projects"))
            .iter()
            .any(|p| p.join("project.json").is_file())
}

/// The data folder for a folder picked in setup: the folder itself when it is one (or empty),
/// a data folder inside it, else a new `Urakatsu` folder in it, or `Urakatsu Data` when
/// `Urakatsu` there is something else (a source checkout, say).
pub fn home_for(folder: PathBuf) -> PathBuf {
    if is_home(&folder) || is_empty(&folder) {
        return folder;
    }
    for name in ["Urakatsu", ".urakatsu"] {
        let inner = folder.join(name);
        if is_home(&inner) {
            return inner;
        }
    }
    let inner = folder.join("Urakatsu");
    if !inner.exists() || is_empty(&inner) {
        inner
    } else {
        folder.join("Urakatsu Data")
    }
}

fn is_empty(dir: &Path) -> bool {
    fs::read_dir(dir).map(|mut entries| entries.next().is_none()).unwrap_or(false)
}

/// The models folder a data folder uses: the one `storage.json` names, else `models\` inside it.
pub fn models_dir_of(home: &Path) -> PathBuf {
    fs::read_to_string(home.join("storage.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')).ok())
        .and_then(|v| v["models_dir"].as_str().map(PathBuf::from))
        .unwrap_or_else(|| home.join("models"))
}

/// A picked folder that holds a models folder rather than being one (a data folder, say).
pub fn models_folder_in(chosen: PathBuf) -> PathBuf {
    let inner = chosen.join("models");
    if !holds_models(&chosen) && holds_models(&inner) {
        inner
    } else {
        chosen
    }
}

fn holds_models(dir: &Path) -> bool {
    dir.join("installed.json").is_file() || dir.join("hf").is_dir() || dir.join("github").is_dir()
}

/// `{path, exists, catalog, downloaded, bytes}`: models added from the Models page, weight
/// repositories the app downloaded, and the folder's size. Totalling a folder of tens of GB
/// on a hard drive takes seconds, so `sizes` false leaves `bytes` null (filled in later).
pub fn scan_models(dir: &Path, sizes: bool) -> Value {
    let catalog = fs::read_to_string(dir.join("installed.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|v| v["models"].as_array().map(Vec::len))
        .unwrap_or(0);
    let downloaded: usize = ["hf", "github"]
        .iter()
        .map(|source| {
            subfolders(&dir.join(source))
                .iter()
                .map(|owner| subfolders(owner).len())
                .sum::<usize>()
        })
        .sum();
    json!({
        "path": dir.display().to_string(),
        "exists": dir.is_dir(),
        "catalog": catalog,
        "downloaded": downloaded,
        "bytes": if sizes { json!(folder_size(dir)) } else { Value::Null },
    })
}

/// `{path, exists, projects, models}` for a data folder and the models folder it uses.
pub fn scan_home(home: &Path, sizes: bool) -> Value {
    let projects = subfolders(&home.join("projects"))
        .iter()
        .filter(|p| p.join("project.json").is_file())
        .count();
    json!({
        "path": home.display().to_string(),
        "exists": is_home(home),
        "projects": projects,
        "models": scan_models(&models_dir_of(home), sizes),
    })
}

/// Data folders from other installs: the saved data folder, `~/.urakatsu`, and `Urakatsu` or
/// `.urakatsu` at the top of each local drive (mounted volumes on macOS and Linux), when they
/// hold projects or models.
/// `current` is left out.
pub fn other_homes(current: &Path, default: &Path, sizes: bool) -> Vec<Value> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.extend(platform::saved_home());
    candidates.push(default.to_path_buf());
    for root in platform::drive_roots() {
        for name in ["Urakatsu", ".urakatsu"] {
            candidates.push(root.join(name));
        }
    }
    let mut seen: Vec<PathBuf> = vec![key(current)];
    let mut found = Vec::new();
    for path in candidates {
        let id = key(&path);
        if seen.contains(&id) || !is_home(&path) {
            continue;
        }
        seen.push(id);
        let home = scan_home(&path, sizes);
        let models = &home["models"];
        let used = home["projects"].as_u64().unwrap_or(0) > 0
            || models["catalog"].as_u64().unwrap_or(0) + models["downloaded"].as_u64().unwrap_or(0) > 0;
        if used {
            found.push(home); // an empty one (a first start that made only settings) is not worth offering
        }
    }
    found
}

/// A comparable form of a folder path (Windows and macOS paths ignore case).
fn key(path: &Path) -> PathBuf {
    let text = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string();
    let text = text.trim_end_matches(['\\', '/']);
    PathBuf::from(if platform::CASE_INSENSITIVE {
        text.to_lowercase()
    } else {
        text.to_owned()
    })
}

pub fn same_place(a: &Path, b: &Path) -> bool {
    key(a) == key(b)
}

fn subfolders(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default()
}

fn folder_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(next) = stack.pop() {
        let Ok(entries) = fs::read_dir(&next) else { continue };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => stack.push(entry.path()),
                Ok(_) => total += entry.metadata().map(|m| m.len()).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("urakatsu-discover-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_data_folder_and_its_models_are_found() {
        let home = temp("home");
        assert!(!is_home(&home));
        fs::create_dir_all(home.join("projects").join("p1")).unwrap();
        fs::write(home.join("projects").join("p1").join("project.json"), "{}").unwrap();
        fs::create_dir_all(home.join("projects").join("not-a-project")).unwrap();
        let models = home.join("models");
        fs::create_dir_all(models.join("hf").join("owner").join("repo-a")).unwrap();
        fs::create_dir_all(models.join("hf").join("owner").join("repo-b")).unwrap();
        fs::create_dir_all(models.join("github").join("org").join("tool")).unwrap();
        fs::write(models.join("hf").join("owner").join("repo-a").join("w.gguf"), vec![0u8; 1000]).unwrap();
        fs::write(models.join("installed.json"), r#"{"models": [{"id": "x"}, {"id": "y"}]}"#).unwrap();
        assert!(is_home(&home));
        assert_eq!(home_for(home.clone()), home);
        let found = scan_home(&home, true);
        assert_eq!(found["projects"], 1);
        assert_eq!(found["models"]["catalog"], 2);
        assert_eq!(found["models"]["downloaded"], 3);
        assert!(found["models"]["bytes"].as_u64().unwrap() >= 1000);
        assert!(scan_home(&home, false)["models"]["bytes"].is_null());
        // Picking the data folder in the models dialog means its models folder.
        assert_eq!(models_folder_in(home.clone()), models);
        assert_eq!(models_folder_in(models.clone()), models);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn other_tools_folders_are_not_data_folders() {
        let drive = temp("drive");
        // A ComfyUI-style models folder and an empty projects folder at a drive's top.
        fs::create_dir_all(drive.join("models").join("checkpoints")).unwrap();
        fs::create_dir_all(drive.join("projects")).unwrap();
        assert!(!is_home(&drive));
        assert_eq!(home_for(drive.clone()), drive.join("Urakatsu"));
        // "Urakatsu" there is a source checkout: the data goes beside it.
        fs::create_dir_all(drive.join("Urakatsu").join("web")).unwrap();
        fs::write(drive.join("Urakatsu").join("pyproject.toml"), "").unwrap();
        assert_eq!(home_for(drive.clone()), drive.join("Urakatsu Data"));
        // An earlier data folder inside is found.
        fs::create_dir_all(drive.join(".urakatsu")).unwrap();
        fs::write(drive.join(".urakatsu").join("pipeline.toml"), "").unwrap();
        assert_eq!(home_for(drive.clone()), drive.join(".urakatsu"));
        let empty = temp("empty");
        assert_eq!(home_for(empty.clone()), empty);
        let _ = fs::remove_dir_all(&drive);
        let _ = fs::remove_dir_all(&empty);
    }

    #[test]
    fn storage_json_names_the_models_folder() {
        let home = temp("storage");
        assert_eq!(models_dir_of(&home), home.join("models"));
        fs::write(home.join("storage.json"), "\u{feff}{\"models_dir\": \"E:\\\\AI\\\\models\"}").unwrap(); // a BOM, as Notepad writes
        assert_eq!(models_dir_of(&home), PathBuf::from(r"E:\AI\models"));
        fs::write(home.join("storage.json"), r#"{"models_dir": null}"#).unwrap();
        assert_eq!(models_dir_of(&home), home.join("models"));
        let _ = fs::remove_dir_all(&home);
    }
}
