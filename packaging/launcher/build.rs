//! Embeds the Urakatsu wheel (built by packaging/build.py) and, on Windows, the icon and the manifest.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=URAKATSU_WHEEL");
    let wheel = std::env::var("URAKATSU_WHEEL")
        .expect("set URAKATSU_WHEEL to the urakatsu wheel; packaging/build.py does this");
    println!("cargo:rerun-if-changed={wheel}");
    let name = Path::new(&wheel)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("wheel file name")
        .to_owned();
    let version = name
        .split('-')
        .nth(1)
        .expect("wheel names are <name>-<version>-...")
        .to_owned();

    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::copy(&wheel, Path::new(&out).join("urakatsu.whl")).expect("copy the wheel");
    println!("cargo:rustc-env=URAKATSU_WHEEL_NAME={name}");
    println!("cargo:rustc-env=URAKATSU_VERSION={version}");

    // uv's release archives are named after Rust target triples (uv-aarch64-apple-darwin.tar.gz).
    println!("cargo:rustc-env=URAKATSU_UV_TARGET={}", std::env::var("TARGET").unwrap());

    // The exe's icon and manifest are Windows resources; the macOS bundle and the Linux menu
    // entry carry the icon there (urakatsu.icns, urakatsu.png).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        for file in ["launcher.rc", "launcher.manifest", "urakatsu.ico"] {
            println!("cargo:rerun-if-changed={file}");
        }
        embed_resource::compile("launcher.rc", embed_resource::NONE)
            .manifest_required()
            .unwrap();
    }
}
