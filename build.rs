// Embeds the source revision: HEAD, then the working-tree diff after a blank line.
// A scratch index includes untracked files without touching the real index. Builds
// from a copied tree (no .git) take the revision from TURBOGPT_REVISION_FILE.
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, index: Option<&Path>, args: &[&str]) -> Option<Vec<u8>> {
    let mut command = Command::new("git");
    command.arg("-C").arg(root).args(args);
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    let output = command.output().ok()?;
    output.status.success().then_some(output.stdout)
}

fn revision(root: &Path, out: &Path) -> Vec<u8> {
    let Some(hash) = git(root, None, &["rev-parse", "HEAD"]) else {
        return b"unknown".to_vec();
    };
    let mut bytes = String::from_utf8_lossy(&hash).trim().as_bytes().to_vec();
    let scratch = out.join("revision.index");
    if let Some(real) =
        git(root, None, &["rev-parse", "--path-format=absolute", "--git-path", "index"])
    {
        let real = PathBuf::from(String::from_utf8_lossy(&real).trim());
        let _ = std::fs::copy(real, &scratch);
    }
    let patch = git(root, Some(&scratch), &["add", "-A"]).and_then(|_| {
        git(root, Some(&scratch), &["diff", "--cached", "--no-color", "--no-ext-diff", "HEAD"])
    });
    let _ = std::fs::remove_file(&scratch);
    if let Some(patch) = patch.filter(|p| !p.is_empty()) {
        bytes.extend_from_slice(b"\n\n");
        bytes.extend_from_slice(&patch);
    }
    bytes
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=TURBOGPT_REVISION_FILE");
    let bytes = match std::env::var_os("TURBOGPT_REVISION_FILE") {
        Some(file) => {
            println!("cargo:rerun-if-changed={}", Path::new(&file).display());
            std::fs::read(&file).expect("TURBOGPT_REVISION_FILE is unreadable")
        }
        None => {
            for watched in ["src", "tests", "build.rs", "Cargo.toml", ".git/HEAD", ".git/index"] {
                println!("cargo:rerun-if-changed={watched}");
            }
            revision(&root, &out)
        }
    };
    std::fs::write(out.join("revision.bin"), bytes).unwrap();
}
