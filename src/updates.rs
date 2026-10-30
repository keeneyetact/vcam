//! Keeps vcam up to date from GitHub Releases.
//!
//! A background thread asks GitHub for the latest release at startup and then
//! every few hours while the app stays open, and tells the UI when there is a
//! newer one. Nothing is downloaded until the user asks for the update: then
//! [`stage`] downloads its `vcam.exe` next to the running exe as
//! `vcam.new.exe` and checks it against the SHA-256 GitHub publishes for it,
//! and [`install`] swaps it in. Windows won't let a running exe be overwritten
//! but will let it be renamed, so the old exe moves aside to `vcam.old.exe`
//! (deleted on the next start) and the new one takes its name. If the update
//! can't be downloaded or checked (offline, a read-only folder, no checksum)
//! the UI falls back to linking the release page. Failures are silent.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

use sha2::{Digest, Sha256};

const LATEST_URL: &str = "https://api.github.com/repos/keeneyetact/vcam/releases/latest";
const CHECK_EVERY: Duration = Duration::from_secs(6 * 60 * 60);
const ASSET: &str = "vcam.exe";

#[derive(Clone, Debug)]
pub struct Update {
    /// e.g. "0.3.0" (no leading "v").
    pub version: String,
    /// The release page, where the exe and zip are.
    pub url: String,
    /// The exe to fetch with [`stage`]; `None` means the user has to download
    /// it from `url`.
    pub asset: Option<Download>,
}

#[derive(Clone, Debug)]
pub struct Download {
    url: String,
    /// Lowercase hex SHA-256.
    sha256: String,
}

#[derive(serde::Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(serde::Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
    /// "sha256:<hex>".
    digest: Option<String>,
}

/// Starts the checker; `on_found` runs (on its thread) after each send so the
/// UI can repaint.
pub fn watch(on_found: impl Fn() + Send + 'static) -> mpsc::Receiver<Update> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        remove_leftover(&sibling("vcam.old.exe"));
        let mut told: Option<String> = None;
        loop {
            match check() {
                Some(up) => {
                    if told.as_ref() != Some(&up.version) {
                        told = Some(up.version.clone());
                        if tx.send(up).is_err() {
                            return; // the window closed
                        }
                        on_found();
                    }
                }
                // Up to date: an exe downloaded by an earlier run is stale.
                None => {
                    let _ = fs::remove_file(sibling("vcam.new.exe"));
                }
            }
            std::thread::sleep(CHECK_EVERY);
        }
    });
    rx
}

/// Puts the staged exe in place of the running one; with `relaunch`, starts
/// it. The caller should stop the camera and then close the window.
pub fn install(staged: &Path, relaunch: bool) -> io::Result<()> {
    let exe = std::env::current_exe()?;
    let old = sibling("vcam.old.exe");
    let _ = fs::remove_file(&old);
    fs::rename(&exe, &old)?;
    if let Err(e) = fs::rename(staged, &exe) {
        let _ = fs::rename(&old, &exe);
        return Err(e);
    }
    refresh_shell_icon(&exe);
    if relaunch {
        Command::new(&exe).spawn()?;
    }
    Ok(())
}

fn check() -> Option<Update> {
    let body = ureq::get(LATEST_URL)
        .header("User-Agent", concat!("vcam/", env!("CARGO_PKG_VERSION")))
        .header("Accept", "application/vnd.github+json")
        .call()
        .ok()?
        .into_body()
        .read_to_string()
        .ok()?;
    let rel: Release = serde_json::from_str(&body).ok()?;
    let version = rel.tag_name.trim_start_matches(['v', 'V']).to_string();
    if !is_newer(&version, env!("CARGO_PKG_VERSION")) {
        return None;
    }
    // Debug builds never offer it (a `cargo run` shouldn't replace itself
    // with the release).
    let asset = rel
        .assets
        .into_iter()
        .filter(|_| !cfg!(debug_assertions))
        .find(|a| a.name.eq_ignore_ascii_case(ASSET))
        .and_then(|a| {
            let sha256 = a.digest?.strip_prefix("sha256:")?.to_ascii_lowercase();
            Some(Download { url: a.browser_download_url, sha256 })
        });
    Some(Update { version, url: rel.html_url, asset })
}

/// Downloads `asset` to `vcam.new.exe` beside the running exe and returns its
/// path, ready for [`install`], once its SHA-256 matches. Blocks.
pub fn stage(asset: &Download) -> Option<PathBuf> {
    let want = &asset.sha256;
    let dest = sibling("vcam.new.exe");
    if sha256_file(&dest).is_some_and(|h| h == *want) {
        return Some(dest); // already downloaded on an earlier try
    }
    let part = sibling("vcam.new.exe.part");
    let got = download(&asset.url, &part);
    if got.as_deref() != Some(want.as_str()) {
        let _ = fs::remove_file(&part);
        return None;
    }
    fs::rename(&part, &dest).ok()?;
    Some(dest)
}

/// Streams `url` into `path`, returning the file's SHA-256 (hex).
fn download(url: &str, path: &Path) -> Option<String> {
    let resp = ureq::get(url)
        .header("User-Agent", concat!("vcam/", env!("CARGO_PKG_VERSION")))
        .call()
        .ok()?;
    let mut reader = resp.into_body().into_reader();
    let mut file = File::create(path).ok()?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        file.write_all(&buf[..n]).ok()?;
    }
    file.sync_all().ok()?;
    Some(hex(&hash.finalize()))
}

fn sha256_file(path: &Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut hash = Sha256::new();
    io::copy(&mut file, &mut hash).ok()?;
    Some(hex(&hash.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A file next to the running exe.
/// Tells Explorer the exe at `path` changed, so the taskbar re-reads its icon
/// instead of keeping one it cached while the file was missing mid-swap.
fn refresh_shell_icon(path: &Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::UI::Shell::{SHChangeNotify, SHCNE_UPDATEITEM, SHCNF_PATHW};
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    unsafe { SHChangeNotify(SHCNE_UPDATEITEM, SHCNF_PATHW, Some(wide.as_ptr().cast()), None) };
}

fn sibling(name: &str) -> PathBuf {
    vcam::paths::exe_dir().join(name)
}

/// Deletes the exe an update replaced. Right after an update the old process
/// may still be exiting, so keep trying for a few seconds.
fn remove_leftover(path: &Path) {
    for _ in 0..20 {
        if !path.exists() || fs::remove_file(path).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// `a` > `b`, comparing dotted numeric versions ("0.10.0" > "0.9.2"). Anything
/// that doesn't parse is never newer.
fn is_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Option<Vec<u64>> {
        v.split(['-', '+']).next()?.split('.').map(|n| n.parse().ok()).collect()
    };
    match (parse(a), parse(b)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{hex, is_newer};

    #[test]
    fn versions() {
        assert!(is_newer("0.3.0", "0.2.0"));
        assert!(is_newer("0.10.0", "0.9.2"));
        assert!(is_newer("1.0", "0.9.9"));
        assert!(!is_newer("0.2.0", "0.2.0"));
        assert!(!is_newer("0.1.9", "0.2.0"));
        assert!(!is_newer("nightly", "0.2.0"));
    }

    #[test]
    fn sha256_hex() {
        use sha2::{Digest, Sha256};
        assert_eq!(
            hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
