//! Tells the user when a newer vcam is published on GitHub Releases.
//!
//! A background thread asks GitHub for the latest release at startup and then
//! every few hours while the app stays open; anything newer than this build
//! is handed to the UI. Failures (offline, rate limit) are silent.

use std::sync::mpsc;
use std::time::Duration;

const LATEST_URL: &str = "https://api.github.com/repos/keeneyetact/vcam/releases/latest";
const CHECK_EVERY: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Clone, Debug)]
pub struct Update {
    /// e.g. "0.3.0" (no leading "v").
    pub version: String,
    /// The release page, where the exe and zip are.
    pub url: String,
}

#[derive(serde::Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
}

/// Starts the checker; `on_found` runs (on its thread) after each send so the
/// UI can repaint.
pub fn watch(on_found: impl Fn() + Send + 'static) -> mpsc::Receiver<Update> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut told: Option<String> = None;
        loop {
            if let Some(up) = check() {
                if told.as_deref() != Some(&up.version) {
                    told = Some(up.version.clone());
                    if tx.send(up).is_err() {
                        return; // the window closed
                    }
                    on_found();
                }
            }
            std::thread::sleep(CHECK_EVERY);
        }
    });
    rx
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
    is_newer(&version, env!("CARGO_PKG_VERSION")).then_some(Update { version, url: rel.html_url })
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
    use super::is_newer;

    #[test]
    fn versions() {
        assert!(is_newer("0.3.0", "0.2.0"));
        assert!(is_newer("0.10.0", "0.9.2"));
        assert!(is_newer("1.0", "0.9.9"));
        assert!(!is_newer("0.2.0", "0.2.0"));
        assert!(!is_newer("0.1.9", "0.2.0"));
        assert!(!is_newer("nightly", "0.2.0"));
    }
}
