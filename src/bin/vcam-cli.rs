//! vcam from the command line: real-time webcam background replacement,
//! published as the OBS Virtual Camera.
//!
//!     vcam-cli --background office.jpg     # virtual background
//!     vcam-cli --blur                      # blur the real background
//!     vcam-cli --color 0,177,64            # green screen
//!     vcam-cli -b office.jpg --no-preview  # headless, just feed the virtual cam
//!
//! Quit with q / Esc in the preview window, or Ctrl+C in the terminal.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use windows::core::BOOL;
use windows::Win32::System::Console::SetConsoleCtrlHandler;

use vcam::worker::{BgMode, Config, Status, VCamWorker, VirtualCamUse};
use vcam::{capture, log};

#[derive(Parser)]
#[command(
    name = "vcam-cli",
    about = "Real-time webcam virtual background (RobustVideoMatting) published as the OBS Virtual Camera."
)]
struct Args {
    /// Background image to composite behind you
    #[arg(short, long, value_name = "PATH", conflicts_with_all = ["blur", "color"])]
    background: Option<PathBuf>,
    /// Blur the real background instead of replacing it (the default)
    #[arg(long, conflicts_with = "color")]
    blur: bool,
    /// Solid colour background, e.g. 0,177,64
    #[arg(long, value_name = "R,G,B", value_parser = parse_color)]
    color: Option<[u8; 3]>,
    /// RVM internal downsample_ratio 0-1 (default 0.375; lower = faster, softer hair)
    #[arg(long, value_parser = parse_downsample)]
    downsample: Option<f32>,
    /// CPU threads per RVM stage (default: min(8, cores); more can be slower)
    #[arg(long)]
    threads: Option<usize>,
    /// Webcam index (see --list-cameras)
    #[arg(short, long, default_value_t = 0)]
    camera: usize,
    /// Capture width
    #[arg(long, default_value_t = 1280)]
    width: usize,
    /// Capture height
    #[arg(long, default_value_t = 720)]
    height: usize,
    /// Virtual-cam output width (default: same as capture). Capture small/fast, upscale output.
    #[arg(long)]
    out_width: Option<usize>,
    /// Virtual-cam output height (default: same as capture)
    #[arg(long)]
    out_height: Option<usize>,
    /// Target frames per second
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// Let the webcam's auto-exposure run (default: fix it so dim light can't slow the frame rate)
    #[arg(long)]
    auto_exposure: bool,
    /// Brighten the picture 0-1 to make up for the fixed exposure (default 0: as captured)
    #[arg(long, value_parser = parse_brightness, conflicts_with = "auto_exposure")]
    brightness: Option<f32>,
    /// Paste the background picture as-is (default: match it to your room's light, focus and grain)
    #[arg(long)]
    plain: bool,
    /// Keep everyone in frame (default: keep only the single largest person)
    #[arg(long)]
    multi_person: bool,
    /// Horizontally flip (selfie view)
    #[arg(long)]
    mirror: bool,
    /// Don't open the preview window
    #[arg(long)]
    no_preview: bool,
    /// Preview only; don't open a virtual camera
    #[arg(long)]
    no_virtualcam: bool,
    /// List the available cameras and exit
    #[arg(long)]
    list_cameras: bool,
}

/// Parse 'R,G,B' (0-255) into BGR.
fn parse_color(text: &str) -> Result<[u8; 3], String> {
    let v: Vec<u8> = text
        .split(',')
        .map(|s| s.trim().parse::<u8>())
        .collect::<Result<_, _>>()
        .map_err(|_| "--color must be 'R,G,B' with channels 0-255, e.g. 0,177,64".to_string())?;
    match v[..] {
        [r, g, b] => Ok([b, g, r]),
        _ => Err("--color must be 'R,G,B', e.g. 0,177,64".into()),
    }
}

/// RVM's downsample_ratio must be in (0, 1].
fn parse_downsample(text: &str) -> Result<f32, String> {
    match text.trim().parse::<f32>() {
        Ok(v) if v > 0.0 && v <= 1.0 => Ok(v),
        _ => Err("--downsample must be a number above 0 and at most 1, e.g. 0.375".into()),
    }
}

/// Dim-light brightening must be in [0, 1].
fn parse_brightness(text: &str) -> Result<f32, String> {
    match text.trim().parse::<f32>() {
        Ok(v) if (0.0..=1.0).contains(&v) => Ok(v),
        _ => Err("--brightness must be a number from 0 to 1, e.g. 0.4".into()),
    }
}

static STOP: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn on_ctrl(_kind: u32) -> BOOL {
    STOP.store(true, Ordering::Relaxed);
    BOOL(1)
}

fn main() {
    let args = Args::parse();
    if args.list_cameras {
        for (i, name) in capture::list_cameras().iter().enumerate() {
            println!("{i}: {name}");
        }
        return;
    }
    if let Some(bg) = &args.background {
        if !bg.is_file() {
            eprintln!("Background image not found: {}", bg.display());
            std::process::exit(1);
        }
    }
    let mode = if args.background.is_some() {
        BgMode::Image
    } else if args.color.is_some() {
        BgMode::Color
    } else {
        if !args.blur {
            log!("No background specified -> blurring the real background.");
        }
        BgMode::Blur
    };
    let defaults = Config::default();
    let cfg = Config {
        camera: args.camera,
        width: args.width,
        height: args.height,
        out_width: args.out_width,
        out_height: args.out_height,
        fps: args.fps,
        mode,
        bg_path: args.background.clone(),
        color: args.color.unwrap_or(defaults.color),
        realistic: !args.plain,
        mirror: args.mirror,
        multi_person: args.multi_person,
        fix_exposure: !args.auto_exposure,
        brightness: args.brightness.unwrap_or(defaults.brightness),
        downsample: args.downsample,
        threads: args.threads,
        virtualcam: if args.no_virtualcam { VirtualCamUse::Off } else { VirtualCamUse::Optional },
        ..defaults
    };
    unsafe {
        let _ = SetConsoleCtrlHandler(Some(on_ctrl), true);
    }

    let worker = VCamWorker::new(cfg);
    worker.start();
    let code = run(&worker, !args.no_preview);
    worker.stop();
    std::process::exit(code);
}

fn run(worker: &VCamWorker, show_preview: bool) -> i32 {
    let mut window: Option<minifb::Window> = None;
    let mut shown: Option<std::sync::Arc<vcam::Frame>> = None;
    let mut buf: Vec<u32> = Vec::new();
    let mut last_report = Instant::now();
    loop {
        if STOP.load(Ordering::Relaxed) {
            log!("Interrupted.");
            return 0;
        }
        match worker.status() {
            Status::Error => {
                eprintln!("{}", worker.error().unwrap_or_default());
                return 1;
            }
            Status::Stopped => return 0,
            _ => {}
        }
        if worker.running() && last_report.elapsed() > Duration::from_secs(4) {
            log!("~{:.1} fps", worker.fps());
            last_report = Instant::now();
        }
        if !show_preview {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let frame = worker.latest_frame();
        match (&frame, &mut window) {
            (Some(f), w) if !shown.as_ref().is_some_and(|s| std::sync::Arc::ptr_eq(s, f)) => {
                if w.as_ref().is_none_or(|w| w.get_size() != (f.w, f.h)) {
                    match minifb::Window::new("vcam preview (q/Esc to quit)", f.w, f.h, minifb::WindowOptions::default()) {
                        Ok(mut win) => {
                            win.set_target_fps(120);
                            *w = Some(win);
                        }
                        Err(e) => {
                            log!("could not open a preview window ({e}); continuing without one");
                            return run(worker, false);
                        }
                    }
                }
                buf.clear();
                buf.extend(f.data.chunks_exact(3).map(|p| (p[2] as u32) << 16 | (p[1] as u32) << 8 | p[0] as u32));
                if let Some(win) = w {
                    let _ = win.update_with_buffer(&buf, f.w, f.h);
                }
                shown = frame;
            }
            (_, Some(win)) => win.update(),
            _ => std::thread::sleep(Duration::from_millis(10)),
        }
        if let Some(win) = &window {
            if !win.is_open() || win.is_key_down(minifb::Key::Q) || win.is_key_down(minifb::Key::Escape) {
                return 0;
            }
        }
    }
}
