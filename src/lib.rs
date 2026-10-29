//! vcam — real-time webcam background replacement.
//!
//! Reads frames from a physical webcam (Media Foundation), mattes the person
//! with RobustVideoMatting (ONNX Runtime), composites them over a background of
//! your choice (image, blur, or solid color), and publishes the result through
//! the OBS Virtual Camera so any app (Zoom, Meet, Teams, ...) can use it as a
//! normal webcam.
//!
//! The library holds the whole pipeline; `src/main.rs` is the GUI control panel
//! and `src/bin/vcam-cli.rs` the command line.

#[cfg(not(windows))]
compile_error!("vcam targets Windows (Media Foundation capture + the OBS Virtual Camera)");

pub mod autotune;
pub mod capture;
pub mod compositor;
pub mod faces;
pub mod hw;
pub mod imgproc;
pub mod kernels;
pub mod look;
pub mod models;
pub mod onnx_graph;
pub mod paths;
pub mod pipeline;
pub mod rvm;
pub mod system;
pub mod virtualcam;
pub mod worker;

pub use imgproc::Frame;

/// `[vcam] ...` progress line on stderr (silently dropped by the windowed GUI).
#[macro_export]
macro_rules! log {
    ($($t:tt)*) => { eprintln!("[vcam] {}", format_args!($($t)*)) };
}
