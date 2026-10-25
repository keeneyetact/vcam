//! vcam — the control panel for the virtual camera.
//!
//! Run it, press Start camera, then in Google Meet:
//! Settings -> Video -> Camera -> "OBS Virtual Camera".

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod library;
mod settings;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

use vcam::rvm::RvmEngine;
use vcam::{capture, imgproc, paths, virtualcam, Frame};

/// `vcam.exe --selftest`: exercise the matting engine and drivers without a
/// camera or GUI, writing the result to vcam_selftest.txt (the windowed build
/// has no console).
fn selftest() -> i32 {
    let mut lines = Vec::new();
    let run = || -> anyhow::Result<String> {
        let mut eng = RvmEngine::new(640, 480, None, true, None)?;
        let mut seed = 1u32;
        let mut out = None;
        for _ in 0..3 {
            // A few frames, so the staged pipeline is exercised end to end.
            let mut f = Frame::new(640, 480);
            for v in &mut f.data {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (seed >> 24) as u8;
            }
            out = Some(eng.process(Arc::new(f))?);
        }
        let (alpha, fg) = out.unwrap();
        let preview = imgproc::make_preview(&fg, 320, 240);
        Ok(format!(
            "RVM OK: alpha[{}] fg {}x{} dsr={} threads={} staged={}\npreview OK: {} bytes",
            alpha.len(), fg.w, fg.h, eng.dsr, eng.threads, eng.staged(), preview.len()
        ))
    };
    let ok = match run() {
        Ok(s) => {
            lines.push(s);
            true
        }
        Err(e) => {
            lines.push(format!("RVM FAIL: {e:#}"));
            false
        }
    };
    lines.push(format!("virtualcam_installed={}", virtualcam::installed()));
    lines.push(format!("cameras={:?}", capture::list_cameras()));
    let report = format!("{}\n{}", if ok { "PASS" } else { "FAIL" }, lines.join("\n"));
    let _ = std::fs::write("vcam_selftest.txt", &report);
    println!("{report}");
    if ok {
        0
    } else {
        1
    }
}

/// Segoe UI (the Windows UI font) when available, with Segoe UI Symbol for
/// the few glyphs it lacks and egui's built-in fonts as the last resort. A
/// "semi" family holds the semibold face used for headings and buttons.
fn install_fonts(ctx: &egui::Context) {
    let dir = std::env::var_os("WINDIR").map(PathBuf::from).unwrap_or_else(|| "C:\\Windows".into()).join("Fonts");
    let mut fonts = FontDefinitions::default();
    let mut prop = fonts.families[&FontFamily::Proportional].clone();
    let mut semi = prop.clone();
    let add = |fonts: &mut FontDefinitions, key: &str, file: &str| -> bool {
        match std::fs::read(dir.join(file)) {
            Ok(bytes) => {
                fonts.font_data.insert(key.into(), Arc::new(FontData::from_owned(bytes)));
                true
            }
            Err(_) => false,
        }
    };
    let regular = add(&mut fonts, "segoe", "segoeui.ttf");
    let bold = add(&mut fonts, "segoe-semi", "seguisb.ttf");
    if add(&mut fonts, "segoe-sym", "seguisym.ttf") {
        prop.insert(0, "segoe-sym".into());
        semi.insert(0, "segoe-sym".into());
    }
    if regular {
        prop.insert(0, "segoe".into());
    }
    if bold {
        semi.insert(0, "segoe-semi".into());
    } else if regular {
        semi.insert(0, "segoe".into());
    }
    fonts.families.insert(FontFamily::Proportional, prop);
    fonts.families.insert(FontFamily::Name("semi".into()), semi);
    ctx.set_fonts(fonts);
}

fn main() -> eframe::Result {
    if std::env::args().any(|a| a == "--selftest") {
        std::process::exit(selftest());
    }
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("vcam — Virtual Camera")
        .with_inner_size([1120.0, 680.0])
        .with_min_inner_size([820.0, 540.0]);
    if let Ok(img) = image::load_from_memory(paths::ICON_PNG) {
        let img = img.to_rgba8();
        let (width, height) = img.dimensions();
        viewport = viewport.with_icon(Arc::new(egui::IconData { rgba: img.into_raw(), width, height }));
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native(
        "vcam",
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            Ok(Box::new(app::App::new(cc)))
        }),
    )
}
