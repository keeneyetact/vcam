//! Build script: embeds the RVM and face-detection models into the binary
//! when they're present at build time (so the exe needs no download), and
//! sets the exe icon.

use std::path::Path;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(embedded_model)");
    let model = Path::new("models/rvm_mobilenetv3_fp32.onnx");
    println!("cargo:rerun-if-changed={}", model.display());
    if model.is_file() {
        let abs = model.canonicalize().expect("model path");
        println!("cargo:rustc-cfg=embedded_model");
        println!("cargo:rustc-env=VCAM_EMBEDDED_MODEL={}", abs.display());
    } else {
        println!(
            "cargo:warning=models/rvm_mobilenetv3_fp32.onnx not found: the exe will \
             download the RVM model on first run instead of bundling it"
        );
    }

    println!("cargo:rustc-check-cfg=cfg(embedded_face_model)");
    let face = Path::new("models/face_detection_yunet_2023mar.onnx");
    println!("cargo:rerun-if-changed={}", face.display());
    if face.is_file() {
        let abs = face.canonicalize().expect("face model path");
        println!("cargo:rustc-cfg=embedded_face_model");
        println!("cargo:rustc-env=VCAM_EMBEDDED_FACE_MODEL={}", abs.display());
    }

    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the exe icon: {e}");
        }
    }
}
