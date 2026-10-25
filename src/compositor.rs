//! Alpha-blends the person over a background using the matte.

use std::sync::Arc;

use crate::imgproc::{self, Frame};
use crate::kernels;

enum Kind {
    /// "Original": send the camera frame untouched (no matting at all).
    Passthrough,
    /// Blur the live frame itself.
    Blur { sigma: f64 },
    /// A fixed picture (image or solid colour), already sized to the frame.
    Image(Arc<Frame>),
}

pub struct Compositor {
    kind: Kind,
    w: usize,
    h: usize,
    /// Downscaled blur working size: blur is low-frequency, so blurring a
    /// quarter-size copy and upscaling it looks the same and is ~6x cheaper.
    bw: usize,
    bh: usize,
    /// Edge-feather kernel, scaled to resolution (odd, >= 3): ~3 px at 360p,
    /// ~5 px at 720p. A hard edge reads as an obvious "stencil" cutout against
    /// a sharp background; a few px of softening blends the person in the way
    /// Google Meet's segmentation does, without softening hair detail away.
    ksize: usize,
}

impl Compositor {
    fn with(kind: Kind, w: usize, h: usize) -> Self {
        let radius = ((w.min(h) as f64 / 300.0).round() as usize).max(1);
        Self { kind, w, h, bw: (w / 4).max(160), bh: (h / 4).max(90), ksize: radius * 2 + 1 }
    }

    pub fn passthrough() -> Self {
        Self::with(Kind::Passthrough, 0, 0)
    }

    /// Blur the real background; `strength` 1.0 is the original fixed sigma.
    pub fn blur(w: usize, h: usize, strength: f32) -> Self {
        Self::with(Kind::Blur { sigma: 4.0 * strength.max(0.01) as f64 }, w, h)
    }

    pub fn image(background: Arc<Frame>) -> Self {
        let (w, h) = background.size();
        Self::with(Kind::Image(background), w, h)
    }

    pub fn color(w: usize, h: usize, bgr: [u8; 3]) -> Self {
        Self::image(Arc::new(Frame::filled(w, h, bgr)))
    }

    pub fn is_passthrough(&self) -> bool {
        matches!(self.kind, Kind::Passthrough)
    }

    /// `alpha` is an HxW matte in [0, 1]; 1 == person, 0 == background.
    pub fn render(&self, frame: Arc<Frame>, alpha: Option<&[f32]>) -> Arc<Frame> {
        let Some(alpha) = alpha else { return frame };
        let bg = match &self.kind {
            Kind::Passthrough => return frame,
            Kind::Blur { sigma } => {
                let small = imgproc::gaussian_blur(&imgproc::resize(&frame, self.bw, self.bh), *sigma);
                Arc::new(imgproc::resize_linear(&small, frame.w, frame.h))
            }
            Kind::Image(bg) => bg.clone(),
        };
        if bg.size() != frame.size() || (self.w, self.h) != frame.size() {
            return frame; // background built for another size (being rebuilt)
        }
        Arc::new(kernels::composite(&frame, &bg, alpha, self.ksize))
    }
}
