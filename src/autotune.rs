//! Automatic edge detail: the finest RVM downsample_ratio this PC sustains at
//! the camera's frame rate, re-fitted while running.
//!
//! The start is a guess from the hardware (`initial`). Every few seconds the
//! engine's slowest stage time is compared with the frame budget: near the
//! budget, step coarser; with room for the next finer step (cost grows with
//! the ratio squared), step finer. A step that proved too slow isn't retried
//! for a while, so it settles instead of flapping -- each change resets the
//! recurrent state, a brief softening of the matte.

use std::time::{Duration, Instant};

use crate::hw::Machine;

/// The ratios Auto moves between, coarse to fine. Past 0.5 the matte barely
/// gets sharper while the cost keeps climbing, so Auto stops there.
pub const LEVELS: [f32; 6] = [0.20, 0.25, 0.30, 0.375, 0.40, 0.50];

/// How long each measurement runs.
const WINDOW: Duration = Duration::from_secs(3);
/// Step coarser once the slowest stage takes this share of a frame...
const DOWN_AT: f32 = 0.90;
/// ...and finer only if the next step is predicted below this share, leaving
/// room for the video-call app.
const UP_BELOW: f32 = 0.70;
/// A step that was too slow is tried again after this long (the load or the
/// thermals may have changed).
const RETRY_AFTER: Duration = Duration::from_secs(120);

/// Starting ratio before anything is measured: fast cores against pixels.
/// onnxruntime stops scaling past ~8 threads on this model.
pub fn initial(m: &Machine, w: usize, h: usize) -> f32 {
    let pixels = (w * h) as f32 / (1280.0 * 720.0);
    let score = m.fast_cores.min(8) as f32 / pixels.max(0.1);
    LEVELS[match score {
        s if s >= 6.0 => 3,
        s if s >= 4.0 => 2,
        s if s >= 2.5 => 1,
        _ => 0,
    }]
}

pub struct AutoQuality {
    level: usize,
    /// One frame at the camera's rate.
    budget: Duration,
    window: Instant,
    /// The first window after a change is discarded (the state reset and the
    /// frames already in flight blur it).
    settle: bool,
    /// The level that proved too slow, and when.
    ceiling: Option<(usize, Instant)>,
}

impl AutoQuality {
    pub fn new(dsr: f32, fps: f64, now: Instant) -> Self {
        let level = (0..LEVELS.len())
            .min_by(|&a, &b| (LEVELS[a] - dsr).abs().total_cmp(&(LEVELS[b] - dsr).abs()))
            .unwrap_or(0);
        Self {
            level,
            budget: Duration::from_secs_f64(1.0 / fps.clamp(5.0, 120.0)),
            window: now,
            settle: true,
            ceiling: None,
        }
    }

    pub fn dsr(&self) -> f32 {
        LEVELS[self.level]
    }

    /// Call every frame; `stage_time` (the slowest stage's mean run time since
    /// its last call) is read once per window. Returns the new ratio when it
    /// changes.
    pub fn poll(&mut self, now: Instant, stage_time: impl FnOnce() -> Option<Duration>) -> Option<f32> {
        if now.duration_since(self.window) < WINDOW {
            return None;
        }
        self.window = now;
        let stage = stage_time()?;
        if std::mem::take(&mut self.settle) {
            return None;
        }
        if self.ceiling.is_some_and(|(_, t)| now.duration_since(t) >= RETRY_AFTER) {
            self.ceiling = None;
        }
        let load = stage.as_secs_f32() / self.budget.as_secs_f32();
        if load > DOWN_AT && self.level > 0 {
            self.ceiling = Some((self.level, now));
            return Some(self.step(-1));
        }
        let next = self.level + 1;
        let allowed = self.ceiling.is_none_or(|(c, _)| next < c);
        if next < LEVELS.len() && allowed {
            let k = LEVELS[next] / LEVELS[self.level];
            if load * k * k < UP_BELOW {
                return Some(self.step(1));
            }
        }
        None
    }

    fn step(&mut self, by: isize) -> f32 {
        self.level = self.level.saturating_add_signed(by).min(LEVELS.len() - 1);
        self.settle = true;
        self.dsr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs windows where a stage takes `cost(dsr)` and returns where it settles.
    fn settle(start: f32, cost: impl Fn(f32) -> f32) -> f32 {
        let t0 = Instant::now();
        let mut a = AutoQuality::new(start, 30.0, t0);
        let mut changes = 0;
        for i in 1..=60 {
            let now = t0 + WINDOW * i;
            let ms = cost(a.dsr());
            if a.poll(now, || Some(Duration::from_secs_f32(ms / 1000.0))).is_some() {
                changes += 1;
            }
        }
        assert!(changes <= 4, "flapping: {changes} changes");
        a.dsr()
    }

    #[test]
    fn fast_pc_climbs_to_finest() {
        assert_eq!(settle(0.375, |d| 40.0 * d * d), 0.50); // 10 ms at 0.5
    }

    #[test]
    fn slow_pc_drops_until_it_keeps_up() {
        // 33 ms budget; 0.30 costs 31.5 (too close), 0.25 costs ~22.
        let dsr = settle(0.375, |d| 350.0 * d * d);
        assert_eq!(dsr, 0.25);
    }

    #[test]
    fn stays_put_when_it_fits() {
        // 0.375 costs 25 ms (fits); 0.40 would be ~28.4 (over 70%).
        assert_eq!(settle(0.375, |d| 178.0 * d * d), 0.375);
    }

    #[test]
    fn initial_guess() {
        let m = |fast| Machine { cpu: String::new(), logical: 8, cores: 4, fast_cores: fast, avx2: true, avx512: false };
        assert_eq!(initial(&m(8), 1280, 720), 0.375);
        assert_eq!(initial(&m(4), 1280, 720), 0.30);
        assert_eq!(initial(&m(2), 1280, 720), 0.20);
        assert_eq!(initial(&m(2), 640, 360), 0.375);
    }
}
