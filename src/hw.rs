//! What this PC's CPU can give the matting: its fast cores and vector units.
//!
//! Detected once per process. Thread counts and the starting quality are
//! sized from this; `autotune` then fits the quality to the speed measured
//! while running (other apps, thermals, battery all show up there).

use std::sync::OnceLock;

use windows::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

#[derive(Clone, Debug)]
pub struct Machine {
    pub cpu: String,
    /// Logical processors (hardware threads).
    pub logical: usize,
    /// Physical cores.
    pub cores: usize,
    /// Physical cores of the fastest class: the P-cores on a hybrid CPU
    /// (12th-gen Intel and later), every core otherwise.
    pub fast_cores: usize,
    pub avx2: bool,
    pub avx512: bool,
}

impl Machine {
    /// One line for the log and the settings panel.
    pub fn summary(&self) -> String {
        let cores = if self.fast_cores < self.cores {
            format!("{} performance + {} efficiency cores", self.fast_cores, self.cores - self.fast_cores)
        } else {
            format!("{} cores", self.cores)
        };
        let simd = if self.avx512 {
            ", AVX-512"
        } else if self.avx2 {
            ", AVX2"
        } else {
            ""
        };
        format!("{cores}, {} threads{simd}", self.logical)
    }
}

pub fn machine() -> &'static Machine {
    static M: OnceLock<Machine> = OnceLock::new();
    M.get_or_init(detect)
}

fn detect() -> Machine {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    // Without the core list, assume 2-way SMT.
    let classes = core_classes().filter(|c| !c.is_empty()).unwrap_or_else(|| vec![0; (logical / 2).max(1)]);
    let best = classes.iter().copied().max().unwrap_or(0);
    Machine {
        cpu: cpu_name(),
        logical,
        cores: classes.len(),
        fast_cores: classes.iter().filter(|&&c| c == best).count(),
        avx2: std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        avx512: std::arch::is_x86_feature_detected!("avx512f"),
    }
}

/// The EfficiencyClass of each physical core (higher == faster).
fn core_classes() -> Option<Vec<u8>> {
    let mut len = 0u32;
    unsafe {
        let _ = GetLogicalProcessorInformationEx(RelationProcessorCore, None, &mut len);
        // u64s keep the records aligned.
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            Some(buf.as_mut_ptr().cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()),
            &mut len,
        )
        .ok()?;
        let base = buf.as_ptr().cast::<u8>();
        let (mut off, mut classes) = (0usize, Vec::new());
        // Records are variable-size (shorter than the struct's union), so
        // only the header has to fit: Relationship + Size.
        while off + 8 <= len as usize {
            let rec = &*base.add(off).cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
            if rec.Relationship == RelationProcessorCore {
                classes.push(rec.Anonymous.Processor.EfficiencyClass);
            }
            if rec.Size == 0 {
                break;
            }
            off += rec.Size as usize;
        }
        Some(classes)
    }
}

/// The CPU's brand string, e.g. "Intel(R) Core(TM) i7-14700".
fn cpu_name() -> String {
    use std::arch::x86_64::__cpuid;
    if __cpuid(0x8000_0000).eax < 0x8000_0004 {
        return "Unknown CPU".into();
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002..=0x8000_0004u32 {
        let r = __cpuid(leaf);
        for v in [r.eax, r.ebx, r.ecx, r.edx] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    String::from_utf8_lossy(&bytes).trim_matches(char::from(0)).trim().to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn detects_this_machine() {
        let m = super::machine();
        eprintln!("{}: {}", m.cpu, m.summary());
        assert!(m.fast_cores >= 1 && m.fast_cores <= m.cores && m.cores <= m.logical);
    }
}
