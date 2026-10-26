//! Output to the OBS Virtual Camera, and its driver's setup.
//!
//! The OBS Virtual Camera DirectShow source (installed with OBS Studio 26+)
//! reads frames from a named shared-memory queue, `OBSVirtualCamVideo`: a small
//! header followed by three NV12 frame slots. This writes that queue directly,
//! the way OBS itself (and pyvirtualcam) do, so OBS doesn't need to be running.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use anyhow::{bail, Result};
use rayon::prelude::*;
use windows::core::{w, HSTRING, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS, FILE_MAP_READ,
    MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};
use windows::Win32::System::Registry::{
    RegCloseKey, RegGetValueW, RegOpenKeyExW, HKEY, HKEY_LOCAL_MACHINE, KEY_READ, RRF_RT_REG_SZ,
};

use crate::imgproc::Frame;

/// The name apps list the camera under.
pub const DEVICE_NAME: &str = "OBS Virtual Camera";

const STATE_STARTING: u32 = 1;
const STATE_READY: u32 = 2;
const STATE_STOPPING: u32 = 3;
const FRAME_HEADER_SIZE: usize = 32;

/// `struct queue_header` from OBS's plugins/win-dshow/shared-memory-queue.c.
#[repr(C)]
struct QueueHeader {
    write_idx: AtomicU32,
    read_idx: AtomicU32,
    state: AtomicU32,
    offsets: [u32; 3],
    kind: u32,
    cx: u32,
    cy: u32,
    interval: u64,
    reserved: [u32; 8],
}

// Must match the C struct the DirectShow filter reads.
const _: () = assert!(std::mem::size_of::<QueueHeader>() == 80);

const fn align32(n: usize) -> usize {
    (n + 31) & !31
}

/// A live virtual-camera output.
pub struct VirtualCam {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    width: usize,
    height: usize,
    epoch: Instant,
    nv12: Vec<u8>,
}

// The mapping is process-wide memory; nothing about it is tied to a thread.
unsafe impl Send for VirtualCam {}

impl VirtualCam {
    pub fn open(width: usize, height: usize, fps: f64) -> Result<Self> {
        if width % 2 != 0 || height % 2 != 0 {
            bail!("virtual camera size must be even, got {width}x{height}");
        }
        if !installed() {
            bail!("The OBS Virtual Camera is not installed / registered (virtual camera backend missing)");
        }
        let frame_size = width * height * 3 / 2;
        let mut offsets = [0u32; 3];
        let mut size = align32(std::mem::size_of::<QueueHeader>());
        for o in &mut offsets {
            *o = size as u32;
            size = align32(size + frame_size + FRAME_HEADER_SIZE);
        }
        let name = w!("OBSVirtualCamVideo");
        unsafe {
            // Already in use (OBS's own virtual camera, or another vcam).
            if let Ok(h) = OpenFileMappingW(FILE_MAP_READ.0, false, name) {
                let _ = CloseHandle(h);
                bail!("The OBS Virtual Camera is already in use by another program (is OBS's virtual camera running?)");
            }
            let handle = CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE, 0, size as u32, name)?;
            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
            if view.Value.is_null() {
                let _ = CloseHandle(handle);
                bail!("could not map the virtual camera's shared memory");
            }
            std::ptr::write(
                view.Value as *mut QueueHeader,
                QueueHeader {
                    write_idx: AtomicU32::new(0),
                    read_idx: AtomicU32::new(0),
                    state: AtomicU32::new(STATE_STARTING),
                    offsets,
                    kind: 0, // SHARED_QUEUE_TYPE_VIDEO
                    cx: width as u32,
                    cy: height as u32,
                    interval: (10_000_000.0 / fps.max(1.0)) as u64,
                    reserved: [0; 8],
                },
            );
            Ok(Self { handle, view, width, height, epoch: Instant::now(), nv12: vec![0; frame_size] })
        }
    }

    pub fn device(&self) -> &'static str {
        DEVICE_NAME
    }

    fn header(&self) -> &QueueHeader {
        unsafe { &*(self.view.Value as *const QueueHeader) }
    }

    /// Publish one BGR frame (must be the size the camera was opened with).
    pub fn send(&mut self, frame: &Frame) -> Result<()> {
        if frame.size() != (self.width, self.height) {
            bail!("frame is {}x{}, virtual camera is {}x{}", frame.w, frame.h, self.width, self.height);
        }
        bgr_to_nv12(frame, &mut self.nv12);
        let hdr = self.header();
        let inc = hdr.write_idx.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        let slot = (inc % 3) as usize;
        let base = self.view.Value as *mut u8;
        unsafe {
            let off = hdr.offsets[slot] as usize;
            let ts = self.epoch.elapsed().as_nanos() as u64;
            std::ptr::write_unaligned(base.add(off) as *mut u64, ts);
            std::ptr::copy_nonoverlapping(self.nv12.as_ptr(), base.add(off + FRAME_HEADER_SIZE), self.nv12.len());
        }
        hdr.read_idx.store(inc, Ordering::Release);
        hdr.state.store(STATE_READY, Ordering::Release);
        Ok(())
    }
}

impl Drop for VirtualCam {
    fn drop(&mut self) {
        self.header().state.store(STATE_STOPPING, Ordering::Release);
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            let _ = CloseHandle(self.handle);
        }
    }
}

/// BGR -> NV12, BT.601 limited range (what the virtual camera's consumers assume).
fn bgr_to_nv12(f: &Frame, out: &mut [u8]) {
    let (w, h) = (f.w, f.h);
    let (y_plane, uv_plane) = out.split_at_mut(w * h);
    y_plane
        .par_chunks_mut(w * 2)
        .zip(uv_plane.par_chunks_mut(w))
        .zip(f.data.par_chunks(w * 3 * 2))
        .for_each(|((y2, uv), src)| {
            let (r0, r1) = src.split_at(w * 3);
            let (y0, y1) = y2.split_at_mut(w);
            let luma = |p: &[u8]| -> u8 {
                let (b, g, r) = (p[0] as i32, p[1] as i32, p[2] as i32);
                (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16) as u8
            };
            for x in 0..w {
                y0[x] = luma(&r0[x * 3..]);
                y1[x] = luma(&r1[x * 3..]);
            }
            for x in (0..w).step_by(2) {
                let mut s = [0i32; 3];
                for p in [&r0[x * 3..], &r0[x * 3 + 3..], &r1[x * 3..], &r1[x * 3 + 3..]] {
                    for c in 0..3 {
                        s[c] += p[c] as i32;
                    }
                }
                let (b, g, r) = ((s[0] + 2) >> 2, (s[1] + 2) >> 2, (s[2] + 2) >> 2);
                uv[x] = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
                uv[x + 1] = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
            }
        });
}

// --------------------------------------------------------------------------- //
// Driver (the OBS Virtual Camera DirectShow source)
// --------------------------------------------------------------------------- //

/// CLSID of the OBS Virtual Camera DirectShow filter (stable across OBS 26+).
pub const OBS_VCAM_CLSID: &str = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}";
/// The module DLLs (64- and 32-bit), relative to the OBS Studio folder.
const OBS_VCAM_DLLS: [&str; 2] = [
    r"data\obs-plugins\win-dshow\obs-virtualcam-module64.dll",
    r"data\obs-plugins\win-dshow\obs-virtualcam-module32.dll",
];

/// OBS Studio's install folder as its installer recorded it
/// (HKLM\SOFTWARE\OBS Studio), so a non-default location is found too.
fn obs_install_dir() -> Option<PathBuf> {
    let mut buf = [0u16; 1024];
    let mut len = std::mem::size_of_val(&buf) as u32;
    unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!(r"SOFTWARE\OBS Studio"),
            PCWSTR::null(),
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr().cast()),
            Some(&mut len),
        )
        .ok()
        .ok()?;
    }
    let s = String::from_utf16_lossy(&buf[..len as usize / 2]);
    let s = s.trim_end_matches('\0').trim();
    (!s.is_empty()).then(|| PathBuf::from(s))
}

/// Existing OBS Virtual Camera module DLL paths (64- and 32-bit), from the
/// first OBS folder that has them: the registered install, else the default.
pub fn find_dlls() -> Vec<PathBuf> {
    let default = std::env::var_os("ProgramFiles").map_or_else(|| PathBuf::from(r"C:\Program Files"), PathBuf::from);
    obs_install_dir()
        .into_iter()
        .chain([default.join("obs-studio")])
        .map(|dir| OBS_VCAM_DLLS.iter().map(|d| dir.join(d)).filter(|p| p.is_file()).collect::<Vec<_>>())
        .find(|dlls| !dlls.is_empty())
        .unwrap_or_default()
}

/// True if the OBS Virtual Camera DirectShow source is registered system-wide.
pub fn installed() -> bool {
    let key = HSTRING::from(format!(r"SOFTWARE\Classes\CLSID\{OBS_VCAM_CLSID}"));
    let mut hkey = HKEY::default();
    unsafe {
        let ok = RegOpenKeyExW(HKEY_LOCAL_MACHINE, &key, None, KEY_READ, &mut hkey).is_ok();
        if ok {
            let _ = RegCloseKey(hkey);
        }
        ok
    }
}

pub struct Status {
    pub installed: bool,
    pub obs_found: bool,
}

pub fn status() -> Status {
    Status { installed: installed(), obs_found: !find_dlls().is_empty() }
}

/// regsvr32 the DLLs with a single UAC elevation prompt, then verify.
fn run_regsvr(unregister: bool) -> (bool, String) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let dlls = find_dlls();
    if dlls.is_empty() {
        return (false, "OBS Virtual Camera not found. Install OBS Studio \
                        (https://obsproject.com) — it provides the virtual-camera driver."
            .into());
    }
    let flag = if unregister { "/u /s" } else { "/s" };
    let mut lines = vec!["@echo off".to_string()];
    lines.extend(dlls.iter().map(|d| format!("regsvr32 {flag} \"{}\"", d.display())));
    let bat = std::env::temp_dir().join("vcam_virtualcam_setup.bat");
    if let Err(e) = std::fs::write(&bat, lines.join("\r\n")) {
        return (false, format!("Could not write the setup script: {e}"));
    }
    // Single-quoted PowerShell string: a quote in the path (e.g. a user
    // named O'Brien) is escaped by doubling it.
    let ps = format!("Start-Process -FilePath '{}' -Verb RunAs -Wait", bat.display().to_string().replace('\'', "''"));
    let run = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &ps])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    let _ = std::fs::remove_file(&bat);
    if let Err(e) = run {
        return (false, format!("Elevation failed or was cancelled: {e}"));
    }
    // Trust the registry, not regsvr32's (often-zero) exit code.
    let now = installed();
    if unregister {
        (!now, if now { "Could not unregister (elevation cancelled?)." } else { "Virtual camera uninstalled." }.into())
    } else {
        (now, if now {
            "Virtual camera installed — pick 'OBS Virtual Camera' in your app."
        } else {
            "Registration did not take effect (elevation cancelled?)."
        }
        .into())
    }
}

pub fn install() -> (bool, String) {
    run_regsvr(false)
}

pub fn uninstall() -> (bool, String) {
    run_regsvr(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_of_grey_is_neutral() {
        let f = Frame::filled(4, 2, [128, 128, 128]);
        let mut out = vec![0; 12];
        bgr_to_nv12(&f, &mut out);
        assert!(out[..8].iter().all(|&y| y == 126));
        assert!(out[8..].iter().all(|&c| c == 128));
    }
}
