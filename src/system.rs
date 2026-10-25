//! Process-level Windows tweaks.

use std::sync::Once;

use windows::core::HSTRING;
use windows::Win32::Media::timeBeginPeriod;
use windows::Win32::System::Threading::{
    GetCurrentProcess, ProcessPowerThrottling, SetPriorityClass, SetProcessInformation,
    ABOVE_NORMAL_PRIORITY_CLASS, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
    PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
};
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

/// Stop Windows from throttling us while the window is minimized/hidden.
///
/// Windows 11 treats a process with no visible window as background work: it
/// applies EcoQoS (on hybrid CPUs like 12th-14th gen Intel the matting threads
/// get parked on the slow E-cores) and ignores its timer-resolution request
/// (sleeps round up to ~15.6 ms). Either one makes the virtual camera stutter
/// in the meeting as soon as the app is minimized. Opt out of both, and run at
/// above-normal priority so the meeting app's own load can't starve us.
/// Best-effort: older Windows lacks some of these calls.
pub fn keep_realtime_when_background() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        let proc = GetCurrentProcess();
        // Control the bits but leave them off in StateMask == "never throttle me".
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            StateMask: 0,
        };
        let _ = SetProcessInformation(
            proc,
            ProcessPowerThrottling,
            &state as *const _ as *const _,
            std::mem::size_of_val(&state) as u32,
        );
        let _ = SetPriorityClass(proc, ABOVE_NORMAL_PRIORITY_CLASS);
        timeBeginPeriod(1);
    });
}

/// Open a URL in the default browser.
pub fn open_url(url: &str) {
    unsafe {
        ShellExecuteW(None, &HSTRING::from("open"), &HSTRING::from(url), None, None, SW_SHOWNORMAL);
    }
}
