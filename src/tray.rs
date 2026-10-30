//! The notification-area (tray) icon. Closing the window hides vcam here
//! instead of quitting, so the virtual camera keeps running; clicking the icon
//! brings the window back and its right-click menu quits.
//!
//! The icon lives on its own thread with a hidden window and message loop, so
//! it answers clicks while the egui window is hidden (and not repainting).

use std::cell::RefCell;
use std::sync::mpsc;

use eframe::egui;
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Shell::{
    Shell_NotifyIconW, NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_TIP, NIIF_INFO, NIM_ADD, NIM_DELETE, NIM_MODIFY,
    NOTIFYICONDATAW,
};
use windows::Win32::UI::WindowsAndMessaging::*;

const WM_TRAY: u32 = WM_APP + 1;
const ID_OPEN: usize = 1;
const ID_QUIT: usize = 2;
const TIP: &str = "vcam — Virtual Camera";

pub enum Event {
    /// "Quit" in the tray menu (the window is already shown again).
    Quit,
}

pub struct Tray {
    /// The tray thread's hidden window, which owns the icon.
    icon_wnd: isize,
    /// The egui window.
    main: isize,
    pub events: mpsc::Receiver<Event>,
}

struct State {
    main: isize,
    ctx: egui::Context,
    tx: mpsc::Sender<Event>,
    nid: NOTIFYICONDATAW,
    taskbar_created: u32,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

impl Tray {
    /// Adds the icon for the egui window `main`; `None` if the shell refuses.
    pub fn new(ctx: &egui::Context, main: isize) -> Option<Self> {
        let (tx, events) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || unsafe { run(main, ctx, tx, ready_tx) });
        let icon_wnd = ready_rx.recv().ok()??;
        Some(Self { icon_wnd, main, events })
    }

    pub fn hide_window(&self) {
        unsafe {
            let _ = ShowWindow(HWND(self.main as _), SW_HIDE);
        }
    }

    /// A balloon notification from the icon.
    pub fn notify(&self, title: &str, text: &str) {
        let mut nid = self.id();
        nid.uFlags = NIF_INFO;
        nid.dwInfoFlags = NIIF_INFO;
        copy_wide(&mut nid.szInfoTitle, title);
        copy_wide(&mut nid.szInfo, text);
        unsafe {
            let _ = Shell_NotifyIconW(NIM_MODIFY, &nid);
        }
    }

    fn id(&self) -> NOTIFYICONDATAW {
        NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: HWND(self.icon_wnd as _),
            uID: 1,
            ..Default::default()
        }
    }
}

impl Drop for Tray {
    fn drop(&mut self) {
        // Remove the icon now: one left behind lingers until hovered.
        unsafe {
            let _ = Shell_NotifyIconW(NIM_DELETE, &self.id());
            let _ = PostMessageW(Some(HWND(self.icon_wnd as _)), WM_CLOSE, WPARAM(0), LPARAM(0));
        }
    }
}

fn copy_wide<const N: usize>(dst: &mut [u16; N], s: &str) {
    let wide: Vec<u16> = s.encode_utf16().take(N - 1).collect();
    dst[..wide.len()].copy_from_slice(&wide);
    dst[wide.len()] = 0;
}

unsafe fn run(main: isize, ctx: egui::Context, tx: mpsc::Sender<Event>, ready: mpsc::Sender<Option<isize>>) {
    let inst = GetModuleHandleW(None).map(|m| m.into()).ok();
    let class = w!("vcam-tray");
    let wc = WNDCLASSW { lpfnWndProc: Some(wndproc), hInstance: inst.unwrap_or_default(), lpszClassName: class, ..Default::default() };
    RegisterClassW(&wc);
    let Ok(hwnd) =
        CreateWindowExW(WINDOW_EX_STYLE(0), class, w!("vcam tray"), WINDOW_STYLE(0), 0, 0, 0, 0, None, None, inst, None)
    else {
        let _ = ready.send(None);
        return;
    };

    // The exe's own icon (resource 1, set by build.rs) at the tray's size.
    let icon = LoadImageW(
        inst,
        PCWSTR(1 as _),
        IMAGE_ICON,
        GetSystemMetrics(SM_CXSMICON),
        GetSystemMetrics(SM_CYSMICON),
        LR_DEFAULTCOLOR,
    )
    .map(|h| HICON(h.0))
    .or_else(|_| LoadIconW(None, IDI_APPLICATION))
    .unwrap_or_default();
    let mut nid = NOTIFYICONDATAW {
        cbSize: size_of::<NOTIFYICONDATAW>() as u32,
        hWnd: hwnd,
        uID: 1,
        uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
        uCallbackMessage: WM_TRAY,
        hIcon: icon,
        ..Default::default()
    };
    copy_wide(&mut nid.szTip, TIP);
    if !Shell_NotifyIconW(NIM_ADD, &nid).as_bool() {
        let _ = DestroyWindow(hwnd);
        let _ = ready.send(None);
        return;
    }
    let taskbar_created = RegisterWindowMessageW(w!("TaskbarCreated"));
    STATE.with_borrow_mut(|s| *s = Some(State { main, ctx, tx, nid, taskbar_created }));
    let _ = ready.send(Some(hwnd.0 as isize));

    let mut msg = MSG::default();
    while GetMessageW(&mut msg, None, 0, 0).as_bool() {
        let _ = TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_TRAY => match lp.0 as u32 {
            WM_LBUTTONUP | WM_LBUTTONDBLCLK => show_main(),
            WM_RBUTTONUP => menu(hwnd),
            _ => {}
        },
        WM_COMMAND => match wp.0 & 0xffff {
            ID_OPEN => show_main(),
            ID_QUIT => {
                // The window handles quitting (it may ask to confirm), and
                // a hidden window doesn't run, so show it first.
                show_main();
                STATE.with_borrow(|s| {
                    if let Some(s) = s {
                        let _ = s.tx.send(Event::Quit);
                        s.ctx.request_repaint();
                    }
                });
            }
            _ => {}
        },
        WM_DESTROY => PostQuitMessage(0),
        _ => {
            // Explorer restarted: the icon has to be added again.
            let readd = STATE.with_borrow(|s| s.as_ref().is_some_and(|s| msg == s.taskbar_created && msg != 0));
            if readd {
                STATE.with_borrow(|s| {
                    if let Some(s) = s {
                        let _ = Shell_NotifyIconW(NIM_ADD, &s.nid);
                    }
                });
            }
            return DefWindowProcW(hwnd, msg, wp, lp);
        }
    }
    LRESULT(0)
}

unsafe fn show_main() {
    STATE.with_borrow(|s| {
        let Some(s) = s else { return };
        let main = HWND(s.main as _);
        let _ = ShowWindow(main, if IsIconic(main).as_bool() { SW_RESTORE } else { SW_SHOW });
        let _ = SetForegroundWindow(main);
        s.ctx.request_repaint();
    });
}

unsafe fn menu(hwnd: HWND) {
    let Ok(menu) = CreatePopupMenu() else { return };
    let _ = AppendMenuW(menu, MF_STRING, ID_OPEN, w!("Open vcam"));
    let _ = AppendMenuW(menu, MF_SEPARATOR, 0, None);
    let _ = AppendMenuW(menu, MF_STRING, ID_QUIT, w!("Quit"));
    let _ = SetMenuDefaultItem(menu, ID_OPEN as u32, 0);
    let mut pt = POINT::default();
    let _ = GetCursorPos(&mut pt);
    // Without this the menu doesn't close when clicking elsewhere.
    let _ = SetForegroundWindow(hwnd);
    let _ = TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, pt.x, pt.y, None, hwnd, None);
    let _ = PostMessageW(Some(hwnd), WM_NULL, WPARAM(0), LPARAM(0));
    let _ = DestroyMenu(menu);
}
