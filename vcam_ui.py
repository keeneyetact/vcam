#!/usr/bin/env python3
"""
vcam_ui — the control panel for the virtual camera.

  * Live preview front and centre; one big Start / Stop button.
  * One-time virtual-camera setup that only shows up when it's needed (so the
    camera appears in Google Meet, Zoom, Teams, ... as "OBS Virtual Camera").
  * Background picker: Blur, your own image, or a green screen.
  * Advanced settings (performance, edge detail, engine) tucked away.
  * Follows the Windows light/dark theme, is DPI-aware, and remembers settings.

Run with:  python vcam_ui.py
Then in Google Meet:  Settings -> Video -> Camera -> "OBS Virtual Camera".
"""
from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# A PyInstaller --windowed .exe has no console: sys.stdout/stderr are None, and
# the library's print() calls would crash. Route them to a sink before anything
# else imports / runs.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk

import vcam

APP_NAME = "vcam"
OBS_URL = "https://obsproject.com/download"
VCAM_NAME = "OBS Virtual Camera"

# label -> (capture_w, capture_h, out_w, out_h). "Smooth" captures at a fast
# low-res the webcam can deliver at 30 fps, then upscales the matted output.
# The capture size is 16:9 (matches the 1280x720 output) so the upscale is
# uniform — a 4:3 capture would stretch you horizontally — and 360p mattes
# ~20% faster than 480p (fewer pixels through the net).
SPEED_MODES = {
    "Sharp · 720p": (1280, 720, None, None),
    "Smooth · 30 fps": (640, 360, 1280, 720),
}
SPEED_DEFAULT = "Sharp · 720p"

# label -> RVM downsample_ratio (the quality<->speed dial; resolution unchanged).
# Higher = sharper hair/edges but slower; lower = faster but softer. Applies live
# while the camera is on (RVM engine only). 0.375 is RVM's recommended 720p point.
# NOTE: above ~0.5 you leave RVM's trained coarse-resolution range, so the matte
# barely gets sharper while speed craters (0.75 ≈ 6x slower for ~no visible gain).
# The two slowest rows are included so you can judge for yourself; "Fine" is the
# practical sharpest.
QUALITY_PRESETS = {
    "Finest (very slow)": 0.75,
    "Extra fine (slow)": 0.60,
    "Fine": 0.50,
    "Sharper": 0.40,
    "Balanced (recommended)": 0.375,
    "Smoother": 0.30,
    "Fast": 0.25,
    "Fastest": 0.20,
}
QUALITY_DEFAULT = "Balanced (recommended)"

ENGINES = {
    "Auto (recommended)": "auto",
    "RVM — best quality": "rvm",
    "MediaPipe — lightest": "mediapipe",
}
ENGINE_DEFAULT = "Auto (recommended)"

CAMERAS = [f"Camera {i + 1}" for i in range(6)]   # label index == device index

# Blur sigma multiplier range for the "Blur amount" slider. Governs both the
# live "Blur" background mode and the "Soften image" checkbox; 1.0 reproduces
# the original fixed blur amount. Applies live (rebuilds the compositor, no
# camera reconnect).
BLUR_MIN = 0.1
BLUR_MAX = 3.0
BLUR_DEFAULT = 0.25

PALETTES = {
    "light": dict(
        bg="#f6f7f9", surface="#ffffff", field="#ffffff", border="#dfe3e8",
        text="#1f2328", muted="#5f6b7a", accent="#2563eb", accent_hover="#1d4ed8",
        on_accent="#ffffff", danger="#dc2626", danger_hover="#b91c1c",
        success="#15803d", warn="#b45309", tile="#ffffff", tile_sel="#eaf1fe",
        notice="#fff7ed", notice_border="#fed7aa", preview="#111317",
        preview_text="#9aa2ad", pill="#000000",
    ),
    "dark": dict(
        bg="#17191e", surface="#1f2228", field="#272b33", border="#343943",
        text="#e7e9ec", muted="#9aa3af", accent="#3b82f6", accent_hover="#2f6fe0",
        on_accent="#ffffff", danger="#ef4444", danger_hover="#dc2626",
        success="#4ade80", warn="#fbbf24", tile="#272b33", tile_sel="#1d2b48",
        notice="#2a2216", notice_border="#5b4520", preview="#0b0c0f",
        preview_text="#8b939e", pill="#000000",
    ),
}


# ------------------------------------------------------------- platform bits --
def _enable_dpi_awareness():
    """Crisp (non-bitmap-stretched) rendering on high-DPI Windows displays."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def _system_prefers_dark() -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return val == 0
    except OSError:
        return False


def _dark_title_bar(root, dark: bool):
    """Match the Windows 10/11 title bar to the app theme."""
    if not (dark and sys.platform.startswith("win")):
        return
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        val = ctypes.c_int(1)
        for attr in (20, 19):   # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)) == 0:
                break
    except Exception:  # noqa: BLE001
        pass


def _settings_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / APP_NAME / "settings.json"


def friendly_error(raw: str) -> str:
    """Turn a worker exception into a plain-language next step."""
    m = (raw or "").lower()
    if "could not open camera" in m:
        return ("Can't access your webcam. Close other apps that might be using it "
                "(Meet, Zoom, Teams, OBS), or pick a different camera, then try again.")
    if "stopped delivering" in m:
        return "Your webcam stopped sending video. Check it's connected, then press Start camera."
    if "download" in m:
        return "Couldn't download the background model. Check your internet connection and try again."
    if "mediapipe" in m:
        return "The MediaPipe engine isn't included in this version. Set Engine to Auto in Advanced settings."
    if "onnxruntime" in m or "engine" in m:
        return "The background engine couldn't start. Try Engine → Auto in Advanced settings."
    if "pyvirtualcam" in m or "obs" in m or "virtual camera" in m or "backend" in m:
        return ("The virtual camera isn't set up yet. Use “Set up virtual camera” on the right, "
                "then press Start camera.")
    return f"Something went wrong: {raw}"


class Tooltip:
    """Small hover hint. `text` may be a callable for live content."""

    def __init__(self, widget, text, ui):
        self.widget, self.text, self.ui = widget, text, ui
        self._tip = self._job = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._job = self.widget.after(450, self._show)

    def _show(self):
        text = self.text() if callable(self.text) else self.text
        if not text or self._tip:
            return
        c = self.ui.c
        x = self.widget.winfo_rootx() + self.ui.px(12)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + self.ui.px(4)
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=text, justify="left", wraplength=self.ui.px(300),
                 bg=c["field"], fg=c["text"], font=self.ui.f_small,
                 highlightthickness=1, highlightbackground=c["border"],
                 padx=self.ui.px(8), pady=self.ui.px(5)).pack()

    def _hide(self, _e=None):
        if self._job:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


class VCamUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("vcam — Virtual Camera")
        self.worker = vcam.VCamWorker()
        self._preview_imgtk = None
        self._busy = False
        self._last_state = None
        self._save_job = None

        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0)
        self.dark = _system_prefers_dark()
        self.c = PALETTES["dark" if self.dark else "light"]

        s = self._load_settings()
        # Default to the user's image background if one sits next to the app.
        default_img = vcam.HERE / "2.png"
        saved_img = s.get("bg_path")
        if saved_img and Path(saved_img).is_file():
            self.bg_path = saved_img
        else:
            self.bg_path = str(default_img) if default_img.is_file() else None

        def pick(key, allowed, default):
            v = s.get(key, default)
            return v if v in allowed else default

        self.cam_choice = tk.StringVar(value=pick("camera", CAMERAS, CAMERAS[0]))
        self.mirror = tk.BooleanVar(value=bool(s.get("mirror", False)))
        mode = pick("mode", ("blur", "image", "color"), "image" if self.bg_path else "blur")
        self.mode = tk.StringVar(value="blur" if mode == "image" and not self.bg_path else mode)
        self.blur_image = tk.BooleanVar(value=bool(s.get("blur_image", False)))
        try:
            blur = min(BLUR_MAX, max(BLUR_MIN, float(s.get("blur_strength", BLUR_DEFAULT))))
        except (TypeError, ValueError):
            blur = BLUR_DEFAULT
        self.blur_strength = tk.DoubleVar(value=blur)
        self.speed = tk.StringVar(value=pick("speed", SPEED_MODES, SPEED_DEFAULT))
        self.qbalance = tk.StringVar(value=pick("quality", QUALITY_PRESETS, QUALITY_DEFAULT))
        self.engine = tk.StringVar(value=pick("engine", ENGINES, ENGINE_DEFAULT))
        self.one_person = tk.BooleanVar(value=bool(s.get("one_person", True)))
        self.adv_open = bool(s.get("advanced_open", False))

        self._setup_fonts()
        self._setup_style()
        self._build()
        self._sync_bg_controls()
        self._refresh_vc_status()
        self._bind_keys()

        root.configure(bg=self.c["bg"])
        root.minsize(self.px(820), self.px(540))
        root.geometry(f"{self.px(1120)}x{self.px(680)}")
        _dark_title_bar(root, self.dark)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # --------------------------------------------------------------- theme --
    def px(self, n: float) -> int:
        return int(round(n * self.scale))

    def _setup_fonts(self):
        fam = "Segoe UI" if sys.platform.startswith("win") else None
        base = tkfont.nametofont("TkDefaultFont")
        if fam:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                tkfont.nametofont(name).configure(family=fam, size=10)
        family = base.actual("family")
        semi = "Segoe UI Semibold" if fam else family
        self.f_body = tkfont.Font(family=family, size=10)
        self.f_small = tkfont.Font(family=family, size=9)
        self.f_title = tkfont.Font(family=semi, size=14, weight="normal" if fam else "bold")
        self.f_section = tkfont.Font(family=semi, size=9, weight="normal" if fam else "bold")
        self.f_button = tkfont.Font(family=semi, size=12, weight="normal" if fam else "bold")
        self.f_preview = tkfont.Font(family=family, size=12)
        self.f_pill = tkfont.Font(family=semi, size=9, weight="normal" if fam else "bold")

    def _setup_style(self):
        c, p = self.c, self.px
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure(".", background=c["surface"], foreground=c["text"], font=self.f_body,
                     bordercolor=c["border"], lightcolor=c["surface"], darkcolor=c["surface"],
                     troughcolor=c["border"], focuscolor=c["accent"],
                     selectbackground=c["accent"], selectforeground=c["on_accent"],
                     fieldbackground=c["field"], insertcolor=c["text"])
        st.configure("TFrame", background=c["surface"])
        st.configure("Bg.TFrame", background=c["bg"])
        st.configure("TLabel", background=c["surface"], foreground=c["text"])
        st.configure("Title.TLabel", font=self.f_title)
        st.configure("Muted.TLabel", foreground=c["muted"], font=self.f_small)
        st.configure("Footer.TLabel", background=c["bg"], foreground=c["muted"], font=self.f_small)
        st.configure("Section.TLabel", foreground=c["muted"], font=self.f_section)
        st.configure("Ok.TLabel", foreground=c["success"], font=self.f_small)
        st.configure("TSeparator", background=c["border"])

        # Buttons: primary (start), danger (stop), secondary, and link-style.
        for name, bg, hover in (("Primary", c["accent"], c["accent_hover"]),
                                ("Danger", c["danger"], c["danger_hover"])):
            st.configure(f"{name}.TButton", background=bg, foreground=c["on_accent"],
                         bordercolor=bg, lightcolor=bg, darkcolor=bg, font=self.f_button,
                         padding=(p(16), p(11)), focuscolor=c["on_accent"])
            st.map(f"{name}.TButton",
                   background=[("disabled", c["border"]), ("pressed", hover), ("active", hover)],
                   bordercolor=[("active", hover)], lightcolor=[("active", hover)],
                   darkcolor=[("active", hover)],
                   foreground=[("disabled", c["muted"])])
        st.configure("TButton", background=c["field"], foreground=c["text"],
                     bordercolor=c["border"], lightcolor=c["field"], darkcolor=c["field"],
                     padding=(p(10), p(5)))
        st.map("TButton", background=[("disabled", c["surface"]), ("active", c["tile_sel"])],
               foreground=[("disabled", c["muted"])],
               bordercolor=[("active", c["accent"])])
        st.configure("Link.TButton", background=c["surface"], foreground=c["accent"],
                     bordercolor=c["surface"], lightcolor=c["surface"], darkcolor=c["surface"],
                     padding=(0, p(4)), font=self.f_section)
        st.map("Link.TButton", background=[("active", c["surface"])],
               bordercolor=[("active", c["surface"])],
               foreground=[("active", c["accent_hover"])])

        # Tiles / segmented controls: radio buttons drawn as bordered cards.
        st.layout("Tile.TRadiobutton", [
            ("Button.border", {"sticky": "nswe", "border": "1", "children": [
                ("Radiobutton.focus", {"sticky": "nswe", "children": [
                    ("Radiobutton.padding", {"sticky": "nswe", "children": [
                        ("Radiobutton.label", {"sticky": "nswe"})]})]})]})])
        st.configure("Tile.TRadiobutton", background=c["tile"], foreground=c["text"],
                     bordercolor=c["border"], lightcolor=c["tile"], darkcolor=c["tile"],
                     padding=(p(6), p(6)), anchor="center", justify="center",
                     font=self.f_small)
        st.map("Tile.TRadiobutton",
               background=[("selected", c["tile_sel"]), ("active", c["tile_sel"])],
               lightcolor=[("selected", c["tile_sel"]), ("active", c["tile_sel"])],
               darkcolor=[("selected", c["tile_sel"]), ("active", c["tile_sel"])],
               bordercolor=[("selected", c["accent"]), ("active", c["muted"])],
               foreground=[("selected", c["accent"])])

        st.configure("TCheckbutton", background=c["surface"], foreground=c["text"],
                     indicatorbackground=c["field"], indicatorforeground=c["on_accent"],
                     padding=(0, p(3)))
        st.map("TCheckbutton", background=[("active", c["surface"])],
               indicatorbackground=[("selected", c["accent"]), ("active", c["tile_sel"])])
        st.configure("TCombobox", fieldbackground=c["field"], background=c["field"],
                     foreground=c["text"], arrowcolor=c["muted"], padding=(p(6), p(4)))
        st.map("TCombobox", fieldbackground=[("readonly", c["field"])],
               selectbackground=[("readonly", c["field"])],
               selectforeground=[("readonly", c["text"])],
               bordercolor=[("focus", c["accent"]), ("hover", c["muted"])])
        self.root.option_add("*TCombobox*Listbox.background", c["field"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", c["on_accent"])
        self.root.option_add("*TCombobox*Listbox.font", self.f_body)
        st.configure("Horizontal.TScale", background=c["accent"], troughcolor=c["border"],
                     bordercolor=c["surface"], lightcolor=c["accent"], darkcolor=c["accent"])
        st.map("Horizontal.TScale", background=[("active", c["accent_hover"])])
        st.configure("Vertical.TScrollbar", background=c["border"], troughcolor=c["surface"],
                     bordercolor=c["surface"], lightcolor=c["border"], darkcolor=c["border"],
                     arrowcolor=c["muted"])

        st.configure("Notice.TFrame", background=c["notice"], bordercolor=c["notice_border"])
        st.configure("Notice.TLabel", background=c["notice"], foreground=c["text"])
        st.configure("NoticeTitle.TLabel", background=c["notice"], foreground=c["warn"],
                     font=self.f_section)

    # ------------------------------------------------------------------ UI --
    def _build(self):
        c, p = self.c, self.px
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left: preview ---------------------------------------------------------
        left = ttk.Frame(self.root, style="Bg.TFrame", padding=(p(16), p(16), p(8), p(8)))
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.preview_area = tk.Frame(left, bg=c["preview"], highlightthickness=0)
        self.preview_area.grid(row=0, column=0, sticky="nsew")
        self.preview = tk.Label(self.preview_area, bg=c["preview"], fg=c["preview_text"],
                                font=self.f_preview, justify="center", bd=0)
        self.preview.place(relx=0.5, rely=0.5, anchor="center")
        self.pill = tk.Label(self.preview_area, bg=c["pill"], fg="#ffffff", font=self.f_pill,
                             padx=p(10), pady=p(4))
        self.error_bar = tk.Label(self.preview_area, bg=c["danger"], fg="#ffffff",
                                  font=self.f_body, justify="left", anchor="w",
                                  padx=p(12), pady=p(8))
        Tooltip(self.error_bar, lambda: f"Details: {self.worker.error}"
                if self.worker.error else "", self)
        self.preview_area.bind("<Configure>", self._on_preview_resize)

        footer = ttk.Frame(left, style="Bg.TFrame")
        footer.grid(row=1, column=0, sticky="ew", pady=(p(10), 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, style="Footer.TLabel",
                  text=f"In Google Meet, Zoom or Teams, open video settings and choose "
                       f"“{VCAM_NAME}” as your camera.").grid(row=0, column=0, sticky="w")

        # Right: scrollable sidebar --------------------------------------------
        side_wrap = tk.Frame(self.root, bg=c["surface"], highlightthickness=1,
                             highlightbackground=c["border"])
        side_wrap.grid(row=0, column=1, sticky="ns", padx=(p(8), p(16)), pady=p(16))
        side_wrap.rowconfigure(0, weight=1)
        canvas = tk.Canvas(side_wrap, bg=c["surface"], highlightthickness=0, bd=0,
                           width=p(330))
        canvas.grid(row=0, column=0, sticky="ns")
        sb = ttk.Scrollbar(side_wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        side = ttk.Frame(canvas, padding=(p(20), p(18)))
        win = canvas.create_window(0, 0, window=side, anchor="nw")

        def _fit(_e=None):
            canvas.itemconfigure(win, width=canvas.winfo_width())
            canvas.configure(scrollregion=canvas.bbox("all"))
            if side.winfo_reqheight() > canvas.winfo_height():
                sb.grid(row=0, column=1, sticky="ns")
            else:
                sb.grid_remove()
                canvas.yview_moveto(0)
        side.bind("<Configure>", _fit)
        canvas.bind("<Configure>", _fit)

        def _wheel(e):
            if sb.winfo_ismapped():
                canvas.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units")
        side_wrap.bind("<Enter>", lambda _e: self.root.bind_all("<MouseWheel>", _wheel))
        side_wrap.bind("<Leave>", lambda _e: self.root.unbind_all("<MouseWheel>"))
        side.columnconfigure(0, weight=1)
        row = 0

        # Brand + virtual-camera status
        brand = ttk.Frame(side)
        brand.grid(row=row, column=0, sticky="ew")
        self._logo = self._load_logo(p(30))
        if self._logo:
            ttk.Label(brand, image=self._logo).grid(row=0, column=0, rowspan=2, padx=(0, p(10)))
        ttk.Label(brand, text="vcam", style="Title.TLabel").grid(row=0, column=1, sticky="w")
        self.vc_chip = ttk.Label(brand, text="", style="Muted.TLabel")
        self.vc_chip.grid(row=1, column=1, sticky="w")
        row += 1

        # Setup notice (only shown when the virtual camera isn't usable yet)
        self.notice = tk.Frame(side, bg=c["notice"], highlightthickness=1,
                               highlightbackground=c["notice_border"])
        self.notice.grid(row=row, column=0, sticky="ew", pady=(p(14), 0))
        self.notice.columnconfigure(0, weight=1)
        ttk.Label(self.notice, text="ONE-TIME SETUP", style="NoticeTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=p(12), pady=(p(10), 0))
        self.notice_text = ttk.Label(self.notice, style="Notice.TLabel", wraplength=p(260),
                                     justify="left")
        self.notice_text.grid(row=1, column=0, sticky="w", padx=p(12), pady=(p(4), p(8)))
        nbtns = tk.Frame(self.notice, bg=c["notice"])
        nbtns.grid(row=2, column=0, sticky="ew", padx=p(12), pady=(0, p(12)))
        self.notice_btn = ttk.Button(nbtns, text="")
        self.notice_btn.pack(side="left")
        self.recheck_btn = ttk.Button(nbtns, text="Check again", style="Link.TButton",
                                      command=self._refresh_vc_status)
        self.recheck_btn.pack(side="left", padx=(p(12), 0))
        self._notice_row = row
        row += 1

        # Start / Stop
        self.toggle_btn = ttk.Button(side, text="Start camera", style="Primary.TButton",
                                     command=self._toggle)
        self.toggle_btn.grid(row=row, column=0, sticky="ew", pady=(p(16), 0))
        Tooltip(self.toggle_btn, "Ctrl+Enter", self)
        row += 1

        # Camera
        row = self._section(side, row, "CAMERA")
        cam_cb = ttk.Combobox(side, textvariable=self.cam_choice, values=CAMERAS,
                              state="readonly")
        cam_cb.grid(row=row, column=0, sticky="ew")
        cam_cb.bind("<<ComboboxSelected>>", lambda _e: self._apply_camera())
        Tooltip(cam_cb, "Using a laptop with an external webcam? Try Camera 2.", self)
        row += 1
        mirror_cb = ttk.Checkbutton(side, text="Mirror image", variable=self.mirror,
                                    command=self._apply_opts)
        mirror_cb.grid(row=row, column=0, sticky="w", pady=(p(6), 0))
        Tooltip(mirror_cb, "Flips the video horizontally — for everyone in the call, "
                           "not just your preview.", self)
        row += 1

        # Background
        row = self._section(side, row, "BACKGROUND")
        tiles = ttk.Frame(side)
        tiles.grid(row=row, column=0, sticky="ew")
        self._thumbs = {}
        self._tiles = {}
        for i, (val, label) in enumerate((("blur", "Blur"), ("image", "Image"),
                                          ("color", "Green screen"))):
            tiles.columnconfigure(i, weight=1, uniform="tile")
            t = ttk.Radiobutton(tiles, text=label, value=val, variable=self.mode,
                                style="Tile.TRadiobutton", compound="top",
                                command=self._apply_bg)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else p(6), 0))
            self._tiles[val] = t
        Tooltip(self._tiles["color"], "Solid green — for apps (like OBS) that key it out.", self)
        self._render_thumbs()
        row += 1

        self.img_row = ttk.Frame(side)
        self.img_row.grid(row=row, column=0, sticky="ew", pady=(p(10), 0))
        self.img_row.columnconfigure(0, weight=1)
        self.img_label = ttk.Label(self.img_row, text=self._img_name(), style="Muted.TLabel")
        self.img_label.grid(row=0, column=0, sticky="w")
        change = ttk.Button(self.img_row, text="Change image…", command=self._choose_image)
        change.grid(row=0, column=1, sticky="e")
        Tooltip(change, "Ctrl+O", self)
        self.soften_cb = ttk.Checkbutton(self.img_row, text="Soften image (out-of-focus look)",
                                         variable=self.blur_image, command=self._apply_bg)
        self.soften_cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=(p(6), 0))
        row += 1

        self.blur_row = ttk.Frame(side)
        self.blur_row.grid(row=row, column=0, sticky="ew", pady=(p(8), 0))
        self.blur_row.columnconfigure(0, weight=1)
        ttk.Label(self.blur_row, text="Blur amount").grid(row=0, column=0, sticky="w")
        self.blur_value_label = ttk.Label(self.blur_row, style="Muted.TLabel",
                                          text=f"{self.blur_strength.get():.1f}×")
        self.blur_value_label.grid(row=0, column=1, sticky="e")
        ttk.Scale(self.blur_row, from_=BLUR_MIN, to=BLUR_MAX, orient="horizontal",
                  variable=self.blur_strength, command=self._on_blur_scale).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(p(4), 0))
        ends = ttk.Frame(self.blur_row)
        ends.grid(row=2, column=0, columnspan=2, sticky="ew")
        ends.columnconfigure(0, weight=1)
        ttk.Label(ends, text="Subtle", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(ends, text="Strong", style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        row += 1

        # Advanced (collapsible)
        ttk.Separator(side).grid(row=row, column=0, sticky="ew", pady=(p(18), p(6)))
        row += 1
        self.adv_btn = ttk.Button(side, style="Link.TButton", command=self._toggle_advanced)
        self.adv_btn.grid(row=row, column=0, sticky="w")
        row += 1
        self.adv = ttk.Frame(side)
        self.adv.grid(row=row, column=0, sticky="ew", pady=(p(6), 0))
        self.adv.columnconfigure(0, weight=1)
        self._build_advanced(self.adv)
        self._show_advanced(self.adv_open)

    def _build_advanced(self, f):
        p = self.px
        r = 0
        ttk.Label(f, text="Performance").grid(row=r, column=0, sticky="w")
        r += 1
        seg = ttk.Frame(f)
        seg.grid(row=r, column=0, sticky="ew", pady=(p(4), 0))
        for i, label in enumerate(SPEED_MODES):
            seg.columnconfigure(i, weight=1, uniform="seg")
            ttk.Radiobutton(seg, text=label, value=label, variable=self.speed,
                            style="Tile.TRadiobutton", command=self._apply_speed).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else p(6), 0))
        r += 1
        ttk.Label(f, style="Muted.TLabel", wraplength=p(280), justify="left",
                  text="Sharp: full 720p detail. Smooth: steadier motion on slower "
                       "computers or webcams.").grid(row=r, column=0, sticky="w", pady=(p(4), 0))
        r += 1

        ttk.Label(f, text="Edge detail").grid(row=r, column=0, sticky="w", pady=(p(12), 0))
        r += 1
        qb = ttk.Combobox(f, textvariable=self.qbalance, values=list(QUALITY_PRESETS),
                          state="readonly")
        qb.grid(row=r, column=0, sticky="ew", pady=(p(4), 0))
        qb.bind("<<ComboboxSelected>>", lambda _e: self._apply_quality())
        Tooltip(qb, "Finer = crisper hair and edges but more CPU. Applies instantly "
                    "(RVM engine).", self)
        r += 1

        ttk.Label(f, text="Engine").grid(row=r, column=0, sticky="w", pady=(p(12), 0))
        r += 1
        ec = ttk.Combobox(f, textvariable=self.engine, values=list(ENGINES), state="readonly")
        ec.grid(row=r, column=0, sticky="ew", pady=(p(4), 0))
        ec.bind("<<ComboboxSelected>>", lambda _e: self._apply_engine())
        r += 1

        op = ttk.Checkbutton(f, text="Only show one person", variable=self.one_person,
                             command=self._apply_oneperson)
        op.grid(row=r, column=0, sticky="w", pady=(p(10), 0))
        Tooltip(op, "Hides people walking behind you. Turn off to keep everyone in frame.", self)
        r += 1

        ttk.Label(f, text="Virtual camera driver").grid(row=r, column=0, sticky="w",
                                                         pady=(p(12), 0))
        r += 1
        vb = ttk.Frame(f)
        vb.grid(row=r, column=0, sticky="w", pady=(p(4), 0))
        self.install_btn = ttk.Button(vb, text="Reinstall", command=self._install)
        self.install_btn.pack(side="left")
        self.uninstall_btn = ttk.Button(vb, text="Uninstall", command=self._uninstall)
        self.uninstall_btn.pack(side="left", padx=(p(6), 0))
        r += 1

        ttk.Label(f, style="Muted.TLabel", wraplength=p(280), justify="left",
                  text="Changes apply instantly. Changing performance, engine or camera "
                       "briefly reconnects the camera.").grid(
            row=r, column=0, sticky="w", pady=(p(14), 0))

    def _section(self, parent, row, title) -> int:
        ttk.Label(parent, text=title, style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(self.px(20), self.px(6)))
        return row + 1

    def _load_logo(self, size):
        for name in ("icon-256.png", "logo.png"):
            path = vcam.BUNDLE_DIR / "assets" / name
            try:
                if path.is_file():
                    img = Image.open(path).convert("RGBA")
                    img.thumbnail((size, size), Image.LANCZOS)
                    return ImageTk.PhotoImage(img)
            except Exception:  # noqa: BLE001
                pass
        return None

    # ---------------------------------------------------------- thumbnails --
    def _thumb_size(self):
        return self.px(84), self.px(48)

    def _with_person(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        d = ImageDraw.Draw(img)
        col = (38, 42, 50)
        hr = h * 0.17
        cx, cy = w / 2, h * 0.42
        d.ellipse((cx - hr, cy - hr, cx + hr, cy + hr), fill=col)
        d.ellipse((cx - w * 0.24, h * 0.66, cx + w * 0.24, h * 1.35), fill=col)
        return img

    def _render_thumbs(self):
        tw, th = self._thumb_size()
        # Blur: a soft, out-of-focus room.
        blur = Image.new("RGB", (tw, th), (196, 184, 164))
        d = ImageDraw.Draw(blur)
        d.rectangle((0, 0, tw * 0.45, th), fill=(120, 146, 176))
        d.rectangle((tw * 0.62, th * 0.15, tw * 0.9, th * 0.7), fill=(232, 220, 196))
        d.rectangle((0, th * 0.78, tw, th), fill=(140, 112, 90))
        blur = blur.filter(ImageFilter.GaussianBlur(self.px(5)))
        # Image: the chosen picture (or a "+" placeholder).
        if self.bg_path:
            try:
                pic = ImageOps.fit(Image.open(self.bg_path).convert("RGB"), (tw, th))
            except Exception:  # noqa: BLE001
                pic = None
        else:
            pic = None
        if pic is None:
            pic = Image.new("RGB", (tw, th), (210, 214, 220) if not self.dark else (60, 66, 76))
            d = ImageDraw.Draw(pic)
            m, a = (tw / 2, th / 2), self.px(9)
            fg = (110, 118, 130)
            d.line((m[0] - a, m[1], m[0] + a, m[1]), fill=fg, width=self.px(2))
            d.line((m[0], m[1] - a, m[0], m[1] + a), fill=fg, width=self.px(2))
            image = pic
        else:
            image = self._with_person(pic)
        green = self._with_person(Image.new("RGB", (tw, th), (0, 177, 64)))
        for key, img in (("blur", self._with_person(blur)), ("image", image), ("color", green)):
            self._thumbs[key] = ImageTk.PhotoImage(img)
            self._tiles[key].configure(image=self._thumbs[key])

    # -------------------------------------------------------- persistence --
    def _load_settings(self) -> dict:
        try:
            data = json.loads(_settings_path().read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _schedule_save(self):
        if self._save_job:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(500, self._save_settings)

    def _save_settings(self):
        self._save_job = None
        data = dict(
            camera=self.cam_choice.get(), mirror=self.mirror.get(), mode=self.mode.get(),
            bg_path=self.bg_path, blur_image=self.blur_image.get(),
            blur_strength=round(self.blur_strength.get(), 3), speed=self.speed.get(),
            quality=self.qbalance.get(), engine=self.engine.get(),
            one_person=self.one_person.get(), advanced_open=self.adv_open,
        )
        try:
            path = _settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass   # settings are a convenience; never block the app on them

    # ----------------------------------------------------------- keyboard --
    def _bind_keys(self):
        self.root.bind("<Control-Return>", lambda _e: self._toggle())
        self.root.bind("<Control-o>", lambda _e: self._choose_image())
        self.root.bind("<Control-O>", lambda _e: self._choose_image())
        # Re-check the driver when the user comes back (e.g. after installing OBS).
        self.root.bind("<FocusIn>", lambda e: e.widget is self.root and not self._busy
                       and self._refresh_vc_status())

    # -------------------------------------------------------------- actions --
    def _img_name(self):
        return Path(self.bg_path).name if self.bg_path else "No image chosen"

    def _install(self):
        self._vcam_action(vcam.install_virtualcam, "Install", confirm=False)

    def _uninstall(self):
        self._vcam_action(vcam.uninstall_virtualcam, "Uninstall", confirm=True)

    def _vcam_action(self, fn, verb, confirm):
        if self._busy:
            return
        if verb == "Uninstall" and self.worker.running:
            messagebox.showwarning("Virtual camera", "Stop the camera first.")
            return
        if confirm and not messagebox.askyesno(
                "Virtual camera",
                f"Uninstall the virtual camera?\n\nMeet, Zoom and Teams won't be able to "
                f"see vcam until you set it up again. Windows will ask for administrator "
                f"permission."):
            return
        self._set_busy(True, "Waiting for Windows permission…")

        def work():
            ok, msg = fn()
            self.root.after(0, lambda: self._vcam_done(ok, msg))

        threading.Thread(target=work, daemon=True).start()

    def _vcam_done(self, ok, msg):
        self._set_busy(False)
        self._refresh_vc_status()
        if not ok:
            messagebox.showerror("Virtual camera", msg)

    def _set_busy(self, busy, text=None):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.install_btn, self.uninstall_btn, self.notice_btn):
            b.config(state=state)
        if text:
            self.notice_text.config(text=text)
            self.vc_chip.config(text=text, style="Muted.TLabel")

    def _refresh_vc_status(self):
        st = vcam.virtualcam_status()
        if st["installed"]:
            self.vc_chip.config(text="● Ready for Meet, Zoom & Teams", style="Ok.TLabel")
            self.notice.grid_remove()
        else:
            self.vc_chip.config(text="Virtual camera not set up", style="Muted.TLabel")
            if st["obs_found"]:
                self.notice_text.config(
                    text="Let Meet, Zoom and Teams see vcam as a camera. Takes one click — "
                         "Windows will ask for administrator permission.")
                self.notice_btn.config(text="Set up virtual camera", command=self._install)
                self.recheck_btn.pack_forget()
            else:
                self.notice_text.config(
                    text="vcam uses the free OBS Studio virtual-camera driver. Install OBS "
                         "Studio, then come back here. You can still preview without it.")
                self.notice_btn.config(text="Get OBS Studio",
                                       command=lambda: webbrowser.open(OBS_URL))
                self.recheck_btn.pack(side="left", padx=(self.px(12), 0))
            self.notice.grid()
        self.uninstall_btn.config(state="normal" if st["installed"] and not self._busy
                                  else "disabled")
        self.install_btn.config(state="normal" if st["obs_found"] and not self._busy
                                else "disabled")

    def _toggle_advanced(self):
        self.adv_open = not self.adv_open
        self._show_advanced(self.adv_open)
        self._schedule_save()

    def _show_advanced(self, show):
        self.adv_btn.config(text=("▾  Advanced settings" if show else "▸  Advanced settings"))
        (self.adv.grid if show else self.adv.grid_remove)()

    def _sync_bg_controls(self):
        """Only show the controls that matter for the chosen background."""
        mode = self.mode.get()
        (self.img_row.grid if mode == "image" else self.img_row.grid_remove)()
        show_blur = mode == "blur" or (mode == "image" and self.blur_image.get())
        (self.blur_row.grid if show_blur else self.blur_row.grid_remove)()

    def _toggle(self):
        if self.worker.running or self.worker.status == "starting":
            self.worker.stop()
            return
        mode = self.mode.get()
        if mode == "image" and not self.bg_path and not self._choose_image():
            return
        cw, ch, ow, oh = SPEED_MODES[self.speed.get()]
        self.worker.configure(
            camera=self._cam_index(), engine=ENGINES[self.engine.get()], mode=self.mode.get(),
            bg_path=self.bg_path, blur_image=self.blur_image.get(),
            blur_strength=self.blur_strength.get(),
            mirror=self.mirror.get(), multi_person=not self.one_person.get(),
            downsample=QUALITY_PRESETS[self.qbalance.get()],
            width=cw, height=ch, out_width=ow, out_height=oh)
        self.worker.start()

    def _cam_index(self) -> int:
        try:
            return CAMERAS.index(self.cam_choice.get())
        except ValueError:
            return 0

    def _apply_bg(self):
        mode = self.mode.get()
        if mode == "image" and not self.bg_path and not self._choose_image():
            self.mode.set("blur")
            mode = "blur"
        self.worker.set_background(mode, path=self.bg_path, blur_image=self.blur_image.get())
        self._sync_bg_controls()
        self._schedule_save()

    def _on_blur_scale(self, value):
        # Live: rebuilds the compositor with the new sigma (Blur mode and/or
        # Soften image), no camera reconnect.
        v = float(value)
        self.blur_value_label.config(text=f"{v:.1f}×")
        self.worker.set_blur_strength(v)
        self._schedule_save()

    def _apply_opts(self):
        self.worker.configure(mirror=self.mirror.get())
        self._schedule_save()

    def _apply_quality(self):
        # Live quality<->speed balance (RVM only); takes effect on the next frame.
        self.worker.set_quality(QUALITY_PRESETS[self.qbalance.get()])
        self._schedule_save()

    def _apply_oneperson(self):
        # Live (in-place): keep only the largest person, or everyone.
        self.worker.configure(multi_person=not self.one_person.get())
        self._schedule_save()

    def _apply_speed(self):
        # Capture/output size change -> worker reconnects the camera automatically.
        cw, ch, ow, oh = SPEED_MODES[self.speed.get()]
        self.worker.configure(width=cw, height=ch, out_width=ow, out_height=oh)
        self._schedule_save()

    def _apply_engine(self):
        # Engine swap -> worker rebuilds the matting engine (auto reconnect).
        self.worker.configure(engine=ENGINES[self.engine.get()])
        self._schedule_save()

    def _apply_camera(self):
        # Camera change -> worker reopens the device (auto reconnect).
        self.worker.configure(camera=self._cam_index())
        self._schedule_save()

    def _choose_image(self) -> bool:
        path = filedialog.askopenfilename(
            title="Choose a background image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return False
        self.bg_path = path
        self.img_label.config(text=self._img_name())
        self.mode.set("image")
        self._render_thumbs()
        self.worker.set_background("image", path=path, blur_image=self.blur_image.get())
        self._sync_bg_controls()
        self._schedule_save()
        return True

    # ----------------------------------------------------------- main loop --
    def _on_preview_resize(self, _e=None):
        self._last_state = None   # re-layout overlays on the next tick

    def _set_state(self, state):
        """Update button / pill / placeholder only when the state changes."""
        if state == self._last_state:
            return
        self._last_state = state
        c, p = self.c, self.px
        if state in ("running", "starting"):
            self.toggle_btn.config(text="Stop camera", style="Danger.TButton")
        else:
            self.toggle_btn.config(text="Start camera", style="Primary.TButton")

        if state == "starting":
            self.pill.config(text="●  Starting camera…", fg=c["warn"])
            self.pill.place(x=p(12), y=p(12))
            self.preview.config(image="", text="Starting your camera…\n"
                                               "The first start can take a few seconds.")
        elif state == "running":
            self.pill.place(x=p(12), y=p(12))
        else:
            self.pill.place_forget()

        if state == "error":
            msg = friendly_error(self.worker.error or "")
            width = max(p(200), self.preview_area.winfo_width() - p(24))
            self.error_bar.config(text=msg, wraplength=width - p(24))
            self.error_bar.place(x=p(12), rely=1.0, y=-p(12), anchor="sw", width=width)
        else:
            self.error_bar.place_forget()

        if state in ("stopped", "error"):
            self._preview_imgtk = None
            self.preview.config(image="", text="Your camera is off\n\n"
                                               "Press Start camera (Ctrl+Enter) to see your preview")

    def _tick(self):
        st = self.worker.status
        state = "running" if (self.worker.running and st == "running") else st
        self._set_state(state)
        if state == "running":
            dev = f"  ·  {self.worker.device}" if self.worker.device else ""
            self.pill.config(text=f"●  LIVE  ·  {self.worker.fps:.0f} fps{dev}",
                             fg="#ff5a5a")

        frame = self.worker.latest_frame()
        if frame is not None:
            aw = self.preview_area.winfo_width()
            ah = self.preview_area.winfo_height()
            h, w = frame.shape[:2]
            k = min(aw / w, ah / h)
            tw, th = int(w * k), int(h * k)
            if tw > 2 and th > 2:
                interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
                rgb = cv2.cvtColor(cv2.resize(frame, (tw, th), interpolation=interp),
                                   cv2.COLOR_BGR2RGB)
                self._preview_imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.preview.config(image=self._preview_imgtk, text="")

        self.root.after(40, self._tick)

    def _on_close(self):
        if self.worker.running and not messagebox.askokcancel(
                "Quit vcam", "Your virtual camera is on. Quit and turn it off?",
                icon="question"):
            return
        if self._save_job:
            self.root.after_cancel(self._save_job)
        self._save_settings()
        try:
            self.worker.stop()
        finally:
            self.root.destroy()


def _selftest() -> int:
    """`vcam.exe --selftest`: exercise the bundle (onnxruntime + cv2 + RVM model)
    without a camera/GUI, writing the result to vcam_selftest.txt (handy because
    the windowed build has no console)."""
    import numpy as np
    lines, ok = [], True
    try:
        eng = vcam.RvmEngine(640, 480)
        a, fg = eng.process((np.random.rand(480, 640, 3) * 255).astype("uint8"))
        lines.append(f"RVM OK: alpha{a.shape} fg{fg.shape} dsr={eng.dsr} threads={eng.threads}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        lines.append(f"RVM FAIL: {exc!r}")
    lines.append(f"virtualcam_installed={vcam.virtualcam_installed()}")
    report = ("PASS\n" if ok else "FAIL\n") + "\n".join(lines)
    try:
        Path("vcam_selftest.txt").write_text(report, encoding="ascii", errors="replace")
    except OSError:
        pass
    print(report)
    return 0 if ok else 1


def _set_window_icon(root):
    """Set the window/taskbar icon from assets (works from source and frozen)."""
    assets = vcam.BUNDLE_DIR / "assets"
    try:
        ico = assets / "icon.ico"
        if ico.is_file():
            root.iconbitmap(default=str(ico))
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        png = assets / "icon-256.png"
        if png.is_file():
            root._icon_img = tk.PhotoImage(file=str(png))   # keep a reference
            root.iconphoto(True, root._icon_img)
    except Exception:  # noqa: BLE001
        pass


def main():
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _enable_dpi_awareness()
    root = tk.Tk()
    _set_window_icon(root)
    VCamUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
