#!/usr/bin/env python3
"""
vcam_ui — a small control panel for the virtual camera.

  * Install / Uninstall the OBS Virtual Camera (so it appears in Google Meet,
    Zoom, Teams, ... as a camera option).
  * Turn the camera ON / OFF (start/stop feeding it; ON == webcam is in use).
  * Pick the background: blur, an image, or a green screen.
  * Live preview.

Run with:  python vcam_ui.py
Then in Google Meet:  Settings -> Video -> Camera -> "OBS Virtual Camera".
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import vcam

PREVIEW_W = 480


class VCamUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("vcam — Virtual Camera")
        root.resizable(False, False)
        self.worker = vcam.VCamWorker()
        self._preview_imgtk = None
        self._busy = False

        # Default to the user's image background if one sits next to the app.
        default_img = vcam.HERE / "2.png"
        self.bg_path = str(default_img) if default_img.is_file() else None

        self._build()
        self._refresh_vc_status()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # ------------------------------------------------------------------ UI --
    def _build(self):
        pad = dict(padx=8, pady=4)

        # Left column: controls --------------------------------------------
        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="n")

        # Virtual camera install/uninstall
        vc = ttk.LabelFrame(left, text="1. Virtual camera (for Google Meet)")
        vc.grid(row=0, column=0, sticky="ew", **pad)
        self.vc_status = ttk.Label(vc, text="…", wraplength=300, justify="left")
        self.vc_status.grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        self.install_btn = ttk.Button(vc, text="Install", command=self._install)
        self.install_btn.grid(row=1, column=0, sticky="ew", **pad)
        self.uninstall_btn = ttk.Button(vc, text="Uninstall", command=self._uninstall)
        self.uninstall_btn.grid(row=1, column=1, sticky="ew", **pad)

        # On / Off
        oc = ttk.LabelFrame(left, text="2. Camera on / off")
        oc.grid(row=1, column=0, sticky="ew", **pad)
        self.toggle_btn = ttk.Button(oc, text="Turn ON", command=self._toggle)
        self.toggle_btn.grid(row=0, column=0, sticky="ew", **pad)
        self.run_status = ttk.Label(oc, text="Stopped", foreground="gray")
        self.run_status.grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(oc, text="Camera #").grid(row=1, column=0, sticky="e", **pad)
        self.cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(oc, from_=0, to=8, width=4, textvariable=self.cam_idx).grid(
            row=1, column=1, sticky="w", **pad)

        # Background
        bgf = ttk.LabelFrame(left, text="3. Background")
        bgf.grid(row=2, column=0, sticky="ew", **pad)
        self.mode = tk.StringVar(value="image" if self.bg_path else "blur")
        ttk.Radiobutton(bgf, text="Blur", value="blur", variable=self.mode,
                        command=self._apply_bg).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(bgf, text="Image", value="image", variable=self.mode,
                        command=self._apply_bg).grid(row=0, column=1, sticky="w", **pad)
        ttk.Radiobutton(bgf, text="Green screen", value="color", variable=self.mode,
                        command=self._apply_bg).grid(row=0, column=2, sticky="w", **pad)
        ttk.Button(bgf, text="Choose image…", command=self._choose_image).grid(
            row=1, column=0, sticky="ew", **pad)
        self.img_label = ttk.Label(bgf, text=self._img_name(), foreground="#555")
        self.img_label.grid(row=1, column=1, columnspan=2, sticky="w", **pad)

        # Options
        op = ttk.LabelFrame(left, text="4. Options")
        op.grid(row=3, column=0, sticky="ew", **pad)
        ttk.Label(op, text="Engine").grid(row=0, column=0, sticky="e", **pad)
        self.engine = tk.StringVar(value="auto")
        ttk.Combobox(op, textvariable=self.engine, values=["auto", "rvm", "mediapipe"],
                     width=10, state="readonly").grid(row=0, column=1, sticky="w", **pad)
        self.mirror = tk.BooleanVar(value=False)
        ttk.Checkbutton(op, text="Mirror", variable=self.mirror,
                        command=self._apply_opts).grid(row=0, column=2, sticky="w", **pad)
        self.one_person = tk.BooleanVar(value=True)
        ttk.Checkbutton(op, text="Only one person", variable=self.one_person).grid(
            row=1, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(op, text="(engine/camera/one-person apply on next ON)",
                  foreground="#888").grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        # Right column: preview --------------------------------------------
        right = ttk.LabelFrame(self.root, text="Preview")
        right.grid(row=0, column=1, sticky="n", **pad)
        self.preview = ttk.Label(right, text="(camera off)", anchor="center",
                                 width=40, background="#222", foreground="#aaa")
        self.preview.grid(row=0, column=0, padx=6, pady=6)
        self.fps_label = ttk.Label(right, text="")
        self.fps_label.grid(row=1, column=0, sticky="w", padx=6)

        ttk.Label(self.root,
                  text="In Google Meet: Settings ▸ Video ▸ Camera ▸ \"OBS Virtual Camera\"",
                  foreground="#444").grid(row=1, column=0, columnspan=2, sticky="w", **pad)

    # -------------------------------------------------------------- actions --
    def _img_name(self):
        return Path(self.bg_path).name if self.bg_path else "(none)"

    def _install(self):
        self._vcam_action(vcam.install_virtualcam, "Install")

    def _uninstall(self):
        self._vcam_action(vcam.uninstall_virtualcam, "Uninstall")

    def _vcam_action(self, fn, verb):
        if self._busy:
            return
        if verb == "Uninstall" and self.worker.running:
            messagebox.showwarning("Virtual Camera", "Turn the camera OFF first.")
            return
        if not messagebox.askyesno(
            "Virtual Camera",
            f"{verb} the OBS Virtual Camera now?\n\n"
            "This needs administrator rights — a Windows UAC prompt will appear."):
            return
        self._set_busy(True, f"{verb}ing…")

        def work():
            ok, msg = fn()
            self.root.after(0, lambda: self._vcam_done(ok, msg))

        threading.Thread(target=work, daemon=True).start()

    def _vcam_done(self, ok, msg):
        self._set_busy(False)
        self._refresh_vc_status()
        (messagebox.showinfo if ok else messagebox.showerror)("Virtual Camera", msg)

    def _set_busy(self, busy, text=None):
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.install_btn.config(state=state)
        self.uninstall_btn.config(state=state)
        if text:
            self.vc_status.config(text=text)

    def _refresh_vc_status(self):
        st = vcam.virtualcam_status()
        if st["installed"]:
            self.vc_status.config(text="Installed ✓  (\"OBS Virtual Camera\" is available)",
                                  foreground="green")
        elif st["obs_found"]:
            self.vc_status.config(text="Not registered — click Install.", foreground="#a60")
        else:
            self.vc_status.config(
                text="OBS Studio not found. Install it from obsproject.com, then Install here.",
                foreground="red")

    def _toggle(self):
        if self.worker.running or self.worker.status == "starting":
            self.toggle_btn.config(text="Turn ON")
            self.worker.stop()
            return
        mode = self.mode.get()
        if mode == "image" and not self.bg_path and not self._choose_image():
            return
        self.worker.configure(
            camera=self.cam_idx.get(), engine=self.engine.get(), mode=mode,
            bg_path=self.bg_path, mirror=self.mirror.get(),
            multi_person=not self.one_person.get())
        self.toggle_btn.config(text="Turn OFF")
        self.worker.start()

    def _apply_bg(self):
        mode = self.mode.get()
        if mode == "image" and not self.bg_path and not self._choose_image():
            self.mode.set("blur")
            mode = "blur"
        self.worker.set_background(mode, path=self.bg_path)

    def _apply_opts(self):
        self.worker.configure(mirror=self.mirror.get())

    def _choose_image(self) -> bool:
        path = filedialog.askopenfilename(
            title="Choose a background image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return False
        self.bg_path = path
        self.img_label.config(text=self._img_name())
        self.mode.set("image")
        self.worker.set_background("image", path=path)
        return True

    # ----------------------------------------------------------- main loop --
    def _tick(self):
        st = self.worker.status
        if self.worker.running and st == "running":
            self.run_status.config(text="● Running", foreground="green")
            self.toggle_btn.config(text="Turn OFF")
            self.fps_label.config(text=f"{self.worker.fps:.0f} fps · {self.worker.device or ''}")
        elif st == "starting":
            self.run_status.config(text="Starting…", foreground="#a60")
            self.toggle_btn.config(text="Turn OFF")
        elif st == "error":
            self.run_status.config(text="Error (see below)", foreground="red")
            self.toggle_btn.config(text="Turn ON")
            self.fps_label.config(text=self.worker.error or "")
        else:
            self.run_status.config(text="Stopped", foreground="gray")
            self.toggle_btn.config(text="Turn ON")
            self.fps_label.config(text="")

        frame = self.worker.latest_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            disp = cv2.resize(frame, (PREVIEW_W, int(h * PREVIEW_W / w)))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._preview_imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.config(image=self._preview_imgtk, text="")
        elif not self.worker.running:
            self.preview.config(image="", text="(camera off)")

        self.root.after(40, self._tick)

    def _on_close(self):
        try:
            self.worker.stop()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    VCamUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
