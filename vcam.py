#!/usr/bin/env python3
"""
vcam — real-time webcam background replacement.

Reads frames from a physical webcam, isolates the person with Google
MediaPipe Selfie Segmentation, composites them over a background of your
choice (image, blur, or solid color), and publishes the result as a system
virtual camera via pyvirtualcam so any app (Zoom, Meet, Teams, OBS, ...) can
use it as a normal webcam.

Examples
--------
    # Blur the real background (default when no --background is given)
    python vcam.py

    # Replace background with an image
    python vcam.py --background backgrounds/beach.jpg

    # Solid green screen, mirrored preview, camera #1
    python vcam.py --color 0,177,64 --camera 1 --mirror

    # Headless (no preview window), just feed the virtual cam
    python vcam.py --background office.png --no-preview

Press 'q' (or Esc) in the preview window to quit, or Ctrl+C in the terminal.

Notes
-----
MediaPipe 0.10+ replaced the old ``mp.solutions.selfie_segmentation`` API with
the Tasks API, which loads a ``.tflite`` model file. The required model is
downloaded automatically on first run into ``models/``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.request
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# MediaPipe is optional: the RVM engine is the default and doesn't need it, and
# the packaged .exe ships without it. None == "MediaPipe engine unavailable".
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    mp = mp_python = mp_vision = None

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:  # pragma: no cover
    pyvirtualcam = None
    PixelFormat = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

# Optional Rust kernels (native/, built with maturin) for the per-frame pre/post-
# processing and compositing. Same results as the numpy/OpenCV code they replace,
# which stays as the fallback. None == not built; everything still works.
try:
    import vcam_native
except ImportError:  # pragma: no cover
    vcam_native = None


HERE = Path(__file__).resolve().parent
# When frozen by PyInstaller, bundled data lives under sys._MEIPASS.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", HERE))
MODEL_DIR = HERE / "models"


def _writable_model_dir() -> Path:
    """Where to cache a downloaded model (the .exe's extract dir is read-only/
    temporary, so write to %LOCALAPPDATA%/vcam when frozen)."""
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "vcam"
        return base / "models"
    return MODEL_DIR

# Official Google MediaPipe selfie-segmentation models (free / open-source).
MODELS = {
    "general": {
        "file": "selfie_segmenter.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
               "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    },
    "landscape": {
        "file": "selfie_segmenter_landscape.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
               "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    },
}

# RobustVideoMatting — real-time video matting with temporal coherence. Far
# better edges/hair/hands than the lightweight selfie segmenter. Free/open-source.
RVM_MODEL = {
    "file": "rvm_mobilenetv3_fp32.onnx",
    "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
           "v1.0.0/rvm_mobilenetv3_fp32.onnx",
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _download_model(file: str, url: str, label: str) -> Path:
    """Return a local model path. Prefer a copy bundled into the .exe, then a
    cached download, otherwise download it on first use."""
    for cand in (BUNDLE_DIR / "models" / file, MODEL_DIR / file, _writable_model_dir() / file):
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    path = _writable_model_dir() / file
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vcam] Downloading {label} model -> {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download model from {url}: {exc}")
    print("[vcam] Model ready.")
    return path


def ensure_model(name: str) -> Path:
    spec = MODELS[name]
    return _download_model(spec["file"], spec["url"], f"selfie '{name}'")


def ensure_rvm_model() -> Path:
    return _download_model(RVM_MODEL["file"], RVM_MODEL["url"], "RobustVideoMatting")


# RVM cut just before its deep-guided-filter tail: outputs the coarse guided-
# filter coefficients A (tensor "777") and b ("779") instead of fgr/pha. The
# tail -- upsample A and b to full resolution, out = A*x + b, clip -- is ~20% of
# RVM's time at 720p (two full-res Resize ops plus elementwise passes), so the
# native module does it in one fused pass (vcam_native.guided_upsample) on the
# output thread instead. Built once from the full model; needs the `onnx`
# package only to build it (the .exe ships it prebuilt).
RVM_TRUNK_FILE = "rvm_mobilenetv3_fp32_trunk.onnx"
RVM_TRUNK_OUTPUTS = ["777", "779", "r1o", "r2o", "r3o", "r4o"]
RVM_REC_IN = ["r1i", "r2i", "r3i", "r4i"]
RVM_REC_OUT = ["r1o", "r2o", "r3o", "r4o"]
# The trunk split into three pipeline stages (see _RvmStagePipeline).
RVM_STAGE_FILES = ("rvm_mobilenetv3_fp32_enc.onnx", "rvm_mobilenetv3_fp32_core.onnx",
                   "rvm_mobilenetv3_fp32_tail.onnx")


def _find_model(file: str) -> Path | None:
    for cand in (BUNDLE_DIR / "models" / file, MODEL_DIR / file, _writable_model_dir() / file):
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None


def ensure_rvm_trunk() -> Path | None:
    """Path of the truncated RVM model, building it if needed; None when it's
    unavailable (not bundled and `onnx` isn't installed)."""
    found = _find_model(RVM_TRUNK_FILE)
    if found:
        return found
    try:
        import onnx
        import onnx.utils
    except ImportError:
        return None
    path = _writable_model_dir() / RVM_TRUNK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    full = ensure_rvm_model()
    inputs = [i.name for i in onnx.load(str(full), load_external_data=False).graph.input]
    tmp = path.with_suffix(".tmp")
    onnx.utils.extract_model(str(full), str(tmp), inputs, RVM_TRUNK_OUTPUTS)
    tmp.replace(path)
    return path


def ensure_rvm_stages() -> list[Path] | None:
    """Paths of the encoder / core / tail stage models, building them from the
    trunk if needed; None when unavailable (not bundled, no `onnx`)."""
    found = [_find_model(f) for f in RVM_STAGE_FILES]
    if all(found):
        return found
    try:
        import onnx
        import onnx.utils
    except ImportError:
        return None
    trunk = ensure_rvm_trunk()
    if trunk is None:
        return None
    g = onnx.load(str(trunk)).graph
    init = {i.name for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    # Tensors that depend on the recurrent state (the graph is topologically sorted).
    dep = set(RVM_REC_IN)
    for n in g.node:
        if any(i in dep for i in n.input):
            dep.update(n.output)
    stateful = [n for n in g.node if any(o in dep for o in n.output)]
    # Encoder = the state-free part; its outputs are what stateful nodes read from it.
    enc_out = []
    for n in stateful:
        enc_out += [i for i in n.input if i and i not in dep and i not in init and i not in enc_out]
    # Core = stateful nodes the next frame's state needs; tail = the rest of A/b.
    core, stack = set(), list(RVM_REC_OUT)
    while stack:
        n = prod.get(stack.pop())
        if n is not None and n.name not in core and any(o in dep for o in n.output):
            core.add(n.name)
            stack += list(n.input)
    tail = [n for n in stateful if n.name not in core]
    tail_made = {o for n in tail for o in n.output}
    tail_in = []
    for n in tail:
        tail_in += [i for i in n.input
                    if i and i not in init and i not in tail_made and i not in tail_in]
    core_out = RVM_REC_OUT + [t for t in tail_in if t in prod and prod[t].name in core
                              and t not in RVM_REC_OUT]
    paths = [_writable_model_dir() / f for f in RVM_STAGE_FILES]
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, ins, outs in zip(paths, (["src", "downsample_ratio"], enc_out + RVM_REC_IN, tail_in),
                               (enc_out, core_out, RVM_TRUNK_OUTPUTS[:2])):
        tmp = path.with_suffix(".tmp")
        onnx.utils.extract_model(str(trunk), str(tmp), ins, outs)
        tmp.replace(path)
    return paths


def make_segmenter(model_path: Path) -> mp_vision.ImageSegmenter:
    options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return mp_vision.ImageSegmenter.create_from_options(options)


# --------------------------------------------------------------------------- #
# Background helpers
# --------------------------------------------------------------------------- #
def resize_cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize `img` to fill (width, height), cropping the overflow (no distortion)."""
    ih, iw = img.shape[:2]
    scale = max(width / iw, height / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (nw - width) // 2
    y0 = (nh - height) // 2
    return resized[y0:y0 + height, x0:x0 + width]


def load_background_image(path: str, width: int, height: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read background image: {path}")
    return resize_cover(img, width, height)


def blur_background_image(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Soft, out-of-focus version of a chosen background picture (the Meet-style
    "blurred office photo" look, applied to your own image). Built once per
    background choice rather than per frame, so unlike the live `--blur` mode
    there's no speed budget: downscale->blur->upscale gives a smoother, more
    photographic result than one big-sigma blur on the full-res image.

    `strength` scales the blur sigma; 1.0 reproduces the original fixed amount.
    """
    h, w = img.shape[:2]
    sw, sh = max(1, w // 8), max(1, h // 8)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = 6.0 * max(0.01, strength)
    cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=small)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def solid_color_background(color_bgr: tuple[int, int, int], width: int, height: int) -> np.ndarray:
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = color_bgr
    return bg


def parse_color(text: str) -> tuple[int, int, int]:
    """Parse 'R,G,B' (0-255) into a BGR tuple for OpenCV."""
    try:
        r, g, b = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--color must be 'R,G,B', e.g. 0,177,64")
    for v in (r, g, b):
        if not 0 <= v <= 255:
            raise argparse.ArgumentTypeError("color channels must be 0-255")
    return (b, g, r)  # OpenCV is BGR


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    # CAP_DSHOW is far more reliable for USB/integrated webcams on Windows.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low: always the freshest frame
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera {index}. A webcam can only be used by one app "
            "at a time; close anything else using it (Meet/Zoom/Teams/OBS, or another "
            "script), or try a different camera index."
        )
    return cap


_realtime_done = False


def keep_realtime_when_background() -> None:
    """Stop Windows from throttling us while the window is minimized/hidden.

    Windows 11 treats a process with no visible window as background work: it
    applies EcoQoS (on hybrid CPUs like 12th-14th gen Intel the matting threads
    get parked on the slow E-cores) and ignores its timer-resolution request
    (sleeps round up to ~15.6 ms). Either one makes the virtual camera stutter
    in the meeting as soon as the app is minimized. Opt out of both, and run at
    above-normal priority so the meeting app's own load can't starve us.
    No-op off Windows; best-effort (older Windows lacks some of these calls)."""
    global _realtime_done
    if _realtime_done or not sys.platform.startswith("win"):
        return
    _realtime_done = True
    import ctypes
    from ctypes import wintypes

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declare HANDLE types: ctypes' default int would truncate the 64-bit handle.
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    try:
        k32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD)
    except AttributeError:
        pass   # pre-Windows 8
    proc = k32.GetCurrentProcess()
    EXECUTION_SPEED, IGNORE_TIMER_RESOLUTION = 0x1, 0x4
    # Control the bits but leave them off in StateMask == "never throttle me".
    state = _PowerThrottling(1, EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, 0)
    try:
        k32.SetProcessInformation(proc, 4,   # ProcessPowerThrottling
                                  ctypes.byref(state), ctypes.sizeof(state))
    except (AttributeError, OSError):
        pass
    try:
        k32.SetPriorityClass(proc, 0x8000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except (AttributeError, OSError):
        pass


class FrameGrabber:
    """Continuously pulls frames in a background thread so the main loop never
    blocks on the camera. `cap.read()` can block for a full frame interval
    (~55 ms at 18 fps); overlapping that with segmentation/compositing lets us
    run at the camera's true frame rate instead of read+process serialized.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Condition()   # notified on every new frame
        self._frame: np.ndarray | None = None
        self._seq = 0
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "FrameGrabber":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._lock.notify_all()
        with self._lock:
            self.running = False
            self._lock.notify_all()

    def read(self) -> tuple[int, np.ndarray | None]:
        """Return (sequence_number, latest_frame_copy). seq lets the caller skip
        re-processing a frame it has already handled."""
        with self._lock:
            if self._frame is None:
                return self._seq, None
            return self._seq, self._frame.copy()

    def wait_next(self, last_seq: int, timeout: float = 0.5) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than `last_seq` arrives (woken by the camera
        thread, not by polling with sleep — sleeps get coarse when the app is in
        the background). Returns (seq, copy), or (last_seq, None) on timeout or
        when the camera has stopped."""
        with self._lock:
            self._lock.wait_for(lambda: self._seq != last_seq and self._frame is not None
                                or not self.running, timeout)
            if self._seq == last_seq or self._frame is None:
                return last_seq, None
            return self._seq, self._frame.copy()

    def stop(self) -> None:
        self.running = False
        self._thread.join(timeout=1.0)


_HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")


class MaskRefiner:
    """Turns MediaPipe's soft confidence mask into a clean, sharp, single-person
    alpha matte — the difference between an obvious "cut-out" and a background
    that doesn't look virtual.

    Pipeline (raw confidence mask -> final alpha):
      1. guided filter — snaps the fuzzy mask boundary onto the real image edges
         (hair, shoulders) using the camera frame as the guide. This is what
         makes the cut "hug the guy".
      2. temporal EMA on the continuous mask — kills frame-to-frame flicker
         (very visible against a still background) WITHOUT softening the final
         edge, because we smooth before thresholding.
      3. threshold -> hard binary.
      4. keep only the largest connected component => exactly ONE person; every
         other blob/object is dropped.
      5. close internal holes, then erode 1 px so no rim of the original
         background bleeds through (removes the halo "outside the edge").
      6. a 1 px feather just to anti-alias the otherwise hard edge.
    """

    def __init__(self, width, height, *, threshold=0.5, feather=1, erode=1,
                 radius=12, eps=1e-4, temporal=0.5, single_person=True, scale=1.0):
        self.w, self.h = width, height
        self.threshold = float(threshold)
        self.feather = int(feather)
        self.erode = int(erode)
        self.eps = float(eps)
        self.temporal = float(temporal)          # weight of the current frame; 1.0 disables EMA
        self.single = bool(single_person)
        self.scale = float(scale)
        self.sw = max(64, int(round(width * self.scale)))
        self.sh = max(64, int(round(height * self.scale)))
        self.radius = max(1, int(round(radius * self.scale)))
        self._prev = None                        # previous continuous mask (working res)
        self._close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if not _HAS_XIMGPROC:
            print("[vcam] NOTE: cv2.ximgproc unavailable; edges will be softer. "
                  "Install opencv-contrib-python for sharp edge refinement.")

    def refine(self, frame_bgr, raw_mask):
        """Return a float32 alpha matte in [0, 1] at full (w, h)."""
        mask = np.ascontiguousarray(raw_mask, dtype=np.float32)

        # --- work at `scale` resolution for speed ---
        if (self.sw, self.sh) != (self.w, self.h):
            guide = cv2.resize(frame_bgr, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.sw, self.sh), interpolation=cv2.INTER_LINEAR)
        else:
            guide = frame_bgr
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 1. edge-aware refinement: pull the boundary onto real image edges.
        if _HAS_XIMGPROC:
            mask = cv2.ximgproc.guidedFilter(guide_gray, mask, self.radius, self.eps)

        # 2. temporal stabilization on the continuous mask (before thresholding).
        #    Motion-adaptive: where the mask is stable we smooth hard (kills the
        #    flicker that screams "virtual"); where it's moving we follow the
        #    current frame so there are no ghost trails behind a moving head.
        if self.temporal < 1.0 and self._prev is not None and self._prev.shape == mask.shape:
            motion = cv2.absdiff(mask, self._prev)
            w = self.temporal + (1.0 - self.temporal) * np.minimum(motion * 4.0, 1.0)
            mask = w * mask + (1.0 - w) * self._prev
        self._prev = mask

        # 3. hard threshold.
        binary = (mask > self.threshold).astype(np.uint8)

        # 4. keep only the largest blob => one person.
        if self.single:
            binary = self._largest_component(binary)

        # 5. fill internal holes, then pull the edge in to kill the background halo.
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close_k)
        if self.erode > 0:
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode * 2 + 1,) * 2)
            binary = cv2.erode(binary, ek)

        # back to full resolution.
        alpha = cv2.resize(binary * 255, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # 6. thin feather just to anti-alias the hard edge.
        if self.feather > 0:
            k = self.feather * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)

        return alpha.astype(np.float32) / 255.0

    @staticmethod
    def _largest_component(binary):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return binary                         # nothing detected; leave as-is
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == biggest).astype(np.uint8)


class Compositor:
    """Alpha-blends the person over a background using a precomputed alpha matte.

    Optimised hot path (~8 ms/frame at 720p vs ~19 ms for the naive float
    version):
      * integer math instead of float32 (out = bg + (fg-bg)*alpha/255), and
      * reused scratch buffers so we don't reallocate ~10 MB per frame.

    `--blur` mode blurs a quarter-size copy and upscales it (~2.4 ms vs ~15 ms);
    blur is low-frequency so the result is visually identical.

    The matte's edge also gets a light Gaussian feather (see `_feather_ksize`)
    before blending. A hard edge is barely noticeable in blur mode -- a blurred
    background hides it -- but against a sharp Image/Color background it reads
    as an obvious "stencil" cutout. A few px of softening blends the person into
    any background the way Google Meet's segmentation does, at negligible cost
    (single-channel blur vs. the 3-channel ops below).
    """

    def __init__(self, width, height, background_bgr, blur_bg, blur_strength=1.0):
        self.w, self.h = width, height
        # None in blur mode. Made contiguous once (resize_cover returns a cropped
        # view) so the native compositor can take it as-is every frame.
        self.background = None if background_bgr is None else np.ascontiguousarray(background_bgr)
        self.blur_bg = blur_bg
        # Reused scratch buffers (allocated once).
        self._diff = np.empty((height, width, 3), np.int16)
        self._out = np.empty((height, width, 3), np.uint8)
        self._alpha_u8 = np.empty((height, width), np.uint8)
        # Downscaled blur working size (≥ 160x90).
        self._bw = max(160, width // 4)
        self._bh = max(90, height // 4)
        # Blur intensity (--blur mode only); 1.0 reproduces the original fixed
        # sigma. Exposed as the UI's "Blur strength" control.
        self._blur_sigma = 4.0 * max(0.01, blur_strength)
        # Edge-feather kernel, scaled to resolution (odd, ≥3): ~3px @360p,
        # ~5px @720p. Gentle on purpose -- enough to kill the hard edge without
        # softening hair/hand detail away.
        radius = max(1, round(min(width, height) / 300))
        k = radius * 2 + 1
        self._feather_ksize = (k, k)

    def _blurred_background(self, frame_bgr):
        small = cv2.resize(frame_bgr, (self._bw, self._bh), interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(small, (0, 0), sigmaX=self._blur_sigma, sigmaY=self._blur_sigma, dst=small)
        return cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def render(self, frame_bgr, alpha):
        """`alpha` is a HxW float32 matte in [0, 1]; 1 == person, 0 == background."""
        bg = self._blurred_background(frame_bgr) if self.blur_bg else self.background

        if vcam_native is not None and alpha.dtype == np.float32 and alpha.flags.c_contiguous:
            # Feather + blend fused in one multi-threaded pass (~7 ms -> ~2 ms).
            vcam_native.composite(np.ascontiguousarray(frame_bgr), bg, alpha,
                                  self._out, self._feather_ksize[0])
            return self._out

        alpha_u8 = self._alpha_u8
        alpha_u8[:] = (alpha * 255.0).astype(np.uint8)
        cv2.GaussianBlur(alpha_u8, self._feather_ksize, 0, dst=alpha_u8)
        alpha3 = cv2.merge([alpha_u8, alpha_u8, alpha_u8])

        # out = bg + (fg - bg) * alpha / 255, all in int16 then saturated to uint8.
        cv2.subtract(frame_bgr, bg, dst=self._diff, dtype=cv2.CV_16S)
        cv2.multiply(self._diff, alpha3, dst=self._diff, dtype=cv2.CV_16S, scale=1 / 255.0)
        cv2.add(bg, self._diff, dst=self._out, dtype=cv2.CV_8U)
        return self._out


def make_preview(frame_bgr, width, height):
    """BGR frame -> new RGB array of (height, width) for the GUI preview.
    (OpenCV's SIMD resize beats a hand-written Rust one here, and releases the
    GIL too, so there's no native version.)"""
    h, w = frame_bgr.shape[:2]
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.cvtColor(cv2.resize(frame_bgr, (width, height), interpolation=interp),
                        cv2.COLOR_BGR2RGB)


class PassthroughCompositor:
    """"Original" mode: send the camera frame untouched (no background effect)."""

    def render(self, frame_bgr, alpha):
        return frame_bgr


# --------------------------------------------------------------------------- #
# Matting engines — each turns a BGR frame into (alpha matte, foreground|None)
# --------------------------------------------------------------------------- #
class _RvmStagePipeline:
    """RVM run as three models in a pipeline, one thread + onnxruntime session
    each: encoder (state-free) | recurrent decoder core | decoder tail (state-
    free). Frame N+2's encoder, N+1's core and N's tail run at the same time, so
    throughput is set by the slowest stage rather than the sum: +25-35% at 720p
    on an i7-14700 (onnxruntime stops scaling past ~8 threads on this model, so
    three sessions use a many-core CPU better; shared memory bandwidth keeps it
    from the ideal 2.5x). Output is identical to the single model; the cost is
    two frames more latency. Each frame carries its downsample_ratio, so the
    core resets the recurrent state exactly at the frame where it changes.
    """

    def __init__(self, paths, threads):
        self._enc, self._core, self._tail = (
            _build_rvm_session(p, ["CPUExecutionProvider"], t) for p, t in zip(paths, threads))
        self._enc_out = [o.name for o in self._enc.get_outputs()]
        self._core_in = [i.name for i in self._core.get_inputs()]
        self._core_out = [o.name for o in self._core.get_outputs()]
        self._tail_in = [i.name for i in self._tail.get_inputs()]
        self.error = None
        # maxsize=1: bounded latency, and back-pressure reaches the caller.
        self._qs = [queue.Queue(maxsize=1) for _ in range(3)]
        self._threads = [threading.Thread(target=self._stage, args=(k,), daemon=True,
                                          name=f"rvm-stage-{k}") for k in range(3)]
        for th in self._threads:
            th.start()

    def submit(self, src, dsr):
        """Queue one frame's (1,3,H,W) input; returns a Future of (A, b). `src`
        must stay untouched until the encoder has read it (the caller rotates
        buffers). Blocks while the pipeline is full."""
        if self.error is not None:
            raise self.error
        fut = concurrent.futures.Future()
        self._qs[0].put((fut, dsr, src))
        return fut

    def _stage(self, k):
        rec, last_dsr = None, None
        q_in, q_out = self._qs[k], (self._qs[k + 1] if k < 2 else None)
        while (item := q_in.get()) is not None:
            fut, dsr, data = item
            if self.error is not None:
                fut.set_exception(self.error)    # an earlier stage/frame failed
                continue
            try:
                if k == 0:
                    feats = self._enc.run(self._enc_out, {
                        "src": data, "downsample_ratio": np.array([dsr], np.float32)})
                    q_out.put((fut, dsr, dict(zip(self._enc_out, feats))))
                elif k == 1:
                    if dsr != last_dsr:          # new coarse size: state can't carry over
                        rec = [np.zeros([1, 1, 1, 1], np.float32) for _ in range(4)]
                        last_dsr = dsr
                    data.update(zip(RVM_REC_IN, rec))
                    out = self._core.run(self._core_out, {n: data[n] for n in self._core_in})
                    rec = out[:4]
                    data.update(zip(self._core_out, out))
                    q_out.put((fut, dsr, data))
                else:
                    a, b = self._tail.run(RVM_TRUNK_OUTPUTS[:2], {n: data[n] for n in self._tail_in})
                    fut.set_result((a, b))
            except Exception as exc:  # noqa: BLE001  (surfaced via the futures/submit)
                self.error = exc
                fut.set_exception(exc)
        if q_out is not None:
            q_out.put(None)

    def close(self):
        self._qs[0].put(None)
        for th in self._threads:
            th.join(timeout=2.0)


class RvmEngine:
    """RobustVideoMatting via onnxruntime — high-quality, temporally-coherent
    video matting. The recurrent state (r1..r4, fed back every frame) is what
    keeps hair/hands/edges stable under motion, so no extra temporal smoothing
    or guided-filter hacks are needed. Returns a soft alpha and a colour-
    decontaminated foreground (clean hair edges, no old-background colour bleed).
    """

    def __init__(self, width, height, *, downsample=None, single_person=True, threads=None):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed (pip install onnxruntime)")
        # More threads than physical cores hurts this small recurrent net (on a
        # 14700 all-28-logical = 77 ms vs 8 = 48 ms), so cap it. (Only the CPU
        # provider uses these; a GPU provider ignores them.)
        self.threads = int(threads) if threads else _rvm_thread_count()
        # With the native module, run the truncated model and do RVM's full-res
        # tail in Rust on the output thread (see ensure_rvm_trunk); otherwise the
        # stock model. Either way the outputs are (2 tensors, r1o..r4o).
        model = None
        if vcam_native is not None and hasattr(vcam_native, "guided_upsample"):
            try:
                model = ensure_rvm_trunk()
            except Exception as exc:  # noqa: BLE001  (fall back to the stock model)
                print(f"[vcam] RVM trunk model unavailable ({exc}); using the full model")
        self.trunk = model is not None
        # Pick the fastest WORKING execution provider. A GPU (DirectML/CUDA/...)
        # can be several times faster than the CPU — but only a capable one: an
        # integrated or old GPU is often *slower* than a many-core CPU for this
        # small net, and some GPU providers build a session fine yet fail at run
        # time. So we verify and time-trial the GPU and only keep it when it
        # clearly beats the CPU; otherwise we stay on the (reliable) CPU.
        self.sess, self.provider, self.bench = _select_rvm_session(
            model or ensure_rvm_model(), self.threads, width, height)
        self.w, self.h = width, height
        self.single = bool(single_person)
        # downsample_ratio drives the coarse network's resolution and dominates
        # speed. RVM's recommended ~0.375 (0.25 for >=1080p) keeps the refined
        # full-res edges sharp while running fast; the old 512/max formula made
        # low-res inputs *slower* (0.8 at 640px = 62 ms vs 0.375 = 18 ms).
        if downsample:
            self.dsr = float(downsample)
        else:
            self.dsr = 0.25 if max(width, height) >= 1600 else 0.375
        self._rec = [np.zeros([1, 1, 1, 1], np.float32) for _ in range(4)]
        self._dsr = np.array([self.dsr], np.float32)
        self._src = None        # reused NCHW input buffer (no per-frame realloc)
        # On the CPU, run the trunk as a 3-stage pipeline when the stage models
        # are available (see _RvmStagePipeline). A GPU provider keeps the single
        # session. Each in-flight frame needs its own input buffer, so rotate a
        # few: at most one waits in the encoder's queue, one is being encoded and
        # one is being filled.
        self._stages = None
        if self.trunk and self.provider == "CPUExecutionProvider":
            try:
                paths = ensure_rvm_stages()
                if paths:
                    self._stages = _RvmStagePipeline(
                        paths, (max(2, self.threads // 2), self.threads, self.threads))
            except Exception as exc:  # noqa: BLE001  (fall back to the single session)
                print(f"[vcam] RVM stage pipeline unavailable ({exc}); using one session")
        self._src_ring, self._ring_i = [], 0

    def process(self, frame_bgr):
        return self.finish(self.infer(frame_bgr))

    def infer(self, frame_bgr):
        """The serial half: input conversion + inference. Returns the raw model
        outputs for `finish`. Only this has to run frame-by-frame in order (it
        carries the recurrent state); `finish` can run on another thread, so the
        next inference starts while this frame is still being post-processed."""
        h, w = frame_bgr.shape[:2]
        # Fill a reused float32 NCHW buffer directly (avoids astype + transpose
        # + ascontiguousarray copies: ~7.5 ms -> ~2 ms at 720p). RVM expects
        # normalised RGB, so index the BGR frame's channels in RGB order
        # (R=2, G=1, B=0) — this also skips an intermediate cvtColor copy.
        if self._stages is not None:
            if not self._src_ring or self._src_ring[0].shape[2:] != (h, w):
                self._src_ring = [np.empty((1, 3, h, w), np.float32) for _ in range(4)]
            src = self._src_ring[self._ring_i]
            self._ring_i = (self._ring_i + 1) % len(self._src_ring)
        else:
            if self._src is None or self._src.shape[2:] != (h, w):
                self._src = np.empty((1, 3, h, w), np.float32)
            src = self._src
        if vcam_native is not None:
            vcam_native.fill_src(np.ascontiguousarray(frame_bgr), src)
        else:
            src[0, 0], src[0, 1], src[0, 2] = frame_bgr[:, :, 2], frame_bgr[:, :, 1], frame_bgr[:, :, 0]
            np.multiply(src, np.float32(1.0 / 255.0), out=src)
        if self._stages is not None:
            return self._stages.submit(src, self.dsr), frame_bgr
        out0, out1, *self._rec = self.sess.run(
            [], {"src": src, "r1i": self._rec[0], "r2i": self._rec[1],
                 "r3i": self._rec[2], "r4i": self._rec[3], "downsample_ratio": self._dsr}
        )
        # Trunk model: coarse guided-filter coefficients (A, b) + the frame they
        # apply to. Stock model: (fgr, pha).
        return (out0, out1, frame_bgr) if self.trunk else (out0, out1)

    def finish(self, raw):
        """Raw outputs from `infer` -> (alpha HxW float [0,1], foreground BGR uint8)."""
        if self.trunk:
            if self._stages is not None:
                fut, frame_bgr = raw
                a, b = fut.result(timeout=10)
            else:
                a, b, frame_bgr = raw
            alpha, fg = vcam_native.guided_upsample(a, b, np.ascontiguousarray(frame_bgr))
            if self.single:
                alpha = _keep_largest(alpha)
            return alpha, fg
        fgr, pha = raw
        alpha = pha[0, 0]                                            # HxW float [0,1]
        if self.single:
            alpha = _keep_largest(alpha)
        if vcam_native is not None:
            return alpha, vcam_native.fgr_to_bgr(fgr)
        # Foreground RGB (3,H,W) float -> BGR uint8 HWC via per-channel saturating
        # scale + merge (~9.3 ms -> ~1.3 ms at 720p; no transpose copy / cvtColor).
        fg = fgr[0]
        fg_bgr = cv2.merge([cv2.convertScaleAbs(fg[2], alpha=255.0),
                            cv2.convertScaleAbs(fg[1], alpha=255.0),
                            cv2.convertScaleAbs(fg[0], alpha=255.0)])
        return alpha, fg_bgr

    def set_downsample(self, dsr):
        """Live-tune the quality/speed dial. downsample_ratio is a per-frame model
        input, so the change takes effect on the next frame with no session
        rebuild. The recurrent state is sized to the *old* coarse resolution, so
        it must be reset — otherwise the next inference fails on a shape mismatch
        (one frame re-initialises temporal coherence; imperceptible)."""
        dsr = float(dsr)
        if abs(dsr - self.dsr) < 1e-6:
            return
        self.dsr = dsr
        self._dsr = np.array([dsr], np.float32)
        self._rec = [np.zeros([1, 1, 1, 1], np.float32) for _ in range(4)]

    def set_single_person(self, single):
        """Live toggle for keeping only the single largest person."""
        self.single = bool(single)

    def close(self):
        if self._stages is not None:
            self._stages.close()


class MediaPipeEngine:
    """Lightweight MediaPipe selfie segmentation + the sharpening MaskRefiner.
    Fast and dependency-light, but weaker on hands-near-camera, motion and hair.
    """

    def __init__(self, width, height, args):
        if mp is None:
            raise RuntimeError("MediaPipe engine is not available in this build; use the RVM engine.")
        self.segmenter = make_segmenter(ensure_model(args.model))
        self.refiner = MaskRefiner(
            width, height,
            threshold=args.threshold, feather=args.feather, erode=args.erode,
            radius=args.edge_radius, temporal=args.temporal,
            single_person=not args.multi_person, scale=args.mask_scale,
        )
        self._t0 = time.time()
        self._ts = 0

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts = max(self._ts + 1, int((time.time() - self._t0) * 1000))
        result = self.segmenter.segment_for_video(mp_image, self._ts)
        mask = result.confidence_masks[0].numpy_view()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return self.refiner.refine(frame_bgr, mask), None       # no foreground decontamination

    def set_single_person(self, single):
        """Live toggle for keeping only the single largest person."""
        self.refiner.single = bool(single)

    def close(self):
        self.segmenter.close()


def _rvm_thread_count() -> int:
    """A good onnxruntime intra-op thread count for RVM: enough to use the fast
    cores, capped so a many-thread machine doesn't oversubscribe this small net."""
    import os
    return max(2, min(8, os.cpu_count() or 4))


def _rvm_providers() -> list[str]:
    """onnxruntime execution providers for RVM, fastest-available first with a
    guaranteed CPU fallback. A plain ``pip install onnxruntime`` ships CPU only;
    installing ``onnxruntime-directml`` (Windows GPU, incl. Intel/AMD iGPUs) or
    ``onnxruntime-gpu`` (NVIDIA CUDA) lights up the GPU and makes matting several
    times faster. Unavailable providers are filtered out, so this is always safe."""
    if ort is None:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    preferred = ["DmlExecutionProvider", "CUDAExecutionProvider",
                 "ROCMExecutionProvider", "CoreMLExecutionProvider"]
    return [p for p in preferred if p in available] + ["CPUExecutionProvider"]


def _build_rvm_session(model_path, providers, threads, *, quiet=False):
    """Build an RVM InferenceSession on `providers` (fastest first, CPU last)."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if any(p != "CPUExecutionProvider" for p in providers):
        so.enable_mem_pattern = False          # required by the DirectML provider
    if quiet:
        so.log_severity_level = 4              # hush probe-time errors from a bad GPU
    return ort.InferenceSession(str(model_path), so, providers=providers)


def _bench_rvm_session(sess, width, height, n=6) -> float:
    """Median ms/frame over a few dummy inferences. Raises if the provider builds
    a session but can't actually run the model (e.g. some integrated/old GPUs)."""
    src = np.zeros((1, 3, height, width), np.float32)
    rec = [np.zeros([1, 1, 1, 1], np.float32) for _ in range(4)]
    dsr = np.array([0.375], np.float32)

    def one():
        nonlocal rec
        t = time.perf_counter()
        _, _, *rec = sess.run([], {"src": src, "r1i": rec[0], "r2i": rec[1],
                                   "r3i": rec[2], "r4i": rec[3], "downsample_ratio": dsr})
        return (time.perf_counter() - t) * 1000.0

    one(); one()                               # warm up (shader compile / arena alloc)
    times = sorted(one() for _ in range(n))
    return times[len(times) // 2]


def _select_rvm_session(model_path, threads, width, height):
    """Return (session, provider_name, bench) on the fastest working provider.

    Always falls back to the CPU, which is reliable and — for this small net on a
    modern many-core CPU — frequently faster than an integrated/old GPU. A GPU is
    only adopted when it both runs the model *and* clearly beats the CPU.
    `bench` is None when no GPU was tried, else (cpu_ms, gpu_ms, gpu_name)."""
    cpu_sess = _build_rvm_session(model_path, ["CPUExecutionProvider"], threads)
    gpus = [p for p in _rvm_providers() if p != "CPUExecutionProvider"]
    if not gpus:
        return cpu_sess, "CPUExecutionProvider", None
    cpu_ms = _bench_rvm_session(cpu_sess, width, height)
    for prov in gpus:
        try:
            gpu_sess = _build_rvm_session(model_path, [prov, "CPUExecutionProvider"],
                                          threads, quiet=True)
            if gpu_sess.get_providers()[0] != prov:
                continue                       # provider silently declined the model
            gpu_ms = _bench_rvm_session(gpu_sess, width, height)
        except Exception:                      # builds but can't run -> ignore this GPU
            continue
        if gpu_ms < cpu_ms * 0.9:              # only switch if clearly (>10%) faster
            return gpu_sess, prov, (cpu_ms, gpu_ms, prov)
        return cpu_sess, "CPUExecutionProvider", (cpu_ms, gpu_ms, prov)
    return cpu_sess, "CPUExecutionProvider", None


def _keep_largest(alpha):
    """Keep only the single largest person blob in a soft alpha matte; suppress
    the rest while preserving the kept subject's soft edges. Connected components
    run at half resolution (blobs are large) — ~6 ms -> ~3 ms at 720p."""
    if vcam_native is not None and alpha.dtype == np.float32 and alpha.flags.c_contiguous:
        vcam_native.keep_largest(alpha)   # in place, no full-size temporaries
        return alpha
    h, w = alpha.shape[:2]
    small = cv2.resize(alpha, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) if max(h, w) > 640 else alpha
    binary = (small > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return alpha
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((labels == biggest).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    if keep.shape != alpha.shape:
        keep = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST)
    return alpha * keep


def make_engine(args, width, height):
    """Pick the matting engine: RVM if available (best quality), else MediaPipe."""
    want = args.engine
    if want == "auto":
        want = "rvm" if ort is not None else "mediapipe"
    if want == "rvm":
        if ort is None:
            print("[vcam] onnxruntime not installed -> falling back to MediaPipe. "
                  "For best quality: pip install onnxruntime")
            want = "mediapipe"
        else:
            try:
                engine = RvmEngine(width, height,
                                   downsample=args.downsample,
                                   single_person=not args.multi_person,
                                   threads=getattr(args, "threads", None))
                on_cpu = engine.provider == "CPUExecutionProvider"
                accel = f"CPU x{engine.threads} threads" if on_cpu else engine.provider
                print(f"[vcam] Engine: RobustVideoMatting "
                      f"(downsample_ratio={engine.dsr:.2f}, {accel})")
                if engine.bench:               # a GPU was time-trialled
                    cpu_ms, gpu_ms, gpu = engine.bench
                    print(f"[vcam] Provider trial: CPU {cpu_ms:.0f} ms/frame vs "
                          f"{gpu} {gpu_ms:.0f} ms/frame -> using "
                          f"{'GPU' if not on_cpu else 'CPU (faster here)'}.")
                elif on_cpu:
                    print("[vcam] Running on CPU. If you have a powerful (discrete) "
                          "GPU, installing onnxruntime-directml (Windows) or "
                          "onnxruntime-gpu (NVIDIA/CUDA) can speed matting up.")
                return engine
            except Exception as exc:  # noqa: BLE001
                if mp is None:
                    raise RuntimeError(f"RVM engine failed to start: {exc}") from exc
                print(f"[vcam] RVM unavailable ({exc}) -> falling back to MediaPipe.")
                want = "mediapipe"
    if mp is None:
        raise RuntimeError("No matting engine available (onnxruntime/RVM required).")
    print("[vcam] Engine: MediaPipe selfie segmentation")
    return MediaPipeEngine(width, height, args)


# --------------------------------------------------------------------------- #
# Virtual-camera driver (Windows / OBS Virtual Camera DirectShow source)
# --------------------------------------------------------------------------- #
# CLSID of the OBS Virtual Camera DirectShow filter (stable across OBS 26+).
OBS_VCAM_CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
_OBS_VCAM_DLLS = [
    r"C:\Program Files\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module64.dll",
    r"C:\Program Files\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module32.dll",
]


def find_virtualcam_dlls() -> list[str]:
    """Existing OBS Virtual Camera module DLL paths (64- and 32-bit)."""
    return [p for p in _OBS_VCAM_DLLS if Path(p).is_file()]


def virtualcam_installed() -> bool:
    """True if the OBS Virtual Camera DirectShow source is registered system-wide."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SOFTWARE\Classes\CLSID\\" + OBS_VCAM_CLSID)
        winreg.CloseKey(key)
        return True
    except OSError:
        return False


def virtualcam_status() -> dict:
    dlls = find_virtualcam_dlls()
    return {"installed": virtualcam_installed(), "obs_found": bool(dlls), "dlls": dlls}


def _run_regsvr(dlls: list[str], unregister: bool) -> tuple[bool, str]:
    """regsvr32 the DLLs with a single UAC elevation prompt, then verify."""
    if not dlls:
        return False, ("OBS Virtual Camera not found. Install OBS Studio "
                       "(https://obsproject.com) — it provides the virtual-camera driver.")
    flag = "/u /s" if unregister else "/s"
    lines = ["@echo off"] + [f'regsvr32 {flag} "{d}"' for d in dlls]
    bat = Path(tempfile.gettempdir()) / "vcam_virtualcam_setup.bat"
    bat.write_text("\r\n".join(lines), encoding="ascii")
    ps = f"Start-Process -FilePath '{bat}' -Verb RunAs -Wait"
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=180)
    except Exception as exc:  # noqa: BLE001  (includes UAC cancellation)
        return False, f"Elevation failed or was cancelled: {exc}"
    finally:
        try:
            bat.unlink()
        except OSError:
            pass
    # Trust the registry, not regsvr32's (often-zero) exit code.
    now = virtualcam_installed()
    if unregister:
        return (not now), ("Virtual camera uninstalled." if not now
                           else "Could not unregister (elevation cancelled?).")
    return now, ("Virtual camera installed — pick 'OBS Virtual Camera' in your app."
                 if now else "Registration did not take effect (elevation cancelled?).")


def install_virtualcam() -> tuple[bool, str]:
    return _run_regsvr(find_virtualcam_dlls(), unregister=False)


def uninstall_virtualcam() -> tuple[bool, str]:
    return _run_regsvr(find_virtualcam_dlls(), unregister=True)


class OutputPipeline:
    """Runs compositing + virtual-cam send on a worker thread so that work
    overlaps the *next* frame's inference instead of waiting behind it.

    The matte (engine.infer) is a serial ~40 ms wall — each frame feeds the
    recurrent state into the next, so inferences can't overlap each other. But
    the engine's post-processing (single-person isolation, foreground conversion)
    and the compositing + cam.send AFTER inference don't depend on the next
    frame, so doing them on a second thread hides them behind the next
    inference. Frames are handled in submit order and the output is byte-for-byte
    identical to doing it inline — a pure throughput win (~10-15% in Sharp mode),
    costing one extra frame of latency (imperceptible for a webcam).
    """

    def __init__(self, cam, on_output=None):
        self._cam = cam
        self._on_output = on_output          # optional callback(out_bgr)
        self._q = queue.Queue(maxsize=2)     # small => bounded latency + back-pressure
        self._latest = None
        self._lock = threading.Lock()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, compositor, source_bgr, alpha, upscale=None, finish=None):
        """Hand one frame's render+send to the worker. Blocks (back-pressure) if
        the worker is two frames behind; raises if the worker has failed.
        `finish`, if given, is called on the worker first and returns
        (alpha, foreground|None) -- the engine's deferred post-processing."""
        if self.error is not None:
            raise self.error
        self._q.put((compositor, source_bgr, alpha, upscale, finish))

    def latest(self):
        """Most recently sent output frame (for preview); None until the first."""
        with self._lock:
            return self._latest

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            if self.error is not None:
                continue                     # drain quietly so submit() never blocks
            try:
                compositor, source_bgr, alpha, upscale, finish = item
                if finish is not None:
                    alpha, fg = finish()
                    if fg is not None:
                        source_bgr = fg
                out = compositor.render(source_bgr, alpha)
                if upscale is not None and (out.shape[1], out.shape[0]) != upscale:
                    out = cv2.resize(out, upscale, interpolation=cv2.INTER_LINEAR)
                if self._cam is not None:
                    # No sleep_until_next_frame(): the webcam already paces us,
                    # and pacing twice adds a sleep that turns into a stall
                    # (and back-pressure on inference) when timers get coarse.
                    self._cam.send(out)
                with self._lock:
                    self._latest = out
                if self._on_output is not None:
                    self._on_output(out)
            except Exception as exc:  # noqa: BLE001  (surfaced to producer via submit)
                self.error = exc

    def close(self):
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:                   # worker wedged; drop a slot and signal
            try:
                self._q.get_nowait()
                self._q.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Worker — runs the capture→matte→composite→virtual-cam loop in a thread so a
# GUI can start/stop it and swap the background live.
# --------------------------------------------------------------------------- #
class VCamWorker:
    DEFAULTS = dict(
        camera=0, width=1280, height=720, out_width=None, out_height=None,
        fps=30, engine="auto", mode="image", bg_path=None, color=(0, 177, 64),
        blur_image=False, blur_strength=1.0, mirror=False, multi_person=False,
        downsample=None, threads=None, model="general", threshold=0.5, feather=1, erode=1,
        edge_radius=12, temporal=0.5, mask_scale=1.0,
    )

    # Params that need the capture/engine/virtual-cam pipeline rebuilt (vs applied
    # in-place each frame). Changing one while running triggers a quick automatic
    # reconnect; everything else (mirror, downsample, one-person, background)
    # applies live on the next frame with no rebuild.
    _RESTART_KEYS = frozenset({"camera", "width", "height", "out_width",
                               "out_height", "engine", "fps", "threads"})

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._lock = threading.Lock()
        self._latest = None          # latest processed BGR frame (for preview)
        self._preview = None         # it, fitted into preview_box, as RGB
        # (width, height) the GUI wants the preview to fit, or None for no
        # preview (e.g. while minimized). Set from the GUI thread.
        self.preview_box = None
        self._bg_dirty = False
        self._bg_gen = 0             # bumps per background change; stale loads are dropped
        self._pending = None         # compositor finished by a loader thread
        self._bg_cache = OrderedDict()   # (path, mtime, w, h) -> fitted background (LRU)
        self.running = False
        self.status = "stopped"      # stopped | starting | running | error
        self.error = None
        self.fps = 0.0
        self.device = None
        self.cfg = dict(self.DEFAULTS)

    # ---- public, thread-safe ----
    def configure(self, **kw):
        with self._lock:
            restart = any(k in self._RESTART_KEYS and self.cfg.get(k) != v
                          for k, v in kw.items())
            self.cfg.update(kw)
        # A pipeline-shape change while live -> reconnect (no manual OFF/ON).
        if restart and self.running:
            self._restart.set()

    def set_background(self, mode, path=None, color=None, blur_image=None):
        with self._lock:
            self.cfg["mode"] = mode
            if path is not None:
                self.cfg["bg_path"] = path
            if color is not None:
                self.cfg["color"] = color
            if blur_image is not None:
                self.cfg["blur_image"] = blur_image
            self._bg_dirty = True

    def set_blur_strength(self, strength):
        """Blur intensity for both live --blur mode and a blurred image
        background. Rebuilds the compositor (like set_background) rather than
        applying per-frame, since the sigma is baked into the compositor."""
        with self._lock:
            self.cfg["blur_strength"] = strength
            self._bg_dirty = True

    def set_quality(self, downsample):
        """Live quality/speed balance (RVM downsample_ratio). Takes effect on the
        next frame while running; also remembered for the next start."""
        with self._lock:
            self.cfg["downsample"] = downsample

    def latest_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def latest_preview(self):
        """Latest output frame as an RGB array fitted into `preview_box`, or
        None. A fresh array per frame that nothing writes to again, so the GUI
        can use it without copying."""
        with self._lock:
            return self._preview

    def start(self):
        if self.running:
            return
        keep_realtime_when_background()   # keep streaming smoothly when minimized
        self._stop.clear()
        self.error = None
        self.status = "starting"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    # ---- internal ----
    def _build_compositor(self, cfg, width, height):
        """Build from a cfg snapshot (no lock held: image decoding can be slow)."""
        mode = cfg["mode"]
        strength = cfg.get("blur_strength", 1.0)
        if mode == "none":
            return PassthroughCompositor()
        if mode == "color":
            return Compositor(width, height, solid_color_background(cfg["color"], width, height), False)
        if mode == "image" and cfg["bg_path"] and Path(cfg["bg_path"]).is_file():
            try:
                bg = self._cached_background(cfg["bg_path"], width, height)
                if cfg.get("blur_image"):
                    bg = blur_background_image(bg, strength)
                return Compositor(width, height, bg, False)
            except (RuntimeError, OSError):
                pass   # unreadable image -> fall back to blur
        # blur (and fallback when no image chosen)
        return Compositor(width, height, None, True, blur_strength=strength)

    def _cached_background(self, path, width, height):
        """Fitted background image; re-selecting a recent one is instant."""
        key = (str(path), os.path.getmtime(path), width, height)
        with self._lock:
            bg = self._bg_cache.get(key)
            if bg is not None:
                self._bg_cache.move_to_end(key)
                return bg
        bg = load_background_image(path, width, height)
        with self._lock:
            self._bg_cache[key] = bg
            while len(self._bg_cache) > 8:          # ~2.7 MB each at 720p
                self._bg_cache.popitem(last=False)
        return bg

    def _load_compositor_async(self, gen, cfg, width, height):
        """Build the new compositor off the frame loop; the old background keeps
        streaming until it's ready, and a newer selection supersedes this one."""
        def work():
            comp = self._build_compositor(cfg, width, height)
            with self._lock:
                if gen == self._bg_gen:
                    self._pending = comp
        threading.Thread(target=work, daemon=True).start()

    def _store_latest(self, out):
        # Runs on the output thread: build the GUI preview here (resized to the
        # box the GUI asked for, RGB) so the Tk thread only has to display it.
        box = self.preview_box
        preview = None
        if box is not None:
            h, w = out.shape[:2]
            k = min(box[0] / w, box[1] / h)
            tw, th = int(w * k), int(h * k)
            if tw > 2 and th > 2:
                preview = make_preview(out, tw, th)
        with self._lock:
            self._latest = out
            self._preview = preview

    def _loop(self):
        # Outer loop: a session runs until stop or until a pipeline-shape change
        # (camera/resolution/engine) requests a reconnect, then rebuilds — so the
        # user can change any option without clicking OFF/ON.
        try:
            while not self._stop.is_set():
                self._restart.clear()
                self.status = "starting"
                self._run_session()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.status = "error"
        finally:
            self.running = False
            self.device = None
            with self._lock:
                self._latest = self._preview = None
            if self.status != "error":
                self.status = "stopped"

    def _run_session(self):
        cap = cam = grabber = engine = pipeline = None
        try:
            with self._lock:
                cfg = dict(self.cfg)
            if pyvirtualcam is None:
                raise RuntimeError("pyvirtualcam is not installed")
            cap = open_camera(cfg["camera"], cfg["width"], cfg["height"], cfg["fps"])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg["width"]
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg["height"]
            engine = make_engine(types.SimpleNamespace(**cfg), width, height)
            with self._lock:
                bg_cfg = dict(self.cfg)
                self._bg_dirty = False
                self._pending = None
                self._bg_gen += 1
            compositor = self._build_compositor(bg_cfg, width, height)
            # Output can be a different (larger) size than capture: matte the
            # small/fast frame, upscale the finished frame -> smooth 30 fps.
            out_w = cfg["out_width"] or width
            out_h = cfg["out_height"] or height
            upscale = (out_w, out_h) if (out_w, out_h) != (width, height) else None
            cam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=cfg["fps"], fmt=PixelFormat.BGR)
            self.device = cam.device
            # Compositing + cam.send run on a worker thread, overlapping the next
            # frame's inference (identical output, higher fps).
            pipeline = OutputPipeline(cam, on_output=self._store_latest)
            grabber = FrameGrabber(cap).start()
            self.running = True
            self.status = "running"
            frames, t0, last_seq = 0, time.time(), -1
            while not self._stop.is_set() and not self._restart.is_set():
                if pipeline.error is not None:
                    raise pipeline.error
                seq, frame = grabber.wait_next(last_seq, timeout=0.1)
                if frame is None:
                    if not grabber.running:
                        raise RuntimeError("camera stopped delivering frames")
                    continue
                last_seq = seq
                load = None
                with self._lock:
                    if self._bg_dirty:
                        self._bg_dirty = False
                        self._bg_gen += 1
                        load = (self._bg_gen, dict(self.cfg))
                    if self._pending is not None:
                        compositor, self._pending = self._pending, None
                    mirror = self.cfg["mirror"]
                    want_dsr = self.cfg.get("downsample")
                    want_multi = self.cfg.get("multi_person")
                if load:
                    self._load_compositor_async(load[0], load[1], width, height)
                # Follow the compositor actually in use (a new one may still be loading).
                passthrough = isinstance(compositor, PassthroughCompositor)
                if mirror:
                    frame = cv2.flip(frame, 1)
                if passthrough:
                    # Original camera: skip matting entirely (saves the CPU);
                    # the engine stays loaded so switching back is instant.
                    pipeline.submit(compositor, frame, None, upscale)
                    frames += 1
                    if frames % 15 == 0:
                        self.fps = frames / (time.time() - t0)
                    continue
                # Live, in-place option changes (no rebuild needed):
                if want_dsr is not None and hasattr(engine, "set_downsample"):
                    engine.set_downsample(want_dsr)          # RVM quality dial
                if want_multi is not None and hasattr(engine, "set_single_person"):
                    engine.set_single_person(not want_multi)  # only-one-person
                if hasattr(engine, "infer"):
                    # Only inference runs here; the post-processing happens on
                    # the output thread, overlapping the next inference.
                    raw = engine.infer(frame)
                    pipeline.submit(compositor, frame, None, upscale,
                                    finish=functools.partial(engine.finish, raw))
                else:
                    alpha, fg = engine.process(frame)
                    pipeline.submit(compositor, fg if fg is not None else frame, alpha, upscale)
                frames += 1
                if frames % 15 == 0:
                    self.fps = frames / (time.time() - t0)
        finally:
            if grabber:
                grabber.stop()
            if pipeline:
                pipeline.close()         # drain + join worker before closing the cam
            if cap:
                cap.release()
            if engine:
                engine.close()
            if cam:
                cam.close()


def run(args: argparse.Namespace) -> int:
    keep_realtime_when_background()
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    # The camera may not honor the requested size; use what we actually get.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[vcam] Capturing {width}x{height} from camera {args.camera}"
          + (f" (reports {actual_fps:.0f} fps)" if actual_fps else ""))

    # Output size may differ from capture: capture fast/small, upscale the result.
    out_w = args.out_width or width
    out_h = args.out_height or height
    upscaling = (out_w, out_h) != (width, height)
    if upscaling:
        print(f"[vcam] Scaling output to {out_w}x{out_h}")

    # Build the static background (blur recomputes from the live frame each tick).
    blur_bg = False
    background_bgr = None
    if args.blur:
        blur_bg = True
    elif args.background:
        background_bgr = load_background_image(args.background, width, height)
    elif args.color:
        background_bgr = solid_color_background(args.color, width, height)
    else:
        blur_bg = True  # sensible default: blur the real background
        print("[vcam] No background specified -> blurring the real background.")

    engine = make_engine(args, width, height)
    compositor = Compositor(width, height, background_bgr, blur_bg)

    # Try to open the virtual camera; degrade gracefully to preview-only.
    cam = None
    if not args.no_virtualcam and pyvirtualcam is not None:
        try:
            cam = pyvirtualcam.Camera(
                width=out_w, height=out_h, fps=args.fps, fmt=PixelFormat.BGR
            )
            print(f"[vcam] Virtual camera live -> '{cam.device}' (select it in your video app)")
        except Exception as exc:  # noqa: BLE001
            print(f"[vcam] WARNING: could not start a virtual camera: {exc}")
            print("[vcam] Install OBS Studio (provides 'OBS Virtual Camera') and retry.")
            print("[vcam] Continuing in preview-only mode.")
    elif pyvirtualcam is None:
        print("[vcam] pyvirtualcam not installed -> preview-only mode.")

    grabber = FrameGrabber(cap).start()
    # Compositing + cam.send run on a worker thread so they overlap the next
    # frame's inference (identical output, higher fps). The preview window stays
    # on this (main) thread and shows the worker's latest finished frame.
    upscale = (out_w, out_h) if upscaling else None
    pipeline = OutputPipeline(cam)

    show_preview = not args.no_preview
    frames, t0 = 0, time.time()
    last_seq = -1
    try:
        while True:
            if pipeline.error is not None:
                raise pipeline.error
            seq, frame_bgr = grabber.wait_next(last_seq, timeout=0.1)
            if frame_bgr is None:
                if not grabber.running:
                    print("[vcam] Failed to read frame from camera; stopping.")
                    break
                if show_preview and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_seq = seq

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            if hasattr(engine, "infer"):
                # RVM: infer here, post-process on the output thread (overlaps
                # the next inference); it returns a colour-decontaminated foreground.
                pipeline.submit(compositor, frame_bgr, None, upscale,
                                finish=functools.partial(engine.finish, engine.infer(frame_bgr)))
            else:
                alpha, foreground = engine.process(frame_bgr)
                pipeline.submit(compositor, foreground if foreground is not None else frame_bgr,
                                alpha, upscale)

            if show_preview:
                preview = pipeline.latest()
                if preview is not None:
                    cv2.imshow("vcam preview (q/Esc to quit)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break

            frames += 1
            if frames % 120 == 0:
                fps = frames / (time.time() - t0)
                print(f"[vcam] ~{fps:.1f} fps")
    except KeyboardInterrupt:
        print("\n[vcam] Interrupted.")
    finally:
        grabber.stop()
        pipeline.close()         # drain + join worker before closing the cam
        cap.release()
        engine.close()
        if cam is not None:
            cam.close()
        if show_preview:
            cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time webcam virtual background (RobustVideoMatting / MediaPipe) "
                    "published as a system virtual camera via pyvirtualcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bg = p.add_mutually_exclusive_group()
    bg.add_argument("--background", "-b", metavar="PATH", help="background image to composite behind you")
    bg.add_argument("--blur", action="store_true", help="blur the real background instead of replacing it")
    bg.add_argument("--color", type=parse_color, metavar="R,G,B", help="solid color background, e.g. 0,177,64")

    p.add_argument("--engine", choices=("auto", "rvm", "mediapipe"), default="auto",
                   help="matting engine: 'rvm' (RobustVideoMatting, best quality) or 'mediapipe' "
                        "(faster, lighter). 'auto' uses rvm when onnxruntime is installed.")
    p.add_argument("--downsample", type=float, default=None,
                   help="[rvm] internal downsample_ratio 0-1 (default 0.375; lower = faster, softer hair)")
    p.add_argument("--threads", type=int, default=None,
                   help="[rvm] onnxruntime CPU threads (default: min(8, cores); more can be slower)")

    p.add_argument("--camera", "-c", type=int, default=0, help="webcam device index")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    p.add_argument("--out-width", type=int, default=None,
                   help="virtual-cam output width (default: same as capture). Capture small/fast, upscale output.")
    p.add_argument("--out-height", type=int, default=None,
                   help="virtual-cam output height (default: same as capture)")
    p.add_argument("--fps", type=int, default=30, help="target frames per second")
    p.add_argument("--model", choices=("general", "landscape"), default="general",
                   help="[mediapipe] model: 'general' (square) or 'landscape' (wide, slightly faster)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="[mediapipe] mask cutoff (0-1); lower keeps more of you")
    p.add_argument("--feather", type=int, default=1,
                   help="[mediapipe] edge anti-alias radius in px (0 = hardest edge)")
    p.add_argument("--edge-radius", type=int, default=12,
                   help="[mediapipe] edge-refinement strength (guided filter); higher hugs broad edges")
    p.add_argument("--erode", type=int, default=1,
                   help="[mediapipe] shrink the cutout by N px to remove the background halo (0 = none)")
    p.add_argument("--temporal", type=float, default=0.5,
                   help="temporal smoothing 0-1 (weight of current frame); lower = steadier, 1 = off")
    p.add_argument("--multi-person", action="store_true",
                   help="keep everyone in frame (default: keep only the single largest person)")
    p.add_argument("--mask-scale", type=float, default=1.0,
                   help="resolution scale for edge refinement (e.g. 0.5 = faster, slightly softer)")
    p.add_argument("--mirror", action="store_true", help="horizontally flip (selfie view)")
    p.add_argument("--no-preview", action="store_true", help="don't open the OpenCV preview window")
    p.add_argument("--no-virtualcam", action="store_true", help="preview only; don't open a virtual camera")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.background and not Path(args.background).is_file():
        sys.exit(f"Background image not found: {args.background}")
    try:
        return run(args)
    except RuntimeError as exc:           # camera busy, model download failed, ...
        sys.exit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
