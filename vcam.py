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
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    sys.exit("mediapipe is not installed. Run:  pip install -r requirements.txt")

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


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

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
    """Return a local model path, downloading it on first use."""
    path = MODEL_DIR / file
    if path.is_file() and path.stat().st_size > 0:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
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


class FrameGrabber:
    """Continuously pulls frames in a background thread so the main loop never
    blocks on the camera. `cap.read()` can block for a full frame interval
    (~55 ms at 18 fps); overlapping that with segmentation/compositing lets us
    run at the camera's true frame rate instead of read+process serialized.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
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
                self.running = False
                break
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self) -> tuple[int, np.ndarray | None]:
        """Return (sequence_number, latest_frame_copy). seq lets the caller skip
        re-processing a frame it has already handled."""
        with self._lock:
            if self._frame is None:
                return self._seq, None
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
    """

    def __init__(self, width, height, background_bgr, blur_bg):
        self.w, self.h = width, height
        self.background = background_bgr        # None in blur mode
        self.blur_bg = blur_bg
        # Reused scratch buffers (allocated once).
        self._diff = np.empty((height, width, 3), np.int16)
        self._out = np.empty((height, width, 3), np.uint8)
        # Downscaled blur working size (≥ 160x90).
        self._bw = max(160, width // 4)
        self._bh = max(90, height // 4)

    def _blurred_background(self, frame_bgr):
        small = cv2.resize(frame_bgr, (self._bw, self._bh), interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(small, (0, 0), sigmaX=4, sigmaY=4, dst=small)
        return cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def render(self, frame_bgr, alpha):
        """`alpha` is a HxW float32 matte in [0, 1]; 1 == person, 0 == background."""
        bg = self._blurred_background(frame_bgr) if self.blur_bg else self.background

        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        alpha3 = cv2.merge([alpha_u8, alpha_u8, alpha_u8])

        # out = bg + (fg - bg) * alpha / 255, all in int16 then saturated to uint8.
        cv2.subtract(frame_bgr, bg, dst=self._diff, dtype=cv2.CV_16S)
        cv2.multiply(self._diff, alpha3, dst=self._diff, dtype=cv2.CV_16S, scale=1 / 255.0)
        cv2.add(bg, self._diff, dst=self._out, dtype=cv2.CV_8U)
        return self._out


# --------------------------------------------------------------------------- #
# Matting engines — each turns a BGR frame into (alpha matte, foreground|None)
# --------------------------------------------------------------------------- #
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
        so = ort.SessionOptions()
        # More threads than physical cores hurts this small recurrent net (on a
        # 14700 all-28-logical = 77 ms vs 8 = 48 ms), so cap it.
        so.intra_op_num_threads = int(threads) if threads else _rvm_thread_count()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.threads = so.intra_op_num_threads
        self.sess = ort.InferenceSession(
            str(ensure_rvm_model()), so, providers=["CPUExecutionProvider"]
        )
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

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)             # HWC uint8
        # Fill a reused float32 NCHW buffer directly (avoids astype + transpose
        # + ascontiguousarray copies: ~7.5 ms -> ~2 ms at 720p).
        if self._src is None or self._src.shape[2:] != (h, w):
            self._src = np.empty((1, 3, h, w), np.float32)
        src = self._src
        src[0, 0], src[0, 1], src[0, 2] = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        np.multiply(src, np.float32(1.0 / 255.0), out=src)
        fgr, pha, *self._rec = self.sess.run(
            [], {"src": src, "r1i": self._rec[0], "r2i": self._rec[1],
                 "r3i": self._rec[2], "r4i": self._rec[3], "downsample_ratio": self._dsr}
        )
        alpha = pha[0, 0]                                            # HxW float [0,1]
        if self.single:
            alpha = _keep_largest(alpha)
        # Foreground RGB (3,H,W) float -> BGR uint8 HWC via per-channel saturating
        # scale + merge (~9.3 ms -> ~1.3 ms at 720p; no transpose copy / cvtColor).
        fg = fgr[0]
        fg_bgr = cv2.merge([cv2.convertScaleAbs(fg[2], alpha=255.0),
                            cv2.convertScaleAbs(fg[1], alpha=255.0),
                            cv2.convertScaleAbs(fg[0], alpha=255.0)])
        return alpha, fg_bgr

    def close(self):
        pass


class MediaPipeEngine:
    """Lightweight MediaPipe selfie segmentation + the sharpening MaskRefiner.
    Fast and dependency-light, but weaker on hands-near-camera, motion and hair.
    """

    def __init__(self, width, height, args):
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

    def close(self):
        self.segmenter.close()


def _rvm_thread_count() -> int:
    """A good onnxruntime intra-op thread count for RVM: enough to use the fast
    cores, capped so a many-thread machine doesn't oversubscribe this small net."""
    import os
    return max(2, min(8, os.cpu_count() or 4))


def _keep_largest(alpha):
    """Keep only the single largest person blob in a soft alpha matte; suppress
    the rest while preserving the kept subject's soft edges. Connected components
    run at half resolution (blobs are large) — ~6 ms -> ~3 ms at 720p."""
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
                print(f"[vcam] Engine: RobustVideoMatting "
                      f"(downsample_ratio={engine.dsr:.2f}, threads={engine.threads})")
                return engine
            except Exception as exc:  # noqa: BLE001
                print(f"[vcam] RVM unavailable ({exc}) -> falling back to MediaPipe.")
                want = "mediapipe"
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


# --------------------------------------------------------------------------- #
# Worker — runs the capture→matte→composite→virtual-cam loop in a thread so a
# GUI can start/stop it and swap the background live.
# --------------------------------------------------------------------------- #
class VCamWorker:
    DEFAULTS = dict(
        camera=0, width=1280, height=720, out_width=None, out_height=None,
        fps=30, engine="auto", mode="image", bg_path=None, color=(0, 177, 64),
        mirror=False, multi_person=False, downsample=None, threads=None,
        model="general", threshold=0.5, feather=1, erode=1, edge_radius=12,
        temporal=0.5, mask_scale=1.0,
    )

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None          # latest processed BGR frame (for preview)
        self._bg_dirty = False
        self.running = False
        self.status = "stopped"      # stopped | starting | running | error
        self.error = None
        self.fps = 0.0
        self.device = None
        self.cfg = dict(self.DEFAULTS)

    # ---- public, thread-safe ----
    def configure(self, **kw):
        with self._lock:
            self.cfg.update(kw)

    def set_background(self, mode, path=None, color=None):
        with self._lock:
            self.cfg["mode"] = mode
            if path is not None:
                self.cfg["bg_path"] = path
            if color is not None:
                self.cfg["color"] = color
            self._bg_dirty = True

    def latest_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def start(self):
        if self.running:
            return
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
    def _build_compositor(self, width, height):
        mode = self.cfg["mode"]
        if mode == "color":
            return Compositor(width, height, solid_color_background(self.cfg["color"], width, height), False)
        if mode == "image" and self.cfg["bg_path"] and Path(self.cfg["bg_path"]).is_file():
            try:
                return Compositor(width, height, load_background_image(self.cfg["bg_path"], width, height), False)
            except RuntimeError:
                pass   # unreadable image -> fall back to blur
        return Compositor(width, height, None, True)   # blur (and fallback when no image chosen)

    def _loop(self):
        cap = cam = grabber = engine = None
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
                compositor = self._build_compositor(width, height)
                self._bg_dirty = False
            # Output can be a different (larger) size than capture: matte the
            # small/fast frame, upscale the finished frame -> smooth 30 fps.
            out_w = cfg["out_width"] or width
            out_h = cfg["out_height"] or height
            upscaling = (out_w, out_h) != (width, height)
            cam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=cfg["fps"], fmt=PixelFormat.BGR)
            self.device = cam.device
            grabber = FrameGrabber(cap).start()
            self.running = True
            self.status = "running"
            frames, t0, last_seq = 0, time.time(), -1
            while not self._stop.is_set():
                seq, frame = grabber.read()
                if frame is None:
                    if not grabber.running:
                        raise RuntimeError("camera stopped delivering frames")
                    time.sleep(0.002)
                    continue
                if seq == last_seq:
                    time.sleep(0.002)
                    continue
                last_seq = seq
                with self._lock:
                    if self._bg_dirty:
                        compositor = self._build_compositor(width, height)
                        self._bg_dirty = False
                    mirror = self.cfg["mirror"]
                if mirror:
                    frame = cv2.flip(frame, 1)
                alpha, fg = engine.process(frame)
                out = compositor.render(fg if fg is not None else frame, alpha)
                if upscaling:
                    out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                cam.send(out)
                cam.sleep_until_next_frame()
                with self._lock:
                    self._latest = out
                frames += 1
                if frames % 15 == 0:
                    self.fps = frames / (time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.status = "error"
        finally:
            if grabber:
                grabber.stop()
            if cap:
                cap.release()
            if engine:
                engine.close()
            if cam:
                cam.close()
            self.running = False
            with self._lock:
                self._latest = None
            if self.status != "error":
                self.status = "stopped"


def run(args: argparse.Namespace) -> int:
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

    show_preview = not args.no_preview
    frames, t0 = 0, time.time()
    last_seq = -1
    try:
        while True:
            seq, frame_bgr = grabber.read()
            if frame_bgr is None:
                if not grabber.running:
                    print("[vcam] Failed to read frame from camera; stopping.")
                    break
                time.sleep(0.002)  # camera still warming up
                continue
            if seq == last_seq:
                time.sleep(0.002)  # no new frame yet; don't reprocess
                continue
            last_seq = seq

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            alpha, foreground = engine.process(frame_bgr)
            # RVM returns a colour-decontaminated foreground; otherwise use the frame.
            out_bgr = compositor.render(foreground if foreground is not None else frame_bgr, alpha)

            if upscaling:
                out_bgr = cv2.resize(out_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            if cam is not None:
                cam.send(out_bgr)
                cam.sleep_until_next_frame()

            if show_preview:
                cv2.imshow("vcam preview (q/Esc to quit)", out_bgr)
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
