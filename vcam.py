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
import sys
import threading
import time
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


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def ensure_model(name: str) -> Path:
    """Return the local path to the model, downloading it on first use."""
    spec = MODELS[name]
    path = MODEL_DIR / spec["file"]
    if path.is_file() and path.stat().st_size > 0:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[vcam] Downloading '{name}' model -> {path} ...")
    try:
        urllib.request.urlretrieve(spec["url"], path)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Failed to download model: {exc}\nURL: {spec['url']}")
    print("[vcam] Model ready.")
    return path


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
        sys.exit(f"Could not read background image: {path}")
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
        sys.exit(
            f"Could not open camera index {index}. "
            "Try a different --camera index (0, 1, 2, ...)."
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


class Compositor:
    """Alpha-blends the person over a background using the soft mask.

    Optimised hot path (~8 ms/frame at 720p vs ~19 ms for the naive float
    version):
      * integer math instead of float32 (out = bg + (fg-bg)*alpha/255), and
      * reused scratch buffers so we don't reallocate ~10 MB per frame.

    `--blur` mode blurs a quarter-size copy and upscales it (~2.4 ms vs ~15 ms);
    blur is low-frequency so the result is visually identical.
    """

    def __init__(self, width, height, background_bgr, blur_bg, threshold, feather):
        self.w, self.h = width, height
        self.background = background_bgr        # None in blur mode
        self.blur_bg = blur_bg
        self.threshold = threshold
        self.feather = feather
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

    def render(self, frame_bgr, mask):
        """`mask` is HxW float32 in [0, 1]; 1 == person, 0 == background."""
        bg = self._blurred_background(frame_bgr) if self.blur_bg else self.background

        if self.feather > 0:
            k = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Bias toward foreground around the threshold, then to a uint8 alpha 0-255.
        alpha = np.clip((mask - self.threshold) * 3.0 + 0.5, 0.0, 1.0)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        alpha3 = cv2.merge([alpha_u8, alpha_u8, alpha_u8])

        # out = bg + (fg - bg) * alpha / 255, all in int16 then saturated to uint8.
        cv2.subtract(frame_bgr, bg, dst=self._diff, dtype=cv2.CV_16S)
        cv2.multiply(self._diff, alpha3, dst=self._diff, dtype=cv2.CV_16S, scale=1 / 255.0)
        cv2.add(bg, self._diff, dst=self._out, dtype=cv2.CV_8U)
        return self._out


def run(args: argparse.Namespace) -> int:
    model_path = ensure_model(args.model)
    segmenter = make_segmenter(model_path)

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

    compositor = Compositor(width, height, background_bgr, blur_bg, args.threshold, args.feather)

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
    timestamp_ms = 0
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

            # MediaPipe Tasks wants an mp.Image in SRGB (RGB) order.
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode requires strictly increasing timestamps (ms).
            timestamp_ms = max(timestamp_ms + 1, int((time.time() - t0) * 1000))
            result = segmenter.segment_for_video(mp_image, timestamp_ms)
            mask = result.confidence_masks[0].numpy_view()  # HxWx1 float32, 1 == person
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            out_bgr = compositor.render(frame_bgr, mask)

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
        segmenter.close()
        if cam is not None:
            cam.close()
        if show_preview:
            cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time webcam background replacement via MediaPipe + pyvirtualcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bg = p.add_mutually_exclusive_group()
    bg.add_argument("--background", "-b", metavar="PATH", help="background image to composite behind you")
    bg.add_argument("--blur", action="store_true", help="blur the real background instead of replacing it")
    bg.add_argument("--color", type=parse_color, metavar="R,G,B", help="solid color background, e.g. 0,177,64")

    p.add_argument("--camera", "-c", type=int, default=0, help="webcam device index")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    p.add_argument("--out-width", type=int, default=None,
                   help="virtual-cam output width (default: same as capture). Capture small/fast, upscale output.")
    p.add_argument("--out-height", type=int, default=None,
                   help="virtual-cam output height (default: same as capture)")
    p.add_argument("--fps", type=int, default=30, help="target frames per second")
    p.add_argument("--model", choices=("general", "landscape"), default="general",
                   help="MediaPipe model: 'general' (square) or 'landscape' (wide, slightly faster)")
    p.add_argument("--threshold", type=float, default=0.5, help="mask cutoff (0-1); lower keeps more of you")
    p.add_argument("--feather", type=int, default=3, help="edge softening radius in px (0 = hard edge)")
    p.add_argument("--mirror", action="store_true", help="horizontally flip (selfie view)")
    p.add_argument("--no-preview", action="store_true", help="don't open the OpenCV preview window")
    p.add_argument("--no-virtualcam", action="store_true", help="preview only; don't open a virtual camera")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.background and not Path(args.background).is_file():
        sys.exit(f"Background image not found: {args.background}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
