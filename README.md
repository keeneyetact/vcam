# vcam — real-time webcam virtual background

Replace your real background in real time and expose the result as a **system
virtual camera** (via `pyvirtualcam`) that any app — Google Meet, Zoom, Teams,
Discord, OBS — can select like a normal webcam.

Two matting engines:

- **RobustVideoMatting (RVM)** — *default.* A real-time video matting model with
  **temporal coherence** (recurrent state), so it holds up where the lightweight
  models tear: a **hand raised toward the camera**, fast motion, and hair detail.
  This is what gets close to Meet/Teams quality instead of looking like a cutout.
- **MediaPipe Selfie Segmentation** — lighter/faster fallback, with sharpening
  (guided-filter edge refinement, single-person isolation, motion-adaptive
  temporal smoothing).

It isolates a **single person**, keeps a clean edge with **no halo**, and uses
RVM's colour-decontaminated foreground so hair edges don't bleed the old
background.

100% local. No cloud, no account, no subscription. Models (Google's
`selfie_segmenter.tflite` and PeterL1n's `rvm_mobilenetv3_fp32.onnx`, both free
and open-source) download automatically on first run.

---

## 1. Install

```bash
pip install -r requirements.txt
```

Python 3.9–3.13 supported. `onnxruntime` powers the RVM engine — if it's missing,
vcam automatically falls back to MediaPipe. On Windows/Intel-iGPU stick with the
**CPU** build of onnxruntime (DirectML has correctness/perf issues with RVM's
recurrent state); CPU runs ~20 fps at 720p, which exceeds most webcams' frame rate.

### Virtual-camera backend (required to feed other apps)

`pyvirtualcam` needs an OS virtual-camera driver. On **Windows** the easiest is
**OBS Studio** (26.1+): install it once and start "OBS Virtual Camera" *or* just
let OBS register the driver — `vcam` then writes frames into it directly (you do
not need OBS running). On **macOS** OBS provides the backend too; on **Linux**
install `v4l2loopback`.

> Without a backend, `vcam` still runs and shows a live **preview window** — it
> just won't appear as a camera in other apps.

---

## 2. Run

```bash
# Virtual background from an image (the main use case)
python vcam.py --background 2.png

# Blur your real background instead
python vcam.py --blur

# Solid green screen (chroma key)
python vcam.py --color 0,177,64

# Headless: no preview window, just feed the virtual cam
python vcam.py --background office.png --no-preview
```

### Use it in Google Meet

1. Make sure **no other app is using the webcam** — a camera can only be owned
   by one app at a time. (If vcam says it can't open the camera, that's why.)
2. Run `python vcam.py --background 2.png` and leave it running.
3. In Meet: **⋮ → Settings → Video → Camera → "OBS Virtual Camera"**.
   (Meet uses the *virtual* camera; vcam owns the *physical* one.)

Quit with **q** / **Esc** in the preview window, or **Ctrl+C** in the terminal.

---

## 3. Options

| Flag | Default | Description |
|------|---------|-------------|
| `--background, -b PATH` | — | Background image to composite behind you |
| `--blur` | — | Blur the real background instead of replacing it |
| `--color R,G,B` | — | Solid color background, e.g. `0,177,64` |
| `--engine` | `auto` | `rvm` (best), `mediapipe` (lighter), or `auto` |
| `--downsample` | auto | [rvm] internal ratio 0–1; lower = faster, softer hair |
| `--camera, -c N` | `0` | Webcam device index (try 0, 1, 2…) |
| `--width` / `--height` | `1280` / `720` | **Capture** resolution (what the camera delivers) |
| `--out-width` / `--out-height` | = capture | **Output** resolution. Capture small/fast, upscale the result (see Performance) |
| `--fps` | `30` | Target frame rate |
| `--model` | `general` | `general` (square) or `landscape` (wide, slightly faster) |
| `--threshold` | `0.5` | Mask cutoff 0–1; lower keeps more of you |
| `--mirror` | off | Horizontally flip (selfie view) |
| `--multi-person` | off | Keep everyone in frame (default: keep only the **single** largest person) |
| `--no-preview` | off | Don't open the preview window |
| `--no-virtualcam` | off | Preview only; don't open a virtual camera |

These only affect the **MediaPipe** engine (ignored by RVM): `--model`,
`--threshold`, `--feather`, `--edge-radius`, `--erode`, `--temporal`,
`--mask-scale`. See `python vcam.py --help`.

---

## 4. How it works

```
webcam ─► capture ─► matting engine ─► alpha matte (+ foreground)
                                                  │
                 background image / blur / color ─┤
                                                  ▼
                       alpha composite ─► pyvirtualcam ─► system camera
```

**RVM engine (default).** RobustVideoMatting is a recurrent video matting network:
each frame it consumes the previous frame's hidden state and emits a new one, so
the matte is **temporally coherent** — stable hair, hands, and edges under motion
instead of the per-frame flicker/tearing the lightweight models produce. It also
outputs a colour-decontaminated foreground, so composited hair edges don't carry
the old background's colour. Run on the onnxruntime **CPU** provider
(`models/rvm_mobilenetv3_fp32.onnx`, auto-downloaded). Only post-processing is
optional single-person isolation.

**MediaPipe engine (fallback).** MediaPipe's **Tasks API** `ImageSegmenter`
(VIDEO mode) gives a soft confidence mask, then a sharpening pipeline: guided
filter (`cv2.ximgproc`) snaps the mask to real edges → motion-adaptive temporal
smoothing → threshold → keep the largest blob (one person) → erode (no halo) →
feather. Needs `opencv-contrib-python` for `cv2.ximgproc`; without it edges fall
back to softer feathering.

---

## 5. Performance

Frame rate is bounded by `min(camera fps, pipeline fps)`. The pipeline is fast;
the **camera** is almost always the limit.

**The pipeline** (~10–16 ms/frame, i.e. 60–100 fps of headroom):
- segmentation runs ~5 ms (the model is a fixed 256 px internally);
- compositing uses integer math + reused buffers (~8 ms at 720p, vs ~19 ms naive);
- `--blur` blurs a quarter-size copy and upscales it (~2 ms, vs ~15 ms);
- a background thread grabs camera frames so capture overlaps processing instead
  of serializing with it.

**The camera** is the real ceiling, and it's a per-device **resolution/fps
tradeoff**. Many webcams do e.g. 1280×720 @ only 10 fps but 640×480 @ 30 fps.
Check yours and capture at the resolution that hits your target fps.

To get a smooth **30 fps and a 720p-shaped stream**, capture small and upscale
the output:

```bash
python vcam.py --width 640 --height 480 --out-width 1280 --out-height 720
```

This captures at the camera's fast resolution, runs segmentation/compositing on
the small frame, and upscales the finished frame to 720p before the virtual cam.

Dim light makes it worse: webcams lengthen exposure in low light and drop the
frame rate automatically — add light before blaming the code.

---

## 6. Troubleshooting

- **"Could not open camera"** → another app owns the webcam (Meet/Zoom/Teams/OBS
  or another script). Close it, or use a different `--camera` index.
- **"could not start a virtual camera"** → install OBS Studio (see step 1) so a
  virtual-camera driver exists, then re-run.
- **Torn / tearing edges around a raised hand or on motion** → you're on the
  MediaPipe engine; use the default **RVM** engine (`pip install onnxruntime`,
  `--engine rvm`). RVM is built for exactly this.
- **RVM too slow** → lower `--downsample` (e.g. `0.3`), or drop the **capture**
  resolution (`--width 640 --height 480`, then `--out-width 1280 --out-height 720`).
- **A second person/object shows up** → only the largest person is kept by
  default; pass `--multi-person` to keep everyone.
- **(MediaPipe engine) halo** → raise `--erode`; **soft edge** → `--feather 0`;
  **bits cut off** → `--threshold 0.4`; **flicker** → `--temporal 0.3`;
  **ghost trail on motion** → `--temporal 1`.
- **Low FPS** → see [Performance](#5-performance): lower **capture** resolution
  (and `--out-width/--out-height` for a 720p stream), add light, `--no-preview`.
