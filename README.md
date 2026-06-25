# vcam — real-time webcam background replacement

Isolate yourself from your real background in real time using **Google
MediaPipe Selfie Segmentation**, composite over a background image / blur /
solid color, and expose the result as a **system virtual camera** (via
`pyvirtualcam`) that any app — Zoom, Google Meet, Teams, Discord, OBS — can
select like a normal webcam.

100% local. No cloud, no account, no subscription. The segmentation model is
Google's free, open-source `selfie_segmenter` (`.tflite`), downloaded
automatically on first run.

---

## 1. Install

```bash
pip install -r requirements.txt
```

Python 3.9–3.13 supported.

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
# Blur your real background (default)
python vcam.py

# Replace with an image
python vcam.py --background backgrounds/beach.jpg

# Solid green screen (chroma key), mirrored, second camera
python vcam.py --color 0,177,64 --camera 1 --mirror

# Headless: no preview window, just feed the virtual cam
python vcam.py --background office.png --no-preview
```

Then, in your video app, pick **"OBS Virtual Camera"** as the webcam.

Quit with **q** / **Esc** in the preview window, or **Ctrl+C** in the terminal.

---

## 3. Options

| Flag | Default | Description |
|------|---------|-------------|
| `--background, -b PATH` | — | Background image to composite behind you |
| `--blur` | (default) | Blur the real background instead of replacing it |
| `--color R,G,B` | — | Solid color background, e.g. `0,177,64` |
| `--camera, -c N` | `0` | Webcam device index (try 0, 1, 2…) |
| `--width` / `--height` | `1280` / `720` | **Capture** resolution (what the camera delivers) |
| `--out-width` / `--out-height` | = capture | **Output** resolution. Capture small/fast, upscale the result (see Performance) |
| `--fps` | `30` | Target frame rate |
| `--model` | `general` | `general` (square) or `landscape` (wide, slightly faster) |
| `--threshold` | `0.5` | Mask cutoff 0–1; lower keeps more of you |
| `--feather` | `3` | Edge-softening radius in px (`0` = hard edge) |
| `--mirror` | off | Horizontally flip (selfie view) |
| `--no-preview` | off | Don't open the preview window |
| `--no-virtualcam` | off | Preview only; don't open a virtual camera |

---

## 4. How it works

```
webcam ──► OpenCV capture ──► MediaPipe ImageSegmenter ──► soft person mask
                                                              │
                 background (image / blur / color) ───────────┤
                                                              ▼
                                       alpha composite ──► pyvirtualcam ──► system camera
```

The script uses MediaPipe's **Tasks API** (`ImageSegmenter`, VIDEO mode). The
older `mp.solutions.selfie_segmentation` API was removed in MediaPipe 0.10.30+;
this project uses the current model file (`models/selfie_segmenter.tflite`),
fetched automatically.

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

- **"could not start a virtual camera"** → install OBS Studio (see step 1) so a
  virtual-camera driver exists, then re-run.
- **Wrong / black camera** → change `--camera` index.
- **Jagged edges** → raise `--feather` (e.g. `5`) or lower `--threshold`.
- **Too much of you cut off** → lower `--threshold` (e.g. `0.3`).
- **Low FPS** → see [Performance](#5-performance). Short version: lower the
  **capture** resolution (e.g. `--width 640 --height 480`) and add `--out-width
  1280 --out-height 720` to keep a 720p stream; add light; try `--no-preview`.
