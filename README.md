# vcam — real-time webcam virtual background

Replace your real background in real time using **Google MediaPipe Selfie
Segmentation**, and expose the result as a **system virtual camera** (via
`pyvirtualcam`) that any app — Google Meet, Zoom, Teams, Discord, OBS — can
select like a normal webcam.

Tuned to look real, not "virtual": it isolates a **single person** with a
**sharp, edge-hugging cutout**, removes everything outside that edge (no halo),
and stays flicker-free against a still background image.

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
| `--camera, -c N` | `0` | Webcam device index (try 0, 1, 2…) |
| `--width` / `--height` | `1280` / `720` | **Capture** resolution (what the camera delivers) |
| `--out-width` / `--out-height` | = capture | **Output** resolution. Capture small/fast, upscale the result (see Performance) |
| `--fps` | `30` | Target frame rate |
| `--model` | `general` | `general` (square) or `landscape` (wide, slightly faster) |
| `--threshold` | `0.5` | Mask cutoff 0–1; lower keeps more of you |
| `--feather` | `1` | Edge anti-alias radius in px (`0` = hardest edge) |
| `--edge-radius` | `12` | Edge-refinement strength (guided filter). Higher hugs broad edges; lower keeps fine detail |
| `--erode` | `1` | Shrink the cutout by N px to kill the background halo (`0` = none) |
| `--temporal` | `0.5` | Temporal smoothing 0–1 (weight of current frame). Lower = steadier; `1` = off |
| `--multi-person` | off | Keep everyone in frame (default: keep only the **single** largest person) |
| `--mask-scale` | `1.0` | Resolution scale for edge refinement (`0.5` = faster, slightly softer) |
| `--mirror` | off | Horizontally flip (selfie view) |
| `--no-preview` | off | Don't open the preview window |
| `--no-virtualcam` | off | Preview only; don't open a virtual camera |

---

## 4. How it works

```
webcam ─► OpenCV capture ─► MediaPipe ImageSegmenter ─► raw confidence mask
                                                              │
        guided filter (snap to real edges) ◄──────────────────┤
        motion-adaptive temporal smoothing (no flicker/ghost)  │
        threshold ─► keep largest blob (ONE person)            │
        fill holes ─► erode 1px (no halo) ─► 1px feather        ▼
                                              sharp alpha matte
                                                              │
                 background image / blur / color ─────────────┤
                                                              ▼
                                  alpha composite ─► pyvirtualcam ─► system camera
```

The **mask pipeline** is what makes the cutout look real rather than virtual:

- **Guided filter** (`cv2.ximgproc.guidedFilter`) snaps the fuzzy model mask onto
  the actual image edges — hair, shoulders, headphones — so the cut hugs the guy.
- **Largest connected component** keeps exactly one person; any other object or
  person in frame is dropped (`--multi-person` to keep all).
- **Erode + threshold** remove everything outside the edge completely (no halo).
- **Motion-adaptive temporal smoothing** kills flicker against a still background
  where you're steady, but follows you where you move — so there are no ghost trails.

The script uses MediaPipe's **Tasks API** (`ImageSegmenter`, VIDEO mode). The
older `mp.solutions.selfie_segmentation` API was removed in MediaPipe 0.10.30+;
this project uses the current model file (`models/selfie_segmenter.tflite`),
fetched automatically. Edge refinement needs `opencv-contrib-python` (for
`cv2.ximgproc`); without it edges fall back to softer feathering.

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
- **Halo / background bleeding around you** → raise `--erode` (e.g. `2`).
- **Edge too soft / not crisp** → lower `--feather` to `0`, and/or lower
  `--edge-radius` (e.g. `8`) to keep fine detail.
- **Bits of you get cut off** → lower `--threshold` (e.g. `0.4`).
- **A second person/object shows up** → that shouldn't happen (only the largest
  person is kept); pass `--multi-person` if you actually want to keep everyone.
- **Flicker against the background** → lower `--temporal` (e.g. `0.3`).
- **Ghost trail when you move fast** → raise `--temporal` toward `1` (less smoothing).
- **Low FPS** → see [Performance](#5-performance). Lower the **capture** resolution
  (e.g. `--width 640 --height 480`, add `--out-width 1280 --out-height 720` for a
  720p stream), use `--mask-scale 0.5`, add light, or try `--no-preview`.
