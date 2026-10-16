<p align="center">
  <img src="assets/logo.png" alt="vcam — real-time virtual background" width="540">
</p>

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

## 0. Standalone .exe (Windows, no Python needed)

Want to just double-click and run? Build a single self-contained `vcam.exe`:

```bash
pip install pyinstaller
pyinstaller --noconfirm vcam.spec      # -> dist/vcam.exe
```

`dist/vcam.exe` bundles Python, the RVM model, and all dependencies — copy it to
any **Windows 10/11 (64-bit)** PC and double-click; no install required.

Two caveats for "any computer":
- It's a **Windows x64** binary (build separately for macOS/Linux).
- The **virtual-camera output needs the OBS Virtual Camera** driver on the target
  PC (install [OBS Studio](https://obsproject.com) once — the app's **Set up virtual camera**
  button then registers it). Without it the app still runs and previews; it just
  can't appear as a camera in Meet. The RVM model is bundled, so no internet is
  needed for matting.

---

## 1. Install (from source)

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

### Option A — the GUI (easiest)

```bash
python vcam_ui.py
```

A control panel with the live preview front and centre:

1. **One-time setup** — shown only if the "OBS Virtual Camera" isn't registered
   yet: one click registers it (Windows asks for admin), or it links you to
   OBS Studio if the driver is missing and re-checks when you come back.
2. **Start / Stop camera** (Ctrl+Enter) — feeds the virtual camera; the preview
   shows a LIVE badge with fps. Problems appear as plain-language messages on
   the preview (hover for the technical detail).
3. **Camera** picker and **Mirror image**.
4. **Background** — Original (no effect), Blur, Image, or Green screen —
   switchable live, with a blur-amount slider. Image opens a gallery: a
   built-in default plus any pictures you add (**+ Add** or Ctrl+O; copied to
   `%APPDATA%\vcam\backgrounds`). Click one to use it, ✕ to remove it (your
   original file is untouched), and optionally soften it.
5. **Advanced settings** — Performance (Sharp 720p / Smooth 30 fps), edge
   detail, engine (Auto/RVM/MediaPipe), only-one-person, and driver
   reinstall/uninstall.

The window follows the Windows light/dark theme, is resizable and DPI-aware,
and remembers your settings in `%APPDATA%\vcam\settings.json`.

### Option B — the command line

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
| `--downsample` | `0.375` | [rvm] internal ratio 0–1; lower = faster, softer hair |
| `--threads` | `min(8,cores)` | [rvm] onnxruntime CPU threads (more can be slower) |
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

Frame rate is `min(camera fps, pipeline fps)`. On most webcams the **camera is
the ceiling**, not the matting.

**RVM engine (CPU), per frame:**
- inference ~50 ms at 720p (`downsample_ratio` 0.375) → ~19 fps of headroom;
- pre/post-processing trimmed from ~31 ms to ~6 ms (reused NCHW input buffer,
  per-channel foreground conversion, half-res connected-components);
- onnxruntime threads capped to the fast cores (more threads *slow* this small
  recurrent net — on a 20-core i7-14700, 28 threads = 77 ms vs 8 threads = 48 ms).

**The camera** is a per-device **resolution/fps tradeoff**: many webcams do
1280×720 @ only ~10 fps but 640×480 @ 30 fps. At 720p you're camera-bound at
~10 fps no matter how fast the CPU is.

So for **smooth 30 fps**, capture at the camera's fast resolution and upscale the
matted output — RVM at 640×480 runs ~50 fps on CPU, far above 30:

```bash
python vcam.py --width 640 --height 480 --out-width 1280 --out-height 720
```

In the GUI this is the **"Smooth (30 fps)"** speed preset. The tradeoff is
slightly softer *source* detail (480p webcam read); the matte itself stays sharp.

> GPU (DirectML/OpenVINO on the Intel iGPU) is intentionally **not** used: at 720p
> the camera caps fps anyway, at 480p the CPU already exceeds 30 fps, and DirectML
> is known to mis-compute RVM's recurrent state on Intel iGPUs. CPU is faster-where-
> it-matters *and* correct.

Dim light also drops the frame rate (webcams lengthen exposure) — add light.

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
