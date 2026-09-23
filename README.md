# OpenLander

OpenLander is a lean, local AI-assisted autonomous landing decision demo. It turns a time-ordered aerial image sequence into a self-contained mission run, then replays the result instantly without loading models or recomputing inference.

Its pipeline is:

`Scenario Folder → Real AI Perception → Temporal Drift → Landing Score → Decision State Machine → Explainable Selection → Processed Run → Interactive Replay`

The system fuses **Depth Anything V2 Small**, **CLIPSeg**, **YOLO11n**, OpenCV optical flow, and optional flight telemetry. It deliberately does not use prepared masks or synthetic ground truth as model output.

## Apple Silicon setup

Use Python 3.11 on macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_test_data.py
python run_batch.py
python app.py
```

OpenLander sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing PyTorch. Compute selection is Apple MPS first, CUDA second for portability, and CPU last. Inference uses float32 for MPS compatibility. The application header and mission readout display the selected device.

The first raw analysis downloads the pretrained weights (several hundred MB in total) into `models_cache/`, plus the YOLO nano checkpoint. Once cached, analysis can work offline. **Processed-run replay always works without inference and never loads Transformers or Ultralytics models.**

## Use

Run `python generate_test_data.py` to discover every non-hidden PNG, JPEG, or WebP image in `seed_photos/`. Each untouched seed becomes a normalized scenario with 150 frames, aspect-preserving output, deterministic smooth zoom/drift, filename-derived altitude and view telemetry, and a five-second H.264 raw preview in `outputs/test_generation_preview/`. SHA256 values are checked before and after generation. There is no procedural fallback or generated ground truth. The debug-only `crop_center_x_norm`, `crop_center_y_norm`, and `crop_scale` metadata describe the camera path and are never loaded into the landing decision engine.

Run `python run_batch.py` for the sequential real-seed evaluation. It loads the three AI models once, reuses them across every generated scenario, exports both cached replay videos, and writes `batch_summary.csv`, `batch_summary.json`, `batch_report.md`, and `seed_integrity.json` under `outputs/evaluation/`. Use `--scenario <normalized_name>` for one scenario or `--force` to rerun completed report entries. A failure is recorded per scenario without stopping later scenarios.

Start `python app.py`, then choose one of the two entry points:

- **Analyze Raw Scenario** validates and copies the source, runs real inference on every frame, writes a unique timestamped run under `runs/`, and opens it in Mission Control. Analysis artifacts are never overwritten; requested replay exports are added to that run.
- **Open Processed Run** loads `run.json`, `result.json`, saved views, and compressed score maps immediately. It does not load AI models.

Use Previous, Play/Pause, Next, the slider, and the metadata-timed 0.5×/1×/2× speed control to navigate. Change among Final Decision, Raw Camera, Depth, AI Perception, Landing Score, Optical Flow, and Debug Dashboard. Click the main image to query the cached local score, terrain, clearance, reachability, hazard, and facility maps.

Choose Final Decision or Debug Dashboard under **Export View**, then click **Export Video**. Export reads saved frames only; it never reruns inference. When system `ffmpeg` is available it creates a 30 fps H.264 MP4 (`landing_replay.mp4` or `dashboard_replay.mp4`) in the processed-run folder. Otherwise it creates a GIF fallback with the matching stem.

## Raw scenario contract

```text
scenario_name/
├── scenario.json
├── metadata.csv
└── frames/000000.png ...
```

`metadata.csv` requires `frame_id`, `filename`, and `timestamp_s`. Optional altitude, XYZ velocity, roll/pitch/yaw, and horizontal/vertical camera FOV fields enrich the display and footprint/reachability estimate. Frames may have any dimensions, but all frames in a scenario must match. `scenario.json` accepts `SAFETY` or `RECOVERY` mission mode.

## Processed run contract

Each new analysis creates a unique, self-contained folder:

```text
runs/<scenario>_<timestamp>/
├── run.json              # provenance, models, device, dimensions, timing
├── result.json           # compact decisions, candidates, events, telemetry
├── summary.txt
├── landing.gif
├── landing_replay.mp4    # optional cached-frame export
├── dashboard_replay.mp4  # optional cached-frame export
├── raw/                  # complete copied scenario; no symlinks
├── annotated/ depth/ perception/ heatmap/ flow/ dashboard/
└── maps/                 # compact float16 arrays for point inspection
```

You can move or archive this folder and reopen it by path. The original scenario is not required.

## Scope and limitations

OpenLander is an explainable prototype, not flight software. General pretrained models can misunderstand unusual planetary imagery; its strength is fusing several imperfect signals rather than treating one model as truth. The reachable envelope is an illustrative image-space approximation. The PID-like trajectory is explicitly a **simulated control reference**, not a real or flight-certified spacecraft controller.
