# FoundationPose Model-Free Flow

This folder contains the model-free object-pose path. It is separate from the
older RGB-D point-geometry baseline in `visual_servoing/point_pose`.

## Environment Check

```bash
conda activate visual
export FOUNDATIONPOSE_ROOT=/home/kgs/FoundationPose
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m visual_servoing.scripts.fp_setup_check --foundationpose-path "$FOUNDATIONPOSE_ROOT" --strict
```

## Tkinter Workflow GUI

The recommended first-pass operator flow is the staged Tkinter GUI:

```bash
python -m visual_servoing.scripts.fp_gui --foundationpose-root "$FOUNDATIONPOSE_ROOT"
```

The GUI keeps heavy setup/build/tracking stages behind CLI subprocesses and
shows their stdout/stderr status in the log panel.

The current recommended reference workflow is recording based:

- create or select an object profile
- set a descriptive prompt such as `a white computer mouse`
- use `Board Axis Snapshot` to confirm the ChArUco board origin and axes
- enter the object pose relative to the board with `Obj XYZ m` and `RPY deg`
- click `Start Recording`
- move the camera around the fixed board/object pair from varied viewpoints
- click `Stop Recording`
- click `Processing` to select valid frames and publish `refs/`
- click `Build Assets`
- click `Track Live`

The default SAM3 processing resolution is fixed at `1008` because lower
resolutions missed small white objects in the D435 reference recordings. In the
OpenCV live tracking window, press `R` to reinitialize tracking from a fresh
SAM3 mask.

## Object Registration

Create an object profile:

```bash
python -m visual_servoing.scripts.fp_register_object --name phone --prompt "mobile phone"
```

Use `--camera d435` instead of the default D405. If multiple RealSense cameras
are connected, also pass the target serial with `--serial` where the command
supports it.

Record raw RGB-D frames for calibrated ChArUco processing:

```bash
python -m visual_servoing.scripts.fp_charuco_reference \
  --record \
  --object phone \
  --prompt "mobile phone" \
  --camera d435 \
  --duration-s 30 \
  --object-xyz-m 0.0 -0.065 -0.02
```

Process recordings into FoundationPose reference frames:

```bash
python -m visual_servoing.scripts.fp_charuco_reference \
  --process-recordings \
  --object phone \
  --prompt "mobile phone" \
  --camera d435 \
  --required-keyframes 16 \
  --max-keyframes 48 \
  --object-xyz-m 0.0 -0.065 -0.02
```

## Asset Build

Dry-run the build command:

```bash
python -m visual_servoing.scripts.fp_build_assets --object phone --foundationpose-root "$FOUNDATIONPOSE_ROOT"
```

Run the BundleSDF mesh build:

```bash
python -m visual_servoing.scripts.fp_build_assets --object phone --foundationpose-root "$FOUNDATIONPOSE_ROOT" --execute
```

## Live Tracking

```bash
python -m visual_servoing.scripts.fp_track_live --object phone --foundationpose-root "$FOUNDATIONPOSE_ROOT" --print-timing
```

For D435 live tracking:

```bash
python -m visual_servoing.scripts.fp_track_live --object phone --foundationpose-root "$FOUNDATIONPOSE_ROOT" --camera d435 --print-timing
```

Press `q` or `Esc` in the OpenCV window to stop.
