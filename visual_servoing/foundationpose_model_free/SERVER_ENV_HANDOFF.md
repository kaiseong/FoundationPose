# Server Handoff: Unified `visual` Env For SAM3 + FoundationPose Build

This handoff is for the Codex session running on the remote server, currently assumed to be:

- Server host: `192.168.0.3`
- Server repo: `/home/rby1/FoundationPose`
- SAM3 checkout or package source: usually `/home/rby1/sam3`
- Desired server env: `visual`
- v2 server port: `8081`

The goal is to run **remote processing, SAM3 segmentation, FoundationPose tracking, and BundleSDF Build Assets from one environment** named `visual`.

## Current State

The client GUI can connect to the v2 server:

```text
Connected to 192.168.0.3:8081
```

Earlier failures showed that the server did reach real BundleSDF build execution:

```text
[compute_scene_bounds()] compute_scene_bounds_worker start
[build_octree()] Octree voxel dilate_radius:1
ModuleNotFoundError: No module named 'kaolin'
```

This means the pipeline reached `bundlesdf/run_nerf.py`; the problem was environment dependencies, not the GUI or upload path.

Later, `visual` had the important GPU build dependencies installed:

```text
OK torch 2.8.0+cu128
OK kaolin 0.18.0
OK nvdiffrast 0.4.0
OK pytorch3d 0.7.9
```

but was still missing normal Python packages such as:

```text
joblib
matplotlib
cv2
trimesh
```

When the server was run from the Kaolin-focused env before SAM3 was available, remote processing returned:

```text
Remote processing need_more_recording: accepted=0/16, uploaded_sessions=1
```

That likely means processing ran but every frame was rejected, commonly because SAM3/mask generation was unavailable or failing in the server env.

## Important Design Point

SAM3 and FoundationPose can coexist. The issue is not that SAM3 and FoundationPose are mutually exclusive.

The real compatibility pressure is:

```text
SAM3 wants a modern torch stack
BundleSDF Build Assets needs Kaolin, which is torch/CUDA-version sensitive
```

A unified env is possible if all of these import successfully from the same Python:

```text
sam3
torch
kaolin
nvdiffrast
pytorch3d
joblib
matplotlib
cv2
trimesh
```

The client repo has also been patched so split-env operation is possible through:

```bash
FOUNDATIONPOSE_BUILD_PYTHON=/path/to/build/env/bin/python
```

For this handoff, leave that unset unless you intentionally want split envs. The user's current request is a unified `visual` env.

## Install / Repair The Unified `visual` Env

Activate the server env:

```bash
conda activate visual
cd /home/rby1/FoundationPose
python -V
python -m pip -V
```

The server already confirmed this GPU stack works in `visual`:

```text
torch 2.8.0+cu128
kaolin 0.18.0
nvdiffrast 0.4.0
pytorch3d 0.7.9
```

If those are not present, install them first:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

python -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
```

Then install the normal runtime packages that BundleSDF and processing need:

```bash
python -m pip install \
  joblib matplotlib opencv-python trimesh xatlas imageio PyYAML scipy scikit-learn \
  h5py ruamel.yaml transformations pandas Pillow pyrender \
  pyOpenGL pyOpenGL_accelerate kornia omegaconf psutil tqdm \
  warp-lang scikit-image open3d

# pyrender declares PyOpenGL==3.1.0, but Python 3.12 + EGL textured rendering
# needs the newer PyOpenGL wrapper. Run this after installing pyrender.
python -m pip install --upgrade PyOpenGL==3.1.10 PyOpenGL_accelerate==3.1.10
```

Install or expose SAM3 in this same env. If `/home/rby1/sam3` is a local checkout:

```bash
python -m pip install -e /home/rby1/sam3
```

If editable install is not appropriate, start the server with:

```bash
export PYTHONPATH=/home/rby1/sam3:${PYTHONPATH:-}
```

## Smoke Test

Run this inside `conda activate visual`:

```bash
cd /home/rby1/FoundationPose
python - <<'PY'
import sys
print("python", sys.executable)
for name in [
    "torch",
    "kaolin",
    "nvdiffrast",
    "pytorch3d",
    "joblib",
    "matplotlib",
    "cv2",
    "trimesh",
    "xatlas",
    "yaml",
    "imageio",
    "sam3",
]:
    try:
        mod = __import__(name)
        print("OK", name, getattr(mod, "__version__", ""))
    except Exception as exc:
        print("MISS", name, repr(exc))
PY
```

Expected result: all entries print `OK`.

Also check the exact imports used by the failing Build Assets path:

```bash
python - <<'PY'
from pytorch3d.transforms import so3_log_map, so3_exp_map
import nvdiffrast.torch as dr
import kaolin
import kaolin.ops.spc
import kaolin.render.spc
import joblib
import matplotlib.pyplot as plt
import trimesh
import xatlas
import pyrender
import OpenGL
import cv2
import sam3
print("FoundationPose/BundleSDF/SAM3 import smoke test OK")
PY
```

## Native Extension Checks

FoundationPose also needs compiled local extensions. Check:

```bash
ls /home/rby1/FoundationPose/mycpp/build/mycpp*.so
ls /home/rby1/FoundationPose/bundlesdf/mycuda/build/lib.*/*.so
```

If missing, build them from the active `visual` env:

```bash
cd /home/rby1/FoundationPose
bash build_all_conda.sh
```

If `build_all_conda.sh` fails, preserve the first and last 80 lines of the log for debugging.

## Start The Server From The Unified Env

Use `visual` as the only runtime. Do not set `FOUNDATIONPOSE_BUILD_PYTHON` for the unified-env test.

```bash
conda activate visual
cd /home/rby1/FoundationPose
export FOUNDATIONPOSE_ROOT=/home/rby1/FoundationPose
unset FOUNDATIONPOSE_BUILD_PYTHON
export PYOPENGL_PLATFORM=egl

# Only needed if sam3 was not installed editable into visual:
export PYTHONPATH=/home/rby1/sam3:${PYTHONPATH:-}

python -m visual_servoing.visual_servo_server_v2 --host 0.0.0.0 --port 8081
```

From another shell or from the client machine:

```bash
curl http://192.168.0.3:8081/foundationpose/v2/health
```

Expected:

```json
{"ok": true, "protocol_version": 2, "status": "ready", ...}
```

## Client GUI Settings

On the client GUI:

```text
Host: 192.168.0.3
Port: 8081
```

Click `Connect`. The GUI must show remote/connected state before running remote actions.

Run in this order:

1. `Process Recordings` or the GUI processing button
2. Confirm processing is `ready`, for example `accepted=16/16` or higher
3. `Build Assets`
4. Tracking only after build has produced `model.obj`

## If Processing Returns `accepted=0/16`

That is not a network failure. It means the server processed frames but accepted none.

On the server, inspect the latest processing report:

```bash
cd /home/rby1/FoundationPose
python - <<'PY'
import json
from pathlib import Path

reports = sorted(
    Path("visual_servoing/visual_servoing_data/object_profiles").glob("*/logs/reference_processing_latest.json"),
    key=lambda p: p.stat().st_mtime,
)
report = reports[-1]
data = json.loads(report.read_text())
print("report", report)
print("readiness", data.get("readiness"), "accepted", data.get("accepted"), "required", data.get("required_keyframes"))
for rec in data.get("records", [])[:40]:
    if not rec.get("accepted"):
        print(rec.get("candidate_id"), rec.get("reasons"))
PY
```

Common causes:

- `sam3` does not import in `visual`
- SAM3 model loads but produces no mask for the prompt/object
- ChArUco board was not detected
- depth valid ratio is too low
- recording did not include enough view diversity

ChArUco axes overlay previews, when generated, are under:

```text
visual_servoing/visual_servoing_data/object_profiles/<object>/processing_cache/<run_id>/charuco_axes/
```

For remote processing, these are saved on the server.

## If Build Assets Fails

The Build Assets failure will usually appear as a missing module or a CUDA/build error.

Check the latest build log:

```bash
cd /home/rby1/FoundationPose
find visual_servoing/visual_servoing_data/object_profiles -path '*/logs/build.jsonl' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail
```

Then inspect the relevant file:

```bash
tail -n 1 visual_servoing/visual_servoing_data/object_profiles/<object>/logs/build.jsonl | python -m json.tool
```

Previously seen missing modules and fixes:

```text
joblib       -> python -m pip install joblib
matplotlib   -> python -m pip install matplotlib
pytorch3d    -> install source or wheel compatible with the env torch
nvdiffrast   -> python -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
kaolin       -> torch/CUDA-matched Kaolin wheel, currently torch 2.8.0 cu128 + kaolin 0.18.0
xatlas       -> python -m pip install xatlas
No display   -> export PYOPENGL_PLATFORM=egl before starting the server/build subprocess
glGenTextures TypeError -> python -m pip install --upgrade PyOpenGL==3.1.10 PyOpenGL_accelerate==3.1.10
```

## Split Env Fallback

If unified `visual` becomes unstable for SAM3, use the split model:

```bash
conda activate sam
cd /home/rby1/FoundationPose
export FOUNDATIONPOSE_ROOT=/home/rby1/FoundationPose
export FOUNDATIONPOSE_BUILD_PYTHON=/home/rby1/miniforge3/envs/visual/bin/python
export PYOPENGL_PLATFORM=egl
python -m visual_servoing.visual_servo_server_v2 --host 0.0.0.0 --port 8081
```

In that fallback:

- Server runtime / remote processing / SAM3: `sam`
- Build Assets subprocess / BundleSDF / Kaolin: `visual`

## References

- PyTorch previous version install matrix: https://pytorch.org/get-started/previous-versions/
- NVIDIA Kaolin installation docs: https://kaolin.readthedocs.io/en/stable/notes/installation.html
- NVIDIA Kaolin repository: https://github.com/NVIDIAGameWorks/kaolin
