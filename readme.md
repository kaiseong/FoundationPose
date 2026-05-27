# RBY1 Remote Visual Servoing

This repository is currently used as the RBY1 visual-servo workspace. The
original FoundationPose paper/demo instructions were removed from this README so
the operator-facing entry points are the server/client commands below.

## Runtime Layout

- 5090 workstation: runs `visual_servoing.visual_servo_server`.
  - Receives RGB-D frames from the client over HTTP.
  - Runs SAM3 segmentation on the GPU.
  - Estimates the object center from the masked point cloud.
  - Converts the target to `link_torso_5` / T5 and returns an EE target.
  - Does not import or command `rby1_sdk`.
- Jetson Orin / robot side: runs `visual_servoing.visual_servo_client`.
  - Captures D435 RGB-D frames.
  - Reads robot FK for the live camera mount and current `ee_right` pose.
  - Sends frames and pose metadata to the server.
  - Validates freshness and step limits.
  - Sends right-arm Cartesian commands to the robot.

The remote request path is:

1. Camera point cloud center is estimated in the camera frame on the server from
   the transmitted RGB-D frame and selected mask.
2. The client supplies `t5_T_camera`; in execute mode this comes from live FK of
   `link_head_2` plus `--head-camera-pose`.
3. The server transforms the object center into T5 and plans xyz-only motion.
4. The EE rotation is preserved from the client-side reference pose; only
   `ee_right` xyz is servoed in T5.

## Server Command

Run this on the 5090 workstation:

```bash
cd ~/FoundationPose
git pull --ff-only kaiseong main
conda activate visual

python -m visual_servoing.visual_servo_server \
  --host 0.0.0.0 \
  --port 8080 \
  --device cuda
```

Expected startup output:

```json
{"event":"visual_servo_server_listening","host":"0.0.0.0","port":8080}
```

Keep this process running while the client is active. If the server IP changes,
update the client `--remote-server` value.

## Client Command

Run this on the Jetson Orin / robot-side machine:

```bash
cd ~/FoundationPose
git pull --ff-only kaiseong main
conda activate visual

python -m visual_servoing.visual_servo_client \
  --live-d435 \
  --remote-server 192.168.0.3:8080 \
  --prompt multimeter \
  --address 0.0.0.0:50051 \
  --max-translation-step-m 0.03 \
  --show-camera-window \
  --show-mask-window
```

The command above keeps a conservative 3 cm translation step. The current
default is 6 cm, so this shorter form uses all current defaults:

```bash
python -m visual_servoing.visual_servo_client \
  --live-d435 \
  --remote-server 192.168.0.3:8080 \
  --prompt multimeter \
  --address 0.0.0.0:50051 \
  --show-camera-window \
  --show-mask-window
```

Important defaults:

- `--execute` is enabled by default. Use `--no-execute` for dry-run.
- `--move-to-ready-on-connect` is enabled by default. Use
  `--no-move-to-ready-on-connect` if the current arm pose must be preserved.
- `--width 1280 --height 720 --fps 15` are defaults.
- `--max-iterations 0` means run until interrupted.
- `--remote-timeout-s 2` and `--stale-action-max-age-s 1.0` are defaults.
- Default EE link is `ee_right`; `link_right_arm_6` is still accepted.

## Quick Connectivity Check

With the server running, this sends a synthetic request without camera or robot
motion:

```bash
python -m visual_servoing.visual_servo_client \
  --remote-fixture-request \
  --remote-server 192.168.0.3:8080 \
  --no-execute \
  --max-iterations 1
```

For a lower-level network check, the client must be able to reach:

- `192.168.0.3:8080` for HTTP visual-servo requests.
- The RBY1 SDK endpoint passed by `--address`, commonly port `50051`.

## Jetson Orin Dependencies

The remote client does not need SAM3, FoundationPose rendering extensions, or a
local CUDA inference stack when using `--remote-server`. It does need the robot,
camera, and preview/runtime dependencies below.

Required on Jetson Orin:

- Python environment that can import this repository, recommended Python 3.11.
- `numpy`.
- OpenCV Python bindings if using `--show-camera-window`, `--show-mask-window`,
  or offline image loading. On Jetson, `python3-opencv` from apt is often more
  reliable than a generic pip wheel for GUI/HighGUI support.
- Intel RealSense runtime plus Python binding for D435 capture:
  `pyrealsense2`. If no aarch64 wheel is available, install or build
  librealsense with Python bindings for the Jetson image.
- `rby1_sdk` matching the robot SDK and Jetson Python ABI. This is required only
  when execute mode is used.
- Network route from Jetson to the 5090 server on TCP `8080`.
- Network route from Jetson to the robot/RBY1 control endpoint on TCP `50051`
  or the endpoint used by your deployment.
- USB permission/udev setup for the D435 device.

Recommended Jetson setup sketch:

```bash
cd ~/FoundationPose
conda activate visual
python -m pip install numpy
python -m pip install opencv-python

# Install these from the platform-specific packages/wheels used by the lab:
#   pyrealsense2
#   rby1_sdk
```

If OpenCV GUI windows fail on Jetson, replace the pip OpenCV package with the
Jetson/Ubuntu package and rerun:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
```

Required on the 5090 server:

- Python environment that can import this repository.
- CUDA-capable PyTorch matching the installed NVIDIA driver.
- SAM3 package/model setup. The code prefers `/home/kgs/sam3` when it exists;
  otherwise `sam3` must be importable from the active environment.
- `Pillow`, `numpy`, and OpenCV from `requirements.txt`.
- No `rby1_sdk` and no RealSense device are required on the server.

Server setup sketch:

```bash
cd ~/FoundationPose
conda activate visual
python -m pip install -r requirements.txt

# Install PyTorch and SAM3 according to the GPU/server environment.
python - <<'PY'
import torch
import sam3
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sam3", getattr(sam3, "__file__", "imported"))
PY
```

## Operational Notes

- In remote mode, the client `--prompt` controls SAM3 segmentation. Server
  `--prompt` is only a fallback for requests that omit prompt metadata.
- Client logs are concise by default: frame index, status, fps, action latency,
  encode latency, command result, and reason. Use `--debug` for full JSON.
- `--show-mask-window` requests the selected server mask preview and overlays it
  on the client camera window.
- If the server logs `BrokenPipeError`, the client disconnected before the
  server finished writing the response, usually because `--remote-timeout-s` was
  shorter than SAM3/ZED warmup. Warm the server once with a long timeout, or run
  the first live test with `--remote-timeout-s 10`.
- If the server returns `No usable object mask was produced`, robot motion is
  cancelled for that frame.
- If `remote action wrist step exceeds limit` appears, make sure the server is
  updated to the same commit as the client and that the EE rotation preservation
  path is active.
- If the control manager is disabled or faulted, recover the robot state before
  starting the client; the client cannot move the arm through a disabled/faulted
  controller.
