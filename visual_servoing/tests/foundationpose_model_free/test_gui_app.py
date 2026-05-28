from __future__ import annotations

import base64
import io
import inspect
import json
from pathlib import Path
import sys
from urllib import error as urllib_error
import zipfile

import numpy as np

from visual_servoing.foundationpose_model_free.gui_app import (
    BackgroundCommandRunner,
    FoundationPoseWorkflowGui,
    GuiCommandBuilder,
    GuiConfig,
    create_recordings_archive,
    remote_download_debug_artifacts,
    remote_build_assets,
    remote_download_model_asset,
    remote_process_recordings,
    remote_segmentation_sanity,
    resolve_gui_config,
    summarize_processing_rejections,
    write_segmentation_preview,
    _recording_preview_should_stop,
)
from visual_servoing.visual_servo_protocol_v2 import (
    REQUEST_CONTENT_TYPE,
    decode_foundationpose_segmentation_request,
)


def test_gui_command_builder_constructs_subprocess_commands(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    setup = builder.setup_check(foundationpose_root="/home/kgs/FoundationPose")
    segmentation = builder.segmentation_sanity(prompt="mobile phone")
    segmentation_d435 = builder.segmentation_sanity(prompt="mouse", camera_model="d435", serial="12345")
    turntable = builder.set_reference_poses_turntable(
        object_name="phone",
        axis="y",
        distance_m="0.45",
        start_deg="0",
        step_deg="",
        data_root=str(tmp_path),
    )
    build = builder.build_assets(
        object_name="phone",
        foundationpose_root="/home/kgs/FoundationPose",
        execute=True,
        data_root=str(tmp_path),
    )
    track = builder.track_live(
        object_name="phone",
        prompt="mobile phone",
        foundationpose_root="/home/kgs/FoundationPose",
        auto_reinit=True,
        auto_reinit_after_lost_frames=5,
        camera_model="d435",
        serial="12345",
        width=1280,
        height=720,
        fps=30,
        refine_iterations=1,
        track_iterations=1,
        data_root=str(tmp_path),
    )

    assert setup[:3] == ["python", "-m", "visual_servoing.scripts.fp_setup_check"]
    assert segmentation[:3] == ["python", "-m", "visual_servoing.scripts.point_pose_live"]
    assert "--live" in segmentation
    assert segmentation[segmentation.index("--camera") + 1] == "d405"
    assert "--live" in segmentation_d435
    assert segmentation_d435[segmentation_d435.index("--camera") + 1] == "d435"
    assert segmentation_d435[segmentation_d435.index("--serial") + 1] == "12345"
    assert "--turntable" in turntable
    assert "--data-root" in turntable
    assert "--execute" in build
    assert "--auto-reinit" in track
    assert track[track.index("--prompt") + 1] == "mobile phone"
    assert track[track.index("--camera") + 1] == "d435"
    assert track[track.index("--serial") + 1] == "12345"
    assert track[track.index("--width") + 1] == "1280"
    assert track[track.index("--refine-iterations") + 1] == "1"
    assert track[track.index("--track-iterations") + 1] == "1"


def test_resolve_gui_config_repairs_workspace_parent_foundationpose_root(tmp_path, monkeypatch):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    (foundationpose / "bundlesdf" / "run_nerf.py").write_text("", encoding="utf-8")
    (foundationpose / "estimater.py").write_text("", encoding="utf-8")
    (foundationpose / "learning").mkdir()
    monkeypatch.delenv("FOUNDATIONPOSE_ROOT", raising=False)

    config = resolve_gui_config(GuiConfig(data_root=str(tmp_path / "data"), foundationpose_root=str(tmp_path)))

    assert config.foundationpose_root == str(foundationpose.resolve())


def test_gui_command_builder_supports_pose_dir_import(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    command = builder.set_reference_poses_from_dir(
        object_name="phone",
        pose_dir="/tmp/cam_in_ob",
        data_root=str(tmp_path),
    )

    assert command[:3] == ["python", "-m", "visual_servoing.scripts.fp_set_reference_poses"]
    assert "--pose-dir" in command
    assert "/tmp/cam_in_ob" in command


def test_gui_command_builder_constructs_charuco_offline_command_with_board_defaults(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    command = builder.charuco_reference(
        mode="offline-generate",
        object_name="mouse",
        prompt="wireless mouse",
        object_xyz_m=("0.10", "0.02", "0.00"),
        data_root=str(tmp_path),
    )

    assert command[:3] == ["python", "-m", "visual_servoing.scripts.fp_charuco_reference"]
    assert "--offline-generate" in command
    assert command[command.index("--squares-x") + 1] == "5"
    assert command[command.index("--squares-y") + 1] == "8"
    assert command[command.index("--square-length-m") + 1] == "0.030"
    assert command[command.index("--marker-length-m") + 1] == "0.022"
    assert command[command.index("--charuco-detector-preset") + 1] == "conservative-charuco"
    assert command[command.index("--charuco-origin-convention") + 1] == "charuco_corner_id_0"
    assert command[command.index("--object-xyz-m") + 1 : command.index("--object-xyz-m") + 4] == [
        "0.10",
        "0.02",
        "0.00",
    ]
    assert "--json" in command
    assert command[-2:] == ["--data-root", str(tmp_path)]
    assert command[command.index("--device") + 1] == "auto"
    assert command[command.index("--sam-resolution") + 1] == "1008"


def test_gui_command_builder_constructs_charuco_axis_snapshot_command(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    command = builder.charuco_reference(
        mode="detect-only",
        object_name="mouse",
        prompt="wireless mouse",
        capture_once=True,
        preview_output="/tmp/axis.png",
        data_root=str(tmp_path),
    )

    assert "--detect-only" in command
    assert "--capture-once" in command
    assert command[command.index("--preview-output") + 1] == "/tmp/axis.png"
    assert command[command.index("--axis-length-m") + 1] == "0.05"
    assert command[command.index("--charuco-detector-preset") + 1] == "conservative-charuco"
    assert command[command.index("--device") + 1] == "auto"


def test_gui_command_builder_constructs_record_and_process_commands(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    record = builder.charuco_reference(
        mode="record",
        object_name="mouse",
        prompt="wireless mouse",
        camera_model="d435",
        serial="abc123",
        width=640,
        height=480,
        fps=15,
        frames=16,
        object_xyz_m=("0.01", "0.02", "0.03"),
        object_rpy_deg=("1", "2", "3"),
        required_keyframes="16",
        max_keyframes="48",
        data_root=str(tmp_path),
    )
    process = builder.charuco_reference(
        mode="process-recordings",
        object_name="mouse",
        prompt="wireless mouse",
        camera_model="d435",
        object_xyz_m=("0.01", "0.02", "0.03"),
        object_rpy_deg=("1", "2", "3"),
        required_keyframes="16",
        max_keyframes="48",
        charuco_origin_convention="opencv_board_origin",
        excluded_candidate_ids="session-1:000003",
        data_root=str(tmp_path),
    )
    reselect = builder.charuco_reference(
        mode="reselect-recordings",
        object_name="mouse",
        prompt="wireless mouse",
        camera_model="d435",
        object_xyz_m=("0.01", "0.02", "0.03"),
        object_rpy_deg=("1", "2", "3"),
        required_keyframes="16",
        max_keyframes="24",
        data_root=str(tmp_path),
    )

    assert "--record" in record
    assert "--process-recordings" in process
    assert "--reselect-recordings" in reselect
    assert record[record.index("--charuco-detector-preset") + 1] == "conservative-charuco"
    assert process[process.index("--charuco-detector-preset") + 1] == "conservative-charuco"
    assert reselect[reselect.index("--charuco-detector-preset") + 1] == "conservative-charuco"
    assert record[record.index("--serial") + 1] == "abc123"
    assert record[record.index("--object-xyz-m") + 1 : record.index("--object-xyz-m") + 4] == [
        "0.01",
        "0.02",
        "0.03",
    ]
    assert record[record.index("--object-rpy-deg") + 1 : record.index("--object-rpy-deg") + 4] == [
        "1",
        "2",
        "3",
    ]
    assert process[process.index("--required-keyframes") + 1] == "16"
    assert process[process.index("--max-keyframes") + 1] == "48"
    assert process[process.index("--charuco-origin-convention") + 1] == "opencv_board_origin"
    assert process[process.index("--excluded-candidate-ids") + 1] == "session-1:000003"
    assert reselect[reselect.index("--max-keyframes") + 1] == "24"
    assert process[-2:] == ["--data-root", str(tmp_path)]


def test_gui_main_workflow_hides_legacy_capture_buttons_and_keeps_tracking_focused():
    build_source = inspect.getsource(FoundationPoseWorkflowGui._build)

    assert 'text="3. Recording / Processing"' in build_source
    assert 'text="Start Recording"' in build_source
    assert 'text="Stop Recording"' in build_source
    assert 'text="Processing"' in build_source
    assert 'text="Reselect"' in build_source
    assert 'text="Debug"' in build_source
    assert 'text="Force Build"' in build_source
    assert 'text="Board Axis Snapshot"' in build_source
    assert 'text="Detect Preview"' in build_source
    assert 'text="5. Tracking"' in build_source
    assert 'text="Capture Frame"' not in build_source
    assert 'text="ChArUco Live Capture"' not in build_source
    assert 'text="Generate ChArUco Poses"' not in build_source
    assert 'text="5. Assets / Tracking"' not in build_source


def test_gui_uses_camera_specific_default_resolution():
    from visual_servoing.foundationpose_model_free.gui_app import _default_camera_resolution

    assert _default_camera_resolution("d405") == (640, 480)
    assert _default_camera_resolution("d435") == (640, 480)
    assert _default_camera_resolution("zed") == (672, 376)
    assert _default_camera_resolution("unknown") == (640, 480)


def test_gui_command_builder_uses_zed_live_path_without_forced_default_dimensions(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    segmentation = builder.segmentation_sanity(prompt="mouse", camera_model="zed", width=672, height=376)
    explicit = builder.segmentation_sanity(prompt="mouse", camera_model="zed", width=1280, height=720)
    track = builder.track_live(
        object_name="mouse",
        prompt="wireless mouse",
        foundationpose_root="/home/kgs/FoundationPose",
        auto_reinit=False,
        auto_reinit_after_lost_frames=5,
        camera_model="zed",
        width=672,
        height=376,
    )

    assert "--live-d405" not in segmentation
    assert "--live-d435" not in segmentation
    assert segmentation[segmentation.index("--camera") + 1] == "zed"
    assert "--width" not in segmentation
    assert "--height" not in segmentation
    assert explicit[explicit.index("--width") + 1] == "1280"
    assert explicit[explicit.index("--height") + 1] == "720"
    assert track[track.index("--camera") + 1] == "zed"
    assert track[track.index("--zed-depth-mode") + 1] == "NEURAL"
    assert "--width" not in track
    assert "--height" not in track


def test_gui_command_builder_constructs_remote_track_command(tmp_path):
    builder = GuiCommandBuilder(config=GuiConfig(data_root=str(tmp_path), python_executable="python"))

    command = builder.track_remote_live(
        server_host="192.168.0.3",
        server_port=8081,
        object_name="mouse",
        prompt="wireless mouse",
        foundationpose_root="/home/kgs/FoundationPose",
        auto_reinit=True,
        auto_reinit_after_lost_frames=5,
        camera_model="zed",
        width=672,
        height=376,
        refine_iterations=1,
        track_iterations=1,
        data_root=str(tmp_path),
    )

    assert command[:3] == ["python", "-m", "visual_servoing.visual_servo_client_v2"]
    assert command[command.index("--server-host") + 1] == "192.168.0.3"
    assert command[command.index("--server-port") + 1] == "8081"
    assert command[command.index("--object") + 1] == "mouse"
    assert command[command.index("--camera") + 1] == "zed"
    assert command[command.index("--zed-depth-mode") + 1] == "NEURAL"
    assert "--auto-reinit" in command
    assert command[command.index("--refine-iterations") + 1] == "1"
    assert command[command.index("--track-iterations") + 1] == "1"
    assert "--execute" not in command
    assert "--address" not in command


def test_gui_source_contains_remote_connect_state_flow():
    build_source = inspect.getsource(FoundationPoseWorkflowGui._build)
    connect_source = inspect.getsource(FoundationPoseWorkflowGui.connect_remote_server)
    poll_source = inspect.getsource(FoundationPoseWorkflowGui._poll_queues)
    segmentation_source = inspect.getsource(FoundationPoseWorkflowGui.run_segmentation_check)
    local_tracking_source = inspect.getsource(FoundationPoseWorkflowGui.run_tracking_local)
    remote_tracking_source = inspect.getsource(FoundationPoseWorkflowGui.run_tracking_remote)
    reinit_source = inspect.getsource(FoundationPoseWorkflowGui.reinitialize_tracking_event)
    build_command_source = inspect.getsource(FoundationPoseWorkflowGui.run_build_assets)
    force_build_source = inspect.getsource(FoundationPoseWorkflowGui.run_force_build_assets)
    processing_source = inspect.getsource(FoundationPoseWorkflowGui.run_recording_processing)
    reselect_source = inspect.getsource(FoundationPoseWorkflowGui.run_recording_reselect)
    done_source = inspect.getsource(FoundationPoseWorkflowGui._handle_command_event)

    assert 'text="Server"' in build_source
    assert 'text="Port"' in build_source
    assert 'text="Connect"' in build_source
    assert 'text="Track Local"' in build_source
    assert 'text="Track Remote"' in build_source
    assert "threading.Thread" in connect_source
    assert "remote_events" in poll_source
    assert "_start_remote_segmentation_check" in segmentation_source
    assert "track_live" in local_tracking_source
    assert "_download_remote_model_for_local_tracking" in local_tracking_source
    assert "track_remote_live" in remote_tracking_source
    assert "_last_tracking_mode" in reinit_source
    assert "_start_remote_build" in build_command_source
    assert "_start_remote_build" in force_build_source
    assert build_command_source.index("_start_remote_build") < build_command_source.index("latest_processing_report")
    assert segmentation_source.index("_start_remote_segmentation_check") < segmentation_source.index("segmentation_sanity")
    assert "_start_remote_processing" in processing_source
    assert "_start_remote_processing" in reselect_source
    assert processing_source.index("_start_remote_processing") < processing_source.index("_charuco_command")
    assert "Disconnected" in done_source


def test_summarize_processing_rejections_counts_top_reasons():
    summary = summarize_processing_rejections(
        {
            "records": [
                {"accepted": False, "reasons": ["charuco rejected: corner count 0 below minimum 6"]},
                {"accepted": False, "reasons": ["charuco rejected: corner count 0 below minimum 6"]},
                {"accepted": False, "reason": ["valid depth ratio 0.000 below minimum 0.200"]},
                {"accepted": True, "reasons": ["ignored"]},
            ]
        }
    )

    assert "top rejects:" in summary
    assert "2x charuco rejected" in summary
    assert "1x valid depth ratio" in summary


def test_remote_build_assets_posts_to_server_and_polls_job(monkeypatch):
    calls = []
    responses = [
        {"ok": True, "status": "queued", "job_id": "job-1", "profile": "mouse"},
        {"ok": False, "state": "running", "job_id": "job-1", "profile": "mouse"},
        {"ok": True, "state": "succeeded", "job_id": "job-1", "profile": "mouse"},
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        fake_urlopen,
    )

    result = remote_build_assets(
        host="192.168.0.3",
        port=8081,
        profile="mouse",
        foundationpose_root="/home/kgs/FoundationPose",
        execute=True,
        poll_interval_s=0.0,
        max_wait_s=1.0,
    )

    assert result["state"] == "succeeded"
    post_request = calls[0][0]
    assert post_request.full_url == "http://192.168.0.3:8081/foundationpose/v2/assets/build"
    posted = json.loads(post_request.data.decode("utf-8"))
    assert posted["profile"] == "mouse"
    assert posted["foundationpose_root"] == "/home/kgs/FoundationPose"
    assert posted["execute"] is True
    assert calls[1][0] == "http://192.168.0.3:8081/foundationpose/v2/assets/build/job-1"
    assert calls[2][0] == "http://192.168.0.3:8081/foundationpose/v2/assets/build/job-1"


def test_remote_download_model_asset_saves_model_obj(tmp_path, monkeypatch):
    calls = []

    class FakeHeaders(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class FakeResponse:
        headers = FakeHeaders(
            {
                "X-FoundationPose-Mesh-Sha256": "abc123",
                "X-FoundationPose-Mesh-Size": "7",
            }
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"# obj\n"

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        fake_urlopen,
    )

    target = tmp_path / "assets" / "model" / "model.obj"
    result = remote_download_model_asset(
        host="192.168.0.3",
        port=8081,
        profile="multimeter zed",
        target_path=target,
    )

    assert calls == [("http://192.168.0.3:8081/foundationpose/v2/assets/model/multimeter%20zed", 60.0)]
    assert target.read_bytes() == b"# obj\n"
    assert result["bytes"] == 6
    assert result["sha256"] == "abc123"


def test_remote_download_debug_artifacts_fetches_encoded_profile_and_extracts_zip(tmp_path, monkeypatch):
    calls = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"candidate_count": 1}))
        zf.writestr("masks/session-1_000000.png", b"mask")
        zf.writestr("depth_colormap/session-1_000000.png", b"depth")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return buffer.getvalue()

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        fake_urlopen,
    )

    profile_root = tmp_path / "multimeter zed"
    result = remote_download_debug_artifacts(
        host="192.168.0.3",
        port=8081,
        profile="multimeter zed",
        profile_root=profile_root,
    )

    output_dir = Path(result["output_dir"])
    assert calls == [("http://192.168.0.3:8081/foundationpose/v2/debug/multimeter%20zed", 120.0)]
    assert output_dir.parent == profile_root / "debug_downloads"
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "masks" / "session-1_000000.png").read_bytes() == b"mask"
    assert (output_dir / "depth_colormap" / "session-1_000000.png").read_bytes() == b"depth"
    assert result["file_count"] == 3


def test_remote_download_debug_artifacts_rejects_unsafe_zip_members(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape.txt", b"bad")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return buffer.getvalue()

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        lambda url, timeout: FakeResponse(),
    )

    try:
        remote_download_debug_artifacts(
            host="192.168.0.3",
            port=8081,
            profile="meter",
            profile_root=tmp_path / "meter",
        )
        raise AssertionError("expected unsafe member rejection")
    except RuntimeError as exc:
        assert "unsafe debug artifact member" in str(exc)
    assert not (tmp_path / "escape.txt").exists()


def test_remote_download_debug_artifacts_preserves_server_error_detail(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout):
        payload = json.dumps(
            {
                "ok": False,
                "status": "error",
                "reason": "latest Processing report has no selected candidates",
            }
        ).encode("utf-8")
        raise urllib_error.HTTPError(url, 409, "Conflict", {}, io.BytesIO(payload))

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        fake_urlopen,
    )

    try:
        remote_download_debug_artifacts(
            host="192.168.0.3",
            port=8081,
            profile="meter",
            profile_root=tmp_path / "meter",
        )
        raise AssertionError("expected server error detail")
    except RuntimeError as exc:
        message = str(exc)
    assert "HTTP 409" in message
    assert "latest Processing report has no selected candidates" in message


def test_recording_preview_ignores_stale_stop_state_during_startup_grace():
    assert _recording_preview_should_stop(key=ord("q"), window_visible=True, now_s=1.0, grace_until_s=2.0) is False
    assert _recording_preview_should_stop(key=255, window_visible=False, now_s=1.0, grace_until_s=2.0) is False
    assert _recording_preview_should_stop(key=ord("q"), window_visible=True, now_s=2.1, grace_until_s=2.0) is True
    assert _recording_preview_should_stop(key=255, window_visible=False, now_s=2.1, grace_until_s=2.0) is True
    assert _recording_preview_should_stop(key=255, window_visible=True, now_s=2.1, grace_until_s=2.0) is False


def test_create_recordings_archive_includes_request_and_sessions(tmp_path):
    profile_root = tmp_path / "object_profiles" / "meter"
    session_dir = profile_root / "recordings" / "session-1"
    _write_fake_recorded_frame(session_dir, 0)

    archive, summary = create_recordings_archive(
        profile_root,
        request_payload={"profile": "meter", "prompt": "multimeter"},
    )

    assert summary["session_count"] == 1
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = set(zf.namelist())
        payload = json.loads(zf.read("foundationpose_processing_request.json").decode("utf-8"))
    assert payload["profile"] == "meter"
    assert "recordings/session-1/session.json" in names
    assert "recordings/session-1/frames.jsonl" in names
    assert "recordings/session-1/rgb/000000.png" in names
    assert "recordings/session-1/depth/000000.npy" in names
    assert "recordings/session-1/depth_mm/000000.png" not in names
    assert "recordings/session-1/intrinsics/000000.json" not in names


def test_create_recordings_archive_samples_large_recordings(tmp_path):
    profile_root = tmp_path / "object_profiles" / "meter"
    session_dir = profile_root / "recordings" / "session-1"
    for index in range(10):
        _write_fake_recorded_frame(session_dir, index)

    archive, summary = create_recordings_archive(
        profile_root,
        request_payload={"profile": "meter", "prompt": "multimeter", "max_upload_frames": 4},
    )

    assert summary["source_frame_count"] == 10
    assert summary["frame_count"] == 4
    assert summary["sampled"] is True
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        frames = zf.read("recordings/session-1/frames.jsonl").decode("utf-8").splitlines()
        names = set(zf.namelist())
    selected = [json.loads(line)["index"] for line in frames]
    assert selected == [0, 3, 6, 9]
    assert "recordings/session-1/rgb/000001.png" not in names
    assert "recordings/session-1/rgb/000009.png" in names
    assert "recordings/session-1/depth_mm/000009.png" not in names
    assert "recordings/session-1/intrinsics/000009.json" not in names


def _write_fake_recorded_frame(session_dir: Path, index: int) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    for dirname in ("rgb", "depth", "depth_mm", "intrinsics"):
        (session_dir / dirname).mkdir(exist_ok=True)
    stem = f"{index:06d}"
    files = {
        "rgb_path": f"rgb/{stem}.png",
        "depth_path": f"depth/{stem}.npy",
        "depth_mm_path": f"depth_mm/{stem}.png",
        "intrinsics_path": f"intrinsics/{stem}.json",
    }
    for relative in files.values():
        (session_dir / relative).write_bytes(b"x")
    record = {
        "session_id": session_dir.name,
        "index": index,
        "timestamp_s": float(index),
        "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        **files,
    }
    with (session_dir / "frames.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def test_remote_process_recordings_posts_zip_and_polls_job(monkeypatch):
    calls = []
    responses = [
        {"ok": True, "status": "queued", "job_id": "job-1", "profile": "meter"},
        {"ok": False, "state": "running", "job_id": "job-1", "profile": "meter"},
        {
            "ok": True,
            "state": "succeeded",
            "job_id": "job-1",
            "profile": "meter",
            "result": {"readiness": "ready", "accepted": 16},
        },
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        fake_urlopen,
    )

    result = remote_process_recordings(
        host="192.168.0.3",
        port=8081,
        profile="meter",
        archive=b"zip-bytes",
        poll_interval_s=0.0,
        max_wait_s=1.0,
    )

    assert result["state"] == "succeeded"
    post_request = calls[0][0]
    assert post_request.full_url == "http://192.168.0.3:8081/foundationpose/v2/recordings/process"
    assert post_request.data == b"zip-bytes"
    headers = dict(post_request.header_items())
    assert headers["Content-type"] == "application/x-foundationpose-recordings+zip"
    assert headers["X-foundationpose-profile"] == "meter"
    assert calls[1][0] == "http://192.168.0.3:8081/foundationpose/v2/recordings/process/job-1"
    assert calls[2][0] == "http://192.168.0.3:8081/foundationpose/v2/recordings/process/job-1"


def test_remote_segmentation_sanity_posts_npz_to_server(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "ok": True,
                    "status": "segmented",
                    "mask": {"area": 4, "area_fraction": 0.2},
                    "mask_png_b64": "mask",
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.gui_app.urllib_request.urlopen",
        fake_urlopen,
    )

    result = remote_segmentation_sanity(
        host="192.168.0.3",
        port=8081,
        prompt="multimeter",
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        depth_m=np.ones((4, 5), dtype=np.float32),
        sam_device="cpu",
        sam_resolution=128,
    )

    assert result["status"] == "segmented"
    request = calls[0][0]
    assert request.full_url == "http://192.168.0.3:8081/foundationpose/v2/segmentation"
    assert dict(request.header_items())["Content-type"] == REQUEST_CONTENT_TYPE
    decoded = decode_foundationpose_segmentation_request(request.data)
    assert decoded.prompt == "multimeter"
    assert decoded.rgb.shape == (4, 5, 3)
    assert decoded.depth_m.shape == (4, 5)
    assert decoded.mask_options["device"] == "cpu"
    assert decoded.mask_options["resolution"] == 128


def test_write_segmentation_preview_writes_rgb_and_overlay(tmp_path):
    import pytest

    cv2 = pytest.importorskip("cv2")

    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    rgb_path, overlay_path = write_segmentation_preview(
        tmp_path,
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        mask_png_b64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )

    assert rgb_path.exists()
    assert overlay_path.exists()
    assert "remote_segmentation" in overlay_path.as_posix()


def test_gui_resolves_and_passes_default_data_root(tmp_path, monkeypatch):
    from visual_servoing.common import paths
    from visual_servoing.foundationpose_model_free.gui_app import resolve_gui_config

    monkeypatch.chdir(tmp_path)
    expected_root = Path(paths.__file__).resolve().parents[1] / "visual_servoing_data"
    config = resolve_gui_config(GuiConfig(python_executable="python"))
    builder = GuiCommandBuilder(config=config)
    command = builder.set_reference_poses_turntable(
        object_name="mouse",
        axis="y",
        distance_m="0.22",
        start_deg="0",
        step_deg="",
        data_root=config.data_root,
    )

    assert config.data_root == str(expected_root)
    assert command[-2:] == ["--data-root", str(expected_root)]


def test_background_command_runner_can_stop_and_wait(tmp_path):
    events = []
    runner = BackgroundCommandRunner(on_event=events.append, cwd=tmp_path)

    runner.start([sys.executable, "-c", "import time; time.sleep(30)"])

    assert runner.running is True
    assert runner.stop_and_wait(timeout_s=2.0) is True
    assert runner.running is False
    assert any(event.kind == "stop" for event in events)
