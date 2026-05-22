from __future__ import annotations

import inspect

from visual_servoing.foundationpose_model_free.gui_app import (
    FoundationPoseWorkflowGui,
    GuiCommandBuilder,
    GuiConfig,
    resolve_gui_config,
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
    assert "--live-d405" in segmentation
    assert "--live-d435" in segmentation_d435
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
        data_root=str(tmp_path),
    )

    assert "--record" in record
    assert "--process-recordings" in process
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
    assert process[-2:] == ["--data-root", str(tmp_path)]


def test_gui_main_workflow_hides_legacy_capture_buttons_and_keeps_tracking_focused():
    build_source = inspect.getsource(FoundationPoseWorkflowGui._build)

    assert 'text="3. Recording / Processing"' in build_source
    assert 'text="Start Recording"' in build_source
    assert 'text="Stop Recording"' in build_source
    assert 'text="Processing"' in build_source
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
    assert _default_camera_resolution("unknown") == (640, 480)


def test_gui_resolves_and_passes_default_data_root(tmp_path, monkeypatch):
    from visual_servoing.foundationpose_model_free.gui_app import resolve_gui_config

    monkeypatch.chdir(tmp_path)
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

    assert config.data_root == str(tmp_path / "visual_servoing_data")
    assert command[-2:] == ["--data-root", str(tmp_path / "visual_servoing_data")]
