"""Tkinter workflow GUI for FoundationPose model-free onboarding."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import argparse
import io
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import zipfile

from visual_servoing.common.paths import data_root
from visual_servoing.point_pose.live_camera_config import (
    SUPPORTED_LIVE_CAMERA_MODELS,
    default_camera_resolution,
    is_default_camera_resolution,
)
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera
from visual_servoing.point_pose.zed_camera import DEFAULT_ZED_DEPTH_MODE
from visual_servoing.visual_servo_protocol_v2 import (
    REQUEST_CONTENT_TYPE,
    encode_foundationpose_segmentation_request,
)

from .asset_builder import find_generated_mesh, profile_model_path
from .charuco_reference import (
    CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
    BoardObjectTransform,
    CharucoBoardSpec,
)
from .reference_dataset import count_reference_frames
from .reference_processing import latest_processing_report
from .reference_recording import ReferenceRecordingConfig, ReferenceRecordingSession, list_recording_sessions
from .registry import ObjectProfileRegistry
from .profile_manifest import record_asset_ready


@dataclass(frozen=True)
class GuiConfig:
    data_root: str | None = None
    foundationpose_root: str | None = None
    python_executable: str = sys.executable
    title: str = "FoundationPose Model-Free Workflow"


@dataclass(frozen=True)
class CommandEvent:
    kind: str
    text: str


@dataclass(frozen=True)
class RecordingEvent:
    kind: str
    text: str
    session_id: str | None = None
    frame_count: int = 0


@dataclass(frozen=True)
class RemoteHealthEvent:
    state: str
    text: str
    preview_rgb_path: str | None = None
    preview_overlay_path: str | None = None
    preview_status: str | None = None


DEFAULT_CAMERA_RESOLUTIONS = {
    "d405": (640, 480),
    "d435": (640, 480),
    "zed": (672, 376),
}
RECORDING_PREVIEW_WINDOW = "Reference Recording Preview"
REMOTE_BUILD_POLL_INTERVAL_S = 2.0
REMOTE_BUILD_TIMEOUT_S = 60.0 * 60.0
REMOTE_PROCESS_POLL_INTERVAL_S = 2.0
REMOTE_PROCESS_UPLOAD_TIMEOUT_S = 120.0
REMOTE_PROCESS_TIMEOUT_S = 60.0 * 60.0
REMOTE_PROCESS_UPLOAD_FRAME_CAP = 192
REMOTE_MODEL_DOWNLOAD_TIMEOUT_S = 60.0
REMOTE_DEBUG_DOWNLOAD_TIMEOUT_S = 120.0
REMOTE_SEGMENTATION_WARMUP_FRAMES = 10
RECORDINGS_ZIP_CONTENT_TYPE = "application/x-foundationpose-recordings+zip"
PROCESSING_REQUEST_JSON = "foundationpose_processing_request.json"


class GuiCommandBuilder:
    def __init__(self, *, config: GuiConfig) -> None:
        self.config = config
        self.cwd = Path(__file__).resolve().parents[2]

    def module(self, module: str, *args: str) -> list[str]:
        return [self.config.python_executable, "-m", module, *args]

    def setup_check(self, *, foundationpose_root: str, camera_model: str = "all") -> list[str]:
        command = self.module(
            "visual_servoing.scripts.fp_setup_check",
            "--foundationpose-path",
            foundationpose_root,
            "--camera",
            camera_model,
            "--strict",
        )
        return command

    def segmentation_sanity(
        self,
        *,
        prompt: str,
        camera_model: str = "d405",
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> list[str]:
        command = self.module(
            "visual_servoing.scripts.point_pose_live",
            "--live",
            "--camera",
            camera_model,
            "--prompt",
            prompt,
            "--fps",
            str(fps),
            "--print-timing",
        )
        self._append_live_dimensions(command, camera_model=camera_model, width=width, height=height)
        self._append_serial(command, serial)
        return command

    def set_reference_poses_turntable(
        self,
        *,
        object_name: str,
        axis: str,
        distance_m: str,
        start_deg: str,
        step_deg: str,
        data_root: str | None = None,
    ) -> list[str]:
        command = self.module(
            "visual_servoing.scripts.fp_set_reference_poses",
            "--object",
            object_name,
            "--turntable",
            "--axis",
            axis,
            "--distance-m",
            distance_m,
            "--start-deg",
            start_deg,
        )
        if step_deg.strip():
            command.extend(["--step-deg", step_deg])
        self._append_data_root(command, data_root)
        return command

    def set_reference_poses_from_dir(
        self,
        *,
        object_name: str,
        pose_dir: str,
        data_root: str | None = None,
    ) -> list[str]:
        command = self.module(
            "visual_servoing.scripts.fp_set_reference_poses",
            "--object",
            object_name,
            "--pose-dir",
            pose_dir,
        )
        self._append_data_root(command, data_root)
        return command

    def build_assets(
        self,
        *,
        object_name: str,
        foundationpose_root: str,
        execute: bool,
        data_root: str | None = None,
    ) -> list[str]:
        command = self.module(
            "visual_servoing.scripts.fp_build_assets",
            "--object",
            object_name,
            "--foundationpose-root",
            foundationpose_root,
            "--json",
        )
        if execute:
            command.append("--execute")
        self._append_data_root(command, data_root)
        return command

    def charuco_reference(
        self,
        *,
        mode: str,
        object_name: str,
        prompt: str,
        camera_model: str = "d435",
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        frames: int = 16,
        squares_x: str = "5",
        squares_y: str = "8",
        square_length_m: str = "0.030",
        marker_length_m: str = "0.022",
        dictionary: str = "auto",
        object_xyz_m: tuple[str, str, str] = ("0.0", "0.0", "0.0"),
        object_rpy_deg: tuple[str, str, str] = ("0.0", "0.0", "0.0"),
        capture_once: bool = False,
        preview_output: str | None = None,
        axis_length_m: str = "0.05",
        sam_device: str = "auto",
        sam_resolution: str = "1008",
        required_keyframes: str = "16",
        max_keyframes: str = "32",
        charuco_detector_preset: str = CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
        data_root: str | None = None,
    ) -> list[str]:
        if mode not in {
            "offline-generate",
            "detect-only",
            "live-capture",
            "record",
            "process-recordings",
            "reselect-recordings",
        }:
            raise ValueError(
                "mode must be offline-generate, detect-only, live-capture, record, process-recordings, "
                "or reselect-recordings"
            )
        command = self.module(
            "visual_servoing.scripts.fp_charuco_reference",
            f"--{mode}",
            "--object",
            object_name,
            "--prompt",
            prompt,
            "--camera",
            camera_model,
            "--fps",
            str(fps),
            "--frames",
            str(frames),
            "--squares-x",
            str(squares_x),
            "--squares-y",
            str(squares_y),
            "--square-length-m",
            str(square_length_m),
            "--marker-length-m",
            str(marker_length_m),
            "--dictionary",
            dictionary,
            "--charuco-detector-preset",
            charuco_detector_preset,
            "--object-xyz-m",
            *[str(value) for value in object_xyz_m],
            "--object-rpy-deg",
            *[str(value) for value in object_rpy_deg],
            "--axis-length-m",
            str(axis_length_m),
            "--device",
            str(sam_device),
            "--sam-resolution",
            str(sam_resolution),
            "--required-keyframes",
            str(required_keyframes),
            "--max-keyframes",
            str(max_keyframes),
            "--json",
        )
        self._append_live_dimensions(command, camera_model=camera_model, width=width, height=height)
        if capture_once:
            command.append("--capture-once")
        if preview_output:
            command.extend(["--preview-output", preview_output])
        self._append_serial(command, serial)
        self._append_data_root(command, data_root)
        return command

    def track_live(
        self,
        *,
        object_name: str,
        prompt: str,
        foundationpose_root: str,
        auto_reinit: bool,
        auto_reinit_after_lost_frames: int,
        camera_model: str = "d405",
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        refine_iterations: int = 5,
        track_iterations: int = 2,
        zed_depth_mode: str = DEFAULT_ZED_DEPTH_MODE,
        data_root: str | None = None,
    ) -> list[str]:
        command = self.module(
            "visual_servoing.scripts.fp_track_live",
            "--object",
            object_name,
            "--prompt",
            prompt,
            "--foundationpose-root",
            foundationpose_root,
            "--camera",
            camera_model,
            "--fps",
            str(fps),
            "--print-timing",
            "--auto-reinit-after-lost-frames",
            str(auto_reinit_after_lost_frames),
            "--refine-iterations",
            str(refine_iterations),
            "--track-iterations",
            str(track_iterations),
        )
        self._append_live_dimensions(command, camera_model=camera_model, width=width, height=height)
        self._append_zed_depth_mode(command, camera_model=camera_model, depth_mode=zed_depth_mode)
        if auto_reinit:
            command.append("--auto-reinit")
        self._append_serial(command, serial)
        self._append_data_root(command, data_root)
        return command

    def track_remote_live(
        self,
        *,
        server_host: str,
        server_port: int,
        object_name: str,
        prompt: str,
        foundationpose_root: str,
        auto_reinit: bool,
        auto_reinit_after_lost_frames: int,
        camera_model: str = "d405",
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        refine_iterations: int = 5,
        track_iterations: int = 2,
        zed_depth_mode: str = DEFAULT_ZED_DEPTH_MODE,
        data_root: str | None = None,
    ) -> list[str]:
        command = self.module(
            "visual_servoing.visual_servo_client_v2",
            "--server-host",
            str(server_host),
            "--server-port",
            str(int(server_port)),
            "--object",
            object_name,
            "--prompt",
            prompt,
            "--foundationpose-root",
            foundationpose_root,
            "--camera",
            camera_model,
            "--fps",
            str(fps),
            "--refine-iterations",
            str(refine_iterations),
            "--track-iterations",
            str(track_iterations),
        )
        self._append_live_dimensions(command, camera_model=camera_model, width=width, height=height)
        self._append_zed_depth_mode(command, camera_model=camera_model, depth_mode=zed_depth_mode)
        if auto_reinit:
            command.append("--auto-reinit")
        command.extend(["--auto-reinit-after-lost-frames", str(auto_reinit_after_lost_frames)])
        self._append_serial(command, serial)
        self._append_data_root(command, data_root)
        return command

    @staticmethod
    def _append_data_root(command: list[str], data_root: str | None) -> None:
        if data_root:
            command.extend(["--data-root", data_root])

    @staticmethod
    def _append_serial(command: list[str], serial: str | None) -> None:
        if serial and serial.strip():
            command.extend(["--serial", serial.strip()])

    @staticmethod
    def _append_live_dimensions(command: list[str], *, camera_model: str, width: int, height: int) -> None:
        if camera_model.lower() == "zed" and is_default_camera_resolution(camera_model, width, height):
            return
        command.extend(["--width", str(width), "--height", str(height)])

    @staticmethod
    def _append_zed_depth_mode(command: list[str], *, camera_model: str, depth_mode: str) -> None:
        if camera_model.lower() == "zed":
            command.extend(["--zed-depth-mode", str(depth_mode).upper()])


class BackgroundCommandRunner:
    def __init__(
        self,
        *,
        on_event: Callable[[CommandEvent], None],
        cwd: str | Path,
    ) -> None:
        self.on_event = on_event
        self.cwd = str(cwd)
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, command: list[str]) -> None:
        if self.running:
            raise RuntimeError("another command is already running")
        self.on_event(CommandEvent("start", " ".join(command)))
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        conda_prefix = env.get("CONDA_PREFIX")
        if conda_prefix:
            lib_path = str(Path(conda_prefix) / "lib")
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
        self._process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_output, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.running and self._process is not None:
            self.on_event(CommandEvent("stop", "terminating running command"))
            self._process.terminate()

    def stop_and_wait(self, *, timeout_s: float = 6.0) -> bool:
        if not self.running or self._process is None:
            return True
        process = self._process
        self.stop()
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.on_event(CommandEvent("stop", "killing unresponsive running command"))
            process.kill()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return False
        return process.poll() is not None

    def _read_output(self) -> None:
        assert self._process is not None
        process = self._process
        if process.stdout is not None:
            for line in process.stdout:
                self.on_event(CommandEvent("output", line.rstrip()))
        returncode = process.wait()
        self.on_event(CommandEvent("done", f"returncode={returncode}"))


class FoundationPoseWorkflowGui:
    def __init__(self, config: GuiConfig | None = None) -> None:
        self.config = resolve_gui_config(config)
        self.registry = ObjectProfileRegistry(self.config.data_root)
        self.command_builder = GuiCommandBuilder(config=self.config)
        self.command_events: queue.Queue[CommandEvent] = queue.Queue()
        self.recording_events: queue.Queue[RecordingEvent | Exception] = queue.Queue()
        self.remote_events: queue.Queue[RemoteHealthEvent] = queue.Queue()
        self.runner = BackgroundCommandRunner(on_event=self.command_events.put, cwd=self.command_builder.cwd)
        self.recording_session: ReferenceRecordingSession | None = None
        self.recording_busy = False
        self._recording_stop = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._capture_preview_images: tuple[tk.PhotoImage, ...] | None = None
        self._pending_axis_preview_path: Path | None = None
        self._active_remote_command = False
        self._last_tracking_mode = "local"

        self.root = tk.Tk()
        self.root.title(self.config.title)
        self.profile_name = tk.StringVar(value="phone")
        self.prompt = tk.StringVar(value="mobile phone")
        self.status = tk.StringVar(value="Ready")
        self.remote_state = tk.StringVar(value="Local")
        self.server_host = tk.StringVar(value="127.0.0.1")
        self.server_port = tk.IntVar(value=8081)
        self.capture_status = tk.StringVar(value="Capture: 0 / 16")
        self.recording_status = tk.StringVar(value="Recording: no raw frames")
        self.processing_status = tk.StringVar(value="Processing: no report")
        self.capture_preview_status = tk.StringVar(value="No capture preview yet")
        self.foundationpose_root = tk.StringVar(
            value=self.config.foundationpose_root or os.environ.get("FOUNDATIONPOSE_ROOT", "/home/kgs/FoundationPose")
        )
        self.charuco_squares_x = tk.StringVar(value="5")
        self.charuco_squares_y = tk.StringVar(value="8")
        self.charuco_square_length_m = tk.StringVar(value="0.030")
        self.charuco_marker_length_m = tk.StringVar(value="0.022")
        self.charuco_dictionary = tk.StringVar(value="auto")
        self.charuco_object_x_m = tk.StringVar(value="0.0")
        self.charuco_object_y_m = tk.StringVar(value="0.0")
        self.charuco_object_z_m = tk.StringVar(value="0.0")
        self.charuco_object_roll_deg = tk.StringVar(value="0.0")
        self.charuco_object_pitch_deg = tk.StringVar(value="0.0")
        self.charuco_object_yaw_deg = tk.StringVar(value="0.0")
        self.sam_device = tk.StringVar(value="auto")
        self.sam_resolution = tk.IntVar(value=1008)
        self.camera_model = tk.StringVar(value="d405")
        self.camera_serial = tk.StringVar(value="")
        self.camera_width = tk.IntVar(value=640)
        self.camera_height = tk.IntVar(value=480)
        self.camera_fps = tk.IntVar(value=15)
        self.auto_reinit = tk.BooleanVar(value=True)
        self.auto_reinit_after = tk.IntVar(value=5)
        self.refine_iterations = tk.IntVar(value=5)
        self.track_iterations = tk.IntVar(value=2)
        self.reference_target = tk.IntVar(value=16)
        self.max_keyframes = tk.IntVar(value=32)
        self.listbox = tk.Listbox(self.root, width=54, height=12)
        self.log_text = tk.Text(self.root, width=98, height=14, state="disabled")
        self.capture_rgb_preview = ttk.Label(self.root)
        self.capture_mask_preview = ttk.Label(self.root)
        self._build()
        self.refresh()
        self.root.bind("r", self.reinitialize_tracking_event)
        self.root.bind("R", self.reinitialize_tracking_event)
        self.root.after(100, self._poll_queues)

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        profiles = ttk.LabelFrame(main, text="1. Object Profile", padding=6)
        profiles.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(profiles, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(profiles, textvariable=self.profile_name, width=24).grid(row=0, column=1, sticky="ew")
        ttk.Label(profiles, text="Prompt").grid(row=1, column=0, sticky="w")
        ttk.Entry(profiles, textvariable=self.prompt, width=24).grid(row=1, column=1, sticky="ew")
        ttk.Button(profiles, text="Create/Update", command=self.create_profile).grid(row=2, column=0, sticky="ew")
        ttk.Button(profiles, text="Select", command=self.select).grid(row=2, column=1, sticky="ew")
        ttk.Button(profiles, text="Delete", command=self.delete).grid(row=3, column=0, sticky="ew")
        ttk.Button(profiles, text="Refresh", command=self.refresh).grid(row=3, column=1, sticky="ew")
        self.listbox.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        profiles.columnconfigure(1, weight=1)
        profiles.rowconfigure(4, weight=1)

        stages = ttk.Frame(main)
        stages.grid(row=0, column=1, sticky="nsew")
        stages.columnconfigure(0, weight=1)

        setup = ttk.LabelFrame(stages, text="2. Setup / Segmentation", padding=6)
        setup.grid(row=0, column=0, sticky="ew")
        setup.columnconfigure(1, weight=1)
        setup.columnconfigure(3, weight=1)
        ttk.Label(setup, text="FoundationPose").grid(row=0, column=0, sticky="w")
        ttk.Entry(setup, textvariable=self.foundationpose_root).grid(row=0, column=1, sticky="ew")
        ttk.Button(setup, text="Setup Check", command=self.run_setup_check).grid(row=0, column=2, sticky="ew")
        ttk.Button(setup, text="Segmentation Sanity", command=self.run_segmentation_check).grid(
            row=0, column=3, sticky="ew"
        )
        ttk.Label(setup, text="Camera").grid(row=1, column=0, sticky="w")
        camera_combo = ttk.Combobox(
            setup,
            textvariable=self.camera_model,
            values=SUPPORTED_LIVE_CAMERA_MODELS,
            width=8,
            state="readonly",
        )
        camera_combo.grid(row=1, column=1, sticky="w")
        camera_combo.bind("<<ComboboxSelected>>", self._on_camera_model_changed)
        ttk.Label(setup, text="Serial").grid(row=1, column=2, sticky="e")
        ttk.Entry(setup, textvariable=self.camera_serial, width=18).grid(row=1, column=3, sticky="ew")
        ttk.Label(setup, text="Width").grid(row=2, column=0, sticky="w")
        ttk.Entry(setup, textvariable=self.camera_width, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(setup, text="Height").grid(row=2, column=2, sticky="e")
        ttk.Entry(setup, textvariable=self.camera_height, width=8).grid(row=2, column=3, sticky="w")
        ttk.Label(setup, text="FPS").grid(row=2, column=4, sticky="e")
        ttk.Entry(setup, textvariable=self.camera_fps, width=5).grid(row=2, column=5, sticky="w")
        ttk.Label(setup, text="Server").grid(row=3, column=0, sticky="w")
        ttk.Entry(setup, textvariable=self.server_host, width=16).grid(row=3, column=1, sticky="ew")
        ttk.Label(setup, text="Port").grid(row=3, column=2, sticky="e")
        ttk.Entry(setup, textvariable=self.server_port, width=7).grid(row=3, column=3, sticky="w")
        ttk.Button(setup, text="Connect", command=self.connect_remote_server).grid(row=3, column=4, sticky="ew")
        ttk.Label(setup, textvariable=self.remote_state, anchor="w").grid(row=3, column=5, sticky="w")

        capture = ttk.LabelFrame(stages, text="3. Recording / Processing", padding=6)
        capture.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(capture, textvariable=self.recording_status).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(capture, text="Required").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(capture, from_=1, to=80, textvariable=self.reference_target, width=5).grid(row=0, column=3)
        ttk.Label(capture, text="Max Selected").grid(row=0, column=4, sticky="e")
        ttk.Spinbox(capture, from_=1, to=1500, textvariable=self.max_keyframes, width=5).grid(row=0, column=5)
        ttk.Button(capture, text="Start Recording", command=self.start_recording).grid(row=0, column=6)
        ttk.Button(capture, text="Stop Recording", command=self.stop_recording).grid(row=0, column=7)
        ttk.Button(capture, text="Processing", command=self.run_recording_processing).grid(row=0, column=8)
        ttk.Button(capture, text="Reselect", command=self.run_recording_reselect).grid(row=0, column=9)
        ttk.Label(capture, textvariable=self.processing_status).grid(row=1, column=0, columnspan=6, sticky="w")
        ttk.Button(capture, text="Build Dry Run", command=lambda: self.run_build_assets(False)).grid(row=1, column=6)
        ttk.Button(capture, text="Build Assets", command=lambda: self.run_build_assets(True)).grid(row=1, column=7)
        ttk.Button(capture, text="Force Build", command=self.run_force_build_assets).grid(row=1, column=8)
        ttk.Button(capture, text="Debug", command=self.run_debug_download).grid(row=1, column=9)
        preview = ttk.Frame(capture)
        preview.grid(row=2, column=0, columnspan=10, sticky="ew", pady=(8, 0))
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        ttk.Label(preview, text="RGB").grid(row=0, column=0, sticky="w")
        ttk.Label(preview, text="SAM Mask Overlay").grid(row=0, column=1, sticky="w")
        self.capture_rgb_preview = ttk.Label(preview, text="No RGB capture", anchor="center")
        self.capture_rgb_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.capture_mask_preview = ttk.Label(preview, text="No mask overlay", anchor="center")
        self.capture_mask_preview.grid(row=1, column=1, sticky="nsew")
        ttk.Label(preview, textvariable=self.capture_preview_status, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )

        pose = ttk.LabelFrame(stages, text="4. Reference Pose", padding=6)
        pose.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(pose, text="ChArUco").grid(row=0, column=0, sticky="w")
        ttk.Label(pose, text="Squares").grid(row=0, column=1)
        ttk.Entry(pose, textvariable=self.charuco_squares_x, width=4).grid(row=0, column=2)
        ttk.Entry(pose, textvariable=self.charuco_squares_y, width=4).grid(row=0, column=3)
        ttk.Label(pose, text="Size m").grid(row=0, column=4)
        ttk.Entry(pose, textvariable=self.charuco_square_length_m, width=7).grid(row=0, column=5)
        ttk.Label(pose, text="Marker m").grid(row=0, column=6)
        ttk.Entry(pose, textvariable=self.charuco_marker_length_m, width=7).grid(row=0, column=7)
        ttk.Combobox(
            pose,
            textvariable=self.charuco_dictionary,
            values=("auto", "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000"),
            width=14,
            state="readonly",
        ).grid(row=0, column=8)
        ttk.Label(pose, text="Obj XYZ m").grid(row=1, column=0, sticky="w")
        ttk.Entry(pose, textvariable=self.charuco_object_x_m, width=7).grid(row=1, column=1)
        ttk.Entry(pose, textvariable=self.charuco_object_y_m, width=7).grid(row=1, column=2)
        ttk.Entry(pose, textvariable=self.charuco_object_z_m, width=7).grid(row=1, column=3)
        ttk.Label(pose, text="RPY deg").grid(row=1, column=4)
        ttk.Entry(pose, textvariable=self.charuco_object_roll_deg, width=7).grid(row=1, column=5)
        ttk.Entry(pose, textvariable=self.charuco_object_pitch_deg, width=7).grid(row=1, column=6)
        ttk.Entry(pose, textvariable=self.charuco_object_yaw_deg, width=7).grid(row=1, column=7)
        ttk.Button(pose, text="Detect Preview", command=self.run_charuco_detect_preview).grid(row=1, column=8)
        ttk.Button(pose, text="Board Axis Snapshot", command=self.run_charuco_axis_snapshot).grid(row=1, column=9)
        ttk.Label(pose, text="SAM").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            pose,
            textvariable=self.sam_device,
            values=("auto", "cuda", "cpu"),
            width=6,
            state="readonly",
        ).grid(row=2, column=1)
        ttk.Label(pose, text="SAM Res").grid(row=2, column=2)
        ttk.Entry(pose, textvariable=self.sam_resolution, width=7, state="readonly").grid(row=2, column=3)

        build = ttk.LabelFrame(stages, text="5. Tracking", padding=6)
        build.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(build, text="Auto Reinit", variable=self.auto_reinit).grid(row=0, column=0)
        ttk.Label(build, text="Lost frames").grid(row=0, column=1)
        ttk.Spinbox(build, from_=1, to=300, textvariable=self.auto_reinit_after, width=6).grid(row=0, column=2)
        ttk.Label(build, text="Refine iters").grid(row=0, column=3)
        ttk.Spinbox(build, from_=1, to=10, textvariable=self.refine_iterations, width=5).grid(row=0, column=4)
        ttk.Label(build, text="Track iters").grid(row=0, column=5)
        ttk.Spinbox(build, from_=1, to=10, textvariable=self.track_iterations, width=5).grid(row=0, column=6)
        ttk.Button(build, text="Track Local", command=self.run_tracking_local).grid(row=1, column=0)
        ttk.Button(build, text="Track Remote", command=self.run_tracking_remote).grid(row=1, column=1)
        ttk.Button(build, text="Reinit Tracking", command=self.reinitialize_tracking_event).grid(row=1, column=2)
        ttk.Button(build, text="Stop Command", command=self.stop_command).grid(row=1, column=3)

        logs = ttk.LabelFrame(stages, text="Status / Logs", padding=6)
        logs.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(logs, textvariable=self.status, anchor="w").grid(row=0, column=0, sticky="ew")
        self.log_text.grid(row=1, column=0, sticky="nsew")
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(1, weight=1)
        stages.rowconfigure(4, weight=1)

    def refresh(self) -> None:
        self.listbox.delete(0, tk.END)
        for profile in self.registry.list():
            marker = "*" if profile.selected else " "
            self.listbox.insert(
                tk.END,
                f"{marker} {profile.name} | refs={profile.reference_count} | assets={profile.asset_status}",
            )

    def create_profile(self) -> None:
        profile = self.registry.create(self.profile_name.get(), prompt=self.prompt.get(), exist_ok=True)
        self.registry.select(profile.name)
        self.status.set(f"Selected {profile.name}")
        self.refresh()
        self._update_capture_status()

    def select(self) -> None:
        name = self._selected_name()
        if not name:
            self.status.set("Select a profile first")
            return
        profile = self.registry.select(name)
        self.profile_name.set(profile.name)
        self.prompt.set(profile.prompt)
        self.status.set(f"Selected {profile.name}")
        self.refresh()
        self._update_capture_status()

    def delete(self) -> None:
        name = self._selected_name()
        if not name:
            self.status.set("Select a profile first")
            return
        if not messagebox.askyesno(
            "Delete profile",
            f"Delete object profile '{name}' including recordings, references, assets, logs, and processing caches?",
        ):
            return
        self.registry.delete(name, confirm=True)
        self.status.set(f"Deleted {name}")
        self.refresh()
        self._update_capture_status()

    def run_setup_check(self) -> None:
        self._start_command(
            self.command_builder.setup_check(
                foundationpose_root=self.foundationpose_root.get(),
                camera_model=self.camera_model.get(),
            )
        )

    def run_segmentation_check(self) -> None:
        if self.remote_state.get() == "Remote":
            self._start_remote_segmentation_check()
            return
        self._start_command(
            self.command_builder.segmentation_sanity(
                prompt=self.prompt.get(),
                camera_model=self.camera_model.get(),
                serial=self.camera_serial.get(),
                width=int(self.camera_width.get()),
                height=int(self.camera_height.get()),
                fps=int(self.camera_fps.get()),
            )
        )

    def connect_remote_server(self) -> None:
        host = self.server_host.get().strip()
        port = int(self.server_port.get())
        self.status.set("Checking FoundationPose v2 server")
        thread = threading.Thread(target=self._remote_health_worker, args=(host, port), daemon=True)
        thread.start()

    def _start_remote_segmentation_check(self) -> None:
        host = self.server_host.get().strip()
        port = int(self.server_port.get())
        self.status.set("Capturing one frame for remote segmentation")
        thread = threading.Thread(
            target=self._remote_segmentation_worker,
            args=(
                host,
                port,
                self.prompt.get(),
                self.camera_model.get(),
                self.camera_serial.get().strip() or None,
                int(self.camera_width.get()),
                int(self.camera_height.get()),
                int(self.camera_fps.get()),
                self.sam_device.get(),
                int(self.sam_resolution.get()),
                self._current_profile().root,
            ),
            daemon=True,
        )
        thread.start()

    def _remote_segmentation_worker(
        self,
        host: str,
        port: int,
        prompt: str,
        camera_model: str,
        serial: str | None,
        width: int,
        height: int,
        fps: int,
        sam_device: str,
        sam_resolution: int,
        profile_root: Path,
    ) -> None:
        try:
            with LiveRgbdCamera(
                model=camera_model,
                serial=serial,
                width=width,
                height=height,
                fps=fps,
            ) as camera:
                frame = None
                for _ in range(REMOTE_SEGMENTATION_WARMUP_FRAMES):
                    frame = camera.read()
                if frame is None:
                    frame = camera.read()
            payload = remote_segmentation_sanity(
                host=host,
                port=port,
                prompt=prompt,
                rgb=frame.rgb,
                depth_m=frame.depth_m,
                sam_device=sam_device,
                sam_resolution=sam_resolution,
                timeout_s=20.0,
            )
            rgb_path, overlay_path = write_segmentation_preview(
                profile_root,
                rgb=frame.rgb,
                mask_png_b64=str(payload.get("mask_png_b64") or ""),
            )
        except Exception as exc:
            self.remote_events.put(RemoteHealthEvent("Disconnected", f"Remote segmentation failed: {exc}"))
            return
        mask = payload.get("mask") if isinstance(payload.get("mask"), dict) else {}
        area = mask.get("area", 0)
        fraction = float(mask.get("area_fraction", 0.0))
        self.remote_events.put(
            RemoteHealthEvent(
                "Remote",
                f"Remote segmentation ok: area={area} ({fraction:.3%})",
                preview_rgb_path=str(rgb_path),
                preview_overlay_path=str(overlay_path),
                preview_status=f"Remote SAM mask: area={area} ({fraction:.3%})",
            )
        )

    def _remote_health_worker(self, host: str, port: int) -> None:
        try:
            payload = check_remote_health(host=host, port=port, timeout_s=2.0)
            if payload.get("ok") is True and int(payload.get("protocol_version", -1)) == 2:
                self.remote_events.put(RemoteHealthEvent("Remote", f"Connected to {host}:{port}"))
            else:
                self.remote_events.put(RemoteHealthEvent("Local", f"Unexpected server response from {host}:{port}"))
        except Exception as exc:
            self.remote_events.put(RemoteHealthEvent("Local", f"Remote unavailable: {exc}"))

    def start_recording(self) -> None:
        if self.recording_session is not None:
            self.status.set("Recording is already running")
            return
        try:
            profile = self._current_profile()
            if not self._stop_running_command_for_recording():
                return
            if not self._release_live_sessions_for_gpu():
                return
            self._recording_stop.clear()
            config = ReferenceRecordingConfig(
                camera_model=self.camera_model.get(),
                serial=self.camera_serial.get().strip() or None,
                width=int(self.camera_width.get()),
                height=int(self.camera_height.get()),
                fps=int(self.camera_fps.get()),
                board_spec=self._current_board_spec(),
                board_object=self._current_board_object(),
                sam_device=self.sam_device.get(),
                sam_resolution=int(self.sam_resolution.get()),
            )
            self.recording_session = ReferenceRecordingSession(profile, config=config)
            self.recording_session.start()
        except Exception as exc:
            self.recording_session = None
            self.status.set(str(exc))
            return
        self.recording_busy = True
        self._recording_stop.clear()
        self._recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self._recording_thread.start()
        self.status.set("Recording raw RGB-D frames")
        self._update_capture_status()

    def stop_recording(self) -> None:
        if self.recording_session is None:
            self.status.set("No recording is running")
            return
        self._recording_stop.set()
        self.status.set("Stopping recording")

    def run_recording_processing(self) -> None:
        if self.recording_session is not None:
            self.stop_recording()
            self.status.set("Stopping recording first; press Processing again after it stops")
            return
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        if self.remote_state.get() == "Remote":
            self._start_remote_processing(profile=profile, reselect=False)
            return
        self._start_command(self._charuco_command(profile, mode="process-recordings"))

    def run_recording_reselect(self) -> None:
        if self.recording_session is not None:
            self.stop_recording()
            self.status.set("Stopping recording first; press Reselect again after it stops")
            return
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        if self.remote_state.get() == "Remote":
            self._start_remote_processing(profile=profile, reselect=True)
            return
        self._start_command(self._charuco_command(profile, mode="reselect-recordings"))

    def run_debug_download(self) -> None:
        if self.recording_session is not None:
            self.stop_recording()
            self.status.set("Stopping recording first; press Debug again after it stops")
            return
        profile = self._current_profile()
        if self.remote_state.get() != "Remote":
            self.status.set("Debug download requires a connected Remote server")
            return
        self.status.set("Downloading remote Processing debug artifacts")
        thread = threading.Thread(
            target=self._remote_debug_download_worker,
            args=(self.server_host.get().strip(), int(self.server_port.get()), profile.name, str(profile.root)),
            daemon=True,
        )
        thread.start()

    def run_charuco_detect_preview(self) -> None:
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        self._start_command(self._charuco_command(profile, mode="detect-only"))

    def run_charuco_axis_snapshot(self) -> None:
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        preview_path = profile.refs_dir / "preview" / "charuco_board_axes.png"
        self._pending_axis_preview_path = preview_path
        self._start_command(
            self._charuco_command(
                profile,
                mode="detect-only",
                capture_once=True,
                preview_output=str(preview_path),
            )
        )

    def run_build_assets(self, execute: bool) -> None:
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        if self.remote_state.get() == "Remote":
            self._start_remote_build(profile_name=profile.name, execute=execute)
            return
        report = latest_processing_report(profile)
        if not report:
            self.status.set("Run Processing before Build Assets")
            return
        if report.get("readiness") != "ready":
            self.status.set("Need more recording; use Force Build to override")
            return
        self._start_command(
            self.command_builder.build_assets(
                object_name=profile.name,
                foundationpose_root=self.foundationpose_root.get(),
                execute=execute,
                data_root=self.config.data_root,
            )
        )

    def run_force_build_assets(self) -> None:
        profile = self._current_profile()
        report = latest_processing_report(profile)
        reason = "No processing report found"
        if report:
            reason = (
                f"readiness={report.get('readiness')} accepted={report.get('accepted')}/"
                f"{report.get('required_keyframes')}"
            )
        if not messagebox.askyesno("Force Build", f"Build assets despite quality warning?\n\n{reason}"):
            return
        if not self._release_live_sessions_for_gpu():
            return
        if self.remote_state.get() == "Remote":
            self._start_remote_build(profile_name=profile.name, execute=True)
            return
        self._start_command(
            self.command_builder.build_assets(
                object_name=profile.name,
                foundationpose_root=self.foundationpose_root.get(),
                execute=True,
                data_root=self.config.data_root,
            )
        )

    def run_tracking(self) -> None:
        self.run_tracking_local()

    def run_tracking_local(self) -> None:
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        if find_generated_mesh(profile) is None:
            if not self._download_remote_model_for_local_tracking(profile):
                return
        self._last_tracking_mode = "local"
        self._start_command(
            self.command_builder.track_live(
                object_name=profile.name,
                prompt=self.prompt.get(),
                foundationpose_root=self.foundationpose_root.get(),
                auto_reinit=bool(self.auto_reinit.get()),
                auto_reinit_after_lost_frames=int(self.auto_reinit_after.get()),
                camera_model=self.camera_model.get(),
                serial=self.camera_serial.get(),
                width=int(self.camera_width.get()),
                height=int(self.camera_height.get()),
                fps=int(self.camera_fps.get()),
                refine_iterations=int(self.refine_iterations.get()),
                track_iterations=int(self.track_iterations.get()),
                data_root=self.config.data_root,
            )
        )

    def run_tracking_remote(self) -> None:
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
        self._last_tracking_mode = "remote"
        self._start_command(
            self.command_builder.track_remote_live(
                server_host=self.server_host.get().strip(),
                server_port=int(self.server_port.get()),
                object_name=profile.name,
                prompt=self.prompt.get(),
                foundationpose_root=self.foundationpose_root.get(),
                auto_reinit=bool(self.auto_reinit.get()),
                auto_reinit_after_lost_frames=int(self.auto_reinit_after.get()),
                camera_model=self.camera_model.get(),
                serial=self.camera_serial.get(),
                width=int(self.camera_width.get()),
                height=int(self.camera_height.get()),
                fps=int(self.camera_fps.get()),
                refine_iterations=int(self.refine_iterations.get()),
                track_iterations=int(self.track_iterations.get()),
                data_root=self.config.data_root,
            ),
            remote=True,
        )

    def _download_remote_model_for_local_tracking(self, profile) -> bool:
        host = self.server_host.get().strip()
        port = int(self.server_port.get())
        self.status.set(f"Downloading remote model.obj for {profile.name}")
        try:
            target_path = profile_model_path(profile)
            metadata = remote_download_model_asset(
                host=host,
                port=port,
                profile=profile.name,
                target_path=target_path,
                timeout_s=REMOTE_MODEL_DOWNLOAD_TIMEOUT_S,
            )
            record_asset_ready(
                profile,
                generated_assets=[target_path],
                deterministic_validation_report={
                    "ok": True,
                    "source": "remote_model_download",
                    "remote": metadata,
                },
            )
        except Exception as exc:
            message = (
                f"Track Local skipped: profile {profile.name} has no local model.obj and remote download failed: {exc}. "
                "Use Track Remote or run local Build Assets first."
            )
            self.status.set(message)
            self._append_log(message)
            return False
        message = f"Downloaded remote model.obj for local tracking: {metadata.get('bytes', 0)} bytes"
        self.status.set(message)
        self._append_log(message)
        return True

    def _start_remote_processing(self, *, profile, reselect: bool) -> None:
        host = self.server_host.get().strip()
        port = int(self.server_port.get())
        options = self._remote_processing_options(profile_name=profile.name, reselect=reselect)
        self.status.set("Uploading recordings for remote processing")
        thread = threading.Thread(
            target=self._remote_processing_worker,
            args=(host, port, profile.name, str(profile.root), options),
            daemon=True,
        )
        thread.start()

    def _remote_processing_worker(
        self,
        host: str,
        port: int,
        profile_name: str,
        profile_root: str,
        options: dict,
    ) -> None:
        try:
            archive, upload = create_recordings_archive(profile_root, request_payload=options)
            payload = remote_process_recordings(
                host=host,
                port=port,
                profile=profile_name,
                archive=archive,
                timeout_s=REMOTE_PROCESS_UPLOAD_TIMEOUT_S,
                poll_interval_s=REMOTE_PROCESS_POLL_INTERVAL_S,
                max_wait_s=REMOTE_PROCESS_TIMEOUT_S,
            )
        except Exception as exc:
            self.remote_events.put(RemoteHealthEvent("Disconnected", f"Remote processing failed: {exc}"))
            return
        state = str(payload.get("state") or payload.get("status") or "unknown")
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if state == "succeeded":
            readiness = result.get("readiness", "unknown")
            accepted = result.get("accepted", 0)
            required = result.get("required_keyframes", options.get("required_keyframes", "?"))
            sessions = upload.get("session_count", 0)
            rejection_summary = summarize_processing_rejections(result) if int(accepted) <= 0 else ""
            details = f"; {rejection_summary}" if rejection_summary else ""
            self.remote_events.put(
                RemoteHealthEvent(
                    "Remote",
                    f"Remote processing {readiness}: accepted={accepted}/{required}, uploaded_sessions={sessions}{details}",
                )
            )
        else:
            reason = payload.get("error") or payload.get("reason") or payload.get("stderr_tail") or state
            self.remote_events.put(RemoteHealthEvent("Remote", f"Remote processing {state}: {reason}"))

    def _remote_debug_download_worker(
        self,
        host: str,
        port: int,
        profile_name: str,
        profile_root: str,
    ) -> None:
        try:
            metadata = remote_download_debug_artifacts(
                host=host,
                port=port,
                profile=profile_name,
                profile_root=profile_root,
                timeout_s=REMOTE_DEBUG_DOWNLOAD_TIMEOUT_S,
            )
        except Exception as exc:
            self.remote_events.put(RemoteHealthEvent("Remote", f"Remote debug download failed: {exc}"))
            return
        self.remote_events.put(
            RemoteHealthEvent(
                "Remote",
                f"Remote debug artifacts downloaded: {metadata['file_count']} file(s) -> {metadata['output_dir']}",
            )
        )

    def _start_remote_build(self, *, profile_name: str, execute: bool) -> None:
        host = self.server_host.get().strip()
        port = int(self.server_port.get())
        foundationpose_root = self.foundationpose_root.get()
        self.status.set("Requesting remote asset build")
        thread = threading.Thread(
            target=self._remote_build_worker,
            args=(host, port, profile_name, foundationpose_root, bool(execute)),
            daemon=True,
        )
        thread.start()

    def _remote_build_worker(
        self,
        host: str,
        port: int,
        profile_name: str,
        foundationpose_root: str,
        execute: bool,
    ) -> None:
        try:
            payload = remote_build_assets(
                host=host,
                port=port,
                profile=profile_name,
                foundationpose_root=foundationpose_root,
                execute=execute,
                timeout_s=10.0,
                poll_interval_s=REMOTE_BUILD_POLL_INTERVAL_S,
                max_wait_s=REMOTE_BUILD_TIMEOUT_S,
            )
        except Exception as exc:
            self.remote_events.put(RemoteHealthEvent("Disconnected", f"Remote build failed: {exc}"))
            return
        state = str(payload.get("state") or payload.get("status") or "unknown")
        if payload.get("ok") is True:
            self.remote_events.put(RemoteHealthEvent("Remote", f"Remote build {state}: {profile_name}"))
        else:
            reason = payload.get("error") or payload.get("reason") or payload.get("stderr_tail") or state
            self.remote_events.put(RemoteHealthEvent("Remote", f"Remote build {state}: {reason}"))

    def reinitialize_tracking_event(self, event=None) -> None:
        restart_tracking = self.run_tracking_remote if self._last_tracking_mode == "remote" else self.run_tracking_local
        if self.runner.running:
            self.runner.stop()
            self.root.after(700, restart_tracking)
            self.status.set(f"Restarting {self._last_tracking_mode} tracking for reinitialization")
        else:
            restart_tracking()

    def stop_command(self) -> None:
        self.runner.stop()

    def _on_camera_model_changed(self, event=None) -> None:
        width, height = _default_camera_resolution(self.camera_model.get())
        self.camera_width.set(width)
        self.camera_height.set(height)
        self.status.set(f"Camera {self.camera_model.get()} default resolution: {width}x{height}")

    def _start_command(self, command: list[str], *, remote: bool = False) -> None:
        try:
            self._active_remote_command = bool(remote)
            self.runner.start(command)
        except Exception as exc:
            self._active_remote_command = False
            self.status.set(str(exc))

    def _charuco_command(
        self,
        profile,
        *,
        mode: str,
        capture_once: bool = False,
        preview_output: str | None = None,
    ) -> list[str]:
        return self.command_builder.charuco_reference(
            mode=mode,
            object_name=profile.name,
            prompt=self.prompt.get(),
            camera_model=self.camera_model.get(),
            serial=self.camera_serial.get(),
            width=int(self.camera_width.get()),
            height=int(self.camera_height.get()),
            fps=int(self.camera_fps.get()),
            frames=int(self.reference_target.get()),
            squares_x=self.charuco_squares_x.get(),
            squares_y=self.charuco_squares_y.get(),
            square_length_m=self.charuco_square_length_m.get(),
            marker_length_m=self.charuco_marker_length_m.get(),
            dictionary=self.charuco_dictionary.get(),
            object_xyz_m=(
                self.charuco_object_x_m.get(),
                self.charuco_object_y_m.get(),
                self.charuco_object_z_m.get(),
            ),
            object_rpy_deg=(
                self.charuco_object_roll_deg.get(),
                self.charuco_object_pitch_deg.get(),
                self.charuco_object_yaw_deg.get(),
            ),
            capture_once=capture_once,
            preview_output=preview_output,
            sam_device=self.sam_device.get(),
            sam_resolution=str(self.sam_resolution.get()),
            required_keyframes=str(self.reference_target.get()),
            max_keyframes=str(self.max_keyframes.get()),
            data_root=self.config.data_root,
        )

    def _recording_worker(self) -> None:
        session = self.recording_session
        if session is None:
            return
        preview_window_name = f"{RECORDING_PREVIEW_WINDOW} - {session.session_id}"
        cv2 = None
        try:
            import cv2 as cv2_module  # type: ignore

            cv2 = cv2_module
        except Exception:
            cv2 = None
        preview_stop_grace_until_s = time.monotonic() + 0.5
        try:
            while not self._recording_stop.is_set():
                record = session.record_next_frame()
                self.recording_events.put(
                    RecordingEvent(
                        "frame",
                        f"recorded frame {record.index:06d}",
                        session_id=session.session_id,
                        frame_count=session.frame_count,
                    )
                )
                if cv2 is not None:
                    rgb_path = session.session_dir / record.rgb_path
                    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    if image is not None:
                        _draw_recording_preview_status(
                            image,
                            session_id=session.session_id,
                            frame_count=session.frame_count,
                        )
                        cv2.imshow(preview_window_name, image)
                        key = cv2.waitKey(1) & 0xFF
                        should_stop = _recording_preview_should_stop(
                            key=key,
                            window_visible=_opencv_window_visible(cv2, preview_window_name),
                            now_s=time.monotonic(),
                            grace_until_s=preview_stop_grace_until_s,
                        )
                        if should_stop:
                            self._recording_stop.set()
        except Exception as exc:
            self.recording_events.put(exc)
        finally:
            try:
                info = session.stop()
                self.recording_events.put(
                    RecordingEvent(
                        "done",
                        f"recording stopped: {info.frame_count} raw frame(s)",
                        session_id=info.session_id,
                        frame_count=info.frame_count,
                    )
                )
            finally:
                if cv2 is not None:
                    try:
                        cv2.destroyWindow(preview_window_name)
                    except Exception:
                        pass

    def _poll_queues(self) -> None:
        while True:
            try:
                event = self.command_events.get_nowait()
            except queue.Empty:
                break
            self._handle_command_event(event)
        while True:
            try:
                event = self.remote_events.get_nowait()
            except queue.Empty:
                break
            self._handle_remote_event(event)
        while True:
            try:
                event = self.recording_events.get_nowait()
            except queue.Empty:
                break
            self._handle_recording_event(event)
        self.root.after(100, self._poll_queues)

    def _handle_command_event(self, event: CommandEvent) -> None:
        self._append_log(event.text)
        if event.kind == "start":
            self.status.set("Command running")
        elif event.kind == "done":
            remote_failed = False
            if self._active_remote_command and event.text != "returncode=0":
                self.remote_state.set("Disconnected")
                self.status.set("Remote command failed")
                remote_failed = True
            self._active_remote_command = False
            if self.recording_session is None and not remote_failed:
                self.status.set(event.text)
            if self._pending_axis_preview_path is not None:
                self._show_axis_preview(self._pending_axis_preview_path)
                self._pending_axis_preview_path = None
            self.refresh()
            self._update_capture_status()
            self._update_processing_status()
        elif event.kind == "stop":
            self.status.set(event.text)

    def _handle_remote_event(self, event: RemoteHealthEvent) -> None:
        self.remote_state.set(event.state)
        self.status.set(event.text)
        self._append_log(event.text)
        if event.preview_rgb_path and event.preview_overlay_path:
            self._show_remote_segmentation_preview(
                Path(event.preview_rgb_path),
                Path(event.preview_overlay_path),
                status=event.preview_status or event.text,
            )

    def _handle_recording_event(self, event: RecordingEvent | Exception) -> None:
        if isinstance(event, Exception):
            self.recording_busy = False
            self.recording_session = None
            self._append_log(f"recording failed: {event}")
            self.status.set(str(event))
            self._update_capture_status()
            return
        self._append_log(event.text)
        if event.kind == "frame":
            self.recording_status.set(
                f"Recording: {event.frame_count} raw frame(s) in {event.session_id or 'session'}"
            )
        elif event.kind == "done":
            self.recording_busy = False
            self.recording_session = None
            self._recording_thread = None
            self.status.set(event.text)
            self._update_capture_status()

    def _release_live_sessions_for_gpu(self) -> bool:
        if self.recording_session is not None:
            self._recording_stop.set()
            thread = self._recording_thread
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=6.0)
                if thread.is_alive():
                    self.status.set("Recording is still stopping; try again after the preview closes")
                    return False
            self.recording_session = None
            self.recording_busy = False
        self._update_capture_status()
        return True

    def _stop_running_command_for_recording(self) -> bool:
        if not self.runner.running:
            return True
        self.status.set("Stopping running command before recording")
        if self.runner.stop_and_wait(timeout_s=6.0):
            return True
        self.status.set("Running command is still stopping; try Start Recording again")
        return False

    def _current_board_spec(self) -> CharucoBoardSpec:
        return CharucoBoardSpec(
            squares_x=int(self.charuco_squares_x.get()),
            squares_y=int(self.charuco_squares_y.get()),
            square_length_m=float(self.charuco_square_length_m.get()),
            marker_length_m=float(self.charuco_marker_length_m.get()),
            dictionary=self.charuco_dictionary.get(),
        )

    def _current_board_object(self) -> BoardObjectTransform:
        return BoardObjectTransform.from_xyz_rpy_deg(
            (
                float(self.charuco_object_x_m.get()),
                float(self.charuco_object_y_m.get()),
                float(self.charuco_object_z_m.get()),
            ),
            (
                float(self.charuco_object_roll_deg.get()),
                float(self.charuco_object_pitch_deg.get()),
                float(self.charuco_object_yaw_deg.get()),
            ),
        )

    def _remote_processing_options(self, *, profile_name: str, reselect: bool) -> dict:
        return {
            "profile": profile_name,
            "prompt": self.prompt.get(),
            "reselect": bool(reselect),
            "board_spec": self._current_board_spec().to_dict(),
            "board_object": self._current_board_object().to_dict(),
            "charuco_detector_preset": CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
            "sam_device": self.sam_device.get(),
            "sam_resolution": int(self.sam_resolution.get()),
            "sam_threshold": 0.3,
            "required_keyframes": int(self.reference_target.get()),
            "max_keyframes": int(self.max_keyframes.get()),
            "min_mask_area_fraction": 0.0005,
            "min_valid_depth_ratio": 0.05,
        }

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _show_axis_preview(self, path: Path) -> None:
        if not path.exists():
            self.capture_preview_status.set("Board axis snapshot was not saved")
            return
        try:
            preview_path = _write_single_image_preview(path, max_size=(900, 650))
            axis_image = tk.PhotoImage(file=str(preview_path))
        except Exception as exc:
            self.capture_preview_status.set(f"Board axis preview failed: {exc}")
            return
        self._capture_preview_images = (axis_image,)
        self.capture_rgb_preview.configure(image=axis_image, text="")
        self.capture_mask_preview.configure(image="", text="No mask overlay")
        self.capture_preview_status.set(f"Board axes: {path.name}")

    def _show_remote_segmentation_preview(self, rgb_path: Path, overlay_path: Path, *, status: str) -> None:
        if not rgb_path.exists() or not overlay_path.exists():
            self.capture_preview_status.set("Remote segmentation preview was not saved")
            return
        try:
            rgb_image = tk.PhotoImage(file=str(rgb_path))
            overlay_image = tk.PhotoImage(file=str(overlay_path))
        except Exception as exc:
            self.capture_preview_status.set(f"Remote segmentation preview failed: {exc}")
            return
        self._capture_preview_images = (rgb_image, overlay_image)
        self.capture_rgb_preview.configure(image=rgb_image, text="")
        self.capture_mask_preview.configure(image=overlay_image, text="")
        self.capture_preview_status.set(status)

    def _current_profile(self):
        name = self.profile_name.get().strip() or self._selected_name()
        if not name:
            raise RuntimeError("Select or create a profile first")
        return self.registry.get(name)

    def _selected_name(self) -> str | None:
        selection = self.listbox.curselection()
        if not selection:
            return None
        line = self.listbox.get(selection[0])
        return line.split("|", 1)[0].strip().lstrip("*").strip()

    def _update_capture_status(self) -> None:
        try:
            profile = self._current_profile()
            count = count_reference_frames(profile)
            raw_count = sum(session.frame_count for session in list_recording_sessions(profile))
        except (FileNotFoundError, RuntimeError, ValueError):
            count = 0
            raw_count = 0
        self.capture_status.set(f"Capture: {count} / {int(self.reference_target.get())}")
        if self.recording_session is None:
            self.recording_status.set(f"Recording: {raw_count} raw frame(s) saved")
        self._update_processing_status()

    def _update_processing_status(self) -> None:
        try:
            profile = self._current_profile()
            report = latest_processing_report(profile)
        except (FileNotFoundError, RuntimeError, ValueError):
            report = None
        if not report:
            self.processing_status.set("Processing: no report")
            return
        readiness = report.get("readiness", "unknown")
        accepted = report.get("accepted", 0)
        rejected = report.get("rejected", 0)
        required = report.get("required_keyframes", self.reference_target.get())
        self.processing_status.set(
            f"Processing: {readiness} | accepted={accepted}/{required} rejected={rejected}"
        )

    def run(self) -> None:
        self.root.mainloop()


FoundationPoseProfileGui = FoundationPoseWorkflowGui


def resolve_gui_config(config: GuiConfig | None = None) -> GuiConfig:
    input_config = config or GuiConfig()
    return GuiConfig(
        data_root=str(data_root(input_config.data_root)),
        foundationpose_root=_resolve_foundationpose_root(input_config.foundationpose_root),
        python_executable=input_config.python_executable,
        title=input_config.title,
    )


def _resolve_foundationpose_root(value: str | None = None) -> str:
    raw_candidates = [candidate for candidate in (value, os.environ.get("FOUNDATIONPOSE_ROOT"), "/home/kgs/FoundationPose") if candidate]
    for raw_candidate in raw_candidates:
        candidate = Path(raw_candidate).expanduser()
        for root_candidate in (candidate, candidate / "FoundationPose"):
            if _looks_like_foundationpose_root(root_candidate):
                return str(root_candidate.resolve())
    if value:
        return str(Path(value).expanduser())
    return "/home/kgs/FoundationPose"


def _looks_like_foundationpose_root(path: Path) -> bool:
    return (
        (path / "bundlesdf" / "run_nerf.py").exists()
        and (path / "estimater.py").exists()
        and (path / "learning").is_dir()
    )


def _default_camera_resolution(camera_model: str) -> tuple[int, int]:
    try:
        return default_camera_resolution(camera_model)
    except ValueError:
        return DEFAULT_CAMERA_RESOLUTIONS["d405"]


def check_remote_health(*, host: str, port: int, timeout_s: float = 2.0) -> dict:
    url = f"http://{host}:{int(port)}/foundationpose/v2/health"
    try:
        with urllib_request.urlopen(url, timeout=float(timeout_s)) as response:
            payload = response.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("health response must be a JSON object")
    return decoded


def remote_build_assets(
    *,
    host: str,
    port: int,
    profile: str,
    foundationpose_root: str,
    execute: bool,
    timeout_s: float = 10.0,
    poll_interval_s: float = REMOTE_BUILD_POLL_INTERVAL_S,
    max_wait_s: float = REMOTE_BUILD_TIMEOUT_S,
) -> dict:
    base_url = f"http://{host}:{int(port)}/foundationpose/v2"
    request_id = f"gui-build-{time.monotonic_ns()}"
    payload = {
        "request_id": request_id,
        "profile": profile,
        "foundationpose_root": foundationpose_root,
        "execute": bool(execute),
    }
    initial = _post_json(f"{base_url}/assets/build", payload, timeout_s=timeout_s)
    job_id = initial.get("job_id")
    if not execute or not job_id:
        return initial
    deadline = time.monotonic() + float(max_wait_s)
    while True:
        status = _get_json(f"{base_url}/assets/build/{job_id}", timeout_s=timeout_s)
        if status.get("state") in {"succeeded", "failed"}:
            return status
        if time.monotonic() >= deadline:
            raise RuntimeError(f"remote build timed out: job_id={job_id}")
        time.sleep(float(poll_interval_s))


def remote_download_model_asset(
    *,
    host: str,
    port: int,
    profile: str,
    target_path: str | Path,
    timeout_s: float = REMOTE_MODEL_DOWNLOAD_TIMEOUT_S,
) -> dict:
    encoded_profile = urllib_parse.quote(profile, safe="")
    url = f"http://{host}:{int(port)}/foundationpose/v2/assets/model/{encoded_profile}"
    try:
        with urllib_request.urlopen(url, timeout=float(timeout_s)) as response:
            data = response.read()
            headers = response.headers
    except urllib_error.HTTPError as exc:
        detail = exc.reason
        try:
            decoded = json.loads(exc.read().decode("utf-8"))
            if isinstance(decoded, dict):
                detail = decoded.get("reason") or decoded.get("status") or detail
        except Exception:
            pass
        raise RuntimeError(f"remote model download failed: HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    if not data:
        raise RuntimeError("remote model download returned an empty response")
    target = Path(target_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)
    return {
        "profile": profile,
        "path": str(target),
        "bytes": len(data),
        "sha256": headers.get("X-FoundationPose-Mesh-Sha256"),
        "remote_size": headers.get("X-FoundationPose-Mesh-Size"),
    }


def remote_download_debug_artifacts(
    *,
    host: str,
    port: int,
    profile: str,
    profile_root: str | Path,
    timeout_s: float = REMOTE_DEBUG_DOWNLOAD_TIMEOUT_S,
) -> dict:
    encoded_profile = urllib_parse.quote(profile, safe="")
    url = f"http://{host}:{int(port)}/foundationpose/v2/debug/{encoded_profile}"
    try:
        with urllib_request.urlopen(url, timeout=float(timeout_s)) as response:
            data = response.read()
    except urllib_error.HTTPError as exc:
        detail = exc.reason
        try:
            decoded = json.loads(exc.read().decode("utf-8"))
            if isinstance(decoded, dict):
                detail = decoded.get("reason") or decoded.get("status") or detail
        except Exception:
            pass
        raise RuntimeError(f"remote debug artifact download failed: HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    if not data:
        raise RuntimeError("remote debug artifact download returned an empty response")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = Path(profile_root).expanduser() / "debug_downloads"
    output_dir = output_root / f"debug-{timestamp}-{time.monotonic_ns()}"
    file_count = _extract_zip_safely(data, output_dir)
    return {
        "profile": profile,
        "output_dir": str(output_dir),
        "file_count": file_count,
        "bytes": len(data),
    }


def _extract_zip_safely(data: bytes, output_dir: Path) -> int:
    output_dir = output_dir.expanduser()
    tmp_dir = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    file_count = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                parts = Path(member.filename).parts
                if Path(member.filename).is_absolute() or any(part in {"", ".", ".."} for part in parts):
                    raise RuntimeError(f"unsafe debug artifact member: {member.filename}")
                target = tmp_dir.joinpath(*parts)
                if not _is_relative_to(target.resolve(), tmp_dir.resolve()):
                    raise RuntimeError(f"unsafe debug artifact member: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, target.open("wb") as handle:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                file_count += 1
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(tmp_dir, output_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return file_count


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def create_recordings_archive(profile_root: str | Path, *, request_payload: dict) -> tuple[bytes, dict]:
    profile_root = Path(profile_root).expanduser()
    recordings_root = profile_root / "recordings"
    session_dirs = []
    if recordings_root.exists():
        session_dirs = [
            path
            for path in sorted(recordings_root.iterdir())
            if path.is_dir() and (path / "session.json").exists()
        ]
    if not session_dirs:
        raise RuntimeError("No recording sessions found; run Start Recording first")
    max_upload_frames = _remote_processing_upload_frame_limit(request_payload)
    records = []
    for session_dir in session_dirs:
        for record in _load_recording_frame_records(session_dir):
            records.append((session_dir, record))
    if not records:
        raise RuntimeError("No recorded frames found; run Start Recording first")
    selected_records = _sample_evenly(records, max_upload_frames)
    selected_by_session: dict[Path, list[dict]] = {session_dir: [] for session_dir in session_dirs}
    for session_dir, record in selected_records:
        selected_by_session.setdefault(session_dir, []).append(record)
    buffer = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(PROCESSING_REQUEST_JSON, json.dumps(request_payload, sort_keys=True).encode("utf-8"))
        for session_dir in session_dirs:
            selected = selected_by_session.get(session_dir, [])
            if not selected:
                continue
            session_rel = session_dir.relative_to(profile_root).as_posix()
            metadata_path = session_dir / "session.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                metadata = {}
            metadata["frame_count"] = len(selected)
            zf.writestr(f"{session_rel}/session.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"))
            zf.writestr(
                f"{session_rel}/frames.jsonl",
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected).encode("utf-8"),
            )
            file_count += 2
            for record in selected:
                for key in ("rgb_path", "depth_path"):
                    rel = record.get(key)
                    if not rel:
                        continue
                    path = session_dir / str(rel)
                    if not path.is_file():
                        raise RuntimeError(f"Recorded frame file is missing: {path}")
                    archive_name = f"{session_rel}/{Path(str(rel)).as_posix()}"
                    if archive_name in zf.namelist():
                        continue
                    zf.write(path, archive_name)
                    file_count += 1
    archive = buffer.getvalue()
    return archive, {
        "session_count": len([session for session, items in selected_by_session.items() if items]),
        "session_ids": [path.name for path, items in selected_by_session.items() if items],
        "file_count": file_count,
        "source_frame_count": len(records),
        "frame_count": len(selected_records),
        "sampled": len(selected_records) < len(records),
        "max_upload_frames": max_upload_frames,
        "archive_bytes": len(archive),
    }


def summarize_processing_rejections(result: dict, *, limit: int = 3, max_reason_chars: int = 90) -> str:
    counts: dict[str, int] = {}
    for record in result.get("records", []):
        if not isinstance(record, dict) or record.get("accepted"):
            continue
        reasons = record.get("reasons") or record.get("reason") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        if not reasons:
            reasons = ["unknown rejection reason"]
        for reason in reasons:
            text = " ".join(str(reason).split())
            if not text:
                continue
            if len(text) > max_reason_chars:
                text = text[: max_reason_chars - 3] + "..."
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(int(limit), 1)]
    return "top rejects: " + "; ".join(f"{count}x {reason}" for reason, count in top)


def _remote_processing_upload_frame_limit(request_payload: dict) -> int:
    value = request_payload.get("max_upload_frames")
    if value is not None:
        return max(1, int(value))
    required = int(request_payload.get("required_keyframes", 16))
    requested_max = int(request_payload.get("max_keyframes", 32))
    target = max(required * 8, requested_max * 6)
    return max(required, min(REMOTE_PROCESS_UPLOAD_FRAME_CAP, target))


def _load_recording_frame_records(session_dir: Path) -> list[dict]:
    frames_path = session_dir / "frames.jsonl"
    if not frames_path.exists():
        return []
    records: list[dict] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid recorded frame record in {frames_path}")
        records.append(record)
    return records


def _sample_evenly(items: list, limit: int) -> list:
    limit = int(limit)
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    indexes = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[index] for index in indexes]


def remote_process_recordings(
    *,
    host: str,
    port: int,
    profile: str,
    archive: bytes,
    timeout_s: float = 10.0,
    poll_interval_s: float = REMOTE_PROCESS_POLL_INTERVAL_S,
    max_wait_s: float = REMOTE_PROCESS_TIMEOUT_S,
) -> dict:
    base_url = f"http://{host}:{int(port)}/foundationpose/v2"
    request_id = f"gui-process-{time.monotonic_ns()}"
    initial = _post_bytes(
        f"{base_url}/recordings/process",
        archive,
        timeout_s=timeout_s,
        headers={
            "Content-Type": RECORDINGS_ZIP_CONTENT_TYPE,
            "X-FoundationPose-Request-Id": request_id,
            "X-FoundationPose-Profile": profile,
        },
    )
    job_id = initial.get("job_id")
    if not job_id:
        return initial
    deadline = time.monotonic() + float(max_wait_s)
    while True:
        status = _get_json(f"{base_url}/recordings/process/{job_id}", timeout_s=timeout_s)
        if status.get("state") in {"succeeded", "failed"}:
            return status
        if time.monotonic() >= deadline:
            raise RuntimeError(f"remote processing timed out: job_id={job_id}")
        time.sleep(float(poll_interval_s))


def remote_segmentation_sanity(
    *,
    host: str,
    port: int,
    prompt: str,
    rgb,
    depth_m,
    sam_device: str,
    sam_resolution: int,
    timeout_s: float = 20.0,
) -> dict:
    base_url = f"http://{host}:{int(port)}/foundationpose/v2"
    request_id = f"gui-segmentation-{time.monotonic_ns()}"
    body = encode_foundationpose_segmentation_request(
        rgb=rgb,
        depth_m=depth_m,
        request_id=request_id,
        capture_monotonic_ns=time.monotonic_ns(),
        prompt=prompt,
        mask_options={
            "device": sam_device,
            "threshold": 0.3,
            "resolution": int(sam_resolution),
        },
    )
    return _post_bytes(
        f"{base_url}/segmentation",
        body,
        timeout_s=timeout_s,
        headers={"Content-Type": REQUEST_CONTENT_TYPE},
    )


def write_segmentation_preview(profile_root: str | Path, *, rgb, mask_png_b64: str) -> tuple[Path, Path]:
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as exc:
        raise RuntimeError("OpenCV and NumPy are required for remote segmentation preview.") from exc
    if not mask_png_b64:
        raise RuntimeError("remote segmentation response did not include a mask preview")
    mask_bytes = base64.b64decode(mask_png_b64.encode("ascii"))
    mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("failed to decode remote segmentation mask")
    rgb_array = np.asarray(rgb, dtype=np.uint8)
    if mask.shape != rgb_array.shape[:2]:
        raise RuntimeError(f"mask shape {mask.shape} does not match RGB shape {rgb_array.shape[:2]}")
    mask_bool = mask > 0
    overlay_rgb = rgb_array.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)
    overlay_rgb[mask_bool] = (0.55 * overlay_rgb[mask_bool] + 0.45 * color).astype(np.uint8)
    out_dir = Path(profile_root).expanduser() / "logs" / "remote_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"segmentation_{time.monotonic_ns()}"
    rgb_path = out_dir / f"{stem}_rgb.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_path), _resize_preview(rgb_bgr, max_size=(440, 330)))
    cv2.imwrite(str(overlay_path), _resize_preview(overlay_bgr, max_size=(440, 330)))
    return rgb_path, overlay_path


def _post_json(url: str, payload: dict, *, timeout_s: float) -> dict:
    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_s)) as response:
            raw = response.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("remote build response must be a JSON object")
    return decoded


def _post_bytes(url: str, body: bytes, *, timeout_s: float, headers: dict[str, str]) -> dict:
    request = urllib_request.Request(url, data=body, headers=dict(headers))
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_s)) as response:
            raw = response.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("remote response must be a JSON object")
    return decoded


def _get_json(url: str, *, timeout_s: float) -> dict:
    try:
        with urllib_request.urlopen(url, timeout=float(timeout_s)) as response:
            raw = response.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("remote status response must be a JSON object")
    return decoded


def _draw_recording_preview_status(image_bgr, *, session_id: str, frame_count: int) -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        return
    text = f"Recording {frame_count} raw frame(s) | {session_id} | Q/Esc to stop preview"
    cv2.putText(image_bgr, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def _opencv_window_visible(cv2, window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False


def _recording_preview_should_stop(*, key: int, window_visible: bool, now_s: float, grace_until_s: float) -> bool:
    stop_requested = key in {ord("q"), 27}
    window_closed = not window_visible
    if now_s < grace_until_s and (stop_requested or window_closed):
        return False
    return stop_requested or window_closed


def _write_single_image_preview(
    image_path: Path,
    *,
    max_size: tuple[int, int],
) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local GUI environment
        raise RuntimeError("OpenCV is required for preview rendering.") from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    preview = _resize_preview(image, max_size=max_size)
    preview_path = image_path.with_name(f"{image_path.stem}_gui_preview.png")
    cv2.imwrite(str(preview_path), preview)
    return preview_path


def _resize_preview(image, *, max_size: tuple[int, int]):
    max_width, max_height = max_size
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - already checked by caller
        raise RuntimeError("OpenCV is required for capture preview resizing.") from exc
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--foundationpose-root")
    args = parser.parse_args()
    FoundationPoseWorkflowGui(
        GuiConfig(data_root=args.data_root, foundationpose_root=args.foundationpose_root)
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
