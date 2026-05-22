"""Tkinter workflow GUI for FoundationPose model-free onboarding."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from visual_servoing.common.paths import data_root

from .charuco_reference import BoardObjectTransform, CharucoBoardSpec
from .reference_dataset import count_reference_frames
from .reference_processing import latest_processing_report
from .reference_recording import ReferenceRecordingConfig, ReferenceRecordingSession, list_recording_sessions
from .registry import ObjectProfileRegistry


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


DEFAULT_CAMERA_RESOLUTIONS = {
    "d405": (640, 480),
    "d435": (640, 480),
}
RECORDING_PREVIEW_WINDOW = "Reference Recording Preview"


class GuiCommandBuilder:
    def __init__(self, *, config: GuiConfig) -> None:
        self.config = config
        self.cwd = Path(__file__).resolve().parents[2]

    def module(self, module: str, *args: str) -> list[str]:
        return [self.config.python_executable, "-m", module, *args]

    def setup_check(self, *, foundationpose_root: str) -> list[str]:
        command = self.module(
            "visual_servoing.scripts.fp_setup_check",
            "--foundationpose-path",
            foundationpose_root,
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
            _live_camera_flag(camera_model),
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(fps),
            "--print-timing",
        )
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
        data_root: str | None = None,
    ) -> list[str]:
        if mode not in {"offline-generate", "detect-only", "live-capture", "record", "process-recordings"}:
            raise ValueError("mode must be offline-generate, detect-only, live-capture, record, or process-recordings")
        command = self.module(
            "visual_servoing.scripts.fp_charuco_reference",
            f"--{mode}",
            "--object",
            object_name,
            "--prompt",
            prompt,
            "--camera",
            camera_model,
            "--width",
            str(width),
            "--height",
            str(height),
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
            "--width",
            str(width),
            "--height",
            str(height),
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
        if auto_reinit:
            command.append("--auto-reinit")
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
        self.runner = BackgroundCommandRunner(on_event=self.command_events.put, cwd=self.command_builder.cwd)
        self.recording_session: ReferenceRecordingSession | None = None
        self.recording_busy = False
        self._recording_stop = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._capture_preview_images: tuple[tk.PhotoImage, ...] | None = None
        self._pending_axis_preview_path: Path | None = None

        self.root = tk.Tk()
        self.root.title(self.config.title)
        self.profile_name = tk.StringVar(value="phone")
        self.prompt = tk.StringVar(value="mobile phone")
        self.status = tk.StringVar(value="Ready")
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
            values=("d405", "d435"),
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
        ttk.Label(capture, textvariable=self.processing_status).grid(row=1, column=0, columnspan=6, sticky="w")
        ttk.Button(capture, text="Build Dry Run", command=lambda: self.run_build_assets(False)).grid(row=1, column=6)
        ttk.Button(capture, text="Build Assets", command=lambda: self.run_build_assets(True)).grid(row=1, column=7)
        ttk.Button(capture, text="Force Build", command=self.run_force_build_assets).grid(row=1, column=8)
        preview = ttk.Frame(capture)
        preview.grid(row=2, column=0, columnspan=9, sticky="ew", pady=(8, 0))
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
        ttk.Button(build, text="Track Live", command=self.run_tracking).grid(row=1, column=0)
        ttk.Button(build, text="Reinit Tracking", command=self.reinitialize_tracking_event).grid(row=1, column=1)
        ttk.Button(build, text="Stop Command", command=self.stop_command).grid(row=1, column=2)

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
        if not messagebox.askyesno("Delete profile", f"Delete object profile '{name}'?"):
            return
        self.registry.delete(name, confirm=True)
        self.status.set(f"Deleted {name}")
        self.refresh()
        self._update_capture_status()

    def run_setup_check(self) -> None:
        self._start_command(self.command_builder.setup_check(foundationpose_root=self.foundationpose_root.get()))

    def run_segmentation_check(self) -> None:
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

    def start_recording(self) -> None:
        if self.recording_session is not None:
            self.status.set("Recording is already running")
            return
        try:
            profile = self._current_profile()
            if not self._release_live_sessions_for_gpu():
                return
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
        self._start_command(self._charuco_command(profile, mode="process-recordings"))

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
        self._start_command(
            self.command_builder.build_assets(
                object_name=profile.name,
                foundationpose_root=self.foundationpose_root.get(),
                execute=True,
                data_root=self.config.data_root,
            )
        )

    def run_tracking(self) -> None:
        profile = self._current_profile()
        if not self._release_live_sessions_for_gpu():
            return
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

    def reinitialize_tracking_event(self, event=None) -> None:
        if self.runner.running:
            self.runner.stop()
            self.root.after(700, self.run_tracking)
            self.status.set("Restarting tracking for reinitialization")
        else:
            self.run_tracking()

    def stop_command(self) -> None:
        self.runner.stop()

    def _on_camera_model_changed(self, event=None) -> None:
        width, height = _default_camera_resolution(self.camera_model.get())
        self.camera_width.set(width)
        self.camera_height.set(height)
        self.status.set(f"Camera {self.camera_model.get()} default resolution: {width}x{height}")

    def _start_command(self, command: list[str]) -> None:
        try:
            self.runner.start(command)
        except Exception as exc:
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
        cv2 = None
        try:
            import cv2 as cv2_module  # type: ignore

            cv2 = cv2_module
        except Exception:
            cv2 = None
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
                        cv2.imshow(RECORDING_PREVIEW_WINDOW, image)
                        key = cv2.waitKey(1) & 0xFF
                        if key in {ord("q"), 27} or not _opencv_window_visible(cv2, RECORDING_PREVIEW_WINDOW):
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
                        cv2.destroyWindow(RECORDING_PREVIEW_WINDOW)
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
            self.status.set(event.text)
            if self._pending_axis_preview_path is not None:
                self._show_axis_preview(self._pending_axis_preview_path)
                self._pending_axis_preview_path = None
            self.refresh()
            self._update_capture_status()
            self._update_processing_status()
        elif event.kind == "stop":
            self.status.set(event.text)

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


def _live_camera_flag(camera_model: str) -> str:
    model = camera_model.lower()
    if model == "d435":
        return "--live-d435"
    return "--live-d405"


def _default_camera_resolution(camera_model: str) -> tuple[int, int]:
    return DEFAULT_CAMERA_RESOLUTIONS.get(camera_model.lower(), DEFAULT_CAMERA_RESOLUTIONS["d405"])


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
