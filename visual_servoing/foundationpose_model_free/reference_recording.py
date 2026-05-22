"""Raw RGB-D recording sessions for offline reference processing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from visual_servoing.point_pose.live_camera_config import resolve_live_camera_config
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, RealSenseCamera, RgbdFrame
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics

from .charuco_reference import BoardObjectTransform, CharucoBoardSpec
from .profile_schema import ObjectProfile, ProfileStatus, utc_now_iso


RECORDINGS_DIR = "recordings"
FRAMES_JSONL = "frames.jsonl"
SESSION_JSON = "session.json"


@dataclass(frozen=True)
class ReferenceRecordingConfig:
    camera_model: str = "d405"
    serial: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int = 15
    frame_timeout_ms: int = 5000
    board_spec: CharucoBoardSpec = field(default_factory=CharucoBoardSpec)
    board_object: BoardObjectTransform = field(default_factory=BoardObjectTransform.identity)
    sam_device: str = "auto"
    sam_resolution: int = 1008
    sam_confidence_threshold: float = 0.3

    def to_metadata(self) -> dict[str, Any]:
        camera = resolve_live_camera_config(
            model=self.camera_model,
            serial=self.serial,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        return {
            "camera": {
                "model": camera.model,
                "serial": camera.serial,
                "width": camera.width,
                "height": camera.height,
                "fps": int(camera.fps),
                "sdk_resolution": camera.sdk_resolution,
                "native_resolution": bool(camera.native_resolution),
            },
            "frame_timeout_ms": int(self.frame_timeout_ms),
            "board_spec": self.board_spec.to_dict(),
            "board_object_transform": self.board_object.to_dict(),
            "sam": {
                "device": self.sam_device,
                "resolution": int(self.sam_resolution),
                "confidence_threshold": float(self.sam_confidence_threshold),
            },
        }


@dataclass(frozen=True)
class RecordedFrameRecord:
    session_id: str
    index: int
    timestamp_s: float
    rgb_path: str
    depth_path: str
    depth_mm_path: str
    intrinsics_path: str
    intrinsics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "index": int(self.index),
            "timestamp_s": float(self.timestamp_s),
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "depth_mm_path": self.depth_mm_path,
            "intrinsics_path": self.intrinsics_path,
            "intrinsics": dict(self.intrinsics),
        }


@dataclass(frozen=True)
class RecordingSessionInfo:
    session_id: str
    session_dir: Path
    frame_count: int
    metadata: dict[str, Any]


class ReferenceRecordingSession:
    """Append-only raw frame recorder for later ChArUco/SAM processing."""

    def __init__(
        self,
        profile: ObjectProfile,
        *,
        config: ReferenceRecordingConfig | None = None,
        camera: RealSenseCamera | None = None,
        session_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.config = config or ReferenceRecordingConfig()
        self.session_id = session_id or _new_session_id()
        self.session_dir = profile.root / RECORDINGS_DIR / self.session_id
        self._camera = camera
        self._owned_camera = camera is None
        self._started = False
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self) -> None:
        if self._started:
            return
        self._prepare_dirs()
        if self._camera is None:
            self._camera = LiveRgbdCamera(
                model=self.config.camera_model,
                serial=self.config.serial,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            )
            self._owned_camera = True
        self._camera.start()
        self._started = True
        self.profile.status = ProfileStatus.CAPTURING
        self.profile.metadata["active_recording_session"] = self.session_id
        self.profile.touch()
        self.profile.save()

    def stop(self) -> RecordingSessionInfo:
        try:
            if self._started and self._owned_camera and self._camera is not None:
                self._camera.stop()
        finally:
            self._started = False
            self.profile.metadata.pop("active_recording_session", None)
            self.profile.metadata["last_recording_session"] = self.session_id
            self.profile.touch()
            self.profile.save()
        return self.info()

    def record_next_frame(self) -> RecordedFrameRecord:
        if not self._started or self._camera is None:
            raise RuntimeError("ReferenceRecordingSession.start() must be called before recording frames.")
        frame = self._camera.read(timeout_ms=self.config.frame_timeout_ms)
        record = save_raw_recorded_frame(
            self.profile,
            self.session_id,
            self._frame_count,
            frame=frame,
            timestamp_s=time.time(),
        )
        self._frame_count += 1
        _append_frame_record(self.session_dir, record)
        _update_session_frame_count(self.session_dir, self._frame_count)
        return record

    def info(self) -> RecordingSessionInfo:
        return RecordingSessionInfo(
            session_id=self.session_id,
            session_dir=self.session_dir,
            frame_count=count_recorded_frames(self.session_dir),
            metadata=load_session_metadata(self.session_dir),
        )

    def _prepare_dirs(self) -> None:
        for path in (
            self.session_dir,
            self.session_dir / "rgb",
            self.session_dir / "depth",
            self.session_dir / "depth_mm",
            self.session_dir / "intrinsics",
        ):
            path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "session_id": self.session_id,
            "object": self.profile.name,
            "prompt": self.profile.prompt,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "frame_count": 0,
            **self.config.to_metadata(),
        }
        (self.session_dir / SESSION_JSON).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        frames_path = self.session_dir / FRAMES_JSONL
        if not frames_path.exists():
            frames_path.write_text("", encoding="utf-8")

    def __enter__(self) -> "ReferenceRecordingSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def recordings_dir(profile: ObjectProfile) -> Path:
    return profile.root / RECORDINGS_DIR


def list_recording_sessions(profile: ObjectProfile) -> list[RecordingSessionInfo]:
    root = recordings_dir(profile)
    if not root.exists():
        return []
    sessions: list[RecordingSessionInfo] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = session_dir / SESSION_JSON
        if not metadata_path.exists():
            continue
        sessions.append(
            RecordingSessionInfo(
                session_id=session_dir.name,
                session_dir=session_dir,
                frame_count=count_recorded_frames(session_dir),
                metadata=load_session_metadata(session_dir),
            )
        )
    return sessions


def load_frame_records(session_dir: Path) -> list[RecordedFrameRecord]:
    frames_path = session_dir / FRAMES_JSONL
    if not frames_path.exists():
        return []
    records: list[RecordedFrameRecord] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(
            RecordedFrameRecord(
                session_id=str(data["session_id"]),
                index=int(data["index"]),
                timestamp_s=float(data["timestamp_s"]),
                rgb_path=str(data["rgb_path"]),
                depth_path=str(data["depth_path"]),
                depth_mm_path=str(data["depth_mm_path"]),
                intrinsics_path=str(data["intrinsics_path"]),
                intrinsics=dict(data["intrinsics"]),
            )
        )
    return records


def count_recorded_frames(session_dir: Path) -> int:
    return len(load_frame_records(Path(session_dir)))


def load_session_metadata(session_dir: Path) -> dict[str, Any]:
    path = Path(session_dir) / SESSION_JSON
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_raw_recorded_frame(
    profile: ObjectProfile,
    session_id: str,
    index: int,
    *,
    frame: RgbdFrame,
    timestamp_s: float,
) -> RecordedFrameRecord:
    session_dir = profile.root / RECORDINGS_DIR / session_id
    cv2 = _require_cv2()
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    depth_mm_dir = session_dir / "depth_mm"
    intrinsics_dir = session_dir / "intrinsics"
    for path in (rgb_dir, depth_dir, depth_mm_dir, intrinsics_dir):
        path.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(frame.rgb, dtype=np.uint8)
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("depth shape must match RGB size")

    stem = f"{index:06d}"
    rgb_path = rgb_dir / f"{stem}.png"
    depth_path = depth_dir / f"{stem}.npy"
    depth_mm_path = depth_mm_dir / f"{stem}.png"
    intrinsics_path = intrinsics_dir / f"{stem}.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(depth_path, depth)
    cv2.imwrite(str(depth_mm_path), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
    intrinsics = intrinsics_to_dict(frame.intrinsics)
    intrinsics_path.write_text(json.dumps(intrinsics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return RecordedFrameRecord(
        session_id=session_id,
        index=index,
        timestamp_s=timestamp_s,
        rgb_path=str(rgb_path.relative_to(session_dir)),
        depth_path=str(depth_path.relative_to(session_dir)),
        depth_mm_path=str(depth_mm_path.relative_to(session_dir)),
        intrinsics_path=str(intrinsics_path.relative_to(session_dir)),
        intrinsics=intrinsics,
    )


def intrinsics_to_dict(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "width": intrinsics.width,
        "height": intrinsics.height,
        "distortion_coeffs": list(intrinsics.distortion_coeffs)
        if intrinsics.distortion_coeffs is not None
        else None,
        "distortion_model": intrinsics.distortion_model,
    }


def _append_frame_record(session_dir: Path, record: RecordedFrameRecord) -> None:
    with (session_dir / FRAMES_JSONL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _update_session_frame_count(session_dir: Path, frame_count: int) -> None:
    path = session_dir / SESSION_JSON
    metadata = load_session_metadata(session_dir)
    metadata["frame_count"] = int(frame_count)
    metadata["updated_at"] = utc_now_iso()
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _new_session_id() -> str:
    stamp = utc_now_iso().replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"session-{stamp}-{time.time_ns() % 1_000_000_000:09d}"


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("OpenCV is required to write raw reference recordings.") from exc
    return cv2
