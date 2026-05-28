"""Stdlib HTTP server for FoundationPose pose-only remote processing."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field, replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable
import uuid
import zipfile

import numpy as np

from visual_servoing.foundationpose_model_free.asset_builder import (
    AssetBuildResult,
    FoundationPoseAssetBuilder,
    find_generated_mesh,
)
from visual_servoing.foundationpose_model_free.charuco_reference import (
    BoardObjectTransform,
    CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
    CharucoBoardSpec,
    CharucoDetectorConfig,
    CharucoQualityConfig,
)
from visual_servoing.foundationpose_model_free.foundationpose_adapter import (
    FoundationPoseAdapter,
    FoundationPoseConfig,
)
from visual_servoing.foundationpose_model_free.mask_provider import Sam3MaskProvider
from visual_servoing.foundationpose_model_free.profile_schema import ObjectProfile
from visual_servoing.foundationpose_model_free.reference_processing import (
    ReferenceProcessingConfig,
    process_recorded_references,
    reselect_recorded_references,
)
from visual_servoing.foundationpose_model_free.reference_recording import RECORDINGS_DIR
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.foundationpose_model_free.tracker import (
    FoundationPoseLiveTracker,
    TrackingRecoveryConfig,
)
from visual_servoing.visual_servo_protocol_v2 import (
    DEFAULT_MAX_CONTENT_LENGTH,
    FoundationPoseSegmentationRequest,
    PROTOCOL_VERSION,
    REQUEST_CONTENT_TYPE,
    RESPONSE_CONTENT_TYPE,
    FoundationPoseTrackRequest,
    decode_foundationpose_segmentation_request,
    decode_foundationpose_track_request,
    encode_foundationpose_response,
)


HEALTH_PATH = "/foundationpose/v2/health"
BUILD_PATH = "/foundationpose/v2/assets/build"
BUILD_STATUS_PREFIX = "/foundationpose/v2/assets/build/"
PROCESS_RECORDINGS_PATH = "/foundationpose/v2/recordings/process"
PROCESS_RECORDINGS_STATUS_PREFIX = "/foundationpose/v2/recordings/process/"
SEGMENTATION_PATH = "/foundationpose/v2/segmentation"
TRACK_PATH = "/foundationpose/v2/track"
RECORDINGS_ZIP_CONTENT_TYPE = "application/x-foundationpose-recordings+zip"
PROCESSING_REQUEST_JSON = "foundationpose_processing_request.json"
TAIL_LIMIT = 4000
MESH_HASH_LIMIT_BYTES = 50 * 1024 * 1024
SERVER_DEFAULT_MAX_CONTENT_LENGTH = 20 * 1024 * 1024 * 1024


class UnknownProfileError(ValueError):
    pass


@dataclass
class BuildJob:
    job_id: str
    profile: str
    state: str = "queued"
    started_at: float | None = None
    completed_at: float | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None
    validation_report: dict[str, Any] | None = None
    command: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.state == "succeeded",
            "job_id": self.job_id,
            "state": self.state,
            "profile": self.profile,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
            "validation_report": self.validation_report,
            "command": self.command,
        }


class BuildJobManager:
    def __init__(self, *, max_tail_chars: int = TAIL_LIMIT) -> None:
        self.max_tail_chars = int(max_tail_chars)
        self._jobs: dict[str, BuildJob] = {}
        self._lock = threading.Lock()

    def enqueue(self, *, profile: ObjectProfile, builder: FoundationPoseAssetBuilder) -> BuildJob:
        job = BuildJob(job_id=uuid.uuid4().hex, profile=profile.name)
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, args=(job, profile, builder), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> BuildJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job: BuildJob, profile: ObjectProfile, builder: FoundationPoseAssetBuilder) -> None:
        job.state = "running"
        job.started_at = time.time()
        try:
            result = builder.build(profile, execute=True)
            self._apply_result(job, result)
            job.state = "succeeded" if int(result.returncode) == 0 else "failed"
        except Exception as exc:
            job.state = "failed"
            job.error = str(exc)
            job.stderr_tail = _tail(traceback.format_exc(), self.max_tail_chars)
        finally:
            job.completed_at = time.time()

    def _apply_result(self, job: BuildJob, result: AssetBuildResult) -> None:
        job.returncode = int(result.returncode)
        job.stdout_tail = _tail(result.stdout, self.max_tail_chars)
        job.stderr_tail = _tail(result.stderr, self.max_tail_chars)
        job.validation_report = result.validation_report
        job.command = list(result.command)


@dataclass
class ProcessingJob:
    job_id: str
    profile: str
    request_id: str | None = None
    mode: str = "process_recordings"
    state: str = "queued"
    started_at: float | None = None
    completed_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    stderr_tail: str = ""
    upload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.state == "succeeded",
            "job_id": self.job_id,
            "request_id": self.request_id,
            "state": self.state,
            "profile": self.profile,
            "mode": self.mode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "returncode": self.returncode,
            "error": self.error,
            "stderr_tail": self.stderr_tail,
            "upload": dict(self.upload),
            "result": dict(self.result) if self.result is not None else None,
        }


class ProcessingJobManager:
    def __init__(self, *, max_tail_chars: int = TAIL_LIMIT) -> None:
        self.max_tail_chars = int(max_tail_chars)
        self._jobs: dict[str, ProcessingJob] = {}
        self._lock = threading.Lock()

    def enqueue(
        self,
        *,
        profile: ObjectProfile,
        request_id: str | None,
        mode: str,
        upload: dict[str, Any],
        processor: Callable[[], dict[str, Any]],
    ) -> ProcessingJob:
        job = ProcessingJob(
            job_id=uuid.uuid4().hex,
            profile=profile.name,
            request_id=request_id,
            mode=mode,
            upload=dict(upload),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, args=(job, processor), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> ProcessingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job: ProcessingJob, processor: Callable[[], dict[str, Any]]) -> None:
        job.state = "running"
        job.started_at = time.time()
        try:
            result = processor()
            job.result = result
            job.returncode = int(result.get("returncode", 0))
            job.state = "succeeded"
        except Exception as exc:
            job.state = "failed"
            job.returncode = 1
            job.error = str(exc)
            job.stderr_tail = _tail(traceback.format_exc(), self.max_tail_chars)
        finally:
            job.completed_at = time.time()


@dataclass(frozen=True)
class TrackerCacheKey:
    profile: str
    foundationpose_root: str | None
    mesh_identity_json: str
    refine_iterations: int
    track_iterations: int
    mask_options_json: str
    recovery_options_json: str


@dataclass
class TrackerSession:
    session_id: str
    key: TrackerCacheKey
    tracker: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_monotonic_ns: int = field(default_factory=time.monotonic_ns)


class TrackerSessionManager:
    def __init__(
        self,
        *,
        tracker_factory: Callable[[ObjectProfile, Path, FoundationPoseTrackRequest], Any] | None = None,
    ) -> None:
        self._tracker_factory = tracker_factory
        self._sessions: dict[TrackerCacheKey, TrackerSession] = {}
        self._lock = threading.Lock()

    def session_for(
        self,
        *,
        profile: ObjectProfile,
        mesh_path: Path,
        request: FoundationPoseTrackRequest,
    ) -> tuple[TrackerSession, bool, dict[str, Any]]:
        foundationpose_root = _resolve_server_foundationpose_root(request.foundationpose_root)
        request = replace(request, foundationpose_root=foundationpose_root)
        key = TrackerCacheKey(
            profile=profile.name,
            foundationpose_root=foundationpose_root,
            mesh_identity_json=_stable_json(mesh_identity(mesh_path)),
            refine_iterations=int(request.refine_iterations),
            track_iterations=int(request.track_iterations),
            mask_options_json=_stable_json(request.mask_options),
            recovery_options_json=_stable_json(request.recovery_options),
        )
        with self._lock:
            session = self._sessions.get(key)
            if session is not None:
                return session, True, {"cache_key": key.__dict__}
            tracker = self._make_tracker(profile, mesh_path, request)
            session = TrackerSession(session_id=uuid.uuid4().hex, key=key, tracker=tracker)
            self._sessions[key] = session
            return session, False, {"cache_key": key.__dict__}

    def _make_tracker(self, profile: ObjectProfile, mesh_path: Path, request: FoundationPoseTrackRequest) -> Any:
        if self._tracker_factory is not None:
            return self._tracker_factory(profile, mesh_path, request)
        adapter = FoundationPoseAdapter(
            FoundationPoseConfig(
                foundationpose_root=Path(request.foundationpose_root).expanduser().resolve()
                if request.foundationpose_root
                else None,
                mesh_path=mesh_path,
                debug_dir=profile.logs_dir / "debug",
                refinement_iterations=request.refine_iterations,
                tracking_iterations=request.track_iterations,
            )
        )
        mask_options = dict(request.mask_options)
        mask_provider = Sam3MaskProvider(
            prompt=profile.prompt,
            device=str(mask_options.get("device", "cuda")),
            confidence_threshold=float(mask_options.get("threshold", 0.3)),
            resolution=int(mask_options.get("resolution", 1008)),
        )
        return FoundationPoseLiveTracker(
            profile=profile,
            adapter=adapter,
            mask_provider=mask_provider,
            recovery_config=recovery_config_from_options(request.recovery_options),
        )


class FoundationPoseV2Service:
    def __init__(
        self,
        *,
        registry: ObjectProfileRegistry | None = None,
        builder_factory: Callable[[str | Path], FoundationPoseAssetBuilder] | None = None,
        tracker_factory: Callable[[ObjectProfile, Path, FoundationPoseTrackRequest], Any] | None = None,
        processing_runner: Callable[[ObjectProfile, dict[str, Any]], dict[str, Any]] | None = None,
        segmentation_runner: Callable[[FoundationPoseSegmentationRequest], dict[str, Any]] | None = None,
        job_manager: BuildJobManager | None = None,
        processing_job_manager: ProcessingJobManager | None = None,
        tracker_sessions: TrackerSessionManager | None = None,
    ) -> None:
        self.registry = registry or ObjectProfileRegistry()
        self.builder_factory = builder_factory or _make_asset_builder
        self.processing_runner = processing_runner
        self.segmentation_runner = segmentation_runner
        self.job_manager = job_manager or BuildJobManager()
        self.processing_job_manager = processing_job_manager or ProcessingJobManager()
        self.tracker_sessions = tracker_sessions or TrackerSessionManager(tracker_factory=tracker_factory)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "status": "ready",
            "server_time_monotonic_ns": time.monotonic_ns(),
        }

    def build_assets(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        missing = [key for key in ("request_id", "profile", "foundationpose_root", "execute") if key not in payload]
        if missing:
            return 400, {"ok": False, "status": "error", "reason": f"missing required fields: {missing}"}
        try:
            profile = self._profile(str(payload["profile"]))
        except UnknownProfileError as exc:
            return 404, {"ok": False, "status": "error", "reason": str(exc)}
        foundationpose_root = _resolve_server_foundationpose_root(str(payload["foundationpose_root"]))
        builder = self.builder_factory(str(foundationpose_root))
        if not bool(payload.get("execute")):
            try:
                result = builder.build(profile, execute=False)
                return 200, _build_result_payload(result, profile=profile.name)
            except Exception as exc:
                return 200, {
                    "ok": False,
                    "status": "validation_failed",
                    "profile": profile.name,
                    "reason": str(exc),
                }
        job = self.job_manager.enqueue(profile=profile, builder=builder)
        return 202, {"ok": True, "status": "queued", "job_id": job.job_id, "profile": profile.name}

    def build_status(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self.job_manager.get(job_id)
        if job is None:
            return 404, {"ok": False, "status": "error", "reason": f"unknown build job: {job_id}"}
        return 200, job.payload()

    def process_recordings_archive(
        self,
        *,
        archive: bytes,
        request_id: str | None = None,
        profile_name: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        try:
            options, upload = _extract_recordings_archive(
                self.registry,
                archive,
                profile_name=profile_name,
            )
            profile = self.registry.create(
                str(options["profile"]),
                prompt=str(options.get("prompt") or "object"),
                exist_ok=True,
            )
        except Exception as exc:
            return 400, {"ok": False, "status": "error", "reason": str(exc)}
        mode = "reselect_recordings" if bool(options.get("reselect")) else "process_recordings"
        request_id = request_id or str(options.get("request_id") or "")
        job = self.processing_job_manager.enqueue(
            profile=profile,
            request_id=request_id,
            mode=mode,
            upload=upload,
            processor=lambda: self._run_recording_processing(profile.name, options),
        )
        return 202, {
            "ok": True,
            "status": "queued",
            "job_id": job.job_id,
            "profile": profile.name,
            "mode": mode,
            "upload": upload,
        }

    def process_recordings_status(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self.processing_job_manager.get(job_id)
        if job is None:
            return 404, {"ok": False, "status": "error", "reason": f"unknown processing job: {job_id}"}
        return 200, job.payload()

    def segment(self, request: FoundationPoseSegmentationRequest) -> tuple[int, dict[str, Any]]:
        if self.segmentation_runner is not None:
            try:
                return 200, self.segmentation_runner(request)
            except Exception as exc:
                return 500, {
                    "ok": False,
                    "status": "error",
                    "request_id": request.request_id,
                    "prompt": request.prompt,
                    "reason": str(exc),
                    "server_time_monotonic_ns": time.monotonic_ns(),
                }
        mask_options = dict(request.mask_options)
        provider = Sam3MaskProvider(
            prompt=request.prompt,
            device=str(mask_options.get("device", "auto")),
            confidence_threshold=float(mask_options.get("threshold", 0.3)),
            resolution=int(mask_options.get("resolution", 1008)),
        )
        try:
            result = provider.get_mask(request.rgb, depth_m=request.depth_m, object_name=request.prompt)
        except Exception as exc:
            return 500, {
                "ok": False,
                "status": "error",
                "request_id": request.request_id,
                "prompt": request.prompt,
                "reason": str(exc),
                "server_time_monotonic_ns": time.monotonic_ns(),
            }
        finally:
            release = getattr(provider, "release", None)
            if callable(release):
                release()
        mask = np.asarray(result.mask, dtype=bool)
        try:
            mask_png_b64 = _mask_png_b64(mask)
        except Exception as exc:
            return 500, {
                "ok": False,
                "status": "error",
                "request_id": request.request_id,
                "prompt": request.prompt,
                "reason": str(exc),
                "server_time_monotonic_ns": time.monotonic_ns(),
            }
        return 200, {
            "ok": True,
            "status": "segmented",
            "request_id": request.request_id,
            "prompt": request.prompt,
            "mask": _mask_summary(mask),
            "mask_png_b64": mask_png_b64,
            "mask_source": result.source,
            "mask_metadata": result.metadata,
            "server_time_monotonic_ns": time.monotonic_ns(),
        }

    def track(self, request: FoundationPoseTrackRequest) -> tuple[int, dict[str, Any]]:
        server_received_ns = time.monotonic_ns()
        timing_ms: dict[str, float] = {}
        try:
            profile = self._profile(request.profile)
        except UnknownProfileError as exc:
            return 404, _track_error(request, server_received_ns, timing_ms, str(exc))
        mesh_path = find_generated_mesh(profile)
        if mesh_path is None:
            return 409, _track_error(
                request,
                server_received_ns,
                timing_ms,
                f"profile {profile.name} has no fresh generated mesh/model.obj; run asset build first",
            )

        start = time.perf_counter()
        session, cache_hit, cache_metadata = self.tracker_sessions.session_for(
            profile=profile,
            mesh_path=mesh_path,
            request=request,
        )
        timing_ms["session_ms"] = _elapsed_ms(start)
        start = time.perf_counter()
        with session.lock:
            if request.reinit:
                request_reinit = getattr(session.tracker, "request_reinit", None)
                if callable(request_reinit):
                    request_reinit()
            result = session.tracker.process_frame(
                rgb=request.rgb,
                depth_m=request.depth_m,
                intrinsics=request.intrinsics,
            )
        timing_ms["tracking_ms"] = _elapsed_ms(start)
        pose = result.pose.camera_T_object if getattr(result, "pose", None) is not None else None
        camera_T_object = np.asarray(pose, dtype=np.float64) if pose is not None else None
        t5_T_object = request.t5_T_camera @ camera_T_object if camera_T_object is not None else None
        server_completed_ns = time.monotonic_ns()
        return 200, {
            "ok": camera_T_object is not None,
            "status": str(getattr(result, "state", "LOST")).lower(),
            "request_id": request.request_id,
            "frame_index": request.frame_index,
            "profile": profile.name,
            "tracker_session_id": session.session_id,
            "camera_T_object": _matrix_or_none(camera_T_object),
            "t5_T_object": _matrix_or_none(t5_T_object),
            "tracking_state": getattr(result, "state", "LOST"),
            "fresh_pose": bool(getattr(result, "fresh_pose", False)),
            "held_pose": bool(getattr(result, "held_pose", False)),
            "message": getattr(result, "message", None),
            "server_received_monotonic_ns": server_received_ns,
            "server_completed_monotonic_ns": server_completed_ns,
            "server_timing_ms": timing_ms,
            "tracker_cache": {
                "cache_hit": cache_hit,
                "session_created_monotonic_ns": session.created_monotonic_ns,
                **cache_metadata,
            },
            "tracker_metadata": getattr(result, "metadata", None) or {},
        }

    def _profile(self, name: str) -> ObjectProfile:
        try:
            return self.registry.get(name)
        except FileNotFoundError as exc:
            raise UnknownProfileError(str(exc)) from exc

    def _run_recording_processing(self, profile_name: str, options: dict[str, Any]) -> dict[str, Any]:
        profile = self.registry.get(profile_name)
        if self.processing_runner is not None:
            return self.processing_runner(profile, options)
        board_spec = _board_spec_from_options(options.get("board_spec"))
        quality_config = _quality_config_from_options(options.get("quality_config"))
        board_object = _board_object_from_options(options.get("board_object"))
        detector_preset = (
            options.get("charuco_detector_preset")
            or options.get("detector_preset")
            or CHARUCO_DETECTOR_PRESET_CONSERVATIVE
        )
        detector_config = CharucoDetectorConfig(str(detector_preset))
        config = _processing_config_from_options(options)
        if bool(options.get("reselect")):
            report = reselect_recorded_references(
                profile,
                board_spec=board_spec,
                quality_config=quality_config,
                board_object=board_object,
                config=config,
                detector_config=detector_config,
            )
            mode = "reselect_recordings"
        else:
            provider = Sam3MaskProvider(
                prompt=str(options.get("prompt") or profile.prompt),
                device=str(options.get("sam_device") or options.get("device") or "auto"),
                confidence_threshold=float(options.get("sam_threshold", options.get("threshold", 0.3))),
                resolution=int(options.get("sam_resolution", 1008)),
            )
            report = process_recorded_references(
                profile,
                mask_provider=provider,
                board_spec=board_spec,
                quality_config=quality_config,
                board_object=board_object,
                config=config,
                detector_config=detector_config,
            )
            mode = "process_recordings"
        payload = report.to_dict()
        payload["ok"] = report.ok
        payload["returncode"] = 0 if int(report.accepted) > 0 else 1
        payload["mode"] = mode
        payload["detector_preset"] = detector_config.preset
        return payload


def _resolve_server_foundationpose_root(requested: str | None) -> str | None:
    requested_value = str(requested).strip() if requested is not None else ""
    for candidate in _server_foundationpose_root_candidates(requested_value):
        if _looks_like_foundationpose_root(candidate):
            return str(candidate.resolve())
    return requested_value or None


def _make_asset_builder(foundationpose_root: str | Path) -> FoundationPoseAssetBuilder:
    build_python = os.environ.get("FOUNDATIONPOSE_BUILD_PYTHON")
    return FoundationPoseAssetBuilder(
        foundationpose_root=foundationpose_root,
        python_executable=build_python or None,
    )


def _server_foundationpose_root_candidates(requested: str) -> list[Path]:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    env_root = os.environ.get("FOUNDATIONPOSE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path(__file__).resolve().parents[1])
    return candidates


def _looks_like_foundationpose_root(path: Path) -> bool:
    return (path / "bundlesdf" / "run_nerf.py").exists() and (
        path / "bundlesdf" / "config_ycbv.yml"
    ).exists()


def make_handler(
    service: FoundationPoseV2Service,
    *,
    max_content_length: int = SERVER_DEFAULT_MAX_CONTENT_LENGTH,
):
    class FoundationPoseV2Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == HEALTH_PATH:
                self._send_json(200, service.health())
                return
            if self.path.startswith(BUILD_STATUS_PREFIX):
                job_id = self.path[len(BUILD_STATUS_PREFIX) :].strip("/")
                if not job_id:
                    self._send_json(404, {"ok": False, "status": "error", "reason": "missing job_id"})
                    return
                status_code, payload = service.build_status(job_id)
                self._send_json(status_code, payload)
                return
            if self.path.startswith(PROCESS_RECORDINGS_STATUS_PREFIX):
                job_id = self.path[len(PROCESS_RECORDINGS_STATUS_PREFIX) :].strip("/")
                if not job_id:
                    self._send_json(404, {"ok": False, "status": "error", "reason": "missing job_id"})
                    return
                status_code, payload = service.process_recordings_status(job_id)
                self._send_json(status_code, payload)
                return
            self._send_json(404, {"ok": False, "status": "error", "reason": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == BUILD_PATH:
                self._handle_build()
                return
            if self.path == PROCESS_RECORDINGS_PATH:
                self._handle_process_recordings()
                return
            if self.path == SEGMENTATION_PATH:
                self._handle_segmentation()
                return
            if self.path == TRACK_PATH:
                self._handle_track()
                return
            self._send_json(404, {"ok": False, "status": "error", "reason": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_build(self) -> None:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._send_json(415, {"ok": False, "status": "error", "reason": "unsupported content type"})
                return
            try:
                body = self._read_body()
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("build request must be a JSON object")
            except Exception as exc:
                self._send_json(400, {"ok": False, "status": "error", "reason": str(exc)})
                return
            status_code, response = service.build_assets(payload)
            self._send_json(status_code, response)

        def _handle_process_recordings(self) -> None:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != RECORDINGS_ZIP_CONTENT_TYPE:
                self._send_json(415, {"ok": False, "status": "error", "reason": "unsupported content type"})
                return
            try:
                body = self._read_body()
            except Exception as exc:
                self._send_json(400, {"ok": False, "status": "error", "reason": str(exc)})
                return
            status_code, response = service.process_recordings_archive(
                archive=body,
                request_id=self.headers.get("X-FoundationPose-Request-Id"),
                profile_name=self.headers.get("X-FoundationPose-Profile"),
            )
            self._send_json(status_code, response)

        def _handle_segmentation(self) -> None:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != REQUEST_CONTENT_TYPE:
                self._send_json(415, {"ok": False, "status": "error", "reason": "unsupported content type"})
                return
            try:
                body = self._read_body()
                request = decode_foundationpose_segmentation_request(body, max_content_length=max_content_length)
            except Exception as exc:
                self._send_json(400, {"ok": False, "status": "error", "reason": str(exc)})
                return
            status_code, response = service.segment(request)
            self._send_json(status_code, response)

        def _handle_track(self) -> None:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != REQUEST_CONTENT_TYPE:
                self._send_json(415, {"ok": False, "status": "error", "reason": "unsupported content type"})
                return
            try:
                body = self._read_body()
                request = decode_foundationpose_track_request(body, max_content_length=max_content_length)
            except Exception as exc:
                self._send_json(400, {"ok": False, "status": "error", "reason": str(exc)})
                return
            status_code, response = service.track(request)
            self._send_json(status_code, response)

        def _read_body(self) -> bytes:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid content length") from exc
            if content_length <= 0:
                raise ValueError("empty request body")
            if content_length > int(max_content_length):
                raise ValueError("request body too large")
            return self.rfile.read(content_length)

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            data = encode_foundationpose_response(payload)
            try:
                self.send_response(status_code)
                self.send_header("Content-Type", RESPONSE_CONTENT_TYPE)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                return

    return FoundationPoseV2Handler


def _extract_recordings_archive(
    registry: ObjectProfileRegistry,
    archive: bytes,
    *,
    profile_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        zip_buffer = io.BytesIO(archive)
        with zipfile.ZipFile(zip_buffer) as zf:
            options = _processing_options_from_zip(zf)
            profile_name = str(profile_name or options.get("profile") or "").strip()
            if not profile_name:
                raise ValueError("processing archive must include a profile")
            options["profile"] = profile_name
            profile = registry.create(
                profile_name,
                prompt=str(options.get("prompt") or "object"),
                exist_ok=True,
            )
            upload = _extract_recording_members(zf, profile)
    except zipfile.BadZipFile as exc:
        raise ValueError("recordings upload must be a valid zip archive") from exc
    if int(upload.get("file_count", 0)) <= 0:
        raise ValueError("recordings archive did not contain recording files")
    return options, upload


def _processing_options_from_zip(zf: zipfile.ZipFile) -> dict[str, Any]:
    if PROCESSING_REQUEST_JSON not in zf.namelist():
        return {}
    payload = json.loads(zf.read(PROCESSING_REQUEST_JSON).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{PROCESSING_REQUEST_JSON} must contain a JSON object")
    return payload


def _extract_recording_members(zf: zipfile.ZipFile, profile: ObjectProfile) -> dict[str, Any]:
    profile.root.mkdir(parents=True, exist_ok=True)
    session_ids: set[str] = set()
    file_count = 0
    uncompressed_bytes = 0
    for member in zf.infolist():
        if member.is_dir() or member.filename == PROCESSING_REQUEST_JSON:
            continue
        parts = Path(member.filename).parts
        if len(parts) < 3 or parts[0] != RECORDINGS_DIR:
            raise ValueError(f"unsupported archive member: {member.filename}")
        if any(part in {"", ".", ".."} for part in parts) or Path(member.filename).is_absolute():
            raise ValueError(f"unsafe archive member: {member.filename}")
        target = profile.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as source, target.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        session_ids.add(parts[1])
        file_count += 1
        uncompressed_bytes += int(member.file_size)
    profile.touch()
    profile.save()
    return {
        "session_count": len(session_ids),
        "session_ids": sorted(session_ids),
        "file_count": file_count,
        "uncompressed_bytes": uncompressed_bytes,
    }


def _board_spec_from_options(value: Any) -> CharucoBoardSpec:
    data = dict(value) if isinstance(value, dict) else {}
    return CharucoBoardSpec(
        squares_x=int(data.get("squares_x", 5)),
        squares_y=int(data.get("squares_y", 8)),
        square_length_m=float(data.get("square_length_m", 0.030)),
        marker_length_m=float(data.get("marker_length_m", 0.022)),
        dictionary=str(data.get("dictionary", "auto")),
        legacy_pattern=bool(data.get("legacy_pattern", False)),
    )


def _quality_config_from_options(value: Any) -> CharucoQualityConfig:
    data = dict(value) if isinstance(value, dict) else {}
    return CharucoQualityConfig(
        min_corners=int(data.get("min_corners", 6)),
        min_markers=int(data.get("min_markers", 2)),
        max_reprojection_error_px=float(data.get("max_reprojection_error_px", 4.0)),
        min_image_coverage_fraction=float(data.get("min_image_coverage_fraction", 0.005)),
    )


def _board_object_from_options(value: Any) -> BoardObjectTransform:
    data = dict(value) if isinstance(value, dict) else {}
    matrix = data.get("board_T_object", data.get("matrix"))
    if matrix is None:
        return BoardObjectTransform.identity()
    xyz = data.get("xyz_m")
    rpy = data.get("rpy_deg")
    return BoardObjectTransform(
        np.asarray(matrix, dtype=np.float64),
        source=str(data.get("source") or "remote_processing_options"),
        xyz_m=tuple(float(v) for v in xyz) if isinstance(xyz, (list, tuple)) else None,
        rpy_deg=tuple(float(v) for v in rpy) if isinstance(rpy, (list, tuple)) else None,
    )


def _processing_config_from_options(options: dict[str, Any]) -> ReferenceProcessingConfig:
    return ReferenceProcessingConfig(
        required_keyframes=int(options.get("required_keyframes", 16)),
        max_keyframes=int(options.get("max_keyframes", 32)),
        min_mask_area_fraction=float(options.get("min_mask_area_fraction", 0.0005)),
        min_valid_depth_ratio=float(options.get("min_valid_depth_ratio", 0.05)),
        min_depth_m=float(options.get("min_depth_m", 0.05)),
        max_depth_m=float(options.get("max_depth_m", 3.0)),
        publish=not bool(options.get("evaluate_only", False)),
    )


def _mask_summary(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask))
    summary: dict[str, Any] = {
        "area": area,
        "area_fraction": float(area / max(mask.size, 1)),
        "shape": [int(value) for value in mask.shape],
        "box_xyxy": None,
    }
    if area > 0:
        ys, xs = np.nonzero(mask)
        summary["box_xyxy"] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    return summary


def _mask_png_b64(mask: np.ndarray) -> str:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required to encode segmentation masks.") from exc
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    ok, encoded = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("failed to encode segmentation mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def recovery_config_from_options(options: dict[str, Any]) -> TrackingRecoveryConfig:
    return TrackingRecoveryConfig(
        hold_last_pose_frames=int(options.get("hold_last_pose_frames", 0)),
        auto_reinit=bool(options.get("auto_reinit", False)),
        auto_reinit_after_lost_frames=int(options.get("auto_reinit_after_lost_frames", 30)),
        verify_pose_depth=bool(options.get("verify_pose_depth", False)),
        warn_initial_pose_mask_alignment=bool(options.get("warn_initial_pose_mask_alignment", False)),
        pose_depth_tolerance_m=float(options.get("pose_depth_tolerance_m", 0.18)),
        pose_depth_window_radius_px=int(options.get("pose_depth_window_radius_px", 7)),
        max_pose_jump_m=options.get("max_pose_jump_m"),
        implausible_lost_threshold=int(options.get("implausible_lost_threshold", 1)),
    )


def mesh_identity(mesh_path: Path) -> dict[str, Any]:
    path = Path(mesh_path).expanduser().resolve()
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": None,
    }
    if stat.st_size <= MESH_HASH_LIMIT_BYTES:
        payload["sha256"] = _sha256_file(path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FoundationPose v2 pose-only HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--data-root")
    parser.add_argument("--max-content-length", type=int, default=SERVER_DEFAULT_MAX_CONTENT_LENGTH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(args.data_root))
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(service, max_content_length=args.max_content_length))
    print(json.dumps({"event": "foundationpose_v2_server_listening", "host": args.host, "port": args.port}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _build_result_payload(result: AssetBuildResult, *, profile: str) -> dict[str, Any]:
    return {
        "ok": int(result.returncode) == 0,
        "status": "validated" if not result.executed else "succeeded" if int(result.returncode) == 0 else "failed",
        "profile": profile,
        "executed": bool(result.executed),
        "returncode": int(result.returncode),
        "elapsed_ms": float(result.elapsed_ms),
        "stdout_tail": _tail(result.stdout, TAIL_LIMIT),
        "stderr_tail": _tail(result.stderr, TAIL_LIMIT),
        "validation_report": result.validation_report,
        "command": list(result.command),
    }


def _track_error(
    request: FoundationPoseTrackRequest,
    server_received_ns: int,
    timing_ms: dict[str, float],
    reason: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "lost",
        "request_id": request.request_id,
        "frame_index": request.frame_index,
        "profile": request.profile,
        "camera_T_object": None,
        "t5_T_object": None,
        "tracking_state": "LOST",
        "fresh_pose": False,
        "held_pose": False,
        "message": reason,
        "server_received_monotonic_ns": server_received_ns,
        "server_completed_monotonic_ns": time.monotonic_ns(),
        "server_timing_ms": timing_ms,
    }


def _matrix_or_none(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.float64).tolist()


def _tail(text: str | None, limit: int) -> str:
    if not text:
        return ""
    return str(text)[-int(limit) :]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
