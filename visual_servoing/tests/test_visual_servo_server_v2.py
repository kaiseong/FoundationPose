from __future__ import annotations

from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import io
import json
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request
import zipfile

import numpy as np
import pytest

from visual_servoing.foundationpose_model_free.asset_builder import AssetBuildResult, profile_model_path
from visual_servoing.foundationpose_model_free.foundationpose_adapter import PoseEstimate
from visual_servoing.foundationpose_model_free.profile_manifest import record_asset_ready
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_protocol_v2 import (
    REQUEST_CONTENT_TYPE,
    decode_foundationpose_segmentation_request,
    decode_foundationpose_track_request,
    decode_foundationpose_response,
    encode_foundationpose_segmentation_request,
    encode_foundationpose_track_request,
)
from visual_servoing.visual_servo_server_v2 import FoundationPoseV2Service, make_handler, mesh_identity
from visual_servoing.visual_servo_server_v2 import RECORDINGS_ZIP_CONTENT_TYPE
from visual_servoing.visual_servo_server_v2 import _make_asset_builder


class FakeBuilder:
    def __init__(self, foundationpose_root):
        self.foundationpose_root = foundationpose_root

    def build(self, profile, *, execute=False):
        long_stdout = "o" * 5001
        long_stderr = "e" * 5001
        return AssetBuildResult(
            command=["fake-build", profile.name, str(self.foundationpose_root)],
            returncode=0,
            elapsed_ms=1.5,
            stdout=long_stdout,
            stderr=long_stderr,
            executed=bool(execute),
            validation_report={"ok": True},
        )


class FakeTracker:
    def __init__(self, *, state="TRACKING", blocker=None, lock_ref=None):
        self.state = state
        self.blocker = blocker
        self.lock_ref = lock_ref
        self.request_reinit_calls = 0
        self.reinit_lock_states = []
        self.started = []
        self.finished = []

    def request_reinit(self):
        self.request_reinit_calls += 1
        if self.lock_ref is not None:
            self.reinit_lock_states.append(self.lock_ref.locked())

    def process_frame(self, *, rgb, depth_m, intrinsics):
        del rgb, depth_m, intrinsics
        self.started.append(time.monotonic())
        if self.blocker is not None:
            self.blocker.wait(timeout=2.0)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.2, -0.1, 0.5]
        self.finished.append(time.monotonic())
        return SimpleNamespace(
            pose=PoseEstimate(pose, "fake", {}),
            state=self.state,
            fresh_pose=True,
            held_pose=False,
            message=None,
            metadata={"fake": True},
        )


def fake_processing_runner(profile, options):
    assert options["profile"] == profile.name
    assert (profile.root / "recordings" / "session-1" / "session.json").exists()
    return {
        "ok": True,
        "returncode": 0,
        "mode": "process_recordings",
        "object": profile.name,
        "readiness": "ready",
        "accepted": 3,
        "required_keyframes": int(options["required_keyframes"]),
        "detector_preset": options["charuco_detector_preset"],
    }


def fake_segmentation_runner(request):
    assert request.prompt == "multimeter"
    assert request.rgb.shape == (4, 5, 3)
    return {
        "ok": True,
        "status": "segmented",
        "request_id": request.request_id,
        "prompt": request.prompt,
        "mask": {"area": 4, "area_fraction": 0.2, "shape": [4, 5], "box_xyxy": [2, 1, 4, 3]},
        "mask_png_b64": "fake-mask",
    }


def _serve(service):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def _post(url, body, content_type):
    request = urllib_request.Request(url, data=body, headers={"Content-Type": content_type})
    with urllib_request.urlopen(request, timeout=2.0) as response:
        return response.status, decode_foundationpose_response(response.read())


def _profile_with_mesh(tmp_path, *, name="phone"):
    profile = ObjectProfileRegistry(tmp_path).create(name)
    mesh_path = profile_model_path(profile)
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("# obj\n", encoding="utf-8")
    record_asset_ready(profile, generated_assets=[mesh_path])
    return profile


def _foundationpose_root_with_bundlesdf(tmp_path):
    server_root = tmp_path / "server-foundationpose"
    bundlesdf = server_root / "bundlesdf"
    bundlesdf.mkdir(parents=True)
    (bundlesdf / "run_nerf.py").write_text("# fake\n", encoding="utf-8")
    (bundlesdf / "config_ycbv.yml").write_text("fake: true\n", encoding="utf-8")
    return server_root


def _track_body(*, profile="phone", t5_T_camera=None, **metadata):
    transform = np.eye(4, dtype=np.float64) if t5_T_camera is None else t5_T_camera
    return encode_foundationpose_track_request(
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        depth_m=np.ones((4, 5), dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=10.0, fy=11.0, cx=2.0, cy=2.0, width=5, height=4),
        request_id="req-1",
        frame_index=3,
        capture_monotonic_ns=123,
        t5_T_camera=transform,
        profile=profile,
        foundationpose_root="/fp",
        refine_iterations=metadata.pop("refine_iterations", 5),
        track_iterations=metadata.pop("track_iterations", 2),
        reinit=metadata.pop("reinit", False),
        mask_options=metadata.pop("mask_options", {}),
        recovery_options=metadata.pop("recovery_options", {}),
        metadata=metadata,
    )


def _segmentation_body():
    return encode_foundationpose_segmentation_request(
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        depth_m=np.ones((4, 5), dtype=np.float32),
        request_id="seg-1",
        capture_monotonic_ns=456,
        prompt="multimeter",
        mask_options={"device": "cpu", "threshold": 0.3, "resolution": 128},
    )


def _recordings_archive(*, profile="phone"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "foundationpose_processing_request.json",
            json.dumps(
                {
                    "profile": profile,
                    "prompt": "multimeter",
                    "required_keyframes": 3,
                    "max_keyframes": 8,
                    "charuco_detector_preset": "conservative-charuco",
                }
            ),
        )
        zf.writestr("recordings/session-1/session.json", "{}")
        zf.writestr("recordings/session-1/frames.jsonl", "")
    return buffer.getvalue()


def _write_processing_debug_fixture(profile) -> None:
    cv2 = pytest.importorskip("cv2")
    session_dir = profile.root / "recordings" / "session-1"
    for dirname in ("rgb", "depth", "depth_mm", "intrinsics"):
        (session_dir / dirname).mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    with (session_dir / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for index in (0, 1):
            stem = f"{index:06d}"
            (session_dir / "rgb" / f"{stem}.png").write_bytes(b"rgb")
            np.save(
                session_dir / "depth" / f"{stem}.npy",
                np.array([[0.0, 0.10 + index], [0.20 + index, np.nan]], dtype=np.float32),
            )
            (session_dir / "depth_mm" / f"{stem}.png").write_bytes(b"depth-mm")
            (session_dir / "intrinsics" / f"{stem}.json").write_text("{}", encoding="utf-8")
            record = {
                "session_id": "session-1",
                "index": index,
                "timestamp_s": float(index),
                "rgb_path": f"rgb/{stem}.png",
                "depth_path": f"depth/{stem}.npy",
                "depth_mm_path": f"depth_mm/{stem}.png",
                "intrinsics_path": f"intrinsics/{stem}.json",
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    cache_dir = profile.root / "processing_cache" / "process-test"
    axes_dir = cache_dir / "charuco_axes"
    masks_dir = cache_dir / "masks"
    axes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    axes_image = np.full((3, 3, 3), 127, dtype=np.uint8)
    mask_selected = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    mask_unselected = np.ones((2, 2), dtype=np.uint8)
    assert cv2.imwrite(str(axes_dir / "selected.png"), axes_image)
    assert cv2.imwrite(str(axes_dir / "unselected.png"), axes_image)
    assert cv2.imwrite(str(masks_dir / "selected.png"), mask_selected)
    assert cv2.imwrite(str(masks_dir / "unselected.png"), mask_unselected)
    latest_report = {
        "object": profile.name,
        "run_id": "process-test",
        "readiness": "ready",
        "accepted": 2,
        "required_keyframes": 1,
        "thresholds": {"min_depth_m": 0.05, "max_depth_m": 1.0},
        "processing_cache_path": str(cache_dir),
        "records": [
            {
                "candidate_id": "session-1:000000",
                "session_id": "session-1",
                "frame_index": 0,
                "accepted": True,
                "selected_index": 0,
                "cached_mask_path": "masks/selected.png",
                "charuco_axes_preview_path": "charuco_axes/selected.png",
            },
            {
                "candidate_id": "session-1:000001",
                "session_id": "session-1",
                "frame_index": 1,
                "accepted": True,
                "selected_index": None,
                "cached_mask_path": "masks/unselected.png",
                "charuco_axes_preview_path": "charuco_axes/unselected.png",
            },
        ],
    }
    profile.logs_dir.mkdir(parents=True, exist_ok=True)
    (profile.logs_dir / "reference_processing_latest.json").write_text(
        json.dumps(latest_report, sort_keys=True),
        encoding="utf-8",
    )


def test_health_endpoint_returns_protocol_version(tmp_path):
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        with urllib_request.urlopen(f"{base_url}/foundationpose/v2/health", timeout=2.0) as response:
            payload = decode_foundationpose_response(response.read())
    finally:
        _stop(server, thread)

    assert payload["ok"] is True
    assert payload["protocol_version"] == 2
    assert payload["status"] == "ready"
    assert isinstance(payload["server_time_monotonic_ns"], int)


def test_debug_artifacts_endpoint_returns_selected_processing_candidates_only(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)
    profile = registry.create("phone")
    _write_processing_debug_fixture(profile)
    service = FoundationPoseV2Service(registry=registry, builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        with urllib_request.urlopen(f"{base_url}/foundationpose/v2/debug/phone", timeout=2.0) as response:
            data = response.read()
            headers = response.headers
    finally:
        _stop(server, thread)

    assert headers.get("Content-Type") == "application/zip"
    assert headers.get("X-FoundationPose-Debug-Candidate-Count") == "1"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        candidate = manifest["candidates"][0]
        mask_png = zf.read(candidate["mask"])
        depth_png = zf.read(candidate["depth_colormap"])
    assert manifest["profile"] == "phone"
    assert manifest["run_id"] == "process-test"
    assert manifest["candidate_count"] == 1
    assert candidate["candidate_id"] == "session-1:000000"
    assert candidate["charuco_axes"] in names
    assert candidate["mask"] in names
    assert candidate["depth_colormap"] in names
    assert all("000001" not in name for name in names)

    cv2 = pytest.importorskip("cv2")
    mask = cv2.imdecode(np.frombuffer(mask_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert mask.tolist() == [[0, 255], [255, 0]]
    assert depth[0, 0].tolist() == [0, 0, 0]
    assert depth[1, 1].tolist() == [0, 0, 0]


def test_debug_artifacts_endpoint_reports_missing_processing_cache(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)
    profile = registry.create("phone")
    profile.logs_dir.mkdir(parents=True, exist_ok=True)
    (profile.logs_dir / "reference_processing_latest.json").write_text(
        json.dumps({"run_id": "process-test", "processing_cache_path": str(profile.root / "missing"), "records": []}),
        encoding="utf-8",
    )
    service = FoundationPoseV2Service(registry=registry, builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        try:
            urllib_request.urlopen(f"{base_url}/foundationpose/v2/debug/phone", timeout=2.0)
            raise AssertionError("expected HTTP 404")
        except urllib_error.HTTPError as exc:
            assert exc.code == 404
            payload = decode_foundationpose_response(exc.read())
    finally:
        _stop(server, thread)

    assert payload["ok"] is False
    assert payload["status"] == "missing"
    assert "Processing cache not found" in payload["reason"]


def test_debug_artifacts_endpoint_reports_no_selected_candidates(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)
    profile = registry.create("phone")
    cache_dir = profile.root / "processing_cache" / "process-test"
    cache_dir.mkdir(parents=True)
    profile.logs_dir.mkdir(parents=True, exist_ok=True)
    (profile.logs_dir / "reference_processing_latest.json").write_text(
        json.dumps(
            {
                "run_id": "process-test",
                "processing_cache_path": str(cache_dir),
                "records": [
                    {
                        "candidate_id": "session-1:000000",
                        "session_id": "session-1",
                        "frame_index": 0,
                        "accepted": True,
                        "selected_index": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = FoundationPoseV2Service(registry=registry, builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        try:
            urllib_request.urlopen(f"{base_url}/foundationpose/v2/debug/phone", timeout=2.0)
            raise AssertionError("expected HTTP 409")
        except urllib_error.HTTPError as exc:
            assert exc.code == 409
            payload = decode_foundationpose_response(exc.read())
    finally:
        _stop(server, thread)

    assert payload["ok"] is False
    assert payload["status"] == "error"
    assert "no selected candidates" in payload["reason"]


def test_build_rejects_bad_content_type(tmp_path):
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        try:
            _post(f"{base_url}/foundationpose/v2/assets/build", b"{}", "application/octet-stream")
            raise AssertionError("expected HTTP error")
        except urllib_error.HTTPError as exc:
            assert exc.code == 415
    finally:
        _stop(server, thread)


def test_build_dry_run_and_async_status(tmp_path):
    _profile_with_mesh(tmp_path)
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        dry_status, dry_payload = _post(
            f"{base_url}/foundationpose/v2/assets/build",
            json.dumps(
                {"request_id": "build-1", "profile": "phone", "foundationpose_root": "/fp", "execute": False}
            ).encode("utf-8"),
            "application/json",
        )
        run_status, run_payload = _post(
            f"{base_url}/foundationpose/v2/assets/build",
            json.dumps(
                {"request_id": "build-2", "profile": "phone", "foundationpose_root": "/fp", "execute": True}
            ).encode("utf-8"),
            "application/json",
        )
        job_id = run_payload["job_id"]
        status_payload = None
        for _ in range(20):
            with urllib_request.urlopen(
                f"{base_url}/foundationpose/v2/assets/build/{job_id}", timeout=2.0
            ) as response:
                status_payload = decode_foundationpose_response(response.read())
            if status_payload["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)
    finally:
        _stop(server, thread)

    assert dry_status == 200
    assert dry_payload["ok"] is True
    assert dry_payload["executed"] is False
    assert run_status == 202
    assert run_payload["job_id"]
    assert status_payload is not None
    assert status_payload["state"] == "succeeded"
    assert len(status_payload["stdout_tail"]) <= 4000
    assert len(status_payload["stderr_tail"]) <= 4000
    assert status_payload["stdout_tail"].endswith("o")
    assert status_payload["stderr_tail"].endswith("e")


def test_model_asset_endpoint_downloads_generated_mesh(tmp_path):
    _profile_with_mesh(tmp_path)
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)
    server, thread, base_url = _serve(service)
    try:
        with urllib_request.urlopen(f"{base_url}/foundationpose/v2/assets/model/phone", timeout=2.0) as response:
            data = response.read()
            headers = response.headers
    finally:
        _stop(server, thread)

    assert data == b"# obj\n"
    assert headers.get("Content-Type") == "application/octet-stream"
    assert headers.get("X-FoundationPose-Profile") == "phone"
    assert headers.get("X-FoundationPose-Mesh-Size") == "6"


def test_build_uses_server_foundationpose_root_when_client_path_is_invalid(tmp_path, monkeypatch):
    server_root = _foundationpose_root_with_bundlesdf(tmp_path)
    monkeypatch.setenv("FOUNDATIONPOSE_ROOT", str(server_root))

    _profile_with_mesh(tmp_path)
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)

    status, payload = service.build_assets(
        {
            "request_id": "build-1",
            "profile": "phone",
            "foundationpose_root": "/client-only/FoundationPose",
            "execute": False,
        }
    )

    assert status == 200
    assert payload["ok"] is True
    assert payload["command"] == ["fake-build", "phone", str(server_root.resolve())]


def test_tracking_uses_server_foundationpose_root_when_client_path_is_invalid(tmp_path, monkeypatch):
    server_root = _foundationpose_root_with_bundlesdf(tmp_path)
    monkeypatch.setenv("FOUNDATIONPOSE_ROOT", str(server_root))
    _profile_with_mesh(tmp_path)
    seen_roots = []

    def factory(profile, mesh, request):
        del profile, mesh
        seen_roots.append(request.foundationpose_root)
        return FakeTracker()

    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        tracker_factory=factory,
    )

    status, payload = service.track(decode_foundationpose_track_request(_track_body()))

    assert status == 200
    assert payload["ok"] is True
    assert seen_roots == [str(server_root.resolve())]


def test_default_asset_builder_uses_build_python_override(tmp_path, monkeypatch):
    build_python = tmp_path / "fpbuild-python"
    monkeypatch.setenv("FOUNDATIONPOSE_BUILD_PYTHON", str(build_python))

    builder = _make_asset_builder("/fp")

    assert builder.foundationpose_root.as_posix() == "/fp"
    assert builder.python_executable == str(build_python)


def test_build_missing_fields_and_unknown_profile(tmp_path):
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)

    status, payload = service.build_assets({"request_id": "bad"})
    assert status == 400
    assert "profile" in payload["reason"]

    status, payload = service.build_assets(
        {"request_id": "bad", "profile": "missing", "foundationpose_root": "/fp", "execute": False}
    )
    assert status == 404
    assert "Object profile not found" in payload["reason"]


def test_process_recordings_upload_creates_profile_and_polls_job(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)
    service = FoundationPoseV2Service(
        registry=registry,
        builder_factory=FakeBuilder,
        processing_runner=fake_processing_runner,
    )
    server, thread, base_url = _serve(service)
    try:
        run_status, run_payload = _post(
            f"{base_url}/foundationpose/v2/recordings/process",
            _recordings_archive(profile="meter"),
            RECORDINGS_ZIP_CONTENT_TYPE,
        )
        job_id = run_payload["job_id"]
        status_payload = None
        for _ in range(20):
            with urllib_request.urlopen(
                f"{base_url}/foundationpose/v2/recordings/process/{job_id}", timeout=2.0
            ) as response:
                status_payload = decode_foundationpose_response(response.read())
            if status_payload["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)
    finally:
        _stop(server, thread)

    assert run_status == 202
    assert run_payload["profile"] == "meter"
    assert run_payload["upload"]["session_count"] == 1
    assert (registry.root / "meter" / "recordings" / "session-1" / "session.json").exists()
    assert status_payload is not None
    assert status_payload["state"] == "succeeded"
    assert status_payload["result"]["readiness"] == "ready"
    assert status_payload["result"]["accepted"] == 3


def test_segmentation_protocol_round_trip():
    request = decode_foundationpose_segmentation_request(_segmentation_body())

    assert request.request_id == "seg-1"
    assert request.prompt == "multimeter"
    assert request.rgb.shape == (4, 5, 3)
    assert request.depth_m.shape == (4, 5)
    assert request.mask_options["device"] == "cpu"
    assert request.mask_options["resolution"] == 128


def test_segmentation_endpoint_runs_server_side_mask(tmp_path):
    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        segmentation_runner=fake_segmentation_runner,
    )
    server, thread, base_url = _serve(service)
    try:
        status, payload = _post(
            f"{base_url}/foundationpose/v2/segmentation",
            _segmentation_body(),
            REQUEST_CONTENT_TYPE,
        )
    finally:
        _stop(server, thread)

    assert status == 200
    assert payload["ok"] is True
    assert payload["status"] == "segmented"
    assert payload["request_id"] == "seg-1"
    assert payload["prompt"] == "multimeter"
    assert payload["mask"]["area"] == 4
    assert payload["mask_png_b64"] == "fake-mask"


def test_segmentation_rejects_bad_content_type(tmp_path):
    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        segmentation_runner=fake_segmentation_runner,
    )
    server, thread, base_url = _serve(service)
    try:
        try:
            _post(f"{base_url}/foundationpose/v2/segmentation", b"bad", "application/octet-stream")
            raise AssertionError("expected HTTP error")
        except urllib_error.HTTPError as exc:
            assert exc.code == 415
    finally:
        _stop(server, thread)


def test_track_rejects_bad_content_type(tmp_path):
    _profile_with_mesh(tmp_path)
    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        tracker_factory=lambda profile, mesh, request: FakeTracker(),
    )
    server, thread, base_url = _serve(service)
    try:
        try:
            _post(f"{base_url}/foundationpose/v2/track", b"bad", "application/octet-stream")
            raise AssertionError("expected HTTP error")
        except urllib_error.HTTPError as exc:
            assert exc.code == 415
    finally:
        _stop(server, thread)


def test_track_success_computes_t5_pose_and_stays_pose_only(tmp_path):
    _profile_with_mesh(tmp_path)
    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        tracker_factory=lambda profile, mesh, request: FakeTracker(),
    )
    t5_T_camera = np.eye(4, dtype=np.float64)
    t5_T_camera[:3, 3] = [1.0, 2.0, 3.0]

    status, payload = service.track(
        decode_foundationpose_track_request(_track_body(t5_T_camera=t5_T_camera))
    )

    assert status == 200
    assert payload["ok"] is True
    camera_T_object = np.asarray(payload["camera_T_object"], dtype=np.float64)
    t5_T_object = np.asarray(payload["t5_T_object"], dtype=np.float64)
    np.testing.assert_allclose(t5_T_object, t5_T_camera @ camera_T_object)
    assert payload["tracking_state"] == "TRACKING"
    for forbidden in ("action", "servo_step", "command_recommended", "target_t5_T_ee", "address", "power", "servo"):
        assert forbidden not in payload


def test_track_unknown_profile_and_missing_mesh(tmp_path):
    service = FoundationPoseV2Service(registry=ObjectProfileRegistry(tmp_path), builder_factory=FakeBuilder)
    request = decode_foundationpose_track_request(_track_body(profile="missing"))

    status, payload = service.track(request)
    assert status == 404
    assert "Object profile not found" in payload["message"]

    ObjectProfileRegistry(tmp_path).create("phone")
    request = decode_foundationpose_track_request(_track_body(profile="phone"))
    status, payload = service.track(request)
    assert status == 409
    assert "generated mesh" in payload["message"]


def test_tracker_cache_reuses_and_invalidates_by_options(tmp_path):
    _profile_with_mesh(tmp_path)
    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        tracker_factory=lambda profile, mesh, request: FakeTracker(),
    )
    _, first = service.track(decode_foundationpose_track_request(_track_body(mask_options={"threshold": 0.3})))
    _, second = service.track(decode_foundationpose_track_request(_track_body(mask_options={"threshold": 0.3})))
    _, changed = service.track(decode_foundationpose_track_request(_track_body(mask_options={"threshold": 0.5})))

    assert first["tracker_session_id"] == second["tracker_session_id"]
    assert first["tracker_session_id"] != changed["tracker_session_id"]


def test_tracker_reinit_flag_calls_tracker(tmp_path):
    _profile_with_mesh(tmp_path)
    trackers = []

    def factory(profile, mesh, request):
        tracker = FakeTracker()
        trackers.append(tracker)
        return tracker

    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        tracker_factory=factory,
    )
    service.track(decode_foundationpose_track_request(_track_body(reinit=True)))

    assert trackers[0].request_reinit_calls == 1


def test_reinit_request_is_serialized_by_session_lock(tmp_path):
    _profile_with_mesh(tmp_path)
    trackers = []

    def factory(profile, mesh, request):
        tracker = FakeTracker()
        trackers.append(tracker)
        return tracker

    service = FoundationPoseV2Service(
        registry=ObjectProfileRegistry(tmp_path),
        builder_factory=FakeBuilder,
        tracker_factory=factory,
    )

    service.track(decode_foundationpose_track_request(_track_body(reinit=False)))
    session = next(iter(service.tracker_sessions._sessions.values()))
    trackers[0].lock_ref = session.lock

    service.track(decode_foundationpose_track_request(_track_body(reinit=True)))

    assert trackers[0].request_reinit_calls == 1
    assert trackers[0].reinit_lock_states == [True]


def test_mesh_identity_changes_when_mesh_changes(tmp_path):
    profile = _profile_with_mesh(tmp_path)
    mesh_path = profile_model_path(profile)
    first = mesh_identity(mesh_path)
    mesh_path.write_text("# obj\nv 0 0 0\n", encoding="utf-8")
    second = mesh_identity(mesh_path)

    assert first != second
