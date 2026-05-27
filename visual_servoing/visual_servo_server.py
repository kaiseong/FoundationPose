"""Stdlib HTTP server for remote visual-servo action planning."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import time
from typing import Any, Callable

import numpy as np

from visual_servoing.point_pose.sam3_phone_segmenter import Sam3PhoneSegmenter
from visual_servoing.visual_servo_core import (
    DEFAULT_RIGHT_ARM_EE_LINK,
    POSITION_ONLY_ORIENTATION_POLICY,
    REMOTE_ACTION_CONTROL_MODE,
    REMOTE_OFFSET_FRAME,
    RIGHT_ARM_CONTROL_ROOT_LINK,
    RIGHT_ARM_EE_LINKS,
    ServoLimits,
    estimate_visual_observation,
    matrix_list,
    plan_t5_position_servo_action,
    require_vector3,
    to_list,
)
from visual_servoing.visual_servo_protocol import (
    DEFAULT_MAX_CONTENT_LENGTH,
    REQUEST_CONTENT_TYPE,
    RESPONSE_CONTENT_TYPE,
    VisualServoRequest,
    decode_visual_servo_request,
    encode_visual_servo_response,
)


MASK_PREVIEW_ENCODING = "packbits-b64-v1"


def encode_mask_preview(mask: np.ndarray) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    flat = mask_bool.reshape(-1).astype(np.uint8)
    packed = np.packbits(flat)
    return {
        "encoding": MASK_PREVIEW_ENCODING,
        "shape": [int(mask_bool.shape[0]), int(mask_bool.shape[1])],
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


class VisualServoService:
    def __init__(
        self,
        *,
        prompt: str = "object",
        device: str = "cuda",
        threshold: float = 0.5,
        sam_resolution: int = 1008,
        segmenter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.prompt = prompt
        self.device = device
        self.threshold = threshold
        self.sam_resolution = sam_resolution
        self._segmenter_factory = segmenter_factory
        self._segmenter = None

    def handle(self, request: VisualServoRequest) -> dict[str, Any]:
        server_received_ns = time.monotonic_ns()
        timing_ms: dict[str, float] = {}
        try:
            metadata = request.metadata
            ee_link = str(metadata.get("ee_link", DEFAULT_RIGHT_ARM_EE_LINK))
            if ee_link not in RIGHT_ARM_EE_LINKS:
                allowed = ", ".join(sorted(RIGHT_ARM_EE_LINKS))
                raise ValueError(f"ee_link {ee_link!r} is not an allowed right-arm EE link; allowed: {allowed}")

            segment_start = time.perf_counter()
            selection = self._segmenter_instance().segment(request.rgb)
            timing_ms["segmentation_ms"] = (time.perf_counter() - segment_start) * 1000.0

            plan_start = time.perf_counter()
            limits = ServoLimits(
                max_translation_step_m=float(metadata.get("max_translation_step_m", 0.01)),
                max_wrist_step_rad=math.radians(float(metadata.get("max_wrist_step_deg", 5.0))),
                position_tolerance_m=float(metadata.get("position_tolerance_m", 0.005)),
                wrist_tolerance_rad=math.radians(float(metadata.get("wrist_tolerance_deg", 2.0))),
            )
            observation = estimate_visual_observation(
                request.depth_m,
                selection.mask,
                request.intrinsics,
                t5_T_camera=request.t5_T_camera,
                previous_transform=None,
                min_depth_m=float(metadata.get("min_depth_m", 0.05)),
                max_depth_m=float(metadata.get("max_depth_m", 2.0)),
            )
            target_offset_t5 = require_vector3(
                metadata.get("target_offset_t5_m", metadata.get("target_offset_t5", [0.0, 0.0, 0.0])),
                "target_offset_t5",
            )
            step = plan_t5_position_servo_action(
                current_t5_T_ee=request.current_t5_T_ee,
                object_centroid_t5=observation.t5_T_object[:3, 3],
                target_offset_t5=target_offset_t5,
                limits=limits,
            )
            timing_ms["planning_ms"] = (time.perf_counter() - plan_start) * 1000.0
            server_completed_ns = time.monotonic_ns()
            mask_payload: dict[str, Any] = {
                "index": int(selection.index),
                "score": float(selection.score),
                "area": int(selection.area),
                "box_xyxy": selection.box_xyxy,
            }
            if bool(metadata.get("return_mask_preview", False)):
                mask_payload["preview"] = encode_mask_preview(selection.mask)
            return {
                "ok": True,
                "status": step.status,
                "request_id": request.request_id,
                "frame_index": request.frame_index,
                "server_received_monotonic_ns": server_received_ns,
                "server_completed_monotonic_ns": server_completed_ns,
                "server_timing_ms": timing_ms,
                "offset_frame": REMOTE_OFFSET_FRAME,
                "orientation_policy": POSITION_ONLY_ORIENTATION_POLICY,
                "target_offset_t5_m": to_list(target_offset_t5),
                "action": {
                    "root_link": RIGHT_ARM_CONTROL_ROOT_LINK,
                    "ee_link": ee_link,
                    "control_mode": REMOTE_ACTION_CONTROL_MODE,
                    "target_t5_T_ee": matrix_list(step.target_t5_T_ee),
                    "command_recommended": bool(step.command_recommended),
                    "offset_frame": REMOTE_OFFSET_FRAME,
                    "orientation_policy": POSITION_ONLY_ORIENTATION_POLICY,
                },
                "observation": {
                    "masked_points": observation.masked_points,
                    "centroid_camera_m": to_list(observation.centroid_camera_m),
                    "object_long_axis_t5": to_list(observation.object_long_axis_t5),
                    "object_grasp_axis_t5": to_list(observation.object_grasp_axis_t5),
                    "camera_T_object": matrix_list(observation.camera_T_object),
                    "t5_T_object": matrix_list(observation.t5_T_object),
                },
                "servo_step": {
                    "desired_position_t5_m": to_list(step.desired_position_t5_m),
                    "position_error_m": to_list(step.position_error_m),
                    "translation_step_m": to_list(step.translation_step_m),
                    "wrist_error_rad": step.wrist_error_rad,
                    "wrist_step_rad": step.wrist_step_rad,
                    "ignored_offset_rpy_deg": list(step.ignored_offset_rpy_deg),
                    "target_offset_t5_m": to_list(target_offset_t5),
                    "offset_frame": REMOTE_OFFSET_FRAME,
                    "orientation_policy": POSITION_ONLY_ORIENTATION_POLICY,
                    "current_t5_T_ee": matrix_list(step.current_t5_T_ee),
                    "target_t5_T_ee": matrix_list(step.target_t5_T_ee),
                },
                "mask": mask_payload,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "skipped",
                "request_id": request.request_id,
                "frame_index": request.frame_index,
                "server_received_monotonic_ns": server_received_ns,
                "server_completed_monotonic_ns": time.monotonic_ns(),
                "server_timing_ms": timing_ms,
                "reason": str(exc),
            }

    def _segmenter_instance(self):
        if self._segmenter is None:
            if self._segmenter_factory is not None:
                self._segmenter = self._segmenter_factory()
            else:
                self._segmenter = Sam3PhoneSegmenter(
                    prompt=self.prompt,
                    device=self.device,
                    confidence_threshold=self.threshold,
                    resolution=self.sam_resolution,
                )
        return self._segmenter


def make_handler(service: VisualServoService, *, max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH):
    class VisualServoHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/visual-servo/action":
                self._send_json(404, {"ok": False, "status": "error", "reason": "not found"})
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != REQUEST_CONTENT_TYPE:
                self._send_json(415, {"ok": False, "status": "error", "reason": "unsupported content type"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(411, {"ok": False, "status": "error", "reason": "invalid content length"})
                return
            if content_length <= 0:
                self._send_json(400, {"ok": False, "status": "error", "reason": "empty request body"})
                return
            if content_length > int(max_content_length):
                self._send_json(413, {"ok": False, "status": "error", "reason": "request body too large"})
                return
            try:
                body = self.rfile.read(content_length)
                request = decode_visual_servo_request(body, max_content_length=max_content_length)
            except Exception as exc:
                self._send_json(400, {"ok": False, "status": "error", "reason": str(exc)})
                return
            self._send_json(200, service.handle(request))

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            data = encode_visual_servo_response(payload)
            self.send_response(status_code)
            self.send_header("Content-Type", RESPONSE_CONTENT_TYPE)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return VisualServoHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote visual-servo action server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument("--max-content-length", type=int, default=DEFAULT_MAX_CONTENT_LENGTH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    service = VisualServoService(
        prompt=args.prompt,
        device=args.device,
        threshold=args.threshold,
        sam_resolution=args.sam_resolution,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(service, max_content_length=args.max_content_length))
    print(json.dumps({"event": "visual_servo_server_listening", "host": args.host, "port": args.port}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
