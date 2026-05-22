from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from visual_servoing.point_pose import zed_camera
from visual_servoing.point_pose.zed_camera import ZedCamera, ZedUnavailableError


def test_missing_zed_sdk_has_helpful_error(monkeypatch):
    def missing_sdk():
        raise ZedUnavailableError("missing optional backend")

    monkeypatch.setattr(zed_camera, "_import_sl", missing_sdk)

    with pytest.raises(ZedUnavailableError, match="missing optional backend"):
        ZedCamera().start()


def test_zed_camera_start_read_stop_uses_neural_depth_confidence_and_left_rgb(monkeypatch):
    fake_sl, cameras = make_fake_sl()
    monkeypatch.setattr(zed_camera, "_import_sl", lambda: fake_sl)

    camera = ZedCamera(width=1280, height=720)
    camera.start()
    frame = camera.read()
    camera.stop()

    fake_camera = cameras[0]
    assert fake_camera.init_params.coordinate_units == fake_sl.UNIT.METER
    assert fake_camera.init_params.depth_mode == fake_sl.DEPTH_MODE.NEURAL
    assert fake_camera.init_params.camera_fps == 15
    assert fake_camera.init_params.camera_resolution == fake_sl.RESOLUTION.HD720
    assert fake_camera.runtime_params.confidence_threshold == 95
    assert fake_camera.image_views == [fake_sl.VIEW.LEFT]
    assert fake_camera.measure_views == [fake_sl.MEASURE.DEPTH]
    assert fake_camera.closed is True
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.shape == (2, 3, 3)
    assert frame.rgb[0, 0].tolist() == [10, 20, 30]
    assert frame.depth_m.dtype == np.float32
    assert frame.depth_m.shape == (2, 3)
    assert frame.intrinsics.fx == 100.0
    assert frame.intrinsics.fy == 101.0
    assert frame.intrinsics.cx == 1.5
    assert frame.intrinsics.cy == 1.0
    assert frame.intrinsics.width == 3
    assert frame.intrinsics.height == 2


def test_zed_camera_rejects_shape_mismatch(monkeypatch):
    fake_sl, _ = make_fake_sl(depth_shape=(4, 3))
    monkeypatch.setattr(zed_camera, "_import_sl", lambda: fake_sl)

    camera = ZedCamera()
    camera.start()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        camera.read()


def make_fake_sl(*, depth_shape: tuple[int, int] = (2, 3)):
    cameras = []

    class InitParameters:
        def __init__(self):
            self.coordinate_units = None
            self.depth_mode = None
            self.camera_fps = None
            self.camera_resolution = None
            self.serial = None

        def set_from_serial_number(self, serial):
            self.serial = serial

    class RuntimeParameters:
        def __init__(self):
            self.confidence_threshold = None

    class Mat:
        def __init__(self):
            self.data = None

        def get_data(self):
            return self.data

    class Camera:
        def __init__(self):
            self.init_params = None
            self.runtime_params = None
            self.image_views = []
            self.measure_views = []
            self.closed = False
            cameras.append(self)

        def open(self, init_params):
            self.init_params = init_params
            return 0

        def grab(self, runtime_params):
            self.runtime_params = runtime_params
            return 0

        def retrieve_image(self, mat, view, mem=None):
            self.image_views.append(view)
            mat.data = np.array(
                [
                    [[30, 20, 10, 255], [60, 50, 40, 255], [90, 80, 70, 255]],
                    [[3, 2, 1, 255], [6, 5, 4, 255], [9, 8, 7, 255]],
                ],
                dtype=np.uint8,
            )

        def retrieve_measure(self, mat, measure, mem=None):
            self.measure_views.append(measure)
            mat.data = np.ones(depth_shape, dtype=np.float32)

        def get_camera_information(self):
            image_size = SimpleNamespace(width=3, height=2)
            left_cam = SimpleNamespace(
                fx=100.0,
                fy=101.0,
                cx=1.5,
                cy=1.0,
                image_size=image_size,
                disto=np.zeros(5, dtype=np.float32),
            )
            calibration = SimpleNamespace(left_cam=left_cam)
            camera_config = SimpleNamespace(calibration_parameters=calibration)
            return SimpleNamespace(camera_configuration=camera_config)

        def close(self):
            self.closed = True

        @staticmethod
        def get_device_list():
            return []

    fake_sl = SimpleNamespace(
        Camera=Camera,
        InitParameters=InitParameters,
        RuntimeParameters=RuntimeParameters,
        Mat=Mat,
        UNIT=SimpleNamespace(METER="meter"),
        DEPTH_MODE=SimpleNamespace(NEURAL="neural"),
        RESOLUTION=SimpleNamespace(VGA="vga", HD720="hd720"),
        VIEW=SimpleNamespace(LEFT="left"),
        MEASURE=SimpleNamespace(DEPTH="depth"),
        MEM=SimpleNamespace(CPU="cpu"),
        ERROR_CODE=SimpleNamespace(SUCCESS=0),
    )
    return fake_sl, cameras
