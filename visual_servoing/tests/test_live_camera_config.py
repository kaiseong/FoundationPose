from __future__ import annotations

import pytest

from visual_servoing.point_pose.live_camera_config import (
    SUPPORTED_LIVE_CAMERA_MODELS,
    resolve_live_camera_config,
)


def test_supported_live_camera_models_include_zed():
    assert SUPPORTED_LIVE_CAMERA_MODELS == ("d405", "d435", "zed")


def test_realsense_omitted_dimensions_resolve_to_validated_default():
    config = resolve_live_camera_config(model="d435")

    assert config.model == "d435"
    assert config.width == 640
    assert config.height == 480
    assert config.fps == 15
    assert config.sdk_resolution is None


def test_zed_omitted_dimensions_use_native_vga_mode_without_forcing_pixels():
    config = resolve_live_camera_config(model="zed")

    assert config.model == "zed"
    assert config.width is None
    assert config.height is None
    assert config.fps == 15
    assert config.sdk_resolution == "VGA"
    assert config.native_resolution is True


def test_zed_explicit_supported_dimensions_map_to_sdk_mode():
    config = resolve_live_camera_config(model="zed", width=1280, height=720, fps=30)

    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30
    assert config.sdk_resolution == "HD720"


def test_zed_rejects_unsupported_dimensions_with_supported_modes():
    with pytest.raises(ValueError, match="unsupported ZED resolution 640x480"):
        resolve_live_camera_config(model="zed", width=640, height=480)


def test_dimensions_must_be_omitted_or_provided_together():
    with pytest.raises(ValueError, match="width and height"):
        resolve_live_camera_config(model="d405", width=640, height=None)
