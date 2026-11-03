"""Тесты лёгких функций koz3.yolo_runtime (без torch и ultralytics)."""

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import numpy as np
from koz3 import yolo_runtime


def test_to_model_image_gray_to_rgb() -> None:
    gray = np.arange(12, dtype=np.uint8).reshape(1, 3, 4)
    rgb = yolo_runtime.to_model_image(gray)
    assert rgb.shape == (3, 4, 3) and rgb.dtype == np.uint8
    assert (rgb[..., 0] == rgb[..., 2]).all()


def test_to_model_image_drops_alpha_and_normalizes() -> None:
    rgba = np.zeros((4, 5, 6), dtype=np.float32)
    rgba[0, 0, 0] = 10.0
    rgb = yolo_runtime.to_model_image(rgba)
    assert rgb.shape == (5, 6, 3)
    assert rgb.max() == 255


def test_tile_positions_cover_edges() -> None:
    assert yolo_runtime.tile_positions(500, 1024, 0.3) == [0]
    positions = yolo_runtime.tile_positions(3000, 1024, 0.3)
    assert positions[0] == 0 and positions[-1] == 3000 - 1024
    assert all(b - a <= 1024 for a, b in zip(positions, positions[1:], strict=False))
