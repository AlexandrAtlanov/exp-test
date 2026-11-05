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


class _Array:
    """Минимальная замена тензора ultralytics: .cpu().numpy()."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def cpu(self) -> "_Array":
        return self

    def numpy(self) -> np.ndarray:
        return self.data


def _fake_result(class_ids: list[int]) -> object:
    masks = np.ones((len(class_ids), 4, 4), dtype=np.float32)
    result = type("Result", (), {})()
    result.masks = type("Masks", (), {"data": _Array(masks)})()
    result.boxes = type(
        "Boxes",
        (),
        {
            "cls": _Array(np.array(class_ids, dtype=np.float32)),
            "conf": _Array(np.full(len(class_ids), 0.9, dtype=np.float32)),
        },
    )()
    return result


def test_baseline_class_names_mapped_and_unknown_dropped() -> None:
    """Модель бейзлайна: «kurgany» → kurgany_tselye, gorodishcha как есть, «dorogi» — отброс."""
    names = {0: "gorodishcha", 4: "kurgany", 5: "dorogi"}
    found = yolo_runtime._instances(_fake_result([0, 4, 5]), (4, 4), names)
    assert [inst.class_name for inst in found] == ["gorodishcha", "kurgany_tselye"]
