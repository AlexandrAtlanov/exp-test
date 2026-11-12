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
    """model.pt: «kurgany» → оба класса курганов, gorodishcha как есть, «dorogi» — отброс."""
    names = {0: "gorodishcha", 4: "kurgany", 5: "dorogi"}
    found = yolo_runtime._instances(_fake_result([0, 4, 5]), (4, 4), names)
    assert [inst.class_name for inst in found] == [
        "gorodishcha",
        "kurgany_tselye",
        "kurgany_povrezhdennye",
    ]


def test_passport_sets_tiling_for_own_model(tmp_path) -> None:
    """Паспорт .pt.json (наша модель): тайл из паспорта, масштаб (1,); нет паспорта — config."""
    pt = tmp_path / "lidar_yolo.pt"
    assert yolo_runtime._passport_tiling(pt) == (None, None)  # model.pt бейзлайна
    pt.with_suffix(".pt.json").write_text('{"tile_px": 1024, "imgsz": 512}', encoding="utf-8")
    assert yolo_runtime._passport_tiling(pt) == (1024, (1.0,))
    pt.with_suffix(".pt.json").write_text('{"tile_px": 2048, "scales": [1, 2]}', encoding="utf-8")
    assert yolo_runtime._passport_tiling(pt) == (2048, (1.0, 2.0))
    assert yolo_runtime._passport_rot90(pt) is False
    pt.with_suffix(".pt.json").write_text('{"tile_px": 512, "rot90": true}', encoding="utf-8")
    assert yolo_runtime._passport_rot90(pt) is True
    pt.with_suffix(".pt.json").write_text("не json", encoding="utf-8")
    assert yolo_runtime._passport_tiling(pt) == (None, None)
    assert yolo_runtime._passport_rot90(pt) is False


class _BlobModel:
    """Заглушка YOLO: на каждый вход — одна маска «яркие пиксели», запоминает входы."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(self, source, **kwargs):
        self.calls.append(len(source))
        results = []
        for image in source:
            result = type("Result", (), {})()
            mask = (image[..., 0] > 200).astype(np.float32)[None]
            result.masks = type("Masks", (), {"data": _Array(mask)})()
            result.boxes = type(
                "Boxes",
                (),
                {"cls": _Array(np.zeros(1, np.float32)), "conf": _Array(np.full(1, 0.8))},
            )()
            results.append(result)
        return results


def test_rot90_tta_masks_return_to_tile_frame() -> None:
    """TTA: 4 предсказания на тайл, каждая маска повёрнута обратно и совпадает с пятном."""
    import pytest

    pytest.importorskip("torch")
    tile = np.zeros((32, 32, 3), np.uint8)
    tile[3:9, 20:30] = 255  # несимметричное пятно: ошибка поворота сразу видна
    model = _BlobModel()
    loaded = yolo_runtime.LoadedModel(model, "pt", {0: "gorodishcha"}, 8, 32)
    [plain] = list(yolo_runtime.predict_tiles(loaded, [tile]))
    [tta] = list(yolo_runtime.predict_tiles(loaded, [tile], rot90_tta=True))
    assert model.calls == [1, 4]
    assert len(plain) == 1 and len(tta) == 4
    for inst in tta:
        assert (inst.mask == plain[0].mask).all()
