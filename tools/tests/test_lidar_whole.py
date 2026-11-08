"""Режим «вся отмывка целиком» (как бейзлайн): сжатие, выравнивание и обратная привязка.

Главная проверка — объект, найденный на сжатом и повёрнутом кадре, возвращается
в своё место на исходной отмывке (у бейзлайна обратный поворот перепутан по знаку).
"""

from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from koz3 import config, yolo_runtime
from koz3.datatypes import Region
from koz3.detectors import lidar, lidar_dem, lidar_whole
from koz3.timing import Budget
from koz3.yolo_runtime import TileInstance

UTM = "EPSG:32637"
RASTER_T = Affine(0.1, 0.0, 500000.0, 0.0, -0.1, 6000000.0)
"""Отмывка 0.1 м/пикс, как в бейзлайне."""

BLOB_RC = (900, 2100)
"""Центр «объекта» на исходной отмывке (строка, столбец)."""


def _synthetic_hillshade(angle_deg: float) -> np.ndarray:
    """Отмывка 2000x3000: повёрнутая область данных (серый) на чёрном, белое пятно-объект."""
    h, w = 2000, 3000
    gray = np.zeros((h, w), np.uint8)
    box = cv2.boxPoints(((1500.0, 1000.0), (2400.0, 1200.0), angle_deg)).astype(np.int32)
    cv2.fillPoly(gray, [box], 128)
    cv2.circle(gray, (BLOB_RC[1], BLOB_RC[0]), 40, 255, -1)
    return np.repeat(gray[..., None], 3, axis=-1)


def _blob_mask(frame: np.ndarray) -> np.ndarray:
    return frame[..., 0] > 200


@pytest.mark.parametrize("angle_deg", [0.0, 12.0, 35.0, -20.0])
def test_frame_maps_back_to_source_pixel(angle_deg: float) -> None:
    image = _synthetic_hillshade(angle_deg)
    frame, scale = lidar_whole.downscale_image(image, 1024)
    assert max(frame.shape[:2]) == 1024
    frame, angle, offset, center = lidar_whole.align_and_crop_image(frame)
    rows, cols = np.nonzero(_blob_mask(frame))
    assert rows.size, "пятно потерялось при подготовке кадра"
    to_src = lidar_whole.frame_to_raster(Affine.identity(), scale, angle, offset, center)
    x, y = to_src * (cols.mean() + 0.5, rows.mean() + 0.5)
    # допуск — 2 пикселя кадра (≈ 6 пикселей исходника при сжатии ×2.9)
    assert abs(x - (BLOB_RC[1] + 0.5)) < 2 / scale
    assert abs(y - (BLOB_RC[0] + 0.5)) < 2 / scale


def test_no_downscale_for_small_image() -> None:
    image = np.full((300, 500, 3), 128, np.uint8)
    frame, scale = lidar_whole.downscale_image(image, 1024)
    assert scale == 1.0 and frame.shape == image.shape


def test_detector_whole_mode_places_object_in_utm(tmp_path: Path, monkeypatch) -> None:
    """Детектор в режиме whole: один кадр на отмывку, полигон — на месте объекта в UTM."""
    image = _synthetic_hillshade(25.0)

    def fake_hillshades(las_path, region_crs, out_dir, variants=None):
        path = Path(out_dir) / "hs_c.tif"
        profile = dict(driver="GTiff", height=image.shape[0], width=image.shape[1], count=3)
        profile.update(dtype="uint8", crs=UTM, transform=RASTER_T)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.transpose(image, (2, 0, 1)))
        return [path]

    seen: list[tuple[tuple[int, ...], int | None]] = []

    def fake_predict(loaded, tiles, imgsz=None):
        for frame in tiles:
            seen.append((frame.shape, imgsz))
            yield [TileInstance("kurgany_tselye", 0.9, _blob_mask(frame))]

    monkeypatch.setattr(lidar_dem, "hillshades_for_las", fake_hillshades)
    monkeypatch.setattr(yolo_runtime, "predict_tiles", fake_predict)
    monkeypatch.setattr(
        yolo_runtime,
        "load_model",
        lambda weights_dir=None: yolo_runtime.LoadedModel(object(), "stub", {}, 8),
    )
    monkeypatch.setattr(lidar, "_model", None)
    monkeypatch.setattr(lidar, "_model_error", "")
    monkeypatch.setitem(config.LIDAR, "infer_mode", "whole")
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c",))

    las = tmp_path / "r" / "a.las"
    las.parent.mkdir()
    las.write_bytes(b"0" * 1024)
    region = Region(name="r", path=las.parent, utm_crs=UTM, files={"li": [las]})
    detections = lidar.run(region, Budget(600), config)

    assert len(seen) == 1  # одна отмывка — один кадр
    shape, imgsz = seen[0]
    assert max(shape[:2]) <= 1024 and imgsz == config.LIDAR["whole_px"]
    [det] = detections
    expected_x, expected_y = RASTER_T * (BLOB_RC[1] + 0.5, BLOB_RC[0] + 0.5)
    centroid = det.geom.centroid
    assert abs(centroid.x - expected_x) < 1.0  # метры; объект радиусом 4 м
    assert abs(centroid.y - expected_y) < 1.0
    assert 30 < det.geom.area < 75  # круг r=4 м ≈ 50 м²
    assert det.extra == {"edge_cut": False}
