"""Тайлы на нескольких масштабах (LIDAR["scales"]): объект на сжатой отмывке — на своём месте."""

from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine

from koz3 import config, yolo_runtime
from koz3.datatypes import Region
from koz3.detectors import lidar, lidar_dem
from koz3.timing import Budget
from koz3.yolo_runtime import TileInstance

UTM = "EPSG:32637"
RASTER_T = Affine(0.1, 0.0, 500000.0, 0.0, -0.1, 6000000.0)
BLOB_RC = (700, 900)  # центр пятна на исходной отмывке, пиксели


def test_scales_place_object_and_shrink_input(tmp_path: Path, monkeypatch) -> None:
    image = np.full((1200, 1600), 128, np.uint8)
    cv2.circle(image, (BLOB_RC[1], BLOB_RC[0]), 40, 255, -1)

    def fake_hillshades(las_path, region_crs, out_dir, variants=None):
        path = Path(out_dir) / "hs_c.tif"
        profile = dict(driver="GTiff", height=1200, width=1600, count=1, dtype="uint8")
        with rasterio.open(path, "w", crs=UTM, transform=RASTER_T, **profile) as dst:
            dst.write(image, 1)
        return [path]

    sizes: list[tuple[int, int]] = []

    def fake_predict(loaded, tiles, imgsz=None):
        for tile in tiles:
            sizes.append(tile.shape[:2])
            blob = tile[..., 0] > 200
            yield [TileInstance("gorodishcha", 0.9, blob)] if blob.any() else []

    monkeypatch.setattr(lidar_dem, "hillshades_for_las", fake_hillshades)
    monkeypatch.setattr(yolo_runtime, "predict_tiles", fake_predict)
    monkeypatch.setattr(
        yolo_runtime,
        "load_model",
        lambda weights_dir=None: yolo_runtime.LoadedModel(object(), "stub", {}, 64),
    )
    monkeypatch.setattr(lidar, "_model", None)
    monkeypatch.setattr(lidar, "_model_error", "")
    monkeypatch.setitem(config.LIDAR, "infer_mode", "tiles")
    monkeypatch.setitem(config.LIDAR, "scales", (1, 4))
    monkeypatch.setitem(config.LIDAR, "tile_px", 512)
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c",))

    las = tmp_path / "r" / "a.las"
    las.parent.mkdir()
    las.write_bytes(b"0" * 1024)
    region = Region(name="r", path=las.parent, utm_crs=UTM, files={"li": [las]})
    detections = lidar.run(region, Budget(600), config)

    # ×4: отмывка 1600x1200 → 400x300 — один тайл, дополненный до 512
    assert (512, 512) in sizes
    expected = RASTER_T * (BLOB_RC[1] + 0.5, BLOB_RC[0] + 0.5)
    whole = [d for d in detections if not d.extra["edge_cut"]]  # обрезки краем тайла — не целые
    assert len(whole) >= 2  # целое пятно и на ×1, и на ×4
    for det in whole:
        assert det.geom.centroid.distance(type(det.geom.centroid)(expected)) < 0.5
        assert 35 < det.geom.area < 65  # круг r=4 м ≈ 50 м² на любом масштабе


def test_model_passport_overrides_tile_and_scales(tmp_path: Path, monkeypatch) -> None:
    """Наша модель (паспорт): тайл и масштабы из LoadedModel, а не из config."""
    image = np.full((600, 600), 128, np.uint8)

    def fake_hillshades(las_path, region_crs, out_dir, variants=None):
        path = Path(out_dir) / "hs_c.tif"
        profile = dict(driver="GTiff", height=600, width=600, count=1, dtype="uint8")
        with rasterio.open(path, "w", crs=UTM, transform=RASTER_T, **profile) as dst:
            dst.write(image, 1)
        return [path]

    sizes: list[tuple[int, int]] = []

    def fake_predict(loaded, tiles, imgsz=None):
        for tile in tiles:
            sizes.append(tile.shape[:2])
            yield []

    own = yolo_runtime.LoadedModel(object(), "pt", {}, 64, 512, 256, (1.0,))
    monkeypatch.setattr(lidar_dem, "hillshades_for_las", fake_hillshades)
    monkeypatch.setattr(yolo_runtime, "predict_tiles", fake_predict)
    monkeypatch.setattr(yolo_runtime, "load_model", lambda weights_dir=None: own)
    monkeypatch.setattr(lidar, "_model", None)
    monkeypatch.setattr(lidar, "_model_error", "")
    monkeypatch.setitem(config.LIDAR, "infer_mode", "tiles")
    monkeypatch.setitem(config.LIDAR, "scales", (1, 2, 4))
    monkeypatch.setitem(config.LIDAR, "tile_px", 1024)
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c",))
    las = tmp_path / "r" / "a.las"
    las.parent.mkdir()
    las.write_bytes(b"0" * 1024)
    region = Region(name="r", path=las.parent, utm_crs=UTM, files={"li": [las]})
    lidar.run(region, Budget(600), config)
    assert sizes and set(sizes) == {(256, 256)}  # тайл 256 из паспорта, один масштаб
    assert len(sizes) == 9  # 600 px при тайле 256 и перекрытии 0.3 — 3×3, без ×2 и ×4
