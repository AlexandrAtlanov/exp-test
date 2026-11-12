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


def _ensemble_world(tmp_path: Path, monkeypatch, extra_loader):
    """Отмывка с пятном, основная модель (тайл 256) и доп. модели через extra_loader."""
    image = np.full((600, 600), 128, np.uint8)
    cv2.circle(image, (300, 300), 40, 255, -1)

    def fake_hillshades(las_path, region_crs, out_dir, variants=None):
        path = Path(out_dir) / "hs_c.tif"
        profile = dict(driver="GTiff", height=600, width=600, count=1, dtype="uint8")
        with rasterio.open(path, "w", crs=UTM, transform=RASTER_T, **profile) as dst:
            dst.write(image, 1)
        return [path]

    calls: list[tuple[str, int]] = []

    def fake_predict(loaded, tiles, imgsz=None):
        for tile in tiles:
            calls.append((loaded.name, tile.shape[0]))
            blob = tile[..., 0] > 200
            # основная видит курган, доп. — городище: разные классы, дубли не схлопываются
            cls = "kurgany_tselye" if loaded.name == "lidar_yolo.pt" else "gorodishcha"
            yield [TileInstance(cls, 0.9, blob)] if blob.any() else []

    main = yolo_runtime.LoadedModel(object(), "pt", {}, 64, 512, 256, (1.0,), name="lidar_yolo.pt")

    def fake_load(weights_dir=None, pt_name=None):
        return main if pt_name is None else extra_loader(pt_name)

    monkeypatch.setattr(lidar_dem, "hillshades_for_las", fake_hillshades)
    monkeypatch.setattr(yolo_runtime, "predict_tiles", fake_predict)
    monkeypatch.setattr(yolo_runtime, "load_model", fake_load)
    monkeypatch.setattr(lidar, "_model", None)
    monkeypatch.setattr(lidar, "_model_error", "")
    monkeypatch.setattr(lidar, "_extra_models", {})
    monkeypatch.setitem(config.LIDAR, "infer_mode", "tiles")
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c",))
    las = tmp_path / "r" / "a.las"
    las.parent.mkdir()
    las.write_bytes(b"0" * 1024)
    return Region(name="r", path=las.parent, utm_crs=UTM, files={"li": [las]}), calls


def test_ensemble_runs_each_model_with_own_tiling(tmp_path: Path, monkeypatch) -> None:
    """Ансамбль: каждая модель — со своим тайлом; source доп. модели помечен её весами."""
    extra = yolo_runtime.LoadedModel(object(), "pt", {}, 64, 1024, 600, (1.0,), name="baseline.pt")
    region, calls = _ensemble_world(tmp_path, monkeypatch, lambda name: extra)
    monkeypatch.setitem(config.YOLO, "extra_weights_pt", ("baseline.pt",))
    detections = lidar.run(region, Budget(600), config)
    assert {size for name, size in calls if name == "lidar_yolo.pt"} == {256}
    assert {size for name, size in calls if name == "baseline.pt"} == {600}
    sources = {(d.class_name, d.source) for d in detections}
    assert ("kurgany_tselye", "a.las:c") in sources
    assert ("gorodishcha", "a.las:c:baseline.pt") in sources
    assert lidar.estimate_seconds(region) == 2 * 1024 / 1e9 * config.LIDAR["seconds_per_gb"]


def test_ensemble_missing_extra_is_skipped(tmp_path: Path, monkeypatch) -> None:
    """Доп. модель не грузится — предупреждение, основная работает как раньше."""

    def broken(name):
        raise FileNotFoundError(name)

    region, calls = _ensemble_world(tmp_path, monkeypatch, broken)
    monkeypatch.setitem(config.YOLO, "extra_weights_pt", ("missing.pt",))
    detections = lidar.run(region, Budget(600), config)
    assert detections and all(d.class_name == "kurgany_tselye" for d in detections)
    assert {name for name, _ in calls} == {"lidar_yolo.pt"}
