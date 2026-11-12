"""Вариант отмывки «r» (C23): LRM, уклон, SVF; выбор отмывок по паспорту модели."""

from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import rasterio
from rasterio.transform import Affine

from koz3 import config, yolo_runtime
from koz3.datatypes import Region
from koz3.detectors import lidar, lidar_dem, lidar_relief
from koz3.timing import Budget

STEP = 0.1
T = Affine(STEP, 0.0, 500000.0, 0.0, -STEP, 6000000.0)


def _render(tmp_path: Path, dem: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    valid = np.ones(dem.shape, bool) if valid is None else valid
    filled = np.where(valid, dem, np.nanmin(np.where(valid, dem, np.inf)))
    path = lidar_relief.write_relief(
        tmp_path / "x_r.tif", filled, valid, STEP, T, "EPSG:32637", block_rows=97
    )
    with rasterio.open(path) as src:
        assert src.count == 3 and src.dtypes[0] == "uint8"
        return src.read()


def test_mound_bright_flat_neutral(tmp_path: Path) -> None:
    """Курган Ø 20 м, h 1 м: LRM на вершине ярче, на ровном месте — середина шкалы."""
    rows, cols = np.mgrid[0:600, 0:600]
    dist2 = ((rows - 300) ** 2 + (cols - 300) ** 2) * STEP**2
    dem = 100.0 + 1.0 * np.exp(-dist2 / (2 * 4.0**2))
    lrm, slope, svf = _render(tmp_path, dem).astype(int)
    assert lrm[300, 300] > 200  # +0.8 м и выше при клипе ±1 м
    assert abs(lrm[20, 20] - 127) <= 3  # ровно
    assert slope[20, 20] <= 2 and slope[300, 340] > 20  # склон кургана
    assert svf[20, 20] >= 250  # ровное место — всё небо открыто


def test_plane_45_degrees_and_pit(tmp_path: Path) -> None:
    """Плоскость 45° → уклон 254; яма → SVF темнее ровного места."""
    _, cols = np.mgrid[0:200, 0:200]
    plane = cols * STEP * 1.0  # tan 45° = 1
    _, slope, _ = _render(tmp_path, plane.astype(float))
    assert slope[100, 100] == 254

    rows, cols = np.mgrid[0:400, 0:400]
    dist2 = ((rows - 200) ** 2 + (cols - 200) ** 2) * STEP**2
    pit = 100.0 - 2.0 * np.exp(-dist2 / (2 * 3.0**2))
    _, _, svf = _render(tmp_path, pit).astype(int)
    assert svf[200, 200] < svf[20, 20] - 30


def test_nodata_zero_and_no_edge_rim(tmp_path: Path) -> None:
    """Вне данных — (0, 0, 0); у края данных на ровном месте нет ложного вала LRM."""
    dem = np.full((400, 400), 150.0)
    valid = np.ones(dem.shape, bool)
    valid[:, 250:] = False
    lrm, slope, svf = _render(tmp_path, dem, valid).astype(int)
    assert (lrm[:, 260:] == 0).all() and (slope[:, 260:] == 0).all() and (svf[:, 260:] == 0).all()
    assert abs(lrm[200, 245] - 127) <= 3
    image = yolo_runtime.to_model_image(np.stack([lrm, slope, svf]).astype(np.uint8))
    assert yolo_runtime.valid_pixel_mask(image)[:, :250].all()


def test_relief_variant_built_only_on_request(tmp_path: Path) -> None:
    """hillshades_for_las строит «r» только по запросу; бейзлайновые варианты не меняются."""
    from test_lidar_dem import write_synthetic_las

    las = write_synthetic_las(tmp_path / "s_La.las")
    paths = lidar_dem.hillshades_for_las(las, "EPSG:32637", tmp_path / "a", variants=("g", "r"))
    assert [p.stem.rsplit("_", 1)[-1] for p in paths] == ["g", "r"]
    only_g = lidar_dem.hillshades_for_las(las, "EPSG:32637", tmp_path / "b", variants=("g",))
    with rasterio.open(paths[0]) as a, rasterio.open(only_g[0]) as b:
        assert (a.read() == b.read()).all()


def test_models_get_only_trained_hillshades(tmp_path: Path, monkeypatch) -> None:
    """Модель без «hillshades» в паспорте не получает «r»; модель с «r» — получает."""

    def fake_hillshades(las_path, region_crs, out_dir, variants=None):
        paths = []
        for v in variants:
            path = Path(out_dir) / f"hs_{v}.tif"
            profile = dict(driver="GTiff", height=64, width=64, count=1, dtype="uint8")
            with rasterio.open(path, "w", crs="EPSG:32637", transform=T, **profile) as dst:
                dst.write(np.full((64, 64), 128, np.uint8), 1)
            paths.append(path)
        return paths

    calls: list[tuple[str, str]] = []
    built: list[tuple[str, ...]] = []

    def fake_predict(region, las_path, tif_path, variant, model, budget, cfg):
        calls.append((model.name, variant))
        return []

    old = yolo_runtime.LoadedModel(object(), "pt", {}, 8, name="old.pt")
    new = yolo_runtime.LoadedModel(object(), "pt", {}, 8, name="new.pt", hillshades=("g", "r"))

    def record_hillshades(las_path, region_crs, out_dir, variants=None):
        built.append(tuple(variants))
        return fake_hillshades(las_path, region_crs, out_dir, variants)

    monkeypatch.setattr(lidar_dem, "hillshades_for_las", record_hillshades)
    monkeypatch.setattr(lidar, "_predict_hillshade", fake_predict)
    monkeypatch.setattr(
        yolo_runtime,
        "load_model",
        lambda weights_dir=None, pt_name=None: old if pt_name is None else new,
    )
    monkeypatch.setattr(lidar, "_model", None)
    monkeypatch.setattr(lidar, "_model_error", "")
    monkeypatch.setattr(lidar, "_extra_models", {})
    monkeypatch.setitem(config.YOLO, "extra_weights_pt", ("new.pt",))
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c", "g", "r"))
    las = tmp_path / "r" / "a.las"
    las.parent.mkdir()
    las.write_bytes(b"0" * 1024)
    region = Region(name="r", path=las.parent, utm_crs="EPSG:32637", files={"li": [las]})
    lidar.run(region, Budget(600), config)
    assert built == [("c", "g", "r")]
    assert sorted(calls) == [("new.pt", "g"), ("new.pt", "r"), ("old.pt", "c"), ("old.pt", "g")]

    # без моделей, обученных на «r», отмывка «r» не строится вовсе
    built.clear()
    monkeypatch.setitem(config.YOLO, "extra_weights_pt", ())
    lidar.run(region, Budget(600), config)
    assert built == [("c", "g")]
