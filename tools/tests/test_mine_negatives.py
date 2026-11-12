"""Hard negatives (C24): mine_negatives находит ложные срабатывания, build_dataset их сохраняет."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import mine_negatives
import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import Point, box, mapping
from test_augment_data import HILL, _region
from test_training_dataset import CRS, ORIGIN_X, ORIGIN_Y, _dataset_dirs, _raster

import build_dataset
import settings


def _pred(geom_utm, conf: float, markup: str = "li", region: str = "R01") -> dict:
    to_3857 = Transformer.from_crs("EPSG:32637", "EPSG:3857", always_xy=True)
    geom = shapely.transform(geom_utm, to_3857.transform, interleaved=False)
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "class_name": "kurgany_tselye",
            "confidence": conf,
            "markup_type": markup,
            "region_name": region,
        },
    }


def test_mine_keeps_only_confident_lidar_false_positives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "VAL_REGIONS", ("другой",))  # R01 — train
    data = _region(tmp_path / "train", shift_east_m=0.0)
    far = Point(HILL.x - 15.0, HILL.y + 12.0).buffer(2.0)
    preds = [
        _pred(HILL.buffer(3.0), 0.9),  # на объекте разметки — не FP
        _pred(Point(HILL.x + 30.0, HILL.y).buffer(2.0), 0.8),  # далеко — FP
        _pred(Point(HILL.x + 30.0, HILL.y).buffer(2.0), 0.1),  # неуверенное — нет
        _pred(Point(HILL.x + 30.0, HILL.y).buffer(2.0), 0.9, markup="maggr"),  # не лидар
        _pred(far, 0.9, region="нет такого"),
    ]
    pred_path = tmp_path / "pred.geojson"
    pred_path.write_text(json.dumps({"type": "FeatureCollection", "features": preds}))
    result = mine_negatives.mine(data, pred_path, min_conf=0.3, buffer_m=10.0)
    assert set(result) == {"R01"}
    assert result["R01"]["crs"] == "EPSG:32637"
    [b] = result["R01"]["boxes"]
    assert b[0] == pytest.approx(HILL.x + 28.0, abs=0.05)
    assert b[1] == pytest.approx(HILL.y - 2.0, abs=0.05)


def test_hard_negative_tiles_kept_despite_share(tmp_path: Path, monkeypatch) -> None:
    """Пустой тайл на месте ложного срабатывания сохраняется даже при доле пустых 0."""
    monkeypatch.setattr(settings, "NEGATIVE_TILE_SHARE", 0.0)
    _dataset_dirs(tmp_path)
    raster = _raster(tmp_path / "hs_c.tif", 512)
    opts = build_dataset.TileOptions(tile_px=256, overlap=0.0, scales=(1,), rot90=False)
    empty = build_dataset.tile_raster(raster, {}, tmp_path, "train", "r", opts)
    assert empty["tiles"] == 0
    # рамка 2×2 м в правом нижнем тайле (пиксели 300..320)
    neg = [box(ORIGIN_X + 30.0, ORIGIN_Y - 32.0, ORIGIN_X + 32.0, ORIGIN_Y - 30.0)]
    stats = build_dataset.tile_raster(raster, {}, tmp_path, "train", "r", opts, negatives=neg)
    assert stats["tiles"] == stats["tiles_hard_negative"] == 1
    label = tmp_path / "labels" / "train" / "r__hs_c__s1__y256_x256.txt"
    assert label.read_text() == ""


def test_load_negatives_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "negatives.json"
    path.write_text(json.dumps({"R01": {"crs": CRS, "boxes": [[0, 0, 1, 2]]}}))
    loaded = build_dataset.load_negatives(path)
    assert list(loaded) == ["R01"] and loaded["R01"][0].bounds == (0, 0, 1, 2)
    assert np.isclose(loaded["R01"][0].area, 2.0)


def test_unlabeled_mound_not_taken_as_negative(tmp_path: Path, monkeypatch) -> None:
    """Насыпь без разметки (неполная разметка) — не negative; ровное место — negative."""
    monkeypatch.setattr(settings, "VAL_REGIONS", ("другой",))
    data = _region(tmp_path / "train", shift_east_m=1000.0)  # разметка далеко от холма
    flat = Point(HILL.x - 15.0, HILL.y - 10.0).buffer(2.0)
    preds = [_pred(HILL.buffer(3.0), 0.9), _pred(flat, 0.9)]
    pred_path = tmp_path / "pred.geojson"
    pred_path.write_text(json.dumps({"type": "FeatureCollection", "features": preds}))

    checked = mine_negatives.mine(data, pred_path, mound_contrast_m=0.15)
    assert checked["R01"]["skipped_mounds"] == 1
    [b] = checked["R01"]["boxes"]
    assert b[0] == pytest.approx(flat.bounds[0], abs=0.05)

    unchecked = mine_negatives.mine(data, pred_path, mound_contrast_m=None)
    assert len(unchecked["R01"]["boxes"]) == 2
