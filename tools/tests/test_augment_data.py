"""build_dataset + augment_data на синтетике: сдвиг разметки исправляется, данные добавляются."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Point
from test_lidar_dem import write_synthetic_las

import build_dataset
import settings
from koz3.discovery import find_regions

HILL = Point(500020.0, 6000015.0)


def _region(root: Path, shift_east_m: float, broken: bool = False) -> Path:
    region = root / "R01"
    las_dir = region / "01_R01_La_точки"
    las_dir.mkdir(parents=True)
    write_synthetic_las(las_dir / "R01_La.las", n=60000)
    if broken:
        (las_dir / "R01_part2_La.las").write_bytes(b"LASF" + b"\0" * 50)
    (region / "UTM.json").write_text(json.dumps({"crs": "EPSG:32637"}), encoding="utf-8")
    markup = region / "06_R01_разметка"
    markup.mkdir()
    circle = translate(HILL.buffer(4.0), xoff=shift_east_m)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32637"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [list(circle.exterior.coords)]},
            }
        ],
    }
    (markup / "R01_Li_Курганы_целые.geojson").write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.fixture
def small_world(monkeypatch):
    """Настройки под крошечный синтетический участок 40×30 м."""
    monkeypatch.setattr(settings, "VAL_REGIONS", ("другой",))  # R01 — train
    monkeypatch.setattr(settings, "NEGATIVE_TILE_SHARE", 0.0)
    monkeypatch.setattr(settings, "MIN_OBJECTS_FOR_SHIFT", 1)
    monkeypatch.setattr(settings, "SYNTH_PER_KM2", 5000.0)
    monkeypatch.setattr(settings, "SYNTH_DIAMETER_M", (3.0, 4.0))
    monkeypatch.setattr(settings, "SYNTH_MIN_GAP_M", 2.0)
    return build_dataset.TileOptions(tile_px=128, overlap=0.1, scales=(1,), rot90=False)


def _dirs(out: Path) -> None:
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)


def _label_centers_x_m(labels_dir: Path, pattern: str, size: int) -> list[float]:
    """Восток центров объектов в подписях тайлов, м от левого края растра (сетка 0.1 м)."""
    centers = []
    for label in labels_dir.glob(pattern):
        x0 = int(label.stem.rsplit("_x", 1)[1])
        for line in label.read_text().splitlines():
            values = np.asarray(line.split()[1:], float).reshape(-1, 2)
            centers.append((values[:, 0].mean() * size + x0) * 0.1)
    return centers


def test_shift_fixed_and_extra_data_added(tmp_path: Path, small_world) -> None:
    data = _region(tmp_path / "train", shift_east_m=4.0, broken=True)
    out, cache = tmp_path / "ds", tmp_path / "cache"
    _dirs(out)
    [region] = find_regions(data)
    _, split, stats, notes = build_dataset.process_region(region, out, cache, "las", small_world)
    assert split == "train"
    assert stats["las_skipped"] == 1  # битый второй LAS не уронил регион
    assert stats["las_shifted"] == 1
    assert any("сдвинута" in n for n in notes)
    assert stats["tiles_azimuth"] > 0 and stats["tiles_synth"] > 0
    names = {p.name for p in (out / "images" / "train").iterdir()}
    assert any("__az045__" in n for n in names) and any("__synth_g__" in n for n in names)

    # подпись на основной отмывке сидит на холме (а не там, куда её сдвинули)
    centers = _label_centers_x_m(out / "labels" / "train", "R01__R01_g__s1__*.txt", 128)
    assert centers
    hill_x_from_left = HILL.x - 500000.0  # растр начинается в x = 500000
    # целый контур — на холме; обрезки краем тайла сдвинуты к краю, но не дальше контура.
    # Без исправления контур был бы 20–28 м, центры обрезков — 24–27 м.
    assert min(abs(x - hill_x_from_left) for x in centers) < 1.0
    assert max(centers) < hill_x_from_left + 4.5

    # повторная сборка берёт сдвиг и доп. растры из кэша
    out2 = tmp_path / "ds2"
    _dirs(out2)
    _, _, stats2, _ = build_dataset.process_region(region, out2, cache, "las", small_world)
    assert stats2["las_shifted"] == 1 and stats2["tiles_synth"] == stats["tiles_synth"]


def test_no_fix_keeps_annotation_and_val_gets_no_extra(tmp_path: Path, small_world, monkeypatch):
    monkeypatch.setattr(settings, "VAL_REGIONS", ("R01",))
    data = _region(tmp_path / "train", shift_east_m=4.0)
    out = tmp_path / "ds"
    _dirs(out)
    [region] = find_regions(data)
    opts = build_dataset.TileOptions(tile_px=128, overlap=0.1, scales=(1,), rot90=False, fix=False)
    _, split, stats, _ = build_dataset.process_region(region, out, tmp_path / "c", "las", opts)
    assert split == "val"
    assert stats["las_shifted"] == 0 and stats["tiles_synth"] == 0 and stats["tiles_azimuth"] == 0
