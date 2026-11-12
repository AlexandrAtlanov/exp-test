"""training/train_maggr.py на синтетике: трейн «как у организаторов» → joblib → детектор.

Три региона с серой картой MagGr (диполи и пятна) и разметкой FindsPoints /
ObjectPoly. Проверяется, что файл весов пишется в формате, который принимает
koz3/detectors/maggr.py, и что детектор с ним находит объекты.
"""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin, xy

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import train_maggr  # noqa: E402

from koz3 import config  # noqa: E402
from koz3.detectors import maggr  # noqa: E402
from koz3.discovery import find_regions  # noqa: E402
from koz3.timing import Budget  # noqa: E402

CRS = "EPSG:32637"
SIZE = 300


def _region(root: Path, number: int, seed: int) -> None:
    """Регион трейна: серая карта MagGr, точки и полигоны разметки (в UTM)."""
    name = f"{number}_Участок"
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    values = rng.normal(128.0, 3.0, (SIZE, SIZE))
    transform = from_origin(500000.0 + number * 1000, 6000000.0, 0.1, 0.1)
    points, polys = [], []
    for k in range(6):  # диполи-находки
        r, c = 40 + 40 * k, 60 + (k % 2) * 120
        values += 70 * np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / 18.0)
        values -= 70 * np.exp(-((rows - r) ** 2 + (cols - c - 10) ** 2) / 18.0)
        points.append(xy(transform, r, c))
    for r, c in ((80, 240), (220, 240)):  # крупные аномалии-объекты
        values += 50 * np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / 600.0)
        x0, y0 = xy(transform, r, c)
        polys.append([[x0 - 3, y0 - 3], [x0 + 3, y0 - 3], [x0 + 3, y0 + 3], [x0 - 3, y0 + 3]])
    gray = np.clip(values, 1, 254).astype(np.uint8)
    gray[:5], gray[-5:], gray[:, :5], gray[:, -5:] = 0, 0, 0, 0
    maps = root / name / f"07_{name}_MagGr_карты"
    maps.mkdir(parents=True)
    profile = dict(driver="GTiff", height=SIZE, width=SIZE, count=1, dtype="uint8", crs=CRS)
    with rasterio.open(maps / f"{name}_серый.tif", "w", transform=transform, **profile) as dst:
        dst.write(gray, 1)
    markup = root / name / f"06_{name}_разметка"
    markup.mkdir()

    def write(kind: str, geometries: list[dict]) -> None:
        features = [{"type": "Feature", "properties": {}, "geometry": g} for g in geometries]
        payload = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": CRS}},
            "features": features,
        }
        path = markup / f"{name}_MagGr_{kind}.geojson"
        path.write_text(json.dumps(payload), encoding="utf-8")

    write("FindsPoints", [{"type": "Point", "coordinates": list(p)} for p in points])
    write("ObjectPoly", [{"type": "Polygon", "coordinates": [[*q, q[0]]]} for q in polys])
    (root / name / "UTM.json").write_text(json.dumps({"crs": CRS}), encoding="utf-8")


def test_train_writes_detector_compatible_weights(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "train"
    for number, seed in ((201, 1), (202, 2), (203, 3)):
        _region(data, number, seed)
    out = tmp_path / "w"
    assert train_maggr.main(["--data", str(data), "--out", str(out), "--holdout", "203"]) == 0
    weights = out / str(config.MAGGR["models_file"])
    assert weights.is_file()

    monkeypatch.setattr(config, "WEIGHTS_DIR", out)
    monkeypatch.setattr(maggr, "_MODELS_CACHE", {})
    [region] = [r for r in find_regions(data) if r.name.startswith("203")]
    assert maggr.is_available(region)
    detections = maggr.run(region, Budget(600), config)
    assert {d.markup_type for d in detections} == {"maggr"}
    assert "finds_points" in {d.class_name for d in detections}


def test_needs_two_regions(tmp_path: Path) -> None:
    data = tmp_path / "train"
    _region(data, 201, 1)
    assert train_maggr.main(["--data", str(data), "--out", str(tmp_path / "w")]) == 1
