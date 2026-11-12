"""Тесты tools/make_gt.py: сборка GT из синтетической разметки регионов."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest
from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry

import make_gt
import settings

_ALL_CLASSES = set(settings.ALL_CLASSES)


def _write_collection(path: Path, features: list[dict], crs: str) -> None:
    """Пишет синтетический файл разметки (папки создаёт)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _feature(geometry: BaseGeometry, **properties: object) -> dict:
    """GeoJSON-фича: геометрия shapely + произвольные properties."""
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def _add_markup(
    region_dir: Path, file_name: str, features: list[dict], crs: str = "EPSG:3857"
) -> None:
    """Синтетический файл разметки в папке «06_разметка» региона."""
    _write_collection(region_dir / "06_разметка" / file_name, features, crs)


def test_make_gt_full_run(tmp_path: Path) -> None:
    """Два региона → GT: 7 классов, EPSG:3857, точки точками, без Struct*."""
    train = tmp_path / "train"
    first = train / "001_Ясная_Поляна"
    second = train / "002_Луговая"

    _add_markup(
        first,
        "Li_курганы.geojson",
        [
            _feature(
                Polygon(
                    [(500000, 5000000), (500030, 5000000), (500030, 5000030), (500000, 5000030)]
                ),
                class_name="курганы_целые",
            ),
            _feature(Point(500010, 5000010), name="Объект 12"),
        ],
        crs="EPSG:32637",
    )
    _add_markup(
        first,
        "TerFl_ObjectsPoly.geojson",
        [
            _feature(Point(2337576.0, 5639436.0), class_name="FindsPoints"),
            _feature(
                Polygon(
                    [
                        (2337576.0, 5639436.0),
                        (2337606.0, 5639436.0),
                        (2337606.0, 5639466.0),
                        (2337576.0, 5639466.0),
                    ]
                ),
                class_name="StructrPoly",
            ),
        ],
    )
    _add_markup(
        second,
        "Ae_городище.geojson",
        [
            _feature(
                Polygon(
                    [(500000, 5000000), (500400, 5000000), (500400, 5000400), (500000, 5000400)]
                ),
                class_name="городище",
            )
        ],
        crs="EPSG:32637",
    )
    _add_markup(
        second,
        "MagGr_находки.geojson",
        [_feature(Point(500100, 5000100), class_name="FindsPoints")],
        crs="EPSG:32637",
    )

    out = tmp_path / "gt.geojson"
    code = make_gt.main(
        ["--data", str(train), "--regions", first.name, second.name, "--out", str(out)]
    )
    assert code == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert data["crs"]["properties"]["name"] == "EPSG:3857"
    features = data["features"]
    assert len(features) == 5

    classes = [f["properties"]["class_name"] for f in features]
    assert set(classes) <= _ALL_CLASSES
    assert classes.count("kurgany_tselye") == 2
    assert classes.count("finds_points") == 2
    assert classes.count("gorodishcha") == 1

    pairs = {(f["properties"]["class_name"], f["properties"]["markup_type"]) for f in features}
    assert ("kurgany_tselye", "li") in pairs
    assert ("finds_points", "terfl") in pairs
    assert ("finds_points", "maggr") in pairs
    assert ("gorodishcha", "ae") in pairs
    assert ("object_poly", "terfl") not in pairs  # StructrPoly пропущен

    assert {f["properties"]["region_name"] for f in features} == {first.name, second.name}

    points = [f for f in features if f["properties"]["class_name"] == "finds_points"]
    assert {f["geometry"]["type"] for f in points} == {"Point"}

    tr = Transformer.from_crs("EPSG:32637", "EPSG:3857", always_xy=True)
    expected_x, expected_y = tr.transform(500000, 5000000)
    lidar_poly = next(
        f
        for f in features
        if f["properties"]["markup_type"] == "li" and f["geometry"]["type"] == "Polygon"
    )
    ring = lidar_poly["geometry"]["coordinates"][0]
    assert ring[0][0] == pytest.approx(expected_x, abs=1e-3)
    assert ring[0][1] == pytest.approx(expected_y, abs=1e-3)

    terfl_point = next(f for f in points if f["properties"]["markup_type"] == "terfl")
    assert terfl_point["geometry"]["coordinates"] == [2337576.0, 5639436.0]


def test_make_gt_missing_region_fails(tmp_path: Path) -> None:
    """Отсутствующий регион — код 1, файл не пишется."""
    train = tmp_path / "train"
    train.mkdir()
    out = tmp_path / "gt.geojson"
    code = make_gt.main(["--data", str(train), "--regions", "нет_такого", "--out", str(out)])
    assert code == 1
    assert not out.exists()


def test_make_gt_empty_markup_code_1(tmp_path: Path) -> None:
    """Регион без разметки — пустой валидный файл и код 1."""
    region = tmp_path / "train" / "001_Пустой"
    region.mkdir(parents=True)
    out = tmp_path / "gt.geojson"
    code = make_gt.main(
        ["--data", str(region.parent), "--regions", region.name, "--out", str(out)]
    )
    assert code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert data["features"] == []
