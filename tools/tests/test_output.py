"""Тесты koz3.output и согласованности с tools/validate_output.py (синтетика)."""

import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import pytest
from shapely.geometry import MultiPolygon, Polygon, mapping

from koz3 import config, output

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_output  # noqa: E402


def _feature(geom, class_name="kurgany_tselye", conf=0.8, markup="li", region="r1"):
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "class_name": class_name,
            "confidence": conf,
            "markup_type": markup,
            "region_name": region,
        },
    }


def _square(x0: float, y0: float, side: float) -> Polygon:
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_output_uses_package_config() -> None:
    """Регрессия: output читает koz3.config, а не пустой fallback."""
    assert output.config is config
    assert output.COORD_PRECISION == config.COORD_DECIMALS


def test_resolve_output_path_rules(tmp_path: Path) -> None:
    """*.geojson — файл; иначе директория (в том числе несуществующая)."""
    assert output.resolve_output_path(tmp_path / "a.geojson") == tmp_path / "a.geojson"
    assert (
        output.resolve_output_path(tmp_path / "new_dir") == tmp_path / "new_dir" / "result.geojson"
    )
    assert (tmp_path / "new_dir").is_dir()
    assert output.resolve_output_path(tmp_path) == tmp_path / "result.geojson"


def test_polygon_written_rounded_and_closed(tmp_path: Path) -> None:
    geom = _square(4_000_000.123456, 7_000_000.987654, 10.0)
    path = output.write_result([_feature(geom)], tmp_path / "r.geojson")
    data = _read(path)
    assert data["type"] == "FeatureCollection"
    (feat,) = data["features"]
    assert feat["geometry"]["type"] == "Polygon"
    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    for x, y in ring:
        assert round(x, 3) == x and round(y, 3) == y
    assert "fid" not in feat["properties"]


def test_multipolygon_keeps_largest_part(tmp_path: Path) -> None:
    geom = MultiPolygon([_square(0, 0, 10), _square(100, 100, 2)])
    path = output.write_result([_feature(geom)], tmp_path / "r.geojson")
    (feat,) = _read(path)["features"]
    assert feat["geometry"]["type"] == "Polygon"
    assert Polygon(feat["geometry"]["coordinates"][0]).area == pytest.approx(100.0)


def test_point_rejected(tmp_path: Path) -> None:
    point = {"type": "Point", "coordinates": [10.0, 20.0]}
    feature = _feature(_square(0, 0, 1))
    feature["geometry"] = point
    path = output.write_result([feature], tmp_path / "r.geojson")
    assert _read(path)["features"] == []
    assert output.last_stats.get("dropped_bad_geometry_type") == 1


def test_bad_class_dropped(tmp_path: Path) -> None:
    path = output.write_result(
        [_feature(_square(0, 0, 5), class_name="selishcha")], tmp_path / "r.geojson"
    )
    assert _read(path)["features"] == []
    assert output.last_stats.get("dropped_bad_class") == 1


def test_bowtie_fixed_or_dropped(tmp_path: Path) -> None:
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    path = output.write_result([_feature(bowtie)], tmp_path / "r.geojson")
    for feat in _read(path)["features"]:
        assert Polygon(feat["geometry"]["coordinates"][0]).is_valid


def test_empty_list_writes_valid_file(tmp_path: Path) -> None:
    path = output.write_result([], tmp_path / "r.geojson")
    assert _read(path) == {"type": "FeatureCollection", "features": []}


def test_ring_vertex_limit(tmp_path: Path) -> None:
    circle = Polygon(
        [(1000 * math.cos(i / 2000), 1000 * math.sin(i / 2000)) for i in range(12_566)]
    )
    path = output.write_result([_feature(circle)], tmp_path / "r.geojson")
    (feat,) = _read(path)["features"]
    assert len(feat["geometry"]["coordinates"][0]) - 1 <= config.MAX_RING_VERTICES


def test_validator_accepts_written_file(tmp_path: Path) -> None:
    features = [_feature(_square(i * 20.0, 0, 10), conf=0.5 + i / 100) for i in range(10)]
    path = output.write_result(features, tmp_path / "r.geojson")
    assert validate_output.main([str(path)]) == 0
