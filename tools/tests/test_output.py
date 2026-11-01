"""Тесты output v2 (дефекты Д1–Д13) и синхронизации с validate_output.

Запуск: python test_output.py  (без pytest, только stdlib + shapely/pyproj).
"""

from __future__ import annotations

import json
import logging
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import validate_output
import output

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ALL_CLASSES = list(output._CLASSES)
TMP = Path(tempfile.mkdtemp(prefix="koz3_output_test_"))


def ring(
    cx: float, cy: float, radius: float, positions: int
) -> list[tuple[float, float]]:
    pts = [
        (
            cx + radius * math.cos(2 * math.pi * i / positions),
            cy + radius * math.sin(2 * math.pi * i / positions),
        )
        for i in range(positions)
    ]
    return pts + [pts[0]]


def poly_feature(
    cls: str,
    positions: int,
    conf: float,
    cx: float = 1000.0,
    cy: float = 2000.0,
    radius: float = 100.0,
    holes: int = 0,
    markup: str = "li",
    region: str = "Тестовый регион",
    geometry: dict | None = None,
    props: dict | None = None,
) -> dict:
    coordinates = [ring(cx, cy, radius, positions)]
    for h in range(holes):
        coordinates.append(ring(cx, cy, radius * (0.3 - 0.05 * h), 400))
    return {
        "type": "Feature",
        "geometry": geometry or {"type": "Polygon", "coordinates": coordinates},
        "properties": props
        or {
            "class": cls,
            "markup_type": markup,
            "confidence": conf,
            "region_name": region,
        },
    }


def positions_of(geometry: dict) -> int:
    if geometry["type"] == "Polygon":
        rings = geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        rings = [r for poly in geometry["coordinates"] for r in poly]
    else:
        return 1
    return sum(len(r) - 1 for r in rings)


def read_fc(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_happy_path_and_hole_strip():
    """Д2: interiors снимаются; дедуп после округления; валидатор чист."""
    out = TMP / "happy.geojson"
    feats = [
        poly_feature("kurgany_tselye", 50, 0.9, holes=1),
        {
            **poly_feature("gorodishcha", 10, 0.8, cx=500, cy=500, radius=2.0),
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [(0.0, 0.0), (0.001, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 0.0)]
                ],
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1234.56789, 2345.6789)},
            "properties": {
                "class": "finds_points",
                "markup_type": "terfl",
                "confidence": 0.7,
                "region_name": "Тестовый регион",
            },
        },
    ]
    path = output.write_result(feats, out)
    fc = read_fc(path)
    assert len(fc["features"]) == 3
    assert output.last_stats["written"] == 3
    by_cls = {f["properties"]["class"]: f for f in fc["features"]}
    kurgan = by_cls["kurgany_tselye"]["geometry"]
    assert len(kurgan["coordinates"]) == 1, "дырка должна быть снята (Д2)"
    assert positions_of(kurgan) == 50
    tiny = by_cls["gorodishcha"]["geometry"]
    assert positions_of(tiny) == 3, "после округления дубликат склеен"
    pt = by_cls["finds_points"]["geometry"]
    assert pt["type"] == "Point" and pt["coordinates"] == [1234.57, 2345.68]
    report = validate_output.validate(path)
    assert not report["defects"], report


def test_3d_stripped():
    """Д13: z отбрасывается до трансформации."""
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [(0.0, 0.0, 5.0), (10.0, 0.0, 5.0), (10.0, 10.0, 5.0), (0.0, 0.0, 5.0)]
        ],
    }
    path = output.write_result(
        [poly_feature("arkhitektury", 10, 0.5, geometry=geometry)], TMP / "d3.geojson"
    )
    coords = read_fc(path)["features"][0]["geometry"]["coordinates"][0]
    assert all(len(pt) == 2 for pt in coords), coords


def test_bad_region_name_dropped():
    """Д5: region_name проверяется так же, как в валидаторе."""
    path = output.write_result(
        [
            poly_feature("kurgany_tselye", 10, 0.9, region=""),
            poly_feature("kurgany_tselye", 10, 0.9, region="   "),
            poly_feature("kurgany_tselye", 10, 0.9, region=None),
        ],
        TMP / "d5.geojson",
    )
    assert read_fc(path)["features"] == []
    assert output.last_stats.get("dropped_bad_region_name") == 3, output.last_stats


def test_vertex_sum_limit_and_holes():
    """Д2: сумма позиций ВСЕХ колец ≤ 3000; дырки не считаются (снимаются)."""
    dense = poly_feature(
        "gorodishcha", 2300, 0.9, holes=2
    )
    multi = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[ring(3000, 3000, 80, 2000)], [ring(5000, 5000, 80, 2000)]],
        },
        "properties": {
            "class": "object_poly",
            "markup_type": "ae",
            "confidence": 0.8,
            "region_name": "Тестовый регион",
        },
    }
    path = output.write_result([dense, multi], TMP / "d2.geojson")
    fc = read_fc(path)
    by_cls = {f["properties"]["class"]: f for f in fc["features"]}
    assert positions_of(by_cls["gorodishcha"]["geometry"]) == 2300, (
        "дырки сняты, плотное кольцо цело"
    )
    assert positions_of(by_cls["object_poly"]["geometry"]) <= 3000, (
        "сумма по кольцам (Д2)"
    )
    report = validate_output.validate(path)
    assert not report["defects"], report


def test_decimation_fallback():
    """Д2: если simplify не может — равномерная децимация."""
    old_max, old_base = (
        output._VERTEX_SIMPLIFY_MAX_TOL_M3857,
        output._VERTEX_SIMPLIFY_BASE_TOL_M3857,
    )
    try:
        output._VERTEX_SIMPLIFY_MAX_TOL_M3857 = 0.001
        output._VERTEX_SIMPLIFY_BASE_TOL_M3857 = 0.001
        path = output.write_result(
            [poly_feature("fortifikatsii", 10_000, 0.6)], TMP / "d2b.geojson"
        )
        geom = read_fc(path)["features"][0]["geometry"]
        assert positions_of(geom) <= 3000, positions_of(geom)
        assert not validate_output.validate(path)["defects"]
    finally:
        output._VERTEX_SIMPLIFY_MAX_TOL_M3857 = old_max
        output._VERTEX_SIMPLIFY_BASE_TOL_M3857 = old_base


def test_cap_quotas_keep_rare_class():
    """Д3: при обрезке до MAX_FEATURES редкий класс не выметается."""
    features = []
    mass = [
        c for c in ALL_CLASSES if c != "fortifikatsii"
    ]
    for i in range(24_900):
        features.append(poly_feature(mass[i % 6], 12, 0.9 - 1e-9 * i, cx=100 + i))
    for i in range(100):
        features.append(poly_feature("fortifikatsii", 12, 0.1, cx=90_000 + i))
    path = output.write_result(features, TMP / "d3.geojson")
    fc = read_fc(path)
    assert len(fc["features"]) == 20_000, len(fc["features"])
    classes = {f["properties"]["class"] for f in fc["features"]}
    assert "fortifikatsii" in classes, "квоты не спасли редкий класс (Д3)"
    assert len(classes) == 7
    confs = [f["properties"]["confidence"] for f in fc["features"]]
    assert confs == sorted(confs, reverse=True)


def test_size_limit_iterative_cut():
    """Д1: превышение лимита байт -> итеративная обрезка с квотами."""
    old = output.OUTPUT_SIZE_LIMIT_BYTES
    try:
        output.OUTPUT_SIZE_LIMIT_BYTES = 100_000
        features = [
            poly_feature(ALL_CLASSES[i % 7], 40, 0.5 + 1e-6 * i, cx=100 + i)
            for i in range(2_000)
        ]
        path = output.write_result(features, TMP / "d1.geojson")
        size = path.stat().st_size
        fc = read_fc(path)
        assert size <= 100_000, (size, len(fc["features"]))
        assert 0 < len(fc["features"]) < 2_000
        classes = {f["properties"]["class"] for f in fc["features"]}
        assert len(classes) == 7, "квоты должны сработать и при размерной обрезке"
        assert not validate_output.validate(path)["defects"]
    finally:
        output.OUTPUT_SIZE_LIMIT_BYTES = old


def test_atomic_tmp_cleanup():
    """Д10: при исключении записи осиротевший .tmp удаляется."""
    import os

    tmp_path = TMP / "atomic.geojson"
    tmp_file = tmp_path.with_name(tmp_path.name + ".tmp")
    item = {
        "type": "Feature",
        "geometry": None,
        "properties": {"class": "kurgany_tselye"},
    }

    real_replace = os.replace

    def boom_replace(*args, **kwargs):
        raise OSError("boom")

    os.replace = boom_replace
    try:
        try:
            output._write_features_json_atomic([item], tmp_path)
        except OSError:
            pass
        else:
            raise AssertionError("OSError должен был проброситься")
    finally:
        os.replace = real_replace
    assert not tmp_file.exists(), "tmp не убран (Д10)"
    assert not tmp_path.exists(), "целевой файл не должен появиться"


def test_prepare_cache_between_checkpoints():
    """Д4: повторный write_result готовит только новые фичи; контент стабилен."""
    features = [
        poly_feature(ALL_CLASSES[i % 7], 60, 0.9 - 1e-6 * i, cx=100 + i)
        for i in range(3_000)
    ]
    out1 = TMP / "cache1.geojson"
    t0 = time.perf_counter()
    output.write_result(features, out1)
    t_first = time.perf_counter() - t0

    features += [
        poly_feature(ALL_CLASSES[i % 7], 60, 0.05, cx=90_000 + i) for i in range(300)
    ]
    out2 = TMP / "cache2.geojson"
    t0 = time.perf_counter()
    output.write_result(features, out2)
    t_second = time.perf_counter() - t0

    assert t_second < t_first * 0.75, (t_first, t_second)
    fc1, fc2 = read_fc(out1), read_fc(out2)
    assert fc1["features"] == fc2["features"][: len(fc1["features"])], (
        "кэш исказил старые фичи"
    )


def test_reprojection_to_3857():
    """Репроекция пакетная (Д4-адаптер), выход в EPSG:3857."""
    from pyproj import Transformer

    old = output._TRANSFORMER
    try:
        output._TRANSFORMER = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [(37.60, 55.75), (37.61, 55.75), (37.61, 55.76), (37.60, 55.75)]
            ],
        }
        path = output.write_result(
            [poly_feature("kurgany_povrezhdennye", 10, 0.9, geometry=geometry)],
            TMP / "crs.geojson",
        )
        coords = read_fc(path)["features"][0]["geometry"]["coordinates"][0]
        xs, ys = [p[0] for p in coords], [p[1] for p in coords]
        assert 4.0e6 < min(xs) < max(xs) < 4.3e6, xs
        assert 7.3e6 < min(ys) < max(ys) < 7.7e6, ys
    finally:
        output._TRANSFORMER = old


def test_bad_types_and_multi_codes():
    """Дефектные фичи: коды + Д9 (несколько кодов на фичу в статистике)."""
    path = output.write_result(
        [
            poly_feature("unknown_class", 10, 0.5),
            poly_feature("kurgany_tselye", 10, 0.5, markup="nope"),
            poly_feature("kurgany_tselye", 10, float("nan")),
            poly_feature(
                "kurgany_tselye",
                10,
                0.5,
                geometry={"type": "LineString", "coordinates": [(0, 0), (1, 1)]},
            ),
            poly_feature(
                "kurgany_tselye", 10, 0.5, geometry={"type": "Polygon"}
            ),
            poly_feature("kurgany_tselye", 10, 0.5, region=""),
            {
                "type": "Feature",
                "geometry": None,
                "properties": None,
            },
            "не фича вообще",
        ],
        TMP / "bad.geojson",
    )
    assert read_fc(path)["features"] == []
    st = output.last_stats
    assert st.get("dropped_bad_class") == 2
    assert st.get("dropped_bad_markup_type") == 2
    assert st.get("dropped_bad_confidence") == 2
    assert st.get("dropped_bad_geometry_type") == 1
    assert st.get("dropped_bad_geometry") == 2
    assert st.get("dropped_bad_region_name") == 2
    assert st.get("dropped_bad_feature") == 1
    for code in ("bad_class", "bad_markup_type", "bad_confidence", "bad_region_name"):
        assert st.get(f"dropped_{code}", 0) >= 1


def test_validator_catches_broken_features():
    """Д7: bad_feature_type и geometry_error вместо трейсбека."""
    assert validate_output.feature_defects({"geometry": {}, "properties": {}}) == [
        "bad_feature_type"
    ]
    broken = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[(0, 0), (1, 0)]]},
        "properties": {
            "class": "kurgany_tselye",
            "markup_type": "li",
            "confidence": 0.5,
            "region_name": "x",
        },
    }
    assert "geometry_error" in validate_output.feature_defects(
        broken
    ) or "geometry_invalid" in validate_output.feature_defects(broken)
    no_type = {"type": "FeatureCollection", "features": [{"properties": {}}]}
    p = TMP / "broken.geojson"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(no_type, fh)
    report = validate_output.validate(p)
    assert report["defects"]


def test_resolver_errors():
    """Д12: понятная ошибка, если родитель пути — обычный файл."""
    parent_file = TMP / "parent_is_file.txt"
    parent_file.write_text("x", encoding="utf-8")
    try:
        output.resolve_output_path(parent_file / "result.geojson")
    except NotADirectoryError:
        pass
    else:
        raise AssertionError("ожидалась NotADirectoryError (Д12)")
    assert output.resolve_output_path(TMP).name == "result.geojson"


def test_simplification_preserves_validity():
    """Лимитер: длинное кольцо упрощается и остаётся валидным (Д2/Д8)."""
    path = output.write_result(
        [poly_feature("arkhitektury", 5_000, 0.7)], TMP / "simplify.geojson"
    )
    geom = read_fc(path)["features"][0]["geometry"]
    assert positions_of(geom) <= 3_000
    assert not validate_output.validate(path)["defects"]


def main() -> int:
    tests = [
        fn
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} тестов прошло")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())