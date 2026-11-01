"""Контрактные тесты конвейера постобработки и записи (критерий приёмки 4).

Синтетические детекции всех 7 классов проходят полный путь решения:
postprocess.run → solution._to_features → output.write_result, и итоговый
файл проверяется валидатором tools/validate_output.py — 0 замечаний.

Ожидания выводятся из config (пороги, MIN_AREA_M2, SIMPLIFY_M), а не
захардкожены: тест проверяет контракт, а не конкретные значения констант.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "submission", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from shapely.geometry import box

import solution
import validate_output
from koz3 import config, output, postprocess
from koz3.datatypes import CLASSES, Detection, Region
from koz3.geometry import point_to_circle

REGION_NAME = "synth_region"
UTM = "EPSG:32637"
_PROP_KEYS = {"class_name", "confidence", "markup_type", "region_name", "fid"}


def _region() -> Region:
    return Region(name=REGION_NAME, path=Path("."), utm_crs=UTM, files={})


def _det(class_name, markup_type, confidence, geom, region_name=REGION_NAME):
    return Detection(
        geom=geom,
        class_name=class_name,
        confidence=confidence,
        markup_type=markup_type,
        region_name=region_name,
    )


def _threshold_for(class_name, markup_type):
    return config.CONF_THRESHOLD.get(
        (class_name, markup_type), config.DEFAULT_CONF_THRESHOLD
    )


def _side_for(class_name, factor=2.0, minimum=5.0):
    """Сторона квадрата, гарантированно проходящая MIN_AREA_M2 класса."""
    min_area = float(config.MIN_AREA_M2[class_name])
    return max(minimum, math.sqrt(max(min_area, 0.0)) * factor + 1.0)


def test_config_covers_all_classes():
    """MIN_AREA_M2 и SIMPLIFY_M обязаны покрывать все классы CLASSES."""
    for field_name in ("MIN_AREA_M2", "SIMPLIFY_M"):
        mapping = getattr(config, field_name, None)
        assert isinstance(mapping, dict), f"config.{field_name} должен быть словарём"
        missing = sorted(set(CLASSES) - set(mapping))
        assert not missing, f"config.{field_name} не покрывает классы: {missing}"


def test_threshold_contract():
    """Порог пары: ниже — отбрак, на пороге и выше — остаётся."""
    cls = sorted(CLASSES)[0]
    threshold = _threshold_for(cls, "li")
    assert threshold > 0, "нулевые пороги отключают фильтрацию — проверь конфиг"
    side = _side_for(cls)
    geom = box(0.0, 0.0, side, side)
    kept = postprocess.run([_det(cls, "li", min(threshold, 1.0), geom)])
    dropped = postprocess.run([_det(cls, "li", threshold * 0.5, geom)])
    assert len(kept) == 1
    assert len(dropped) == 0


def test_min_area_contract():
    """Минимальная площадь: мельче MIN_AREA_M2 — отбрак, крупнее — остаётся."""
    cls = sorted(CLASSES)[0]
    min_area = float(config.MIN_AREA_M2[cls])
    side = _side_for(cls)
    out_big = postprocess.run([_det(cls, "li", 0.9, box(0.0, 0.0, side, side))])
    out_small = postprocess.run([_det(cls, "li", 0.9, box(0.0, 0.0, 0.1, 0.1))])
    assert len(out_big) == 1
    # 0.01 м²; отношение, а не константа: min_area может быть и нулевым
    assert (len(out_small) == 0) == (0.01 < min_area)


def test_unknown_class_dropped():
    """Класс вне CLASSES отбраковывается постобработкой."""
    geom = box(0.0, 0.0, 10.0, 10.0)
    assert postprocess.run([_det("definitely_not_a_class", "li", 0.99, geom)]) == []


def test_nms_only_within_pair():
    """NMS — только внутри пары (class_name, markup_type) — инвариант 4.5.

    Точная копия полигона с меньшим confidence подавляется в своей паре,
    но тот же полигон с другим markup_type выживает.
    """
    cls = sorted(CLASSES)[0]
    side = _side_for(cls)
    geom = box(0.0, 0.0, side, side)
    same_pair = postprocess.run(
        [
            _det(cls, "li", 0.9, geom),
            _det(cls, "li", 0.5, geom),  # точный дубль в своей паре — подавлен
        ]
    )
    cross_pair = postprocess.run(
        [
            _det(cls, "li", 0.9, geom),
            _det(cls, "or", 0.5, geom),  # та же геометрия, другая пара — выживает
        ]
    )
    assert len(same_pair) == 1
    assert same_pair[0].confidence == 0.9  # выживает более уверенный
    assert len(cross_pair) == 2


def test_finds_points_circle():
    """finds_points: круг point_to_circle проходит MIN_AREA_M2 и конвейер."""
    circle = point_to_circle(100.0, 100.0)
    assert circle.geom_type == "Polygon"
    min_area = float(config.MIN_AREA_M2["finds_points"])
    assert circle.area + 1e-9 >= min_area, (
        f"конфликт конфигурации: круг finds_points (π·POINT_RADIUS_M² = "
        f"{circle.area:.3f} м²) меньше MIN_AREA_M2['finds_points'] = {min_area}; "
        "такие детекции будут теряться на фильтре площади"
    )
    kept = postprocess.run([_det("finds_points", "maggr", 0.9, circle)])
    assert len(kept) == 1
    assert kept[0].geom.geom_type == "Polygon"


def test_full_pipeline_validator_clean(tmp_path):
    """Полный конвейер: 7 классов × 2 markup_type + мусор → файл без дефектов."""
    regions = {REGION_NAME: _region()}
    detections = []
    cursor_x = 0.0
    for cls in sorted(CLASSES):
        for markup in ("li", "or"):
            side = _side_for(cls)
            detections.append(
                _det(cls, markup, 0.9, box(cursor_x, 0.0, cursor_x + side, side))
            )
            cursor_x += side + 10.0
    first_cls = sorted(CLASSES)[0]
    side = _side_for(first_cls)
    detections.append(_det(first_cls, "li", 0.05, box(cursor_x, 0.0, cursor_x + side, side)))
    detections.append(_det(first_cls, "or", 0.5, box(0.0, 0.0, side, side)))
    detections.append(
        _det("definitely_not_a_class", "li", 0.9, box(cursor_x, 100.0, cursor_x + side, 100.0 + side))
    )
    detections.append(
        _det(first_cls, "li", 0.9, box(0.0, 200.0, side, 200.0 + side), region_name="unknown_region")
    )
    detections.append(_det("finds_points", "maggr", 0.9, point_to_circle(cursor_x + 50.0, 50.0)))

    processed = postprocess.run(detections)
    features = solution._to_features(processed, regions)
    out_file = output.write_result(features, tmp_path / "result.geojson")

    report = validate_output.validate(out_file)
    assert report["defects"] == [], report["defects"]

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert all(feat["geometry"]["type"] == "Polygon" for feat in data["features"])

    props = [feat["properties"] for feat in data["features"]]
    assert {p["class_name"] for p in props} == set(CLASSES)
    assert {p["region_name"] for p in props} == {REGION_NAME}
    assert all(set(p) == _PROP_KEYS for p in props)

    fids = [p["fid"] for p in props]
    assert len(fids) == len(set(fids))
    assert all(re.fullmatch(rf"{REGION_NAME}:(li|or|maggr):\d+", fid) for fid in fids)
    for markup in ("li", "or", "maggr"):  # нумерация в паре сплошная, с 1
        numbers = sorted(
            int(fid.rsplit(":", 1)[1]) for fid in fids if f":{markup}:" in fid
        )
        assert numbers == list(range(1, len(numbers) + 1))

    li_first = [p for p in props if p["class_name"] == first_cls and p["markup_type"] == "li"]
    or_first = [p for p in props if p["class_name"] == first_cls and p["markup_type"] == "or"]
    assert len(li_first) == 1  # дубль внутри пары подавлен
    assert len(or_first) == 2  # «дубль» другой пары выжил (инвариант 4.5)
