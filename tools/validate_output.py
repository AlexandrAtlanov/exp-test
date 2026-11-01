"""Оффлайн-валидатор результата (запускается локально, на платформе его нет).

Синхронизирован с output v2:
  Д2 — лимит «≤ 3000» считается по СУММЕ незамкнутых позиций всех колец
        геометрии (та же трактовка, что в output — рассинхрона «3001» больше нет);
  Д5 — region_name обязателен: непустая строка (bad_region_name);
  Д7 — проверка "type": "Feature" (bad_feature_type) и поимка ошибок
        GEOS/shapely при разборе геометрии (geometry_error) вместо трейсбека.

Выход: код 0 — дефектов нет; 1 — есть дефекты (отчёт в stdout).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.errors import ShapelyError
from shapely.geometry import shape

CLASSES = {
    "kurgany_povrezhdennye": 30,
    "kurgany_tselye": 20,
    "gorodishcha": 15,
    "arkhitektury": 10,
    "fortifikatsii": 5,
    "object_poly": 10,
    "finds_points": 10,
}
MARKUP_TYPES = {"li", "ae", "sp_or", "or", "maggr", "etgr", "terfl"}
logger = logging.getLogger("koz3.validate_output")
MAX_FEATURES = 20_000
MAX_RING_POSITIONS = 3_000
SIZE_LIMIT_BYTES = 100 * 1024 * 1024
POLYGONAL = {"Polygon", "MultiPolygon"}
POINTAL = {"Point", "MultiPoint"}


def _iter_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    gt = geometry.get("type")
    coords = geometry.get("coordinates")
    if gt == "Polygon":
        rings.extend(r for r in coords if isinstance(r, list))
    elif gt == "MultiPolygon":
        for poly in coords:
            rings.extend(r for r in poly if isinstance(r, list))
    return rings


def _check_finite(node: Any) -> bool:
    if isinstance(node, bool):
        return False
    if isinstance(node, (int, float)):
        return math.isfinite(node)
    if isinstance(node, list):
        return all(_check_finite(item) for item in node)
    return False


def geometry_defects(geometry: Any) -> list[str]:
    """Дефекты геометрии одной фичи. Д7: ошибки GEOS — кодом, не трейсбеком."""
    defects: list[str] = []
    if not isinstance(geometry, dict):
        return ["bad_geometry"]
    gt = geometry.get("type")
    if gt not in POLYGONAL | POINTAL:
        return ["bad_geometry_type"]
    if not _check_finite(geometry.get("coordinates")):
        defects.append("non_finite_coordinates")
    if gt in POLYGONAL:
        rings = _iter_rings(geometry)
        positions = sum(len(r) - 1 for r in rings)
        if positions > MAX_RING_POSITIONS:
            defects.append("ring_positions_overflow")
        if any(r[0] != r[-1] for r in rings if len(r) >= 2):
            defects.append("unclosed_ring")
    try:
        geom = shape(geometry)
        if not geom.is_valid:
            defects.append("geometry_invalid")
    except (ShapelyError, ValueError, TypeError, KeyError, IndexError) as exc:  # Д7
        defects.append("geometry_error")
        logger.debug("geometry_error: %s: %r", type(exc).__name__, exc)
    return defects


def feature_defects(feature: Any) -> list[str]:
    """Дефекты одной фичи FeatureCollection."""
    defects: list[str] = []
    if not isinstance(feature, dict) or feature.get("type") != "Feature":  # Д7
        return ["bad_feature_type"]
    props = feature.get("properties")
    if not isinstance(props, dict):
        return ["bad_properties"]
    if props.get("class") not in CLASSES:
        defects.append("bad_class")
    if props.get("markup_type") not in MARKUP_TYPES:
        defects.append("bad_markup_type")
    confidence = props.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        defects.append("bad_confidence")
    region = props.get("region_name")
    if not isinstance(region, str) or not region.strip():
        defects.append("bad_region_name")
    defects.extend(geometry_defects(feature.get("geometry")))
    return defects


def validate(path: str | Path) -> dict[str, Any]:
    """Полная проверка файла. Возвращает отчёт; defects пуст <=> файл годен."""
    path = Path(path)
    report: dict[str, Any] = {"path": str(path), "defects": [], "feature_count": 0}
    if not path.is_file():
        return {**report, "defects": ["file_missing"]}
    size = path.stat().st_size
    report["size_mb"] = round(size / 2**20, 2)
    if size > SIZE_LIMIT_BYTES:
        report["defects"].append("file_too_large")
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {**report, "defects": [f"unreadable: {exc}"]}
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        report["defects"].append("not_feature_collection")
        return report
    features = payload.get("features")
    if not isinstance(features, list):
        report["defects"].append("no_features_list")
        return report
    report["feature_count"] = len(features)
    if len(features) > MAX_FEATURES:
        report["defects"].append("too_many_features")
    by_code: Counter[str] = Counter()
    examples: list[str] = []
    for index, feature in enumerate(features):
        for code in feature_defects(feature):
            by_code[code] += 1
            if len(examples) < 20:
                examples.append(f"feature[{index}]: {code}")
    if by_code:
        report["defects"].append("feature_defects")
        report["defects_by_code"] = dict(by_code)
        report["examples"] = examples
    report["classes"] = dict(
        Counter(
            f.get("properties", {}).get("class", "?")
            for f in features
            if isinstance(f, dict)
        )
    )
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "result.geojson"
    report = validate(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["defects"] else 1


if __name__ == "__main__":
    raise SystemExit(main())