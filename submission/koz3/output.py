"""Модуль записи результата (output) — v2.

Контракт: ``write_result`` вызывается после каждой пары детекторов
и обязан оставлять на диске валидный GeoJSON FeatureCollection в EPSG:3857.
Входные фичи считаются ЗАМОРОЖЕННЫМИ после первого вызова ``write_result``
(кэш подготовки привязан к ``id`` объекта).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from pyproj.exceptions import CRSError
from shapely import force_2d, make_valid
from shapely import transform as shapely_transform
from shapely.errors import ShapelyError
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from . import config

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger("koz3.output")

def _cfg(*names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            return value
    return default


_FALLBACK_CLASSES: dict[str, int] = {
    "kurgany_povrezhdennye": 30,
    "kurgany_tselye": 20,
    "gorodishcha": 15,
    "arkhitektury": 10,
    "fortifikatsii": 5,
    "object_poly": 10,
    "finds_points": 10,
}


def _resolve_classes() -> dict[str, int]:
    raw = _cfg("CLASS_WEIGHTS", "CLASSES", "TARGET_CLASSES")
    if raw is None:
        return dict(_FALLBACK_CLASSES)
    if isinstance(raw, Mapping):
        return {str(k): int(v) for k, v in raw.items()}
    return {str(k): 1 for k in raw}


_CLASSES: dict[str, int] = _resolve_classes()
_MARKUP_TYPES: frozenset[str] = frozenset(_cfg("MARKUP_TYPES", "MARKUP_TYPE_VALUES", default=("li", "ae", "sp_or", "or", "maggr", "etgr", "terfl"),))
MAX_FEATURES: int = int(_cfg("MAX_FEATURES", default=20_000))
MAX_RING_POSITIONS: int = int(_cfg("MAX_RING_POSITIONS", "MAX_RING_VERTICES", default=3_000))
COORD_PRECISION: int = int(_cfg("OUTPUT_COORD_PRECISION", "COORD_PRECISION", default=2))
OUTPUT_SIZE_LIMIT_BYTES: int = int(_cfg("OUTPUT_SIZE_LIMIT_BYTES", default=95 * 1024 * 1024))
OUTPUT_SIZE_HARD_LIMIT_BYTES: int = 100 * 1024 * 1024

_VERTEX_SIMPLIFY_START_M3857: int = (2_400)
_VERTEX_SIMPLIFY_BASE_TOL_M3857: float = 5.0
_VERTEX_SIMPLIFY_MAX_TOL_M3857: float = 25.0

_PREPARE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ShapelyError,
    CRSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    OverflowError,
)

_SRC_CRS: Any = _cfg("SRC_CRS", "DETECTION_CRS", "SOURCE_CRS", "CRS")
_TRANSFORMER: Transformer | None = None
if _SRC_CRS not in (None, "", "EPSG:3857"):
    _TRANSFORMER = Transformer.from_crs(_SRC_CRS, "EPSG:3857", always_xy=True)
_warned_no_crs = False

_prepared_cache: dict[int, tuple[dict[str, Any] | None, tuple[str, ...]]] = {}
_pinned: list[Any] = []
_json_cache: dict[int, str] = {}

last_stats: dict[str, Any] = {}

def resolve_output_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Возвращает целевой файл *.geojson и создаёт каталог.

    Д12: если родительский путь существует, но является обычным файлом,
    раньше mkdir бросал невнятный FileExistsError — теперь понятная
    NotADirectoryError с текстом.
    """
    if path is None:
        result = Path("result.geojson")
    else:
        result = Path(path).expanduser()
        if result.is_dir():
            result = result / "result.geojson"
    parent = result.parent
    if parent.exists() and not parent.is_dir():  # Д12
        raise NotADirectoryError(
            f"выходной путь {parent} существует, но не является директорией"
        )
    parent.mkdir(parents=True, exist_ok=True)
    return result


def _write_json_atomic(payload: Any, path: Path) -> None:
    """Атомарная запись произвольного JSON (tmp + os.replace, одна ФС).

    Д10: при исключении осиротевший .tmp удаляется. fsync не делаем:
    целевой файл защищён os.replace, риск — SIGKILL, а не потеря питания.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                payload, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _feature_json(item: dict[str, Any]) -> str:
    """Д4: JSON-строка фичи с кэшем — сериализация каждой фичи один раз.

    id(item) стабилен: prepared-фичи живут в _prepared_cache до конца прогона.
    """
    cached = _json_cache.get(id(item))
    if cached is None:
        cached = json.dumps(
            item, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        _json_cache[id(item)] = cached
    return cached


def _write_features_json_atomic(items: list[dict[str, Any]], path: Path) -> None:
    """Атомарная запись FeatureCollection из кэшированных строк фич.

    Д1 (бонус): компактные разделители (",", ":"). Д10: уборка .tmp при исключении.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write('{"type":"FeatureCollection","features":[')
            for index, item in enumerate(items):
                if index:
                    fh.write(",")
                fh.write(_feature_json(item))
            fh.write("]}")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise

def _reproject(coords: np.ndarray) -> np.ndarray:
    """Адаптер пакетной репроекции для shapely.transform: (N, 2) -> (N, 2)."""
    assert _TRANSFORMER is not None
    x, y = _TRANSFORMER.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def _round_coords(coords: np.ndarray) -> np.ndarray:
    """Д4: векторное округление — один вызов np.round на всю геометрию.

    Заодно отсекаем не-конечные координаты (NaN/inf): в JSON их писать нельзя.
    """
    if not np.isfinite(coords).all():
        raise ValueError("non-finite coordinates")
    return np.round(coords, COORD_PRECISION)


def _polygonal_part(geom: BaseGeometry) -> BaseGeometry | None:
    """Оставляет от результата make_valid только полигональные части."""
    if isinstance(geom, (Polygon, MultiPolygon, Point, MultiPoint)):
        return geom
    if hasattr(geom, "geoms"):
        parts: list[Polygon] = []
        for g in geom.geoms:
            if isinstance(g, Polygon):
                parts.append(g)
            elif isinstance(g, MultiPolygon):
                parts.extend(g.geoms)
        if parts:
            return MultiPolygon(parts) if len(parts) > 1 else parts[0]
    return None


def _fix_valid(geom: BaseGeometry) -> BaseGeometry | None:
    """make_valid + отсечение неполигональных частей; None если не осталось."""
    if geom.is_valid:
        return geom
    return _polygonal_part(make_valid(geom))


def _strip_interiors(geom: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Д2: снимает interiors всегда.

    Риск платформенной трактовки «сумма вершин всех колец» исчезает вместе
    с дырками, а для метрики дырки только мешают: они уменьшают площадь
    пересечения с разметкой без дырок (порог 0.5 по пересечению/меньшая).
    """
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior) if geom.interiors else geom
    if not any(p.interiors for p in geom.geoms):
        return geom
    return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])


def _count_positions(geom: Polygon | MultiPolygon) -> int:
    """Д2: СУММА незамкнутых позиций по всем кольцам всех частей геометрии."""
    total = 0
    polys = geom.geoms if isinstance(geom, MultiPolygon) else (geom,)
    for p in polys:
        total += len(p.exterior.coords) - 1
        total += sum(len(r.coords) - 1 for r in p.interiors)
    return total


def _decimate_ring(coords: np.ndarray, k: int) -> np.ndarray | None:
    """Равномерная децимация замкнутого кольца (каждый k-й), с замыканием."""
    core = coords[:-1]
    picked = core[::k]
    if len(picked) < 3:
        return None
    return np.vstack([picked, picked[:1]])


def _decimate_geometry(geom: Polygon | MultiPolygon, cap: int) -> BaseGeometry | None:
    """Гарантированный рубеж лимитера: равномерная децимация всех колец."""
    n = _count_positions(geom)
    k = max(2, math.ceil(n / cap))
    for _ in range(6):
        polys = geom.geoms if isinstance(geom, MultiPolygon) else (geom,)
        parts: list[Polygon] = []
        for p in polys:
            ring = np.asarray(p.exterior.coords)
            dec = _decimate_ring(ring, k)
            if dec is not None:
                parts.append(Polygon(dec))
        if not parts:
            return None
        out: BaseGeometry = MultiPolygon(parts) if len(parts) > 1 else parts[0]
        if _count_positions(out) <= cap:
            return _fix_valid(out)
        k += 1
    return _fix_valid(out)


def _limit_geometry_vertices(geom: Polygon | MultiPolygon) -> BaseGeometry | None:
    """Д2 + Д8: держит СУММУ позиций всех колец ≤ MAX_RING_POSITIONS.

    Ступени: (1) упредительное simplify при n > _VERTEX_SIMPLIFY_START_M3857;
    (2) бисекция допуска вплоть до _VERTEX_SIMPLIFY_MAX_TOL_M3857;
    (3) равномерная децимация. Все допуски — в метрах EPSG:3857
    (искажены в 1/cos(широты) раз; это аварийный лимитер, не генерализация).
    """
    if _count_positions(geom) <= MAX_RING_POSITIONS:
        return geom
    if _count_positions(geom) > _VERTEX_SIMPLIFY_START_M3857:
        geom = geom.simplify(_VERTEX_SIMPLIFY_BASE_TOL_M3857, preserve_topology=True)
    if _count_positions(geom) <= MAX_RING_POSITIONS:
        return geom
    lo, hi = _VERTEX_SIMPLIFY_BASE_TOL_M3857, _VERTEX_SIMPLIFY_MAX_TOL_M3857
    best: BaseGeometry | None = None
    for _ in range(8):
        mid = (lo + hi) / 2.0
        cand = geom.simplify(mid, preserve_topology=True)
        if _count_positions(cand) <= MAX_RING_POSITIONS:
            best, hi = cand, mid
        else:
            lo = mid
    if best is not None and best.is_valid and not best.is_empty and best.area > 0:
        return best
    return _decimate_geometry(geom, MAX_RING_POSITIONS)


def _dedupe_ring(coords: np.ndarray) -> np.ndarray | None:
    """Убирает подряд идущие дубликаты вершин (появляются после округления)."""
    core = coords[:-1]
    if len(core) < 3:
        return None
    keep = np.empty(len(core), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(core[1:] != core[:-1], axis=1)
    core = core[keep]
    if len(core) < 3:
        return None
    return np.vstack([core, core[:1]])


def _dedupe_rings(geom: Polygon | MultiPolygon) -> BaseGeometry | None:
    polys = geom.geoms if isinstance(geom, MultiPolygon) else (geom,)
    parts: list[Polygon] = []
    for p in polys:
        ring = np.asarray(p.exterior.coords)
        ded = _dedupe_ring(ring)
        if ded is not None:
            parts.append(Polygon(ded))
    if not parts:
        return None
    return MultiPolygon(parts) if len(parts) > 1 else parts[0]

def _prepare_feature(feature: Any) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Готовит одну фичу к записи. Возвращает (json-фича | None, коды дефектов).

    Коды синхронизированы с validate_output.py: bad_class, bad_markup_type,
    bad_confidence, bad_region_name (Д5), bad_geometry, bad_geometry_type.
    """
    global _warned_no_crs
    if not isinstance(feature, Mapping):
        return None, ("bad_feature",)
    props = feature.get("properties")
    props = props if isinstance(props, Mapping) else {}
    cls = props.get("class_name")
    fid = props.get("fid")
    markup_type = props.get("markup_type")
    confidence = props.get("confidence")
    region_name = props.get("region_name")

    codes: list[str] = []
    if cls not in _CLASSES:
        codes.append("bad_class")
    if markup_type not in _MARKUP_TYPES:
        codes.append("bad_markup_type")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        codes.append("bad_confidence")
    if not isinstance(region_name, str) or not region_name.strip():
        codes.append("bad_region_name")
    geom_dict = feature.get("geometry")
    if not isinstance(geom_dict, Mapping) or "coordinates" not in geom_dict:
        codes.append("bad_geometry")
    if codes:
        return None, tuple(codes)

    try:
        geom = shape(geom_dict)
    except (ShapelyError, ValueError, TypeError, KeyError, IndexError) as exc:
        logger.debug("bad_geometry: %s: %r", type(exc).__name__, exc)
        return None, ("bad_geometry",)
    if geom.is_empty:
        return None, ("bad_geometry",)

    if geom.has_z:
        geom = force_2d(geom)

    if isinstance(geom, (Point, MultiPoint)):
        pass
    elif isinstance(geom, (Polygon, MultiPolygon)):
        geom = _strip_interiors(geom)
    else:
        return None, ("bad_geometry_type",)

    if _TRANSFORMER is not None:
        geom = shapely_transform(geom, _reproject)
    elif _SRC_CRS is None and not _warned_no_crs:
        _warned_no_crs = True
        logger.warning("SRC_CRS не задана в config — считаю геометрии уже в EPSG:3857")

    if isinstance(geom, (Polygon, MultiPolygon)):
        geom = _fix_valid(geom)
        if geom is None or geom.is_empty:
            return None, ("bad_geometry",)
        geom = _limit_geometry_vertices(geom)
        if geom is None or geom.is_empty:
            return None, ("bad_geometry",)
    try:
        geom = shapely_transform(geom, _round_coords)
    except ValueError:
        return None, ("bad_geometry",)
    if isinstance(geom, (Polygon, MultiPolygon)):
        geom = _dedupe_rings(geom)
        if geom is None:
            return None, ("bad_geometry",)
        geom = _fix_valid(geom)
        if geom is None or geom.is_empty or geom.area <= 0:
            return None, ("bad_geometry",)

    properties: dict[str, Any] = {
        "class_name": cls,
        "markup_type": markup_type,
        "confidence": round(float(confidence), 6),
        "region_name": region_name,
    }
    if isinstance(fid, str) and fid:
        properties["fid"] = fid
    prepared = {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": properties,
    }
    return prepared, ()

def _feature_confidence(feature: Mapping) -> float:
    return float(feature["properties"]["confidence"])


def _cap_features(
    prepared: list[dict[str, Any]], cap: int, reason: str
) -> list[dict[str, Any]]:
    """Обрезает отсортированный по убыванию confidence список до cap.

    Д3: не вслепую по confidence. Каждому присутствующему классу
    гарантируется квота floor = max(1, cap // (2 * K)); остальной бюджет
    заполняется по глобальному confidence. Пропущенный класс = F1 0
    с полным весом, FP вдвое дешевле пропуска — представительство
    дешевле потери класса. В лог пишется разбивка срезанного по классам.
    """
    if len(prepared) <= cap:
        return list(prepared)
    if cap <= 0:
        return []
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in prepared:
        by_class[feature["properties"]["class_name"]].append(feature)
    k = len(by_class)
    if k >= cap:
        kept = sorted(prepared, key=_feature_confidence, reverse=True)[:cap]
    else:
        floor = max(1, cap // (2 * k))
        kept: list[dict[str, Any]] = []
        used: dict[str, int] = {}
        for cls, items in by_class.items():
            take = min(floor, len(items))
            kept.extend(items[:take])
            used[cls] = take
        rest = [f for cls, items in by_class.items() for f in items[used[cls] :]]
        rest.sort(key=_feature_confidence, reverse=True)
        kept.extend(rest[: cap - len(kept)])
        kept.sort(key=_feature_confidence, reverse=True)
    kept_ids = {id(f) for f in kept}
    cut_by_class = Counter(
        f["properties"]["class_name"] for f in prepared if id(f) not in kept_ids
    )
    logger.warning(
        "обрезка (%s): %d -> %d фич; срезано по классам: %s",
        reason,
        len(prepared),
        len(kept),
        dict(cut_by_class),
    )
    return kept


def _fit_size_limit(features: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Д1: итеративно уменьшает список, пока файл не влезет в лимит.

    Срез идёт с запасом 10% и через _cap_features (квоты классов — Д3),
    а не вслепую. Если даже минимальный файл больше жёсткого лимита —
    записываем лучшее возможное и кричим CRITICAL.
    """
    _write_features_json_atomic(features, path)
    size = path.stat().st_size
    for _ in range(6):
        if size <= OUTPUT_SIZE_LIMIT_BYTES or not features:
            break
        keep = max(1, int(len(features) * OUTPUT_SIZE_LIMIT_BYTES / size * 0.90))
        features = _cap_features(
            features, keep, f"лимит размера {OUTPUT_SIZE_LIMIT_BYTES / 2**20:.1f} МБ"
        )
        _write_features_json_atomic(features, path)
        size = path.stat().st_size
    if size > OUTPUT_SIZE_HARD_LIMIT_BYTES:
        logger.critical(
            "файл %.1f МБ превышает жёсткий лимит 100 МБ даже после обрезки "
            "до %d фич — записано лучшее возможное",
            size / 2**20,
            len(features),
        )
    return features

def write_result(
    features: Any, output_path: str | os.PathLike[str] | None = None
) -> Path:
    """Готовит фичи и атомарно пишет result.geojson.

    Вызывается после каждой пары детекторов (контракт раздела 5): файл на
    диске всегда валиден. Возвращает путь к файлу. Ошибки самой записи
    не глотаются: при их падении предыдущий чекпоинт остаётся цел.
    """
    global last_stats
    t0 = time.perf_counter()
    result = resolve_output_path(output_path)

    feature_list = list(features) if features is not None else []
    prepared: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"input": len(feature_list)}
    for feat in feature_list:
        key = id(feat)
        cached = _prepared_cache.get(key)
        if cached is None:
            try:
                cached = _prepare_feature(feat)
            except _PREPARE_EXCEPTIONS as exc:
                logger.exception(
                    "непредвиденная ошибка подготовки фичи, дропаю: %r", exc
                )
                cached = (None, ("error",))
            _pinned.append(feat)
            _prepared_cache[key] = cached
        item, codes = cached
        if item is None:
            for code in codes:
                stats[f"dropped_{code}"] = stats.get(f"dropped_{code}", 0) + 1
        else:
            prepared.append(item)

    prepared.sort(key=_feature_confidence, reverse=True)
    prepared = _cap_features(
        prepared, MAX_FEATURES, f"MAX_FEATURES={MAX_FEATURES}"
    )
    prepared = _fit_size_limit(prepared, result)

    size = result.stat().st_size
    stats.update(
        written=len(prepared),
        size_mb=round(size / 2**20, 2),
        seconds=round(time.perf_counter() - t0, 2),
        classes=dict(Counter(f["properties"]["class_name"] for f in prepared)),
    )
    last_stats = stats
    logger.info("write_result -> %s: %s", result, stats)
    return result