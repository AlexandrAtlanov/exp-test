"""
Скрипт расчета итоговой метрики Archaeology Assistive F1 для КОЗ-3.

Вход: два GeoJSON FeatureCollection (predictions, ground truth)
Ожидается объединенный формат: у features в properties есть поля:
- region_name (обязательно)
- sub_region_name (опционально)
- class_name (обязательно)
- confidence (опционально, по умолчанию 1.0)

Алгоритм (ТР КОЗ-3, Приложение 4):
- Фильтрует классы: оставляет только целевые
- Координаты: автоприведение EPSG:3857 -> UTM (EPSG:326xx/327xx) для
  метрических расчётов (INPUT_CRS_MODE: auto|webmercator|utm)
- Мягкий матчинг: GT-полигон — cov = area(P∩G)/min(area) >= COVERAGE_THRESHOLD
  И внутренняя (представительная) точка предсказания внутри эталона;
  GT-точка — расстояние центроид-точка <= POINT_R_TOLERANCE
- Жадный матчинг по confidence (one-to-one)
- Precision: лишнее срабатывание учитывается с коэффициентом 0.5
  (tp / (tp + 0.5*fp + epsilon))
- F1-score по классам, взвешенное усреднение по присутствующим в GT классам
  (нормировка весов: если класса в GT нет, его вес исключается, решение 08.10.2026)
- Справочные показатели (auxiliary; на final_score НЕ влияют): качество контура
  по TP-парам (Boundary IoU, Hausdorff95, PoLiS, метры), полнота/точность по
  типам данных (markup_type), размерные категории GT-полигонов по площади (м²)

Вывод: final_score score в stdout (и опционально JSON с деталями).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typing import Union

try:
    from shapely.geometry import Polygon, MultiPolygon, Point, shape
    from shapely.validation import make_valid
except ImportError:
    raise ImportError("Требуется библиотека shapely. Установите её: pip install shapely")

try:
    from pyproj import Transformer
    from shapely.ops import transform as _shapely_transform
except ImportError:
    raise ImportError("Требуется библиотека pyproj. Установите её: pip install pyproj")


logger = logging.getLogger(__name__)


# Реестр классов и параметры метрики — из единого конфига (config.py).
# Имена оставлены прежними: их импортируют metrics_calculator, fp_matching,
# simple_solution/gt_solution.py и тесты (в том числе проверяют тождество объектов).
try:
    from config import (  # noqa: E402  (импорт после определения logger)
        ALL_VALID_CLASSES,
        CLASS_NAME_MAPPING,
        CLASS_WEIGHTS,
        DATA_TYPE_GROUPS,
        DEFAULT_DATA_TYPE_GROUP,
        data_type_group,
        get_config,
        normalize_class_name,
    )

    PLATFORM_CONFIG_LOADED = True
except ImportError as exc:  # одиночная сборка скрипта: config.py рядом нет
    PLATFORM_CONFIG_LOADED = False
    logger.warning(
        "config.py недоступен (%s) — используются встроенные дефолты метрики; "
        "для расчёта ровно как на платформе положите config.py рядом со скриптом",
        exc,
    )

    # --- Копии реестра классов из config.py (при правке config.py менять синхронно) ---
    CLASS_WEIGHTS = {
        "kurgany_povrezhdennye": 30.0,
        "kurgany_tselye": 20.0,
        "gorodishcha": 15.0,
        "arkhitektury": 10.0,
        "object_poly": 10.0,
        "fortifikatsii": 5.0,
        "finds_points": 10.0,
    }

    # Синонимы названий классов (копия config.CLASS_SYNONYMS; ключи — нижний регистр).
    _CLASS_SYNONYMS = {
        # Курганы целые
        "курганы": "kurgany_tselye",
        "курганы_целые": "kurgany_tselye",
        "целые": "kurgany_tselye",
        # Курганы повреждённые
        "курганы_поврежденные": "kurgany_povrezhdennye",
        "курганы_разрушенные": "kurgany_povrezhdennye",
        "поврежденные": "kurgany_povrezhdennye",
        "разрушенные": "kurgany_povrezhdennye",
        # Фортификации
        "фортификация": "fortifikatsii",
        "фортификации": "fortifikatsii",
        # Городища
        "городище": "gorodishcha",
        "городища": "gorodishcha",
        # Архитектуры
        "архитектура": "arkhitektury",
        "архитектуры": "arkhitektury",
        # Геофизика: точечные аномалии (FindsPoints)
        "findspoints": "finds_points",
        "findspoint": "finds_points",
        # Геофизика: полигональные аномалии (ObjectPoly)
        "objectpoly": "object_poly",
        "objectspoly": "object_poly",
    }

    # Игнорируемые названия классов (копия config.IGNORED_CLASSES).
    _IGNORED_CLASSES = (
        "StructrPoly", "StructPoly", "StructurePoly",
    )

    def _build_class_mapping() -> Dict[str, Optional[str]]:
        """Копия config._build_class_mapping: «название -> канонический ключ | None»."""
        mapping: Dict[str, Optional[str]] = {}
        for key in CLASS_WEIGHTS:
            mapping[key] = key
        for raw, key in _CLASS_SYNONYMS.items():
            mapping[raw] = key
            mapping[raw.lower()] = key
        for ignored in _IGNORED_CLASSES:
            mapping[ignored] = None
            mapping[ignored.lower()] = None
        return mapping

    ALL_VALID_CLASSES = list(CLASS_WEIGHTS.keys())
    CLASS_NAME_MAPPING = _build_class_mapping()

    # Группы типов данных (копия config.DATA_TYPE_GROUPS / DEFAULT_DATA_TYPE_GROUP).
    DATA_TYPE_GROUPS = {
        "la": "лидар",
        "li": "лидар",
        "ae": "аэрофото",
        "german_ae": "аэрофото",
        "sp_or": "спутник",
        "spor": "спутник",
        "or": "ортофото",
        "maggr": "геофизика",
        "etgr": "геофизика",
        "geogr": "геофизика",
        "terfl": "геофизика",
    }
    DEFAULT_DATA_TYPE_GROUP = "прочее"

    def normalize_class_name(raw_name: str) -> Optional[str]:
        """Копия config.normalize_class_name (None — только для игнор-листа)."""
        if raw_name in CLASS_NAME_MAPPING:
            return CLASS_NAME_MAPPING[raw_name]
        low = str(raw_name).strip().lower()
        if low in CLASS_NAME_MAPPING:
            return CLASS_NAME_MAPPING[low]
        return raw_name

    def data_type_group(markup_type: Optional[str]) -> str:
        """Копия config.data_type_group (неизвестное -> DEFAULT_DATA_TYPE_GROUP)."""
        if markup_type is None:
            return DEFAULT_DATA_TYPE_GROUP
        key = str(markup_type).strip().lower()
        if not key:
            return DEFAULT_DATA_TYPE_GROUP
        return DATA_TYPE_GROUPS.get(key, DEFAULT_DATA_TYPE_GROUP)

    class _FallbackConfig:
        """Дефолты config.Config для одиночной сборки (config.py рядом нет).

        При настройках по умолчанию скор и config_hash одиночного расчёта
        совпадают с платформенными (проверяет test_standalone_compute_metrics).
        """

        def __init__(self) -> None:
            self.COVERAGE_THRESHOLD = 0.5
            self.POINT_R_TOLERANCE = 2.0
            self.INPUT_CRS_MODE = "auto"
            self.CIRCLE_POINTS = 32
            self.EPSILON = 1e-6
            self.CONTOUR_BAND_RATIO = 0.02
            self.CONTOUR_SAMPLES = 200
            self.SIZE_BINS = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
            self.METRIC_VERSION = "koz3-1.5.0"
            self.MODALITY_MODE = "off"
            self.STRICT_CRS = False
            self.MODALITIES = {
                "la": 1.0, "li": 1.0, "ae": 1.0, "spor": 1.0, "or": 1.0,
                "maggr": 1.0, "etgr": 1.0, "geogr": 1.0, "terfl": 1.0,
                "unknown": 1.0, "default": 1.0,
            }
            self.MAX_FILE_SIZE_MB = 100
            self.MAX_FEATURES_PER_FILE = 20000
            self.MAX_BODY_SIZE_MB = 100
            self.MAX_POLYGON_POINTS = 3000
            self.MAX_POLYGON_AREA = 1e12
            self.MAX_VALIDATION_ISSUES = 50
            self.REPAIR_ENABLED = True
            self.REPAIR_MAX_ATTEMPTS = 3
            self.REPAIR_WAIT_SECONDS = 1.0
            self.REPAIR_STABILITY_CHECKS = 2

        def metric_summary(self) -> Dict[str, Any]:
            """Формат и содержимое — как config.Config.metric_summary()."""
            import hashlib

            payload: Dict[str, Any] = {
                "metric_version": self.METRIC_VERSION,
                "target_classes": list(CLASS_WEIGHTS.keys()),
                "class_weights": dict(CLASS_WEIGHTS),
                "coverage_threshold": self.COVERAGE_THRESHOLD,
                "input_crs_mode": self.INPUT_CRS_MODE,
                "r_tolerance": self.POINT_R_TOLERANCE,
                "epsilon": self.EPSILON,
                "circle_points": self.CIRCLE_POINTS,
                "strict_crs": self.STRICT_CRS,
                "modality_mode": self.MODALITY_MODE,
                "modalities": dict(self.MODALITIES),
                "limits": {
                    "max_file_size_mb": self.MAX_FILE_SIZE_MB,
                    "max_features_per_file": self.MAX_FEATURES_PER_FILE,
                    "max_body_size_mb": self.MAX_BODY_SIZE_MB,
                    "max_polygon_points": self.MAX_POLYGON_POINTS,
                    "max_polygon_area": self.MAX_POLYGON_AREA,
                    "max_validation_issues": self.MAX_VALIDATION_ISSUES,
                },
                "auxiliary_metrics": {
                    "contour_band_ratio": self.CONTOUR_BAND_RATIO,
                    "contour_samples": self.CONTOUR_SAMPLES,
                    "size_bins": list(self.SIZE_BINS),
                    "data_type_groups": dict(DATA_TYPE_GROUPS),
                },
                "repair": {
                    "enabled": self.REPAIR_ENABLED,
                    "max_attempts": self.REPAIR_MAX_ATTEMPTS,
                    "wait_seconds": self.REPAIR_WAIT_SECONDS,
                    "stability_checks": self.REPAIR_STABILITY_CHECKS,
                },
            }
            payload["weights_sum"] = round(sum(payload["class_weights"].values()), 4)
            payload["config_hash"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            return payload

    _FALLBACK_CONFIG = _FallbackConfig()

    def get_config() -> Any:
        """Синглтон запасного конфига (аналог config.get_config, дефолты Config)."""
        return _FALLBACK_CONFIG

_METRIC_CFG = get_config()


def split_merged_by_source_and_class(
    geojson_data: Dict[str, Any],
) -> Dict[str, Dict[str, Dict]]:
    """Разбивает объединенный FeatureCollection на {source: {class: FeatureCollection}}."""
    result: Dict[str, Dict[str, Dict]] = {}
    features = geojson_data.get("features", [])
    for feature in features:
        props = feature.get("properties", {}) or {}
        region_name = props.get("region_name", "default_source")
        sub_region_name = props.get("sub_region_name", "")
        if sub_region_name and sub_region_name.strip():
            source_name = f"{region_name}__{sub_region_name}"
        else:
            source_name = region_name
        raw_class = props.get("class_name", "unknown")
        class_name = normalize_class_name(raw_class)
        if class_name is None:
            continue  # Игнорируемый класс (artefakty_lidara и т.п.)

        if source_name not in result:
            result[source_name] = {}
        if class_name not in result[source_name]:
            result[source_name][class_name] = {"type": "FeatureCollection", "features": []}
        result[source_name][class_name]["features"].append(feature)
    return result


def _extract_geometry(feature: Dict) -> Optional[Union[Polygon, Point]]:
    """Извлекает Shapely Polygon или Point из GeoJSON feature."""
    geometry = feature.get("geometry")
    if not geometry:
        return None
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if not coords:
        return None
    try:
        if geom_type == "Point":
            if len(coords) >= 2:
                return Point(coords[0], coords[1])
            return None
        elif geom_type == "Polygon":
            poly = Polygon(coords[0], coords[1:] if len(coords) > 1 else None)
        elif geom_type == "MultiPolygon":
            # Берем самый большой полигон из мультиполигона
            polys = []
            for poly_coords in coords:
                p = Polygon(poly_coords[0], poly_coords[1:] if len(poly_coords) > 1 else None)
                if not p.is_valid:
                    p = make_valid(p)
                if p.is_valid and not p.is_empty:
                    polys.append(p)
            if not polys:
                return None
            poly = max(polys, key=lambda p: p.area)
        else:
            return None
        if not poly.is_valid:
            poly = make_valid(poly)
        # make_valid может вернуть MultiPolygon
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_valid and not poly.is_empty:
            return poly
    except Exception:
        pass
    return None


def _extract_polygon(feature: Dict) -> Optional[Polygon]:
    """Обратная совместимость: извлекает только Polygon из GeoJSON feature."""
    geom = _extract_geometry(feature)
    if isinstance(geom, Polygon):
        return geom
    return None


def _get_confidence(feature: Dict) -> float:
    """Извлекает confidence из feature, по умолчанию 1.0."""
    props = feature.get("properties", {}) or {}
    conf = props.get("confidence", 1.0)
    try:
        return float(conf)
    except (TypeError, ValueError):
        return 1.0


def _polygon_iou(a: Polygon, b: Polygon) -> float:
    """Вычисляет IoU между двумя полигонами."""
    try:
        if not a.is_valid or not b.is_valid:
            return 0.0
        inter = a.intersection(b)
        union = a.union(b)
        if union.area == 0:
            return 0.0
        return inter.area / union.area
    except Exception:
        return 0.0


def _polygon_coverage(a: Polygon, b: Polygon) -> float:
    """Перекрытие по меньшей фигуре: cov = area(A∩B) / min(area(A), area(B)).

    Приложение 4: критерий находки для полигона — cov >= 0.5; вложение
    любой из фигур даёт cov = 1.0.
    """
    try:
        if not a.is_valid or not b.is_valid:
            return 0.0
        min_area = min(a.area, b.area)
        if min_area <= 0:
            return 0.0
        return a.intersection(b).area / min_area
    except Exception:
        return 0.0


def _point_in_mask(point: Point, mask: Polygon) -> bool:
    """Проверяет, находится ли точка внутри полигона.

    Допуск касания границы тот же, что в _inner_point_in_mask: точка на границе
    считается попавшей.
    """
    try:
        return mask.contains(point) or mask.boundary.distance(point) < 1e-10
    except Exception:
        return False


def _inner_point_in_mask(pred_poly: Polygon, gt_poly: Polygon) -> bool:
    """Проверяет, находится ли внутренняя (представительная) точка предсказания
    внутри GT-полигона.

    В отличие от центра масс (centroid), который у невыпуклых (вогнутых, угловых)
    объектов может лежать вне контура, representative_point() (point-on-surface,
    GEOS InteriorPoint) по построению всегда лежит внутри фигуры. Критерий
    Сценария А (Приложение 4) поэтому устойчив к форме объекта.
    """
    try:
        point = pred_poly.representative_point()
        return gt_poly.contains(point) or gt_poly.boundary.distance(point) < 1e-10
    except Exception:
        return False


# ===========================================================================
# UTM: метрическая проекция для расчёта площадей и расстояний (Приложение 4)
# ===========================================================================

_UTM_ABS_X_THRESHOLD = 1_000_000  # «большие» x = реальные метры EPSG:3857
_TRANSFORMER_3857_TO_WGS84: Optional[Transformer] = None
_TRANSFORMER_3857_TO_UTM: Dict[int, Transformer] = {}


def _geom_max_abs_x(geom: Any) -> float:
    """Максимум |x| по всем координатам геометрии."""
    try:
        minx, _, maxx, _ = geom.bounds
        return max(abs(minx), abs(maxx))
    except Exception:
        return 0.0


def _to_utm_zone_epsg(lon: float, lat: float) -> int:
    """EPSG-код UTM-зоны: 326xx — северное полушарие, 327xx — южное."""
    zone = int((lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if lat >= 0 else 32700
    return base + zone


def _get_transformer_3857_to_wgs84() -> Transformer:
    """Кэшированный трансформер EPSG:3857 -> EPSG:4326 (для выбора зоны)."""
    global _TRANSFORMER_3857_TO_WGS84
    if _TRANSFORMER_3857_TO_WGS84 is None:
        _TRANSFORMER_3857_TO_WGS84 = Transformer.from_crs(3857, 4326, always_xy=True)
    return _TRANSFORMER_3857_TO_WGS84


def _get_transformer_3857_to_utm(epsg: int) -> Transformer:
    """Кэшированный трансформер EPSG:3857 -> EPSG:326xx/327xx по зоне."""
    transformer = _TRANSFORMER_3857_TO_UTM.get(epsg)
    if transformer is None:
        transformer = Transformer.from_crs(3857, epsg, always_xy=True)
        _TRANSFORMER_3857_TO_UTM[epsg] = transformer
    return transformer


def _to_utm(geom: Any, mode: str) -> Any:
    """Приводит геометрию из EPSG:3857 к метрической проекции UTM.

    Режимы (INPUT_CRS_MODE):
    - auto: конвертировать, если координаты «большие» (max |x| > 1e6, т.е. это
      реальные метры EPSG:3857); фикстуры с малыми числами не трогаем;
    - webmercator: конвертировать всегда;
    - utm: данные уже в метрической проекции — passthrough.

    Зона определяется по центроиду геометрии (3857 -> WGS84 -> UTM EPSG).
    """
    if mode == "utm":
        return geom
    if mode == "auto" and _geom_max_abs_x(geom) <= _UTM_ABS_X_THRESHOLD:
        return geom
    try:
        centroid = geom.centroid
        lon, lat = _get_transformer_3857_to_wgs84().transform(centroid.x, centroid.y)
        epsg = _to_utm_zone_epsg(lon, lat)
        return _shapely_transform(_get_transformer_3857_to_utm(epsg).transform, geom)
    except Exception:
        return geom


def _point_to_circle(point: Point, radius: float, num_points: int = 32) -> Polygon:
    """Конвертирует Point в круговой Polygon (аппроксимация окружности).

    Args:
        point: Shapely Point
        radius: Радиус окружности в единицах координат (метры для UTM)
        num_points: Количество точек аппроксимации
    """
    coords = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        coords.append((point.x + radius * math.cos(angle),
                        point.y + radius * math.sin(angle)))
    coords.append(coords[0])
    return Polygon(coords)


# ===========================================================================
# Справочные показатели КОЗ-3 (Приложение 4): качество контура.
# Считаются по матчнутым парам (TP) в метрической проекции UTM — в метрах.
# На final_score НЕ влияют (закреплено тестом).
# ===========================================================================

def _bump(counter: Dict[str, int], key: str) -> None:
    """Инкремент счётчика по ключу (микро-хелпер справочных счётчиков)."""
    counter[key] = counter.get(key, 0) + 1


def _bbox_diagonal(geom: Any) -> float:
    """Диагональ bbox геометрии (единицы координат; для UTM — метры)."""
    minx, miny, maxx, maxy = geom.bounds
    return math.hypot(maxx - minx, maxy - miny)


def _contour_band(geom: Any, width: float) -> Any:
    """Полоса ширины width вдоль границы фигуры: B = X \\ X.buffer(-width).

    join_style=mitre — прямоугольные стыки, как в определении Boundary IoU
    (Cheng et al., 2021). Fallback на вырожденных фигурах: если эрозия пуста
    (фигура тоньше полосы), полосой считается вся фигура — для идентичных
    фигур результат остаётся 1.0, для тонких полоса не «толще» самой фигуры.
    """
    eroded = geom.buffer(-width, join_style="mitre")
    if eroded.is_empty:
        return geom
    band = geom.difference(eroded)
    return geom if band.is_empty else band


def _boundary_iou(pred_geom: Any, gt_geom: Any, band_ratio: float) -> Optional[float]:
    """Boundary IoU пары «полигон-полигон»; None — невычислимо.

    d = band_ratio × max(диагональ bbox pred, диагональ bbox gt);
    BoundaryIoU = area(B_P ∩ B_G) / area(B_P ∪ B_G). Идентичные полигоны → 1.0.
    Вырожденные случаи (нулевая ширина полосы, нулевая площадь объединения,
    исключение shapely) → None: пара пропускается в среднем (не NaN).
    """
    try:
        d = band_ratio * max(_bbox_diagonal(pred_geom), _bbox_diagonal(gt_geom))
        if d <= 0:
            return None
        band_p = _contour_band(pred_geom, d)
        band_g = _contour_band(gt_geom, d)
        union = band_p.union(band_g)
        if union.area <= 0:
            return None
        return band_p.intersection(band_g).area / union.area
    except Exception:
        return None


def _percentile95(values: List[float]) -> float:
    """95-й перцентиль с линейной интерполяцией (как numpy.percentile)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = 0.95 * (n - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _sample_boundary(geom: Any, samples: int) -> Optional[List[Point]]:
    """samples равноудалённых точек внешнего кольца; None — вырожденная граница."""
    ring = geom.exterior
    if ring.length <= 0:
        return None
    return [ring.interpolate(i / samples, normalized=True) for i in range(samples)]


def _hausdorff95(pred_geom: Any, gt_geom: Any, samples: int) -> Optional[float]:
    """Hausdorff95 пары в метрах: max(p95(A→B), p95(B→A)); None — невычислимо.

    Границы сэмплируются равноудалённо по внешнему кольцу (samples точек);
    для каждого сэмпла — расстояние до границы другой фигуры (включая
    внутренние кольца), затем 95-й перцентиль по направлению. Симметризация
    через max — стандартное определение HD95.
    """
    try:
        points_a = _sample_boundary(pred_geom, samples)
        points_b = _sample_boundary(gt_geom, samples)
        if not points_a or not points_b:
            return None
        boundary_b = gt_geom.boundary
        boundary_a = pred_geom.boundary
        forward = _percentile95([boundary_b.distance(pt) for pt in points_a])
        backward = _percentile95([boundary_a.distance(pt) for pt in points_b])
        return max(forward, backward)
    except Exception:
        return None


def _ring_vertices(geom: Any) -> List[Tuple[float, float]]:
    """Вершины внешнего кольца без дубликата замыкания (последняя == первой)."""
    coords = list(geom.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def _polis(pred_geom: Any, gt_geom: Any) -> Optional[float]:
    """PoLiS пары в метрах; None — невычислимо.

    Среднее по m+n вершинам внешних колец обеих фигур расстояний до границы
    другой фигуры: [Σ_{v∈P} d(v, ∂G) + Σ_{v∈G} d(v, ∂P)] / (m + n).
    Идентичные фигуры → 0; замыкающий дубликат кольца не считается дважды.
    """
    try:
        verts_p = _ring_vertices(pred_geom)
        verts_g = _ring_vertices(gt_geom)
        if not verts_p or not verts_g:
            return None
        boundary_g = gt_geom.boundary
        boundary_p = pred_geom.boundary
        total = sum(boundary_g.distance(Point(v)) for v in verts_p)
        total += sum(boundary_p.distance(Point(v)) for v in verts_g)
        return total / (len(verts_p) + len(verts_g))
    except Exception:
        return None


class F1Calculator:
    """Калькулятор Archaeology Assistive F1 для КОЗ-3 (Приложение 4)."""

    METRIC_CRS = "UTM (EPSG:326xx/327xx)"

    def __init__(
        self,
        coverage_threshold: Optional[float] = None,
        r_tolerance: Optional[float] = None,
        input_crs_mode: Optional[str] = None,
    ):
        # Значения по умолчанию — из единого конфига (.env / config.py)
        self.coverage_threshold = (_METRIC_CFG.COVERAGE_THRESHOLD
                                   if coverage_threshold is None else coverage_threshold)
        self.r_tolerance = (_METRIC_CFG.POINT_R_TOLERANCE
                            if r_tolerance is None else r_tolerance)
        self.input_crs_mode = (_METRIC_CFG.INPUT_CRS_MODE
                               if input_crs_mode is None else input_crs_mode)
        self.circle_points = _METRIC_CFG.CIRCLE_POINTS
        self.epsilon = _METRIC_CFG.EPSILON
        self.target_classes = list(ALL_VALID_CLASSES)
        # Справочные показатели (Приложение 4; на final_score не влияют)
        self.contour_band_ratio = _METRIC_CFG.CONTOUR_BAND_RATIO
        self.contour_samples = _METRIC_CFG.CONTOUR_SAMPLES
        self.size_bins = list(_METRIC_CFG.SIZE_BINS)

    def calculate_f1(
        self,
        predictions_geojson: Dict[str, Any],
        gt_geojson: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Рассчитывает F1-score по спецификации КОЗ-3.

        Args:
            predictions_geojson: GeoJSON FeatureCollection предсказаний
            gt_geojson: GeoJSON FeatureCollection ground truth

        Returns:
            Словарь с результатами: final_score, per_class, auxiliary
        """
        # Собираем полигоны по классам (pooling через все источники).
        # Параллельно — карты id(геометрия) -> markup_type для справочных
        # показателей по типам данных (id стабильны в пределах вызова).
        pred_markup: Dict[int, Optional[str]] = {}
        gt_markup: Dict[int, Optional[str]] = {}
        pred_by_class = self._collect_by_class(predictions_geojson, markup_out=pred_markup)
        # GT-точки сохраняем как Point — сценарий Б (расстояние до точки)
        gt_by_class = self._collect_by_class(
            gt_geojson, preserve_points=True, markup_out=gt_markup)

        class_results = {}
        all_tp_pairs = []  # Для auxiliary метрик: (pred_centroid, gt_centroid)
        total_tp = 0
        total_fp = 0
        total_soft_recall_hits = 0
        total_gt_count = 0

        # Справочные показатели ТР КОЗ-3 (Приложение 4; на final_score не влияют):
        # контур — по TP-парам «полигон-полигон»; типы данных и размерные
        # категории — по GT-объектам и предсказаниям.
        contour_pairs_by_class: Dict[str, List[Tuple[Any, Any]]] = {}
        all_contour_pairs: List[Tuple[Any, Any]] = []
        gt_polygons: List[Any] = []
        dtype_gt_count: Dict[str, int] = {}
        dtype_pred_count: Dict[str, int] = {}
        dtype_tp_count: Dict[str, int] = {}
        dtype_matched_pred: Dict[str, int] = {}

        for cls in self.target_classes:
            preds = pred_by_class.get(cls, [])
            gts = gt_by_class.get(cls, [])

            tp, fp, fn, matches = self._soft_match(preds, gts)
            # Лишнее срабатывание учитывается с коэффициентом 0.5 (Приложение 4)
            precision = tp / (tp + 0.5 * fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)

            # Soft recall (IoU > 0)
            soft_hits = self._count_soft_recall(preds, gts)

            class_results[cls] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "num_predictions": len(preds),
                "num_ground_truth": len(gts),
                "soft_recall_hits": soft_hits,
            }

            total_tp += tp
            total_fp += fp
            total_soft_recall_hits += soft_hits
            total_gt_count += len(gts)
            all_tp_pairs.extend(matches)

            # --- Справочные показатели (Приложение 4): не влияют на final_score ---
            # Контур: только пары «полигон-полигон» из TP (у GT-точек контура нет)
            poly_pairs = [
                (pred_geom, gt_geom) for pred_geom, gt_geom in matches
                if isinstance(pred_geom, Polygon) and isinstance(gt_geom, Polygon)
            ]
            if poly_pairs:
                contour_pairs_by_class[cls] = poly_pairs
                all_contour_pairs.extend(poly_pairs)

            # Типы данных: TP/FN — по markup_type GT-объекта,
            # FP — по markup_type предсказания («прочее» при отсутствии)
            for pred_geom, gt_geom in matches:
                _bump(dtype_tp_count, data_type_group(gt_markup.get(id(gt_geom))))
                _bump(dtype_matched_pred, data_type_group(pred_markup.get(id(pred_geom))))
            for gt_geom, _ in gts:
                _bump(dtype_gt_count, data_type_group(gt_markup.get(id(gt_geom))))
                if isinstance(gt_geom, Polygon):
                    gt_polygons.append(gt_geom)
            for pred_geom, _ in preds:
                _bump(dtype_pred_count, data_type_group(pred_markup.get(id(pred_geom))))

        # Weighted F1 (веса от археологов).
        # Нормировка на присутствующие классы (решение 08.10.2026): если в GT
        # набора нет ни одного объекта класса, его вес исключается, а веса
        # остальных масштабируются единым коэффициентом 100 / сумма_присутствующих
        # (пропорции приоритетов между классами сохраняются). Класс, присутствующий
        # в GT, но полностью пропущенный участником (F1=0), сохраняет полный вес — штраф.
        present_classes = [
            cls for cls in self.target_classes
            if class_results[cls]["num_ground_truth"] > 0
        ]
        total_weight = sum(CLASS_WEIGHTS.get(cls, 0.0) for cls in present_classes)
        final_score = (
            sum(class_results[cls]["f1"] * CLASS_WEIGHTS.get(cls, 0.0) for cls in present_classes)
            / total_weight
        ) if total_weight > 0 else 0.0

        # Auxiliary metrics
        soft_recall = total_soft_recall_hits / (total_gt_count + 1e-6) if total_gt_count > 0 else 0.0
        center_deviation = self._calculate_center_deviation(all_tp_pairs)
        fp_ratio = total_fp / (total_tp + 1e-6) if total_tp > 0 else 0.0

        # Справочные показатели ТР КОЗ-3 (Приложение 4). Считаются поверх
        # матчинга и НЕ участвуют в final_score (закреплено тестом).
        contour = self._aggregate_contour(all_contour_pairs)
        contour_by_class = {
            cls: self._aggregate_contour(pairs)
            for cls, pairs in contour_pairs_by_class.items()
        }
        by_data_type = self._build_by_data_type(
            dtype_gt_count, dtype_pred_count, dtype_tp_count, dtype_matched_pred)
        matched_gt_ids = {id(gt_geom) for _, gt_geom in all_tp_pairs}
        size_categories = self._build_size_categories(gt_polygons, matched_gt_ids)

        return {
            "final_score": final_score,
            "per_class": class_results,
            "auxiliary": {
                "soft_recall": soft_recall,
                "center_deviation": center_deviation,
                "fp_ratio": fp_ratio,
                # Справочные показатели (Приложение 4) — на балл не влияют
                "contour": contour,
                "contour_by_class": contour_by_class,
                "by_data_type": by_data_type,
                "size_categories": size_categories,
            },
            "parameters": {
                "coverage_threshold": self.coverage_threshold,
                "input_crs_mode": self.input_crs_mode,
                "metric_crs": self.METRIC_CRS,
                "target_classes": self.target_classes,
                "class_weights": CLASS_WEIGHTS,
                "metric_version": _METRIC_CFG.METRIC_VERSION,
                "config_hash": _METRIC_CFG.metric_summary()["config_hash"],
                "epsilon": self.epsilon,
                "circle_points": self.circle_points,
                "r_tolerance": self.r_tolerance,
                "modality_mode": _METRIC_CFG.MODALITY_MODE,
                "contour_band_ratio": self.contour_band_ratio,
                "contour_samples": self.contour_samples,
                "size_bins": list(self.size_bins),
            },
        }

    def _collect_by_class(
        self,
        geojson_data: Dict[str, Any],
        preserve_points: bool = False,
        markup_out: Optional[Dict[int, Optional[str]]] = None,
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """Собирает геометрии с confidence по целевым классам.

        - координаты приводятся к UTM при необходимости (INPUT_CRS_MODE);
        - Point-геометрии предсказаний конвертируются в круговые полигоны
          (R = r_tolerance, метры);
        - при preserve_points=True (GT) Point-геометрии сохраняются как есть —
          для них работает сценарий Б ТЗ (расстояние центроид-точка);
        - markup_out (если передан) заполняется картой id(геометрия) ->
          markup_type: типы протягиваются вбок для справочных показателей
          по типам данных, сигнатура _soft_match не меняется.
        """
        result: Dict[str, List[Tuple[Any, float]]] = {}
        for feature in geojson_data.get("features", []):
            props = feature.get("properties", {}) or {}
            raw_class = props.get("class_name", "unknown")
            class_name = normalize_class_name(raw_class)
            if class_name is None or class_name not in self.target_classes:
                continue
            geom = _extract_geometry(feature)
            if geom is None:
                continue
            # Метрическая проекция: 3857 -> UTM (до конвертации Point -> круг!)
            geom = _to_utm(geom, self.input_crs_mode)
            # Конвертируем Point предсказаний в круговой Polygon
            # (радиус и число точек — из конфига); GT-точки сохраняем
            if isinstance(geom, Point) and not preserve_points:
                geom = _point_to_circle(geom, self.r_tolerance, self.circle_points)
            confidence = _get_confidence(feature)
            if markup_out is not None:
                # markup_type нужен справочным показателям по типам данных
                # (Приложение 4); у предсказаний ключа может не быть — None.
                markup_out[id(geom)] = props.get("markup_type")
            if class_name not in result:
                result[class_name] = []
            result[class_name].append((geom, confidence))
        return result

    def _soft_match(
        self,
        preds: List[Tuple[Any, float]],
        gts: List[Tuple[Any, float]],
    ) -> Tuple[int, int, int, List[Tuple[Any, Any]]]:
        """
        Жадный мягкий матчинг по ТР КОЗ-3 (Приложение 4).

        Сценарий А (GT — полигон): hit, если
        cov = area(P∩G) / min(area(P), area(G)) >= coverage_threshold
        И внутренняя (представительная) точка предсказания внутри GT.
        Сценарий Б (GT — точка): hit, если расстояние от центроида предсказания
        до GT-точки <= r_tolerance.
        Среди прошедших проверку GT выбирается лучший: для GT-полигонов — по
        максимальному перекрытию, для GT-точек — по минимальному расстоянию.

        Returns:
            (TP, FP, FN, список совпавших пар (pred_geom, gt_geom))
        """
        if not preds and not gts:
            return 0, 0, 0, []
        if not preds:
            return 0, 0, len(gts), []
        if not gts:
            return 0, len(preds), 0, []

        # Сортируем предсказания по confidence (по убыванию)
        sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)

        tp = 0
        matched_gt_indices = set()
        matches = []

        for pred_geom, _ in sorted_preds:
            if not isinstance(pred_geom, Polygon):
                continue

            best_score = float("-inf")
            best_gt_idx = -1

            for j, (gt_geom, _) in enumerate(gts):
                if j in matched_gt_indices:
                    continue

                hit, score = self._match_quality(pred_geom, gt_geom)

                if hit and score > best_score + 1e-9:  # tie-break: шум плавающей точки не решает равные покрытия
                    best_score = score
                    best_gt_idx = j

            if best_gt_idx != -1:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
                matches.append((pred_geom, gts[best_gt_idx][0]))

        fp = len(preds) - tp
        fn = len(gts) - tp
        return tp, fp, fn, matches

    def _match_quality(
        self,
        pred_geom: Polygon,
        gt_geom: Any,
    ) -> Tuple[bool, float]:
        """Проверка пары: (hit, оценка для выбора лучшего GT).

        Для GT-полигона оценка — перекрытие cov (больше — лучше),
        для GT-точки — расстояние со знаком минус (ближе — лучше).
        """
        if isinstance(gt_geom, Point):
            return self._match_point_gt(pred_geom, gt_geom)
        return self._match_polygon_gt(pred_geom, gt_geom)

    def _match_polygon_gt(self, pred_geom: Polygon, gt_geom: Polygon) -> Tuple[bool, float]:
        """Сценарий А (ТР КОЗ-3): cov >= порога И внутренняя (представительная)
        точка предсказания внутри эталона."""
        coverage = _polygon_coverage(pred_geom, gt_geom)
        hit = coverage >= self.coverage_threshold and _inner_point_in_mask(pred_geom, gt_geom)
        return hit, coverage

    def _match_point_gt(self, pred_geom: Polygon, gt_point: Point) -> Tuple[bool, float]:
        """Сценарий Б (ТР КОЗ-3): GT-точка оценивается по расстоянию — объект
        найден, если расстояние от центроида предсказания до GT-точки не
        превышает r_tolerance. Оценка для выбора лучшего GT — расстояние
        (возвращается со знаком минус: ближе — «больше»).
        """
        try:
            centroid = pred_geom.centroid
            distance = math.hypot(centroid.x - gt_point.x,
                                  centroid.y - gt_point.y)
        except Exception:
            return False, 0.0
        if distance <= self.r_tolerance:
            return True, -distance
        return False, 0.0

    def _count_soft_recall(
        self,
        preds: List[Tuple[Any, float]],
        gts: List[Tuple[Any, float]],
    ) -> int:
        """Считает количество GT-объектов, обнаруженных хотя бы одним предсказанием.

        Критерий: пересечение с ненулевой площадью; для GT-точек — пересечение
        с кругом радиуса r_tolerance.
        """
        if not preds or not gts:
            return 0
        count = 0
        for gt_geom, _ in gts:
            target = gt_geom
            if isinstance(gt_geom, Point):
                target = _point_to_circle(gt_geom, self.r_tolerance, self.circle_points)
            for pred_geom, _ in preds:
                if not isinstance(pred_geom, Polygon):
                    continue
                try:
                    if pred_geom.intersects(target) and pred_geom.intersection(target).area > 0:
                        count += 1
                        break
                except Exception:
                    continue
        return count

    def _calculate_center_deviation(
        self, matches: List[Tuple[Any, Any]]
    ) -> float:
        """Среднее расстояние между центрами TP-пар (для GT-точек — до самой точки)."""
        if not matches:
            return 0.0
        total_dist = 0.0
        for pred_poly, gt_poly in matches:
            try:
                pred_c = pred_poly.centroid
                gt_c = gt_poly.centroid
                dist = math.sqrt((pred_c.x - gt_c.x) ** 2 + (pred_c.y - gt_c.y) ** 2)
                total_dist += dist
            except Exception:
                continue
        return total_dist / len(matches) if matches else 0.0

    def _aggregate_contour(
        self, pairs: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Усредняет контурные метрики по парам «полигон-полигон» (Приложение 4).

        Boundary IoU, Hausdorff95 и PoLiS считаются по каждой паре в метрах;
        агрегат — среднее по парам, где метрика вычислима (None отдельной пары
        из-за вырожденной геометрии не ломает среднее). Нет пар или нет
        вычислимых значений → None (не NaN). ``pairs`` — число рассмотренных
        пар; для GT-точек пары сюда не попадают (у точки нет контура).
        """
        boundary_iou_values: List[float] = []
        hausdorff_values: List[float] = []
        polis_values: List[float] = []
        for pred_geom, gt_geom in pairs:
            boundary_iou = _boundary_iou(pred_geom, gt_geom, self.contour_band_ratio)
            hausdorff = _hausdorff95(pred_geom, gt_geom, self.contour_samples)
            polis = _polis(pred_geom, gt_geom)
            if boundary_iou is not None:
                boundary_iou_values.append(boundary_iou)
            if hausdorff is not None:
                hausdorff_values.append(hausdorff)
            if polis is not None:
                polis_values.append(polis)
        return {
            "boundary_iou": (
                sum(boundary_iou_values) / len(boundary_iou_values)
                if boundary_iou_values else None
            ),
            "hausdorff95_m": (
                sum(hausdorff_values) / len(hausdorff_values)
                if hausdorff_values else None
            ),
            "polis_m": (
                sum(polis_values) / len(polis_values) if polis_values else None
            ),
            "pairs": len(pairs),
        }

    def _build_by_data_type(
        self,
        gt_counts: Dict[str, int],
        pred_counts: Dict[str, int],
        tp_counts: Dict[str, int],
        matched_pred_counts: Dict[str, int],
    ) -> Dict[str, Dict[str, Any]]:
        """Справочный блок по типам данных (Приложение 4).

        TP и FN относятся по markup_type GT-объекта, FP — по markup_type
        предсказания (у предсказаний ключа может не быть — группа «прочее»).
        precision = tp / (tp + 0.5*fp + ε) — тот же штраф за лишнее
        срабатывание, что и в основной метрике; recall = tp/(tp+fn+ε).
        Итоговый (смешанный) tp может учитываться в группе GT, а связанный
        FP — в группе предсказания; это осознанная асимметрия ТЗ.

        Группа попадает в вывод, если в ней есть GT-объекты (основа полноты)
        или немотчированные предсказания (FP, основа точности); группы, где
        есть только матчнутые предсказания без markup_type, не создаются —
        их TP уже учтён по группе GT-объекта.
        """
        groups_present = set(gt_counts) | set(pred_counts)
        order = list(dict.fromkeys(DATA_TYPE_GROUPS.values()))
        if DEFAULT_DATA_TYPE_GROUP not in order:
            order.append(DEFAULT_DATA_TYPE_GROUP)
        order += [g for g in groups_present if g not in order]

        result: Dict[str, Dict[str, Any]] = {}
        for group in order:
            gt_count = gt_counts.get(group, 0)
            pred_count = pred_counts.get(group, 0)
            tp_count = tp_counts.get(group, 0)
            # FP — немотчированные предсказания своей группы
            fp_count = pred_count - matched_pred_counts.get(group, 0)
            if gt_count == 0 and fp_count == 0:
                continue
            fn_count = gt_count - tp_count
            result[group] = {
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "precision": tp_count / (tp_count + 0.5 * fp_count + self.epsilon),
                "recall": tp_count / (tp_count + fn_count + self.epsilon),
            }
        return result

    def _build_size_categories(
        self, gt_polygons: List[Any], matched_gt_ids: set
    ) -> List[Dict[str, Any]]:
        """Размерные категории GT-объектов (Приложение 4).

        Категории — интервалы по площади GT-полигонов в м² (UTM):
        (0, b1], (b1, b2], ..., (bN, +∞), границы — SIZE_BINS. GT-точки
        площади не имеют и не учитываются. recall = tp/gt по категории;
        None — если в категории нет GT (показатель не определён, не NaN).
        Список содержит все категории, включая пустые, — форма отчёта стабильна.
        """
        bins = self.size_bins
        stats = [{"gt": 0, "tp": 0} for _ in range(len(bins) + 1)]
        for geom in gt_polygons:
            area = geom.area
            if area <= 0:
                continue  # вырожденная геометрия без площади — вне категорий
            index = len(bins)
            for i, upper in enumerate(bins):
                if area <= upper:
                    index = i
                    break
            stats[index]["gt"] += 1
            if id(geom) in matched_gt_ids:
                stats[index]["tp"] += 1
        categories: List[Dict[str, Any]] = []
        for i, stat in enumerate(stats):
            gt_count = stat["gt"]
            tp_count = stat["tp"]
            categories.append({
                "max_area": bins[i] if i < len(bins) else None,
                "gt": gt_count,
                "tp": tp_count,
                "recall": (tp_count / gt_count) if gt_count else None,
            })
        return categories


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Расчет метрики Archaeology Assistive F1 (КОЗ-3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", "-p", required=True, help="GeoJSON с предсказаниями")
    parser.add_argument("--ground-truth", "-g", required=True, help="GeoJSON с разметкой (GT)")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=None,
        help="Порог перекрытия cov = area(P∩G)/min(area) для мягкого матчинга "
             "(по умолчанию из config: 0.5)",
    )
    parser.add_argument(
        "--r-tolerance",
        type=float,
        default=2.0,
        help="Радиус допуска для точечной разметки GT (в метрах, UTM)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Путь для сохранения детальных результатов в JSON",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Печать диагностики причин F1=0: что принято/отброшено, CRS, подсказки",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pred_path = Path(args.predictions)
    gt_path = Path(args.ground_truth)
    if not pred_path.exists():
        raise FileNotFoundError(f"Файл предсказаний не найден: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Файл ground truth не найден: {gt_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions_geojson = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_geojson = json.load(f)

    if args.diagnostics:
        try:
            from data_validator import build_diagnostics
        except ImportError:
            logger.error(
                "--diagnostics доступна только в полном комплекте сервиса "
                "(рядом со скриптом нужен модуль data_validator.py)."
            )
            return 1
        diag = build_diagnostics(predictions_geojson, gt_geojson, get_config())
        print("=== Диагностика предсказаний (принято/отброшено) ===")
        print(json.dumps(diag["predictions"], ensure_ascii=False, indent=2))
        print("=== Система координат ===")
        print(json.dumps(diag["crs"], ensure_ascii=False, indent=2))
        print("=== Подсказки ===")
        for hint in diag["hints"]:
            print(" -", hint)
        print()

    calculator = F1Calculator(
        coverage_threshold=args.coverage_threshold,
        r_tolerance=args.r_tolerance,
    )
    results = calculator.calculate_f1(predictions_geojson, gt_geojson)

    # Вывод итогового score
    print(f"{results['final_score']:.10f}")

    # Опциональный вывод в JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Детальные результаты сохранены в {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
