"""Геометрический слой решения: маска→полигоны, CRS, круг, чистка, упрощение.

Общий модуль для детекторов, postprocess и output (раздел 5 пакета контекста).
Функции принимают и возвращают shapely Polygon; единицы — единицы исходной
геометрии (для UTM — метры). CRS-преобразования собраны только здесь и всегда
выполняются с always_xy=True.
"""

from functools import lru_cache

from affine import Affine
import numpy as np
import shapely
from pyproj import Transformer
from rasterio import features
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid

from . import config


@lru_cache(maxsize=None)
def _transformer(src_crs: str, dst_crs: str) -> Transformer:
    """Кэшированный Transformer для пары CRS; оси всегда в порядке x, y."""
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def mask_to_polygons(mask: np.ndarray, transform: Affine) -> list[Polygon]:
    """Перевести бинарную маску в полигоны в CRS растра.

    Ненулевой пиксель считается объектом (бинаризация «mask != 0», поэтому
    допустимы bool, uint8, float 0/1); связность 4 — умолчание rasterio.
    Дыры в пятне становятся внутренними кольцами полигона. Обрывки площадью
    меньше config.MASK_MIN_AREA_M2 (единицы CRS растра) отбрасываются.
    Параметр transform — аффинная привязка растра (пиксель → CRS растра).
    Возвращает полигоны в CRS растра; для пустой или нулевой маски — [].
    """
    binary = (mask != 0).astype(np.uint8)
    if not binary.any():
        return []
    polygons = [
        shape(geom)
        for geom, value in features.shapes(binary, transform=transform)
        if value == 1
    ]
    return [poly for poly in polygons if poly.area >= config.MASK_MIN_AREA_M2]


def reproject(geom: Polygon, src_crs: str, dst_crs: str) -> Polygon:
    """Перевести полигон из src_crs в dst_crs; оси x, y (always_xy=True)."""
    transformer = _transformer(src_crs, dst_crs)
    return shapely.transform(geom, transformer.transform, interleaved=False)


def point_to_circle(x: float, y: float) -> Polygon:
    """Построить полигон-круг вокруг точки (x, y); координаты в метрах (UTM).

    Радиус — config.POINT_RADIUS_M, вершин — config.POINT_VERTICES; вершины
    лежат равномерно на окружности, старт с угла 0 (детерминизм). Результат —
    вписанный правильный N-угольник (finds_points).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, config.POINT_VERTICES, endpoint=False)
    xs = x + config.POINT_RADIUS_M * np.cos(angles)
    ys = y + config.POINT_RADIUS_M * np.sin(angles)
    return Polygon(np.column_stack((xs, ys)))


def clean_polygon(geom: Polygon | None) -> Polygon | None:
    """Привести полигон к валидному: make_valid → части Polygon → максимум.

    Из результата make_valid берутся только части Polygon с положительной
    площадью и выбирается самая большая (при равенстве — первая, детерминизм).
    Минимальную площадь по классам (config.MIN_AREA_M2) проверяет postprocess,
    а не эта функция. None на входе, пустая геометрия или результат без
    полигональных частей дают None.
    """
    if geom is None:
        return None
    fixed = make_valid(geom)
    if isinstance(fixed, Polygon):
        return None if fixed.is_empty else fixed
    parts = [
        part
        for part in getattr(fixed, "geoms", ())
        if isinstance(part, Polygon) and part.area > 0.0
    ]
    if not parts:
        return None
    return max(parts, key=lambda part: part.area)


def simplify_m(geom: Polygon, tolerance_m: float) -> Polygon:
    """Упростить кольца полигона с допуском tolerance_m в метрах (UTM).

    Ожидает валидную геометрию (в postprocess вызывается после clean_polygon),
    preserve_topology=True. Если результат не Polygon или пуст, возвращается
    исходная геометрия
    """
    simplified = geom.simplify(tolerance_m, preserve_topology=True)
    if isinstance(simplified, Polygon) and not simplified.is_empty:
        return simplified
    return geom