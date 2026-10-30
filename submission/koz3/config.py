"""Конфигурация решения: все параметры в одном модуле (раздел 8 пакета).

Только данные и константы, без логики. Параметры, калибруемые на данных,
помечены комментарием «подбирает напарник».
"""

from pathlib import Path

TIME_LIMIT_S: int = 22 * 60
"""Бюджет прогона, с; результат пишем всегда (инвариант 4.6)."""

SEED: int = 42
"""Зерно ГПСЧ для детерминизма прогонов."""

WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "weights"
"""Каталог весов моделей: submission/weights, рядом с пакетом koz3."""

LOG_LEVEL: str = "INFO"
"""Уровень logging в solution.py."""

ENABLED_DETECTORS: tuple[str, ...] = ()
"""Имена модулей-детекторов из koz3.detectors; заполняется по мере готовности."""

DEFAULT_CONF_THRESHOLD: float = 0.25
"""Порог уверенности для пар (class_name, markup_type) без своей записи."""

# Заглушка: рабочие пороги подбираются
CONF_THRESHOLD: dict[tuple[str, str], float] = {}
"""Пороги по парам (class_name, markup_type) для postprocess."""

NMS_IOU: float = 0.5
"""NMS: подавлять при IoU > NMS_IOU; только внутри (class_name, markup_type)."""

NMS_COVER: float = 0.7
"""NMS: подавлять, если пересечение / меньшая площадь > NMS_COVER."""

# Значения-заглушки (кроме finds_points = 0.0).
MIN_AREA_M2: dict[str, float] = {
    "kurgany_tselye": 2.0,
    "kurgany_povrezhdennye": 2.0,
    "gorodishcha": 50.0,
    "fortifikatsii": 5.0,
    "arkhitektury": 2.0,
    "finds_points": 0.0,
    "object_poly": 1.0,
}
"""Минимальная площадь фичи, м² (UTM); меньше — отбраковка в postprocess."""

SIMPLIFY_M: dict[str, float] = {
    "kurgany_tselye": 0.3,
    "kurgany_povrezhdennye": 0.3,
    "gorodishcha": 1.0,
    "fortifikatsii": 0.3,
    "arkhitektury": 0.3,
    "finds_points": 0.3,
    "object_poly": 0.3,
}
"""Допуск упрощения кольца полигона (shapely simplify), м, по классам."""

POINT_RADIUS_M: float = 0.75
"""Радиус полигона-круга finds_points вокруг точки, м (инвариант 4.1)."""

POINT_VERTICES: int = 16
"""Число вершин полигона-круга finds_points."""

COORD_DECIMALS: int = 3
"""Знаков после запятой в координатах EPSG:3857 (инвариант 4.3)."""

MAX_FEATURES: int = 20000
"""Максимум объектов в выходном файле (инвариант 4.4); отбор по confidence."""

MAX_RING_VERTICES: int = 3000
"""Максимум вершин кольца полигона (инвариант 4.3)."""

LIDAR: dict[str, object] = {
    "res_m": 0.1,
    "hillshades": ("c", "ch", "g", "i"),
    "tile_px": 1024,
    "overlap": 0.3,
}
"""Параметры лидарного детектора: DEM, хиллшейды, нарезка на тайлы."""