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

ENABLED_DETECTORS: tuple[str, ...] = ("lidar",)
"""Имена модулей-детекторов из koz3.detectors; заполняется по мере готовности."""

DEFAULT_CONF_THRESHOLD: float = 0.25
"""Порог уверенности для пар (class_name, markup_type) без своей записи."""

SRC_CRS: str = "EPSG:3857"

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

MASK_MIN_AREA_M2: float = 0.5
"""Минимальная площадь пятна при маска→полигоны, единицы CRS растра
(для UTM-растров м²); грубый отсев шума, подбирается"""

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

POINT_VERTICES: int = 32
"""Число вершин полигона-круга finds_points.

32 вершины дают площадь вписанного N-угольника 99.36 % площади круга
(16 вершин — 97.45 %, не проходит допуск ±2 % критерия приёмки 3).
"""

COORD_DECIMALS: int = 3
"""Знаков после запятой в координатах EPSG:3857 (инвариант 4.3)."""

COORD_PRECISION: int = COORD_DECIMALS

MAX_FEATURES: int = 20000
"""Максимум объектов в выходном файле (инвариант 4.4); отбор по confidence."""

MAX_RING_VERTICES: int = 3000
"""Максимум вершин кольца полигона (инвариант 4.3)."""

LIDAR: dict[str, object] = {
    "res_m": 0.1,
    "max_pixels": 60_000_000,  # как в бейзлайне
    "hillshades": ("c", "ch", "g", "i"),
    "infer_hillshades": ("c", "ch", "g", "i"),  # отмывки для инференса
    # тайл в пикселях отмывки (0.1 м → 102 м); модели подаётся сжатым до YOLO["imgsz"]
    "tile_px": 1024,
    "overlap": 0.3,
    # порог доли валидных пикселей тайла (не «все 0» / не «все 255»);
    # ниже — тайл пропускается на инференсе; равен settings.MIN_VALID_SHARE трейна
    "min_valid_share": 0.05,
    "seconds_per_gb": 60.0,  # оценка: секунд на ГБ LAS
}
"""Параметры лидарного детектора: DEM, хиллшейды, нарезка на тайлы."""

SURFACE_CLASSES: tuple[str, ...] = (
    "kurgany_tselye",
    "kurgany_povrezhdennye",
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
)
"""Классы модели по поверхностным данным (li, ae, sp_or, or); порядок = id класса YOLO."""

YOLO: dict[str, object] = {
    "weights_pt": "lidar_yolo.pt",
    "weights_engine": "lidar_yolo.engine",
    "prefer_engine": True,  # TensorRT engine, если есть файл и совместимая среда
    # размер входа модели: тайл LIDAR["tile_px"] сжимается до imgsz (1024 → 512 = 0.2 м/пикс).
    # Одно значение для обучения (training/train.py) и инференса; смена — переобучение и engine.
    "imgsz": 512,
    "half": True,  # FP16 для .pt; engine уже собран в FP16
    "batch": 8,  # тайлов за вызов predict (engine с batch=1 гоняется по одному)
    "conf": 0.05,  # низкий порог модели; рабочие пороги — CONF_THRESHOLD в postprocess
    "iou": 0.5,  # NMS внутри тайла
    "max_det": 300,
    "mask_threshold": 0.5,
}
"""Параметры инференса YOLO-seg; веса лежат в WEIGHTS_DIR."""

MODEL_CLASS_ALIASES: dict[str, str] = {"kurgany": "kurgany_tselye"}
"""Имена классов чужих моделей → наши. Модель бейзлайна (model.pt) называет курганы
«kurgany» — без этой записи все её курганы отбрасывались бы. Наша модель обучена
сразу на SURFACE_CLASSES и в переименовании не нуждается."""

HILLSHADE_NODATA: dict[str, int] = {"c": 0, "ch": 0, "g": 255, "i": 255}
"""Значение «нет данных» в хиллшейдах бейзлайна: цветные — 0, серые — 255."""
