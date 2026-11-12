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

ENABLED_DETECTORS: tuple[str, ...] = ("lidar", "maggr", "etgr")
"""Имена модулей-детекторов из koz3.detectors; заполняется по мере готовности."""

DEFAULT_CONF_THRESHOLD: float = 0.25
"""Порог уверенности для пар (class_name, markup_type) без своей записи."""

SRC_CRS: str = "EPSG:3857"

# Заглушка: рабочие пороги подбираются. maggr и etgr: порог уже применён детектором
# (thr_points / thr_polys / thr из файла весов, подобран по OOF) — postprocess не режет повторно.
CONF_THRESHOLD: dict[tuple[str, str], float] = {
    ("finds_points", "maggr"): 0.0,
    ("object_poly", "maggr"): 0.0,
    ("object_poly", "etgr"): 0.0,
}
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

KURGANY_CROSS_COPY: float | None = 0.5
"""Копия кургана во второй класс курганов с confidence × это число (None — выключено).

Модель путает целые и повреждённые курганы (13.11: val cls-лосс 3.2 против 1.8 на train),
а метрика штрафует FP вполовину: P = TP / (TP + 0.5·FP). Копия в «другой» класс — TP,
если модель ошиблась классом, и дешёвый FP, если нет. Пониженный confidence ставит копии
после исходных детекций в жадном сопоставлении. v3 (model.pt в оба класса) дал ×2 к v1.
Копия не создаётся, если в целевом классе уже есть дубль (как в NMS). Порог CONF_THRESHOLD
к копии не применяется: исходная детекция его уже прошла."""

CROSS_COPY_PAIRS: dict[str, str] = {
    "kurgany_tselye": "kurgany_povrezhdennye",
    "kurgany_povrezhdennye": "kurgany_tselye",
}
"""Класс → класс, в который делается копия (KURGANY_CROSS_COPY)."""

LEGACY_HILLSHADES: tuple[str, ...] = ("c", "ch", "g", "i")
"""Отмывки моделей без ключа "hillshades" в паспорте (обучены до 13.11, model.pt бейзлайна)."""

LIDAR: dict[str, object] = {
    "res_m": 0.1,
    "max_pixels": 60_000_000,  # как в бейзлайне
    # «r» (13.11, C23) — визуализация рельефа LRM/уклон/SVF (detectors/lidar_relief.py)
    "hillshades": ("c", "ch", "g", "i", "r"),
    # отмывки для инференса; каждой модели подаются только те, на которых она обучена
    # (паспорт .pt.json, ключ "hillshades"; нет ключа — LEGACY_HILLSHADES)
    "infer_hillshades": ("c", "ch", "g", "i", "r"),
    "relief": {
        "work_res_m": 0.5,  # сетка гладких частей (гаусс LRM, SVF), м
        "lrm_sigma_m": 10.0,  # LRM: DEM − гаусс(σ), м
        "lrm_clip_m": 1.0,  # LRM ±м → 0..255
        "slope_max_deg": 45.0,
        "svf_radius_m": 10.0,
        "svf_dirs": 8,
        "svf_min": 0.6,
    },
    # тайл в пикселях отмывки (0.1 м → 102 м); модели подаётся сжатым до YOLO["imgsz"]
    "tile_px": 1024,
    "overlap": 0.3,
    # порог доли валидных пикселей тайла (не «все 0» / не «все 255»);
    # ниже — тайл пропускается на инференсе; равен settings.MIN_VALID_SHARE трейна
    "min_valid_share": 0.05,
    "seconds_per_gb": 60.0,  # оценка: секунд на ГБ LAS
    # как модель видит отмывку:
    #   "whole" — как бейзлайн: вся отмывка сжата до whole_px по длинной стороне, выровнена,
    #             один кадр (так обучен model.pt бейзлайна; тайлы 0.1 м он почти не видит);
    #   "tiles" — тайлы tile_px с перекрытием, сжатые до YOLO["imgsz"] (так строит датасет
    #             training/build_dataset.py — для наших моделей).
    # 08.11: whole дал 0.0 на платформе (большой тестовый регион сжимается до 1024 —
    # курганы в несколько пикселей); тайлы 0.1 м дали 0.0138 (точность 0.74, полнота 0.03,
    # только объекты < 500 м²). Поэтому тайлы на нескольких масштабах (scales).
    "infer_mode": "tiles",
    # масштабы тайлов: отмывка сжимается в k раз (0.1·k м/пикс), затем тайлы tile_px.
    # v3 (1, 2, 4): 0.02822, но рост дали курганы в двух классах; TP целых курганов те же, что
    # в v1 (Boundary IoU 0.036380 до знака), крупных объектов ни одного, точность лидара
    # 0.74 → 0.56. Масштабы 2 и 4 добавили только FP → (1,). Наша модель — из паспорта.
    "scales": (1,),
    # TTA: тайл ещё и повёрнутым на 90/180/270° (время инференса ×4, полнота выше).
    # None — включается сам для моделей, обученных с поворотами (паспорт "rot90": true —
    # датасет v3); True/False — принудительно. model.pt без паспорта — выключено.
    "tta_rot90": None,
    "whole_px": 1024,  # длинная сторона кадра и imgsz модели в режиме whole (как бейзлайн)
    # выравнивание поворотом, как бейзлайн. Выключено: бейзлайн поворачивает в том же холсте,
    # и у вытянутых отмывок поворот на 90° срезает до трети региона; масштаб от выравнивания
    # не зависит. Включать только по результату holdout (evaluate.py --compare).
    "whole_align": False,
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
    # ансамбль (C22): ещё веса в WEIGHTS_DIR, например ("baseline.pt",) — model.pt бейзлайна.
    # Каждая модель — со своим паспортом <имя>.pt.json (тайл, масштабы, повороты); отмывки
    # строятся один раз, дубли между моделями снимает NMS postprocess. Нет файла — модель
    # пропускается с предупреждением. Время инференса лидара растёт пропорционально.
    "extra_weights_pt": (),
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

MODEL_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "kurgany": ("kurgany_tselye", "kurgany_povrezhdennye"),
}
"""Имена классов чужих моделей → наши (одно имя модели может дать несколько классов).

Модель бейзлайна (model.pt) знает один класс курганов «kurgany» и не отличает
целые от повреждённых. Каждый её курган выдаётся в ОБА класса: метрика считает F1
по классам независимо, поэтому целые не страдают, а повреждённые (вес 30 — самый
большой) перестают давать 0. Лишние предсказания в чужом классе — FP с весом 0.5.
Наша модель обучена сразу на SURFACE_CLASSES и в переименовании не нуждается."""

HILLSHADE_NODATA: dict[str, int] = {"c": 0, "ch": 0, "g": 255, "i": 255}
"""Значение «нет данных» в хиллшейдах бейзлайна: цветные — 0, серые — 255."""

MAGGR: dict[str, object] = {
    "res_m": 0.1,
    "mask_erode_px": 2,
    "gray_glob": "*_серый.tif",
    "pt_smooth": 1.5,
    "pt_min_dist": 5,
    "pt_min_z": 1.0,
    "pt_dipole_win_px": 15,
    "pt_label_m": 0.5,
    "pt_nms_m": 1.5,
    "poly_res": 0.25,
    "smooth": 2,
    "bg_sigmas": (0, 40, 20),
    "levels": (0.5, 1.0, 1.5, 2.0, 2.5),
    "min_cand_m2": 4.0,
    "nms_iou": 0.2,
    "nms_cover": 0.5,
    "label_iou": 0.1,
    "models_file": "maggr_models.joblib",  # файл весов в WEIGHTS_DIR (train_maggr.py)
    # те же деревья без sklearn (koz3/tree_models.py): если sklearn/joblib нет в образе
    # или joblib не грузится — детектор берёт их; пишут train_maggr.py и tools/export_trees.py
    "models_npz": "maggr_models.npz",
    "seconds_per_mpx": 6.0,  # замер 08.11: ~6–8 с на Мпикс рабочей сетки (CPU)
}
"""Магниторазведка (MagGr): сетка 0.1 м, кандидаты точек (pt_*) и полигонов, пороги NMS.
Константы формул признаков — в koz3/geophys/maggr_features.py (_PT_*, _POLY_*)."""

ETGR: dict[str, object] = {
    "work_side": 512,  # длинная сторона рабочей сетки, px (ноутбук 02, ячейка 2)
    "palette_max_dist": 110.0,  # пиксели дальше от шкалы jet — не данные
    "smooth_px": 2,  # сглаживание перед нарезкой кандидатов, px
    "bg_sigmas": (0, 20, 10),  # вычитание фона с такой σ (0 — сам сигнал)
    "levels": (0.5, 1.0, 1.5, 2.0, 2.5),  # уровни нарезки, в std
    "min_cand_px": 30,  # минимальная площадь кандидата, px
    "nms_iou": 0.2,  # подавление дублей среди кандидатов
    "label_iou": 0.1,  # метка: засчитан платформой и IoU с разметкой >= label_iou
    "split_ratio": 0.5,  # пятно режется на детей со score >= split_ratio * порог
    "crop_px": 128,  # окно кандидата для CNN, px
    "models_file": "etgr_maps.pt",  # файл весов в WEIGHTS_DIR (training/train_etgr.py)
    "seconds_per_map": 20.0,  # ЗАГЛУШКА: оценка до замера на платформе; уточнить по логам прогона
    "profiles_models_file": "etgr_profiles.joblib",  # HGB профилей (train_etgr.py --kind profiles)
    "seconds_per_profile": 20.0,  # ЗАГЛУШКА: профиль + поля карт региона для его признаков
}
"""Электроразведка (EtGr): карты-срезы (обращение шкалы jet, кандидаты-изолинии,
окна CNN) и профили (HGB по PROFILE_FEATURE_NAMES) — ноутбук 02 организаторов."""
