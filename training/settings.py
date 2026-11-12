"""Настройки обучения YOLO-seg для поверхностных классов.

Классы, параметры хиллшейдов и подготовка изображения берутся из
submission/koz3 — один источник правды для обучения и инференса.
Всё, что подбирается на данных, — здесь, с комментарием.
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = REPO_DIR / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from koz3 import config as runtime_config  # noqa: E402

CLASSES: tuple[str, ...] = runtime_config.SURFACE_CLASSES
"""Порядок = id класса в YOLO; совпадает с инференсом."""

ALL_CLASSES: tuple[str, ...] = CLASSES + ("finds_points", "object_poly")
"""Все 7 классов платформы (для make_gt и геофизики)."""

# Синонимы имён классов в разметке трейна → канонические имена.
# Ключи — в нижнем регистре, без пробелов по краям; пробелы и дефисы
# заменяются на «_». Неизвестные имена печатает inspect_annotations.py —
# дополните словарь по его выводу.
CLASS_SYNONYMS: dict[str, str] = {
    "kurgany_tselye": "kurgany_tselye",
    "курганы_целые": "kurgany_tselye",
    "курганы": "kurgany_tselye",
    "курган": "kurgany_tselye",
    "целые": "kurgany_tselye",
    "kurgany": "kurgany_tselye",
    "kurgany_povrezhdennye": "kurgany_povrezhdennye",
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_повреждённые": "kurgany_povrezhdennye",
    "курганы_разрушенные": "kurgany_povrezhdennye",
    "поврежденные": "kurgany_povrezhdennye",
    "повреждённые": "kurgany_povrezhdennye",
    "разрушенные": "kurgany_povrezhdennye",
    "распаханные_курганы": "kurgany_povrezhdennye",
    "курганы_распаханные": "kurgany_povrezhdennye",
    "распаханные": "kurgany_povrezhdennye",
    "gorodishcha": "gorodishcha",
    "городища": "gorodishcha",
    "городище": "gorodishcha",
    "fortifikatsii": "fortifikatsii",
    "фортификации": "fortifikatsii",
    "фортификация": "fortifikatsii",
    "arkhitektury": "arkhitektury",
    "архитектуры": "arkhitektury",
    "архитектура": "arkhitektury",
    # геофизика (для tools/make_gt.py и обучения моделей MagGr / EtGr / TerFl)
    "finds_points": "finds_points",
    "findspoints": "finds_points",
    "findspoint": "finds_points",
    "object_poly": "object_poly",
    "objectpoly": "object_poly",
    "objectspoly": "object_poly",
}
"""ВНИМАНИЕ: «распаханные курганы» отнесены к повреждённым по описанию класса
в ТР («задетые распашкой»). В бейзлайне они шли в целые. Сверьте с документом
организаторов «описание классов разметки» и при необходимости поправьте."""

# Поля properties, в которых может лежать имя класса (по порядку проверки).
CLASS_PROPERTY_KEYS: tuple[str, ...] = ("class_name", "class", "Class", "name", "layer", "type")

CLASS_FIELD_KEYS: tuple[str, ...] = ("class_name", "class", "Class")
"""Явные поля класса: главнее имени файла (по class_name классы читает и метрика платформы)."""

NAME_FIELD_KEYS: tuple[str, ...] = ("name", "layer", "type")
"""Свободный текст (имя объекта, слой): только если класс не понятен из имени файла.
13.11: раньше любое поле перекрывало файл, а поиск синонима идёт по подстроке — объект
«Курган №5» из файла «Курганы_поврежденные» уходил в целые (путаница классов)."""

IGNORED_CLASSES: tuple[str, ...] = ("structrpoly", "structpoly", "structurepoly")
"""Классы-«структуры», не входящие в 7 канонических (игнор-лист как в метрике
организаторов): load_objects пропускает фичи, у которых значение
классоподобного свойства или имя файла (нормализованные) содержат одно
из этих имён."""

MARKUP_LABELS: dict[str, str] = {
    "li": "li",
    "la": "li",
    "ae": "ae",
    "spor": "sp_or",
    "or": "or",
    "maggr": "maggr",
    "etgr": "etgr",
    "terfl": "terfl",
}
"""Метки типа данных в имени файла разметки → markup_type."""

HILLSHADES: tuple[str, ...] = tuple(runtime_config.LIDAR["hillshades"])
LEGACY_HILLSHADES: tuple[str, ...] = tuple(runtime_config.LEGACY_HILLSHADES)
"""Варианты отмывки, которые идут в датасет (каждый — отдельное изображение)."""

# Шаг DEM и лимит пикселей — в koz3.config.LIDAR (общие с инференсом).

TILE_PX: int = 512
"""Размер тайла датасета, пикс (09.11: было 1024 → сжатие до 512 — модели не хватало данных).
512 = YOLO["imgsz"]: тайл подаётся модели без сжатия. Инференс берёт тайл и масштабы из
паспорта модели (dataset_meta.json → train.py → lidar_yolo.pt.json), config не правится."""
TILE_OVERLAP: float = 0.1
"""Перекрытие тайлов датасета (на инференсе — LIDAR["overlap"], там больше: стыки)."""
TILE_SCALES: tuple[float, ...] = (1, 2, 4)
"""Масштабы нарезки: отмывка сжимается в k раз перед тайлами (как lidar на инференсе).
1 — 0.1 м/пикс (тайл 51 м: мелкие курганы), 2 — 0.2 м/пикс (тайл 102 м: крупные курганы
до 100 м, части городищ), 4 — 0.4 м/пикс (тайл 205 м: городища ≥ 70×100 м и повреждённые
курганы до 120 м целиком; 13.11 добавлен — v4: recall объектов 500–5000 м² ≈ 0). Тайлов
×4 меньше, чем на масштабе 2, — датасет растёт на ~5 %."""
ROT90_AUGMENT: bool = True
"""Train: к тайлу с объектами добавить копии, повёрнутые на 90/180/270° (×4 тайлов с
объектами). В val копий нет. Отключить: build_dataset.py --no-rot90."""
RARE_CLASSES: tuple[str, ...] = ("gorodishcha", "fortifikatsii", "arkhitektury")
"""Редкие классы: train-тайл с ними сохраняется ещё RARE_TILE_REPEAT раз (и его повороты)."""
RARE_TILE_REPEAT: int = 2
"""Доп. копий train-тайла с редким классом (0 — выкл.). Повтор = больший вес в эпохе;
разные аугментации YOLO (мозаика, масштаб, отражения) делают копии непохожими.
Классы с малым числом примеров иначе почти не учатся (v4: arkhitektury — 0 TP)."""
NEGATIVE_TILE_SHARE: float = 0.15
"""Доля тайлов без объектов, которые оставляем (подбирается: больше — меньше FP)."""
MIN_VALID_SHARE: float = 0.05
"""Тайлы, где данных (не nodata) меньше этой доли, пропускаются."""
MIN_INSTANCE_PX: float = 16.0
"""Обрезки объектов на краю тайла меньше этой площади (пикс²) не размечаем."""
LABEL_SIMPLIFY_PX: float = 0.7
IMAGE_FORMAT: str = "png"
"""png — без потерь (как на инференсе), jpg — в ~5 раз меньше места."""

VAL_REGIONS: tuple[str, ...] = ()
"""Имена регионов (папок) для валидации. Пусто — ~15 % регионов по хешу имени.
Обязательно задайте вручную, чтобы в val были все 5 классов."""
VAL_SHARE: float = 0.15

AUTO_FIX: bool = True
"""build_dataset: сам выбрать CRS разметки (регион или заголовок LAS) и сдвинуть разметку,
если курганы явно садятся на насыпи со сдвигом (training/augment_data.py). --no-fix — выкл."""
MIN_OBJECTS_FOR_SHIFT: int = 5
"""Сдвиг разметки оценивается, только если курганов в LAS не меньше (иначе ненадёжно)."""
MAX_SHIFT_M: float = 8.0
"""Поиск сдвига разметки ±м."""
DROP_LABEL_NODATA_SHARE: float = 0.5
"""Подпись выбрасывается, если больше этой доли объекта в тайле — «нет данных»."""

EXTRA_DATA: bool = True
"""build_dataset: дополнительные данные для train (--no-extra — выкл.)."""
EXTRA_AZIMUTHS: tuple[float, ...] = (45.0, 135.0, 225.0)
"""Азимуты солнца доп. серых отмывок (на инференсе — 315°): та же разметка, другая тень."""
SYNTH_PER_KM2: float = 40.0
"""Синтетических курганов на км² валидной площади (0 — без синтетики)."""
SYNTH_DIAMETER_M: tuple[float, float] = (6.0, 25.0)
SYNTH_HEIGHT_M: tuple[float, float] = (0.4, 2.0)
SYNTH_MIN_GAP_M: float = 15.0
"""Синтетический курган не ближе этого к разметке и к другим синтетическим, м."""

EXCLUDE_REGIONS: tuple[str, ...] = ()
"""Регионы, которые не идут в датасет (битые LAS, смещённая разметка — по
training/audit_data.py). То же разово — build_dataset.py --exclude."""

SEED: int = 42
