"""Инференс YOLO-seg: загрузка модели (TensorRT engine или .pt) и предсказание тайлов.

Общий модуль для детекторов поверхностных данных (lidar, imagery) и для
training/ — одинаковая подготовка изображения и тайлов при обучении и на
инференсе: to_model_image, tile_positions, valid_pixel_mask.

Загрузка модели:
1. Если config.YOLO["prefer_engine"] и рядом лежит .engine с метаданными
   (.engine.json от training/export_engine.py), и метаданные совпадают со
   средой (GPU compute capability, версия TensorRT), загружается engine.
2. Иначе, или если engine не загрузился, — .pt (FP16 на GPU). Причина
   отката пишется в лог предупреждением; прогон не падает.

Диагностика: классы весов вне SURFACE_CLASSES (после алиасов) и маски,
не совпадающие по размеру с тайлом, не отбраковываются молча — предупреждение
в лог и счётчик _rejected_classes. Тихая отбраковка означала бы пустой или
частичный результат без следов в логах.

Тяжёлые импорты (torch, ultralytics) — только внутри функций.
"""

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)

_rejected_classes: Counter = Counter()
"""Число отбракованных экземпляров по имени класса (после алиасов); для тестов."""

_mask_shape_warned: bool = False
"""Предупреждение о несовпадении размера масок уже выдано в этом прогоне."""


@dataclass
class TileInstance:
    """Одна найденная маска в координатах тайла."""

    class_name: str
    confidence: float
    mask: np.ndarray  # bool, (H, W) тайла


@dataclass
class LoadedModel:
    """Загруженная модель и сведения о ней."""

    model: object
    kind: str  # "engine" | "pt"
    names: dict[int, str]
    batch: int
    imgsz: int | None = None  # вход модели при обучении (из .pt); None — config.YOLO["imgsz"]
    tile_px: int | None = None  # тайл обучения из паспорта .pt.json; None — config.LIDAR
    scales: tuple[float, ...] | None = None  # масштабы инференса; None — config.LIDAR["scales"]
    rot90: bool = False  # обучена с поворотами 90° (паспорт) → можно TTA поворотами
    name: str = ""  # имя файла весов (ансамбль: метка в source детекций)
    hillshades: tuple[str, ...] | None = None  # отмывки обучения (паспорт); None — LEGACY


def to_model_image(array: np.ndarray) -> np.ndarray:
    """Растр (bands, H, W) или (H, W) → RGB uint8 (H, W, 3) для модели.

    Та же функция используется при подготовке датасета (training/), поэтому
    обучение и инференс видят одинаковые пиксели. Серые растры дублируются в 3
    канала; лишние каналы (альфа) отбрасываются; не-uint8 нормируются в 0..255.
    """
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[0] <= 4 and data.shape[0] < data.shape[-1]:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        low, high = float(np.nanmin(data)), float(np.nanmax(data))
        scale = 255.0 / (high - low) if high > low else 0.0
        data = np.nan_to_num((data - low) * scale).astype(np.uint8)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    return np.ascontiguousarray(data[..., :3])


def downscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Сжать изображение (H, W, C) в factor раз (cv2.INTER_AREA); factor <= 1 — как есть.

    Общая для датасета (training/build_dataset.py) и инференса (detectors/lidar.py):
    модель видит сжатые отмывки одинаково при обучении и на платформе.
    """
    if factor <= 1.0:
        return image
    import cv2

    height, width = image.shape[:2]
    size = (max(1, int(round(width / factor))), max(1, int(round(height / factor))))
    return np.ascontiguousarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def valid_pixel_mask(image: np.ndarray) -> np.ndarray:
    """Маска валидных пикселей RGB uint8 (H, W, 3): не «все каналы 0» и не «все 255».

    Единый фильтр для обучения (build_dataset) и инференса (lidar): полностью
    чёрные и белые пиксели — заполнение вне охвата данных, модели их не
    подают. Долю валидных пикселей тайла сравнивают с порогом
    LIDAR["min_valid_share"] (инференс) / settings.MIN_VALID_SHARE (трейн).
    """
    data = np.asarray(image)
    # поканальные сравнения вместо np.all(axis=-1): тот же результат, в ~7 раз быстрее
    c0, c1, c2 = data[..., 0], data[..., 1], data[..., 2]
    zero = (c0 == 0) & (c1 == 0) & (c2 == 0)
    full = (c0 == 255) & (c1 == 255) & (c2 == 255)
    return ~(zero | full)


def tile_positions(length_px: int, tile_px: int, overlap: float) -> list[int]:
    """Начала тайлов вдоль оси: шаг tile·(1−overlap), последний тайл прижат к краю."""
    if length_px <= tile_px:
        return [0]
    step = max(1, int(tile_px * (1.0 - overlap)))
    positions = list(range(0, length_px - tile_px, step))
    positions.append(length_px - tile_px)
    return sorted(set(positions))


def _engine_compatible(engine_path: Path) -> tuple[bool, str]:
    """Проверить метаданные engine против текущей среды; (ок, причина).

    Битые или нечитаемые метаданные (.engine.json) — не ошибка прогона:
    (False, причина) и корректный откат load_model на .pt, лежащий рядом.
    """
    meta_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not meta_path.is_file():
        return False, f"нет метаданных {meta_path.name}"
    try:
        import tensorrt as trt
        import torch
    except ImportError as exc:
        return False, f"нет TensorRT/torch в образе: {exc}"
    if not torch.cuda.is_available():
        return False, "нет CUDA"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"метаданные {meta_path.name} не читаются: {exc}"
    if not isinstance(meta, dict):
        return False, f"метаданные {meta_path.name} — не JSON-объект"
    capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    if meta.get("compute_capability") != capability:
        return False, f"engine собран для sm {meta.get('compute_capability')}, GPU — {capability}"
    if meta.get("tensorrt") != trt.__version__:
        return False, f"engine собран TensorRT {meta.get('tensorrt')}, в образе {trt.__version__}"
    return True, "ок"


def _train_imgsz(model: object) -> int | None:
    """imgsz, на котором обучены веса .pt (train_args чекпойнта); None — неизвестно.

    Инференс подаёт модели тот же размер: model.pt бейзлайна — 1024, наша — 512.
    Раньше imgsz брался только из config и для model.pt его приходилось менять руками.
    """
    ckpt = getattr(model, "ckpt", None)
    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    train_imgsz = train_args.get("imgsz") if isinstance(train_args, dict) else None
    return train_imgsz if isinstance(train_imgsz, int) and train_imgsz > 0 else None


def _warn_class_names(names: dict[int, str]) -> None:
    """Предупредить о расхождении классов весов и SURFACE_CLASSES (после алиасов)."""
    surface = set(config.SURFACE_CLASSES)
    model_classes = {alias for name in names.values() for alias in _our_classes(name)}
    unknown = sorted(model_classes - surface)
    if unknown:
        logger.warning(
            "веса содержат классы вне SURFACE_CLASSES (после алиасов): %s — "
            "их экземпляры отбракуются на инференсе; проверьте имена классов "
            "весов (model.names / .engine.json) и MODEL_CLASS_ALIASES",
            unknown,
        )
    missing = sorted(surface - model_classes)
    if missing:
        logger.warning(
            "классы SURFACE_CLASSES, которых нет в весах (после алиасов): %s — "
            "эти классы не будут предсказаны",
            missing,
        )


def load_model(weights_dir: Path | None = None, pt_name: str | None = None) -> LoadedModel:
    """Загрузить YOLO-seg: engine, если совместим, иначе .pt (FP16 на GPU).

    pt_name — другие веса .pt из weights_dir (ансамбль, YOLO["extra_weights_pt"]): только .pt,
    engine не ищется.
    """
    import torch
    from ultralytics import YOLO

    # Детерминизм инференса (требование регламента): без автоподбора ядер cuDNN.
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    weights_dir = Path(weights_dir or config.WEIGHTS_DIR)
    params = config.YOLO
    engine_path = weights_dir / str(params["weights_engine"])
    pt_path = weights_dir / str(pt_name or params["weights_pt"])

    if pt_name is None and params["prefer_engine"] and engine_path.is_file():
        compatible, reason = _engine_compatible(engine_path)
        if compatible:
            try:
                model = YOLO(str(engine_path), task="segment")
                meta = json.loads(
                    engine_path.with_suffix(".engine.json").read_text(encoding="utf-8")
                )
                names = {int(k): v for k, v in meta["names"].items()}
                _warn_class_names(names)
                logger.info("YOLO: загружен engine %s", engine_path.name)
                return LoadedModel(
                    model, "engine", names, int(meta.get("batch", 1)), name=engine_path.name
                )
            except Exception as exc:  # noqa: BLE001  # откат на .pt не должен ронять прогон
                reason = f"engine не загрузился: {exc}"
        logger.warning("YOLO: engine не используется (%s), откат на %s", reason, pt_path.name)

    if not pt_path.is_file():
        raise FileNotFoundError(f"нет весов модели: {pt_path}")
    model = YOLO(str(pt_path), task="segment")
    names = {int(k): v for k, v in model.names.items()}
    _warn_class_names(names)
    imgsz = _train_imgsz(model)
    tile_px, scales = _passport_tiling(pt_path)
    rot90 = _passport_rot90(pt_path)
    logger.info(
        "YOLO: загружен %s, классы %s, imgsz %s, тайл %s, масштабы %s, повороты при обучении %s",
        pt_path.name,
        list(names.values()),
        imgsz if imgsz is not None else f"{params['imgsz']} (из config)",
        tile_px or "из config",
        scales or "из config",
        rot90,
    )
    hillshades = _passport_hillshades(pt_path)
    if hillshades is not None:
        logger.info("YOLO: %s обучена на отмывках %s", pt_path.name, list(hillshades))
    return LoadedModel(
        model,
        "pt",
        names,
        int(params["batch"]),
        imgsz,
        tile_px,
        scales,
        rot90,
        pt_path.name,
        hillshades,
    )


def _passport_hillshades(pt_path: Path) -> tuple[str, ...] | None:
    """Отмывки, на которых обучена модель (ключ "hillshades" паспорта); нет — None."""
    path = pt_path.with_suffix(".pt.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("hillshades")
    except (OSError, ValueError, AttributeError):
        return None
    return tuple(str(v) for v in value) if value else None


def _passport_rot90(pt_path: Path) -> bool:
    """Обучена ли модель с поворотами 90° (ключ "rot90" паспорта .pt.json); нет — False."""
    path = pt_path.with_suffix(".pt.json")
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("rot90", False))
    except (OSError, ValueError, AttributeError):
        return False


def _passport_tiling(pt_path: Path) -> tuple[int | None, tuple[float, ...] | None]:
    """Тайл и масштабы из паспорта <веса>.pt.json (пишет training/train.py).

    Наша модель обучена на тайлах tile_px одного масштаба — инференс повторяет это:
    тот же тайл и масштаб (1,), если в паспорте не указано "scales". Нет паспорта
    (model.pt бейзлайна) — (None, None): берутся LIDAR["tile_px"] и LIDAR["scales"].
    """
    path = pt_path.with_suffix(".pt.json")
    if not path.is_file():
        return None, None
    try:
        passport = json.loads(path.read_text(encoding="utf-8"))
        tile_px = int(passport["tile_px"])
        scales = tuple(float(k) for k in passport.get("scales", (1,)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("паспорт %s не читается (%s) — тайл и масштабы из config", path.name, exc)
        return None, None
    return tile_px, scales


def predict_tiles(
    loaded: LoadedModel,
    tiles: list[np.ndarray],
    imgsz: int | None = None,
    rot90_tta: bool = False,
) -> Iterator[list[TileInstance]]:
    """Предсказать маски для тайлов RGB uint8; по одному списку на тайл, в том же порядке.

    imgsz — размер входа модели; None — imgsz обучения весов (loaded.imgsz), если он
    известен, иначе config.YOLO["imgsz"].
    rot90_tta — каждый тайл предсказывается ещё повёрнутым на 90/180/270°, маски
    поворачиваются обратно и добавляются к списку тайла (дубли снимает NMS postprocess):
    полнота выше, время ×4. Только для моделей, обученных с поворотами (паспорт rot90).

    Имена классов модели переводятся через config.MODEL_CLASS_ALIASES; классы,
    которых и после этого нет в config.SURFACE_CLASSES, отбрасываются с
    предупреждением (см. _instances), как и маски несовпадающего с тайлом
    размера — геометрия такого полигона была бы ненадёжной.
    """
    import torch

    params = config.YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    half = bool(params["half"]) and device == 0 and loaded.kind == "pt"
    batch = max(1, loaded.batch)
    turns = (0, 1, 2, 3) if rot90_tta else (0,)
    for start in range(0, len(tiles), batch):
        chunk = tiles[start : start + batch]
        # ultralytics ждёт BGR для numpy-входа; повороты — подряд для каждого тайла
        sources = [
            np.ascontiguousarray(np.rot90(tile, k)[..., ::-1]) for tile in chunk for k in turns
        ]
        results = loaded.model.predict(
            source=sources,
            imgsz=int(imgsz or loaded.imgsz or params["imgsz"]),
            conf=float(params["conf"]),
            iou=float(params["iou"]),
            max_det=int(params["max_det"]),
            half=half,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        results = list(results)
        if len(results) != len(sources):
            raise RuntimeError(
                f"модель вернула {len(results)} результатов на {len(sources)} входов"
            )
        for i, tile in enumerate(chunk):
            found: list[TileInstance] = []
            for j, k in enumerate(turns):
                result = results[i * len(turns) + j]
                rotated_hw = np.rot90(tile, k).shape[:2]
                for inst in _instances(result, rotated_hw, loaded.names):
                    mask = np.ascontiguousarray(np.rot90(inst.mask, -k)) if k else inst.mask
                    found.append(TileInstance(inst.class_name, inst.confidence, mask))
            yield found


def _count_rejected_class(class_id: int, raw_name: str, class_name: str) -> None:
    """Учесть отбраковку класса; предупреждение — только при первом случае."""
    _rejected_classes[class_name] += 1
    if _rejected_classes[class_name] == 1:
        logger.warning(
            "id класса %d: имя %r (после алиасов %r) не в SURFACE_CLASSES — "
            "экземпляры отбраковываются; проверьте имена классов весов "
            "(model.names / .engine.json) и MODEL_CLASS_ALIASES",
            class_id,
            raw_name,
            class_name,
        )


def _warn_mask_shape(actual: tuple[int, ...], expected: tuple[int, int]) -> None:
    """Предупредить о несовпадении размера маски и тайла (один раз на прогон)."""
    global _mask_shape_warned
    if _mask_shape_warned:
        return
    _mask_shape_warned = True
    logger.warning(
        "размер маски %s не совпадает с тайлом %s — геометрия ненадёжна, "
        "такие экземпляры отбраковываются; проверьте YOLO imgsz против "
        "LIDAR tile_px и фактический размер масок при retina_masks "
        "(ultralytics в образе)",
        actual,
        expected,
    )


def _instances(
    result: object, shape_hw: tuple[int, int], names: dict[int, str]
) -> list[TileInstance]:
    """Результат ultralytics → список TileInstance с масками размера тайла.

    Экземпляры с классом вне SURFACE_CLASSES (после алиасов) и с маской
    другого размера отбраковываются с предупреждением: молчаливая отбраковка
    дала бы пустой или частичный результат без следов в логах.
    """
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.cpu().numpy() > float(config.YOLO["mask_threshold"])
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    height, width = shape_hw
    surface = set(config.SURFACE_CLASSES)
    found: list[TileInstance] = []
    for mask, class_id, conf in zip(masks, classes, confs, strict=True):
        cid = int(class_id)
        raw_name = names.get(cid, "")
        if mask.shape != (height, width):
            _warn_mask_shape(mask.shape, (height, width))
            continue
        for class_name in _our_classes(raw_name):
            if class_name not in surface:
                _count_rejected_class(cid, raw_name, class_name)
                continue
            found.append(TileInstance(class_name, float(conf), mask))
    return found


def _our_classes(raw_name: str) -> tuple[str, ...]:
    """Наши классы для имени класса модели (config.MODEL_CLASS_ALIASES; иначе — как есть)."""
    return tuple(config.MODEL_CLASS_ALIASES.get(raw_name, (raw_name,)))
