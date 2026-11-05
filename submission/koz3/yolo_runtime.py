"""Инференс YOLO-seg: загрузка модели (TensorRT engine или .pt) и предсказание тайлов.

Общий модуль для детекторов поверхностных данных (lidar, imagery) и для
training/ (одинаковая подготовка изображения при обучении и на инференсе).

Загрузка модели:
1. Если config.YOLO["prefer_engine"] и рядом лежит .engine с метаданными
   (.engine.json от training/export_engine.py), и метаданные совпадают со
   средой (GPU compute capability, версия TensorRT), загружается engine.
2. Иначе, или если engine не загрузился, — .pt (FP16 на GPU). Причина
   отката пишется в лог предупреждением; прогон не падает.

Тяжёлые импорты (torch, ultralytics) — только внутри функций.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)


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


def tile_positions(length_px: int, tile_px: int, overlap: float) -> list[int]:
    """Начала тайлов вдоль оси: шаг tile·(1−overlap), последний тайл прижат к краю."""
    if length_px <= tile_px:
        return [0]
    step = max(1, int(tile_px * (1.0 - overlap)))
    positions = list(range(0, length_px - tile_px, step))
    positions.append(length_px - tile_px)
    return sorted(set(positions))


def _engine_compatible(engine_path: Path) -> tuple[bool, str]:
    """Проверить метаданные engine против текущей среды; (ок, причина)."""
    meta_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not meta_path.is_file():
        return False, f"нет метаданных {meta_path.name}"
    try:
        import tensorrt  # noqa: F401  # нужен для загрузки engine
        import torch
    except ImportError as exc:
        return False, f"нет TensorRT/torch в образе: {exc}"
    if not torch.cuda.is_available():
        return False, "нет CUDA"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    if meta.get("compute_capability") != capability:
        return False, f"engine собран для sm {meta.get('compute_capability')}, GPU — {capability}"
    import tensorrt as trt

    if meta.get("tensorrt") != trt.__version__:
        return False, f"engine собран TensorRT {meta.get('tensorrt')}, в образе {trt.__version__}"
    return True, "ок"


def load_model(weights_dir: Path | None = None) -> LoadedModel:
    """Загрузить YOLO-seg: engine, если совместим, иначе .pt (FP16 на GPU)."""
    import torch
    from ultralytics import YOLO

    # Детерминизм инференса (требование регламента): без автоподбора ядер cuDNN.
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    weights_dir = Path(weights_dir or config.WEIGHTS_DIR)
    params = config.YOLO
    engine_path = weights_dir / str(params["weights_engine"])
    pt_path = weights_dir / str(params["weights_pt"])

    if params["prefer_engine"] and engine_path.is_file():
        compatible, reason = _engine_compatible(engine_path)
        if compatible:
            try:
                model = YOLO(str(engine_path), task="segment")
                meta = json.loads(
                    engine_path.with_suffix(".engine.json").read_text(encoding="utf-8")
                )
                names = {int(k): v for k, v in meta["names"].items()}
                logger.info("YOLO: загружен engine %s", engine_path.name)
                return LoadedModel(model, "engine", names, int(meta.get("batch", 1)))
            except Exception as exc:  # noqa: BLE001  # откат на .pt не должен ронять прогон
                reason = f"engine не загрузился: {exc}"
        logger.warning("YOLO: engine не используется (%s), откат на %s", reason, pt_path.name)

    if not pt_path.is_file():
        raise FileNotFoundError(f"нет весов модели: {pt_path}")
    model = YOLO(str(pt_path), task="segment")
    names = {int(k): v for k, v in model.names.items()}
    logger.info("YOLO: загружен %s, классы %s", pt_path.name, list(names.values()))
    return LoadedModel(model, "pt", names, int(params["batch"]))


def predict_tiles(loaded: LoadedModel, tiles: list[np.ndarray]) -> Iterator[list[TileInstance]]:
    """Предсказать маски для тайлов RGB uint8; по одному списку на тайл, в том же порядке.

    Имена классов модели переводятся через config.MODEL_CLASS_ALIASES; классы,
    которых и после этого нет в config.SURFACE_CLASSES, отбрасываются.
    """
    import torch

    params = config.YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    half = bool(params["half"]) and device == 0 and loaded.kind == "pt"
    batch = max(1, loaded.batch)
    for start in range(0, len(tiles), batch):
        chunk = tiles[start : start + batch]
        # ultralytics ждёт BGR для numpy-входа
        sources = [np.ascontiguousarray(tile[..., ::-1]) for tile in chunk]
        results = loaded.model.predict(
            source=sources,
            imgsz=int(params["imgsz"]),
            conf=float(params["conf"]),
            iou=float(params["iou"]),
            max_det=int(params["max_det"]),
            half=half,
            device=device,
            retina_masks=True,
            verbose=False,
        )
        for tile, result in zip(chunk, results, strict=True):
            yield _instances(result, tile.shape[:2], loaded.names)


def _instances(
    result: object, shape_hw: tuple[int, int], names: dict[int, str]
) -> list[TileInstance]:
    """Результат ultralytics → список TileInstance с масками размера тайла."""
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.cpu().numpy() > float(config.YOLO["mask_threshold"])
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    height, width = shape_hw
    surface = set(config.SURFACE_CLASSES)
    found: list[TileInstance] = []
    for mask, class_id, conf in zip(masks, classes, confs, strict=True):
        raw_name = names.get(int(class_id), "")
        class_name = config.MODEL_CLASS_ALIASES.get(raw_name, raw_name)
        if class_name not in surface:
            continue
        found.append(TileInstance(class_name, float(conf), mask[:height, :width]))
    return found
