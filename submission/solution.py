#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from affine import Affine
from pyproj import Transformer
from shapely.affinity import affine_transform as shapely_affine_transform
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.validation import make_valid
from ultralytics import YOLO


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    BG_RED = "\033[41m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and table-style layout."""
    
    FORMATS = {
        logging.DEBUG: f"{Colors.BRIGHT_BLACK}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} %(message)s",
        logging.INFO: f"{Colors.BRIGHT_CYAN}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} %(message)s",
        logging.WARNING: f"{Colors.BRIGHT_YELLOW}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} {Colors.YELLOW}%(message)s{Colors.RESET}",
        logging.ERROR: f"{Colors.BRIGHT_RED}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} {Colors.RED}%(message)s{Colors.RESET}",
        logging.CRITICAL: f"{Colors.BG_RED}{Colors.WHITE}%(levelname)-8s{Colors.RESET} {Colors.DIM}│{Colors.RESET} {Colors.BRIGHT_RED}%(message)s{Colors.RESET}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("solution")


def log_header(title: str, char: str = "═", width: int = 80):
    """Log a formatted header."""
    logger.info(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{char * width}{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{char * width}{Colors.RESET}")


def log_section(title: str, char: str = "─", width: int = 80):
    """Log a formatted section divider."""
    logger.info(f"{Colors.CYAN}{char * width}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{Colors.BOLD}  {title}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{char * width}{Colors.RESET}")


def log_subsection(title: str, width: int = 80):
    """Log a formatted subsection."""
    logger.info(f"{Colors.DIM}  {'·' * width}{Colors.RESET}")
    logger.info(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}    {title}{Colors.RESET}")
    logger.info(f"{Colors.DIM}  {'·' * width}{Colors.RESET}")


def log_table_row(label: str, value: Any, label_width: int = 30):
    """Log a table-style row with label and value."""
    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} "
        f"{Colors.BRIGHT_WHITE}{label:<{label_width}}{Colors.RESET} "
        f"{Colors.DIM}│{Colors.RESET} "
        f"{Colors.BRIGHT_GREEN}{value}{Colors.RESET}"
    )


def log_metric(label: str, value: Any, unit: str = "", label_width: int = 30):
    """Log a metric with highlighting."""
    display_value = f"{value} {unit}".strip() if unit else str(value)
    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} "
        f"{Colors.WHITE}{label:<{label_width}}{Colors.RESET} "
        f"{Colors.DIM}→{Colors.RESET} "
        f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}{display_value}{Colors.RESET}"
    )


#: Ширина рамки, в которую печатается заголовок обрабатываемого файла.
FILE_BOX_WIDTH = 76


def log_file_header(label: str) -> None:
    """Рамка с заголовком файла (паддинг считается, а не подбирается вручную)."""
    text = f"File {label}"
    padding = " " * max(0, FILE_BOX_WIDTH - 1 - len(text))
    logger.info(f"{Colors.DIM}  ┌{'─' * FILE_BOX_WIDTH}┐{Colors.RESET}")
    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} {Colors.BRIGHT_CYAN}{Colors.BOLD}{text}{Colors.RESET}"
        f"{Colors.DIM}{padding}│{Colors.RESET}"
    )
    logger.info(f"{Colors.DIM}  ├{'─' * FILE_BOX_WIDTH}┤{Colors.RESET}")


def log_file_footer() -> None:
    logger.info(f"{Colors.DIM}  └{'─' * FILE_BOX_WIDTH}┘{Colors.RESET}\n")


def log_timing(label: str, elapsed_seconds: float, label_width: int = 30):
    """Log timing information with formatting."""
    if elapsed_seconds < 1:
        time_str = f"{elapsed_seconds * 1000:.0f}ms"
    elif elapsed_seconds < 60:
        time_str = f"{elapsed_seconds:.2f}s"
    else:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        time_str = f"{minutes}m {seconds:.1f}s"
    
    logger.info(
        f"{Colors.DIM}  │{Colors.RESET} "
        f"{Colors.WHITE}{label:<{label_width}}{Colors.RESET} "
        f"{Colors.DIM}⏱{Colors.RESET} "
        f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}{time_str}{Colors.RESET}"
    )


#: Фиксированное зерно по умолчанию: один и тот же вход → один и тот же result.geojson.
DEFAULT_SEED = 42


def configure_determinism(seed: int = DEFAULT_SEED) -> None:
    """Фиксирует все источники случайности — прогон становится воспроизводимым.

    Закрывает генераторы `random`/`numpy`/`torch` (CPU и все GPU), включает
    детерминированные алгоритмы cuDNN и отключает autotuning (`benchmark`),
    который подбирает разные ядра от запуска к запуску. `PYTHONHASHSEED`
    выставляется для дочерних процессов; чтобы он влиял и на текущий
    интерпретатор, его нужно задать в окружении до старта (см. README).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS требует фиксированную рабочую область, иначе часть матмулов
    # недетерминирована; переменную нужно выставить до первой операции на GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 (Ampere+) урезает мантиссу и может зависеть от окружения/сборки —
    # для воспроизводимости считаем в полном FP32.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # torch без аргумента warn_only
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    logger.info("Seed fixed: %s (deterministic mode)", seed)


# Параметры перевода масок в геометрию (раньше были числами по коду).
MASK_THRESHOLD = 0.5          # порог бинаризации маски модели
MAX_CONTOURS_PER_MASK = 100   # сколько контуров берём с одной маски
CONTOUR_EPSILON_PX = 0.5      # упрощение контура (cv2.approxPolyDP), пиксели
MIN_CONTOUR_AREA = 0.1        # отбрасываем вырожденные контуры, px²
SIMPLIFY_TOLERANCE_M = 2.0    # упрощение готового полигона, метры
MIN_FEATURE_AREA = 1.0        # минимальная площадь объекта в результате, м²


@dataclass
class Detection:
    """Одна найденная маска: бинарная маска в координатах кадра + класс и уверенность."""

    mask: np.ndarray
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class InferConfig:
    """Настройки инференса и препроцессинга — общие для лидарных и не-лидарных растров."""

    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    tile_height: int = 1024
    tile_width: int = 1024
    overlap_ratio: float = 0.3
    enable_downscale: bool = True
    enable_alignment: bool = True
    enable_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8


# Standardised class names (matches evaluation metric)
# Сервис метрики принимает ТОЛЬКО эти 7 имён: любой другой class_name в
# result.geojson валит расчёт с UNKNOWN_CLASS и попытка не получает балл.
CLASS_NAMES = [
    "kurgany_tselye", "kurgany_povrezhdennye",
    "fortifikatsii", "gorodishcha", "arkhitektury",
    "finds_points", "object_poly",
]

# Синонимы: 1:1 с реестром метрики платформы (CLASS_SYNONYMS в config.py сервиса
# метрики) плюс латинские имена классов модели. Именно этот реестр — источник
# правды: сервис метрики разворачивает синонимы сам, но валидатор до него
# синонимов не знает и на незнакомом имени падает.
CLASS_NAME_MAP = {
    # курганы целые
    "курганы_целые": "kurgany_tselye",
    "курган": "kurgany_tselye",
    "курганы": "kurgany_tselye",
    "целые": "kurgany_tselye",
    "целый": "kurgany_tselye",
    "kurgany": "kurgany_tselye",
    "kurgani": "kurgany_tselye",
    "распаханные курганы": "kurgany_tselye",
    # курганы повреждённые
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_разрушенные": "kurgany_povrezhdennye",
    "поврежденные": "kurgany_povrezhdennye",
    "повреждённые": "kurgany_povrezhdennye",
    "разрушенные": "kurgany_povrezhdennye",
    # городища
    "городища": "gorodishcha",
    "городище": "gorodishcha",
    "gorodishcha": "gorodishcha",
    "gorodishche": "gorodishcha",
    # архитектуры
    "архитектуры": "arkhitektury",
    "архитектура": "arkhitektury",
    "arkhitektury": "arkhitektury",
    # фортификации
    "фортификации": "fortifikatsii",
    "фортификация": "fortifikatsii",
    "fortifikatsii": "fortifikatsii",
    # геофизика: точечные аномалии
    "findspoints": "finds_points",
    "FindsPoints": "finds_points",
    "FindsPoint": "finds_points",
    "ARTEFACT.POINTS": "finds_points",
    "finds_points": "finds_points",
    # геофизика: полигональные аномалии
    "objectpoly": "object_poly",
    "ObjectPoly": "object_poly",
    "ObjectsPoly": "object_poly",
    "objects_poly": "object_poly",
    "OBJECT.POLYGON": "object_poly",
    "object_poly": "object_poly",
}

# Классы модели бейзлайна, которым нет канонического соответствия в реестре
# метрики (селища, дороги, ямы, пашни, межа, иное). Их объекты отбрасываются:
# в разметке КОЗ3 таких классов нет, а платформа на них отвечает UNKNOWN_CLASS.
UNSUPPORTED_CLASSES = ("selishcha", "dorogi", "yamy", "pashni", "mezha", "inoe")

VALID_CLASS_NAMES = set(CLASS_NAMES)


def normalize_class_name(raw: str) -> str:
    """Normalise a raw model class name to the canonical Latin form."""
    if raw in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[raw]
    low = raw.lower()
    if low in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[low]
    if low in VALID_CLASS_NAMES:
        return low
    return raw


def drop_unsupported_classes(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Оставляет только классы из реестра метрики — иначе платформа не посчитает балл.

    Сервис метрики валидирует `class_name` ДО нормализации синонимов: любое имя
    вне семи канонических (например `selishcha`, `dorogi`, `yamy`) приводит к
    ответу UNKNOWN_CLASS и нулевому результату вместо балла. Поэтому объекты
    классов модели, которым нет соответствия в разметке КОЗ3, отбрасываются.
    """
    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = defaultdict(int)
    for feature in features:
        class_name = (feature.get("properties") or {}).get("class_name")
        if class_name in VALID_CLASS_NAMES:
            kept.append(feature)
        else:
            dropped[str(class_name)] += 1
    if dropped:
        summary = ", ".join(f"{name}: {count}" for name, count in sorted(dropped.items()))
        logger.warning(
            "  ⚠️  Dropped %d feature(s) with classes outside the metric registry (%s); "
            "no canonical counterpart in KOZ3",
            sum(dropped.values()), summary,
        )
    return kept


def drop_empty_geometries(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Страховка: в GeoJSON не должно быть фич без coordinates (их отклоняет валидатор)."""
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") in ("Polygon", "MultiPolygon") and coordinates:
            kept.append(feature)
        else:
            dropped += 1
    if dropped:
        logger.warning("  ⚠️  Dropped %d feature(s) without valid coordinates", dropped)
    return kept


def prepare_image(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3 and data.shape[0] <= 4:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.dtype != np.uint8:
        data = data.astype(np.float32)
        max_val, min_val = np.max(data), np.min(data)
        if max_val > min_val:
            data = (255 * (data - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    if data.shape[-1] > 3:
        data = data[:, :, :3]
    return data


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    if image.shape[-1] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel = clahe.apply(l_channel)
        
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
    
    return enhanced


def downscale_image(image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= target_size:
        return image.copy(), 1.0
    scale = target_size / max_dim
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = Image.fromarray(image).resize(new_size, Image.LANCZOS)
    return np.asarray(resized), scale


def align_and_crop_image(image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[float, float]]:
    img_rgb = image if image.shape[-1] == 3 else np.repeat(image[:, :, :1], 3, axis=-1)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edge = max(1, min(20, h // 10, w // 10))
    edge_regions = [
        gray[:edge, :],
        gray[-edge:, :],
        gray[:, :edge],
        gray[:, -edge:]
    ]
    edge_means = [float(region.mean()) for region in edge_regions if region.size]
    background_sample = sum(edge_means) / len(edge_means) if edge_means else float(gray.mean())
    dark_background = background_sample < 128
    thresh_type = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, 30 if dark_background else 0, 255, thresh_type)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        center = (w / 2.0, h / 2.0)
        return img_rgb, 0.0, (0, 0), center
    main = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main)
    center, size, angle = rect
    if size[0] < size[1]:
        angle += 90.0
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = (0, 0, 0) if dark_background else (255, 255, 255)
    rotated = cv2.warpAffine(img_bgr, rot_mat, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh_rot = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray_rotated, 30 if dark_background else 250, 255, thresh_rot)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), angle, (0, 0), center
    x, y, w_box, h_box = cv2.boundingRect(coords)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(rotated.shape[1] - x, w_box + 2 * margin)
    h_box = min(rotated.shape[0] - y, h_box + 2 * margin)
    crop = rotated[y:y + h_box, x:x + w_box]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), angle, (x, y), center


def load_model(model_path: str, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps:0"
        else:
            device = "cpu"
    log_metric("Device", device, "🚀")
    model = YOLO(model_path)
    target_device = "mps" if device.startswith("mps") else device
    model.to(target_device)
    model.overrides["device"] = target_device
    
    if "cuda" in target_device:
        torch.cuda.empty_cache()
        # Детерминизм: autotuning cuDNN (`benchmark`) выбирает ядра от запуска
        # к запуску — отключаем и фиксируем детерминированные алгоритмы.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model.overrides["half"] = False  # FP32: FP16-ядра недетерминированы на части GPU
    
    names = getattr(model.model, "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, list):
        names = {idx: name for idx, name in enumerate(names)}

    # Verify model classes against the canonical set and log coverage
    mapped, unknown = [], []
    for idx, raw in names.items():
        canonical = normalize_class_name(raw)
        if canonical in VALID_CLASS_NAMES:
            mapped.append(f"{raw} → {canonical}" if canonical != raw else raw)
        else:
            unknown.append(raw)
    log_metric("Model classes (mapped)", len(mapped))
    for entry in mapped:
        logger.info(f"{Colors.DIM}      ✓ {entry}{Colors.RESET}")
    if unknown:
        logger.warning(
            "  ⚠️  %d model class(es) have no canonical mapping and their objects will be "
            "DROPPED (the metric registry rejects any other class_name): %s",
            len(unknown), unknown,
        )
    if not mapped:
        logger.warning("  ⚠️  None of the model's classes match the expected canonical set! "
                       "Expected: %s", CLASS_NAMES)

    return model, names


def _tile_positions(length: int, tile_size: int, step: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = [0]
    while positions[-1] + tile_size < length:
        next_pos = positions[-1] + step
        if next_pos + tile_size >= length:
            positions.append(max(length - tile_size, 0))
            break
        positions.append(next_pos)
    return sorted(dict.fromkeys(max(0, p) for p in positions))


def run_tiled_inference(
    model,
    image: np.ndarray,
    infer: InferConfig,
    class_names: Dict[int, str]
) -> List[Detection]:
    """Режет уже подготовленный кадр (см. prepare_image) на тайлы и гоняет модель."""
    img_h, img_w = image.shape[:2]
    stride_y = max(int(infer.tile_height * (1.0 - infer.overlap_ratio)), 1)
    stride_x = max(int(infer.tile_width * (1.0 - infer.overlap_ratio)), 1)
    ys = _tile_positions(img_h, infer.tile_height, stride_y)
    xs = _tile_positions(img_w, infer.tile_width, stride_x)
    num_tiles = len(ys) * len(xs)
    log_metric("Tiles to Process", f"{len(ys)}×{len(xs)} = {num_tiles}")
    detections: List[Detection] = []
    
    tiles_to_process = []
    tile_coords = []
    for y in ys:
        y_end = min(y + infer.tile_height, img_h)
        for x in xs:
            x_end = min(x + infer.tile_width, img_w)
            tile = image[y:y_end, x:x_end]
            if tile.size == 0:
                continue
            tiles_to_process.append(tile)
            tile_coords.append((y, x, y_end, x_end))
    
    for tile_idx, tile in enumerate(tiles_to_process):
        y, x, y_end, x_end = tile_coords[tile_idx]
        tile_bgr = np.ascontiguousarray(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        results = model.predict(
            source=tile_bgr,
            conf=infer.conf_threshold,
            iou=infer.iou_threshold,
            verbose=False
        )
        if not results:
            continue
        result = results[0]
        if result.masks is None or result.boxes is None:
            continue
        mask_data = result.masks.data.cpu().numpy()
        orig_h, orig_w = result.masks.orig_shape
        box_cls = result.boxes.cls.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy()
        for idx, raw_mask in enumerate(mask_data):
            mask = raw_mask
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            binary = mask > MASK_THRESHOLD
            h_limit = min(orig_h, img_h - y)
            w_limit = min(orig_w, img_w - x)
            if h_limit <= 0 or w_limit <= 0:
                continue
            if not binary[:h_limit, :w_limit].any():
                continue
            full_mask = np.zeros((img_h, img_w), dtype=bool)
            full_mask[y:y + h_limit, x:x + w_limit] = binary[:h_limit, :w_limit]
            class_id = int(box_cls[idx]) if idx < len(box_cls) else 0
            confidence = float(box_conf[idx]) if idx < len(box_conf) else float(infer.conf_threshold)
            class_name = class_names.get(class_id, str(class_id))
            detections.append(
                Detection(mask=full_mask, class_id=class_id,
                          class_name=class_name, confidence=confidence)
            )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return detections


def normalize_crs(crs_value: str) -> str:
    text = crs_value.strip()
    lower = text.lower()
    if lower.startswith("urn:ogc:def:crs:epsg::"):
        return f"EPSG:{text.split('::')[-1]}"
    if lower.startswith("epsg") and ":" not in text:
        suffix = text.replace("epsg", "").strip(": ")
        return f"EPSG:{suffix}" if suffix else "EPSG:32636" #Fallback to 36N for rare cases
    return text


def utm_to_epsg(utm_value: str) -> Optional[str]:
    cleaned = utm_value.strip().upper()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits:
        return None
    hemisphere = next((ch for ch in reversed(cleaned) if ch.isalpha()), "N")
    try:
        zone = int(digits)
    except ValueError:
        return None
    base = 326 if hemisphere == "N" else 327
    return f"EPSG:{base}{zone:02d}"


def load_region_crs(dir: Path) -> Optional[str]:
    utm_file = dir / "UTM.json"
    if not utm_file.exists():
        return None
    try:
        with utm_file.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", utm_file.name, exc)
        return None
    crs_value = data.get("crs") or data.get("CRS")
    if crs_value:
        return normalize_crs(str(crs_value))
    utm_value = data.get("utm") or data.get("UTM")
    return utm_to_epsg(str(utm_value)) if utm_value else None


def _wkt_to_crs(payload: bytes) -> Optional[str]:
    """WKT из VLR LAS → строка CRS ('EPSG:xxxx', если код определяется)."""
    text = payload.decode("utf-8", "ignore").strip("\x00").strip()
    if not text:
        return None
    try:
        from rasterio.crs import CRS

        crs = CRS.from_wkt(text)
        epsg = crs.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        return crs.to_string() or None
    except Exception:
        return None


def load_las_crs(path: Path) -> Optional[str]:
    """CRS, записанный в самом LAS (VLR/EVLR 2112 «OGC WKT»), если он есть.

    Часть площадок датасета не имеет `UTM.json`, но несёт CRS в заголовке LAS —
    это единственный оставшийся источник геопривязки, когда хиллшейды строятся
    из точек, а не берутся готовыми.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(375)
            if len(head) < 227:
                return None
            header_size = struct.unpack_from("<H", head, 94)[0]
            vlr_count = struct.unpack_from("<I", head, 100)[0]
            handle.seek(header_size)
            for _ in range(vlr_count):
                record = handle.read(54)
                if len(record) < 54:
                    break
                _, _, record_id, record_length, _ = struct.unpack_from("<H16sHH32s", record, 0)
                payload = handle.read(record_length)
                if record_id == 2112:
                    crs = _wkt_to_crs(payload)
                    if crs:
                        return crs
            # LAS 1.4: тот же WKT может лежать в EVLR
            if len(head) >= 247:
                evlr_offset = struct.unpack_from("<Q", head, 235)[0]
                evlr_count = struct.unpack_from("<I", head, 243)[0]
                if evlr_offset and evlr_count:
                    handle.seek(evlr_offset)
                    for _ in range(evlr_count):
                        record = handle.read(60)
                        if len(record) < 60:
                            break
                        _, _, record_id, record_length, _ = struct.unpack_from(
                            "<H16sHQ32s", record, 0)
                        payload = handle.read(record_length)
                        if record_id == 2112:
                            crs = _wkt_to_crs(payload)
                            if crs:
                                return crs
    except Exception:
        return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  LAS → стандартный набор хиллшейдов (c / ch / g / i)
#
#  Портировано из Supplementary/hillshade/make_hillshades.py: бейзлайн больше не
#  ищет готовые растры `*Lidar*.tif` / `*_Li_карты_*.tif`, а строит их сам из
#  облака точек LAS/LAZ — ровно тот стандартный набор, что лежит в датасете:
#
#      c    цветная отмывка «Hot to cold» (TerraModeler, 45°/25°), 3 цикла
#      ch   то же, но N циклов (N = 2 · перепад высот)
#      g    серая отмывка (ArcGIS: азимут 315°, высота 45°, Z = 1)
#      i    инвертированная серая (255 − g)
#
#  Поверхность строится только по точкам класса 2 (Ground) — «модель по точкам
#  земли»; если класса 2 в файле нет, берутся все точки (часть облаков датасета
#  уже отфильтрована и сохранена под другим классом). За пределами контура
#  съёмки сохраняется nodata, как в эталонах: у `g`/`i` это 255 (белый фон),
#  у `c`/`ch` — 0 (чёрный).
# ─────────────────────────────────────────────────────────────────────────────

#: Яркость палитры «Hot to cold»: в эталонах S = 1.0, V ≈ 209/255.
HOT_TO_COLD_VALUE = 209.0 / 255.0
#: Сдвиг тона палитры: hue(t = 0) = 0.75 (подгонка по эталонам датасета).
HOT_TO_COLD_PHASE = 0.75
#: Циклов у `c` (инструкция TerraModeler: «указываем 3»).
DEFAULT_CYCLES_C = 3.0
#: Байт classification в записи точки LAS: форматы 0–5 и 6–10 различаются.
_LAS_CLASS_BYTE = {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15,
                   6: 16, 7: 16, 8: 16, 9: 16, 10: 16}
#: Порядок и состав стандартного набора.
HILLSHADE_MODALITIES = ("c", "ch", "g", "i")


def find_las_files(region_dir: Path) -> List[Path]:
    """Все LAS/LAZ площадки; в каждой папке предпочитает отфильтрованный ground."""
    files = [
        path for path in region_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".las", ".laz")
    ]
    if not files:
        return []
    chosen: Dict[Path, Path] = {}
    for path in sorted(files):
        parent = path.parent
        current = chosen.get(parent)
        if current is None or ("ground" in path.name.lower() and "ground" not in current.name.lower()):
            chosen[parent] = path
    return list(chosen.values())


def _las_header_info(path: Path) -> Dict[str, Any]:
    """Читает заголовок LAS/LAZ (без разбора точек)."""
    with path.open("rb") as handle:
        head = handle.read(375)
    if len(head) < 227:
        raise RuntimeError(f"{path.name}: файл короче заголовка LAS")
    version = (head[24], head[25])
    point_offset = struct.unpack_from("<I", head, 96)[0]
    point_format = head[104]
    record_length = struct.unpack_from("<H", head, 105)[0]
    point_count = struct.unpack_from("<I", head, 107)[0]
    if version >= (1, 4) and len(head) >= 255:
        count_64 = struct.unpack_from("<Q", head, 247)[0]
        if count_64:
            point_count = count_64
    # 179..227: max X, min X, max Y, min Y, max Z, min Z (float64)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", head, 179)
    if not (max_x > min_x and max_y > min_y) or point_count <= 0:
        raise RuntimeError(
            f"{path.name}: некорректный заголовок LAS "
            f"(точек {point_count}, габариты {min_x}..{max_x} × {min_y}..{max_y})"
        )
    return {
        "version": version,
        "point_offset": point_offset,
        "point_format": point_format,
        "record_length": record_length,
        "point_count": int(point_count),
        "scale": struct.unpack_from("<3d", head, 131),
        "offset": struct.unpack_from("<3d", head, 155),
        "bounds": (min_x, min_y, max_x, max_y),
        "z_range": (min_z, max_z),
    }


def _iter_las_chunks(path: Path, header: Dict[str, Any], chunk_points: int):
    """Отдаёт порции точек (x, y, z, classification) без загрузки всего файла.

    Собственный разбор записи LAS (форматы 0–10) — чтобы не тянуть laspy в
    образ; для .laz используется laspy, если он установлен.
    """
    point_format = header["point_format"]
    record_length = header["record_length"]
    if path.suffix.lower() == ".laz":
        import laspy  # LAZ требует распаковки — без laspy никак

        with laspy.open(str(path)) as handle:
            for points in handle.chunk_iterator(chunk_points):
                yield (np.asarray(points.x, dtype=np.float64),
                       np.asarray(points.y, dtype=np.float64),
                       np.asarray(points.z, dtype=np.float64),
                       np.asarray(points.classification))
        return
    if point_format not in _LAS_CLASS_BYTE:
        raise RuntimeError(f"{path.name}: формат точек {point_format} не поддерживается")
    dtype = np.dtype({
        "names": ["x", "y", "z", "cls"],
        "formats": ["<i4", "<i4", "<i4", "u1"],
        "offsets": [0, 4, 8, _LAS_CLASS_BYTE[point_format]],
        "itemsize": record_length,
    })
    scale_x, scale_y, scale_z = header["scale"]
    off_x, off_y, off_z = header["offset"]
    remaining = header["point_count"]
    with path.open("rb") as handle:
        handle.seek(header["point_offset"])
        while remaining > 0:
            count = int(min(chunk_points, remaining))
            buffer = handle.read(count * record_length)
            count = len(buffer) // record_length
            if count <= 0:
                break
            records = np.frombuffer(buffer, dtype=dtype, count=count)
            classification = np.asarray(records["cls"])
            if point_format <= 5:
                # форматы 0–5: класс занимает младшие 5 бит, старшие — флаги
                classification = classification & 0x1F
            yield (records["x"].astype(np.float64) * scale_x + off_x,
                   records["y"].astype(np.float64) * scale_y + off_y,
                   records["z"].astype(np.float64) * scale_z + off_z,
                   classification)
            remaining -= count


def _fill_nan_pyramid(array: np.ndarray, max_levels: int = 16) -> np.ndarray:
    """Заполняет NaN-пустоты пирамидой уменьшенных копий (без scipy)."""
    current = np.asarray(array, dtype=np.float32)
    levels: List[np.ndarray] = []
    for _ in range(max_levels):
        if np.isfinite(current).all() or min(current.shape) < 2:
            break
        levels.append(current)
        pad_h, pad_w = current.shape[0] % 2, current.shape[1] % 2
        if pad_h or pad_w:
            current = np.pad(current, ((0, pad_h), (0, pad_w)),
                             mode="constant", constant_values=np.nan)
        valid = np.isfinite(current)
        values = np.where(valid, current, 0.0).astype(np.float32)
        counts = valid.reshape(current.shape[0] // 2, 2,
                               current.shape[1] // 2, 2).sum(axis=(1, 3))
        sums = values.reshape(current.shape[0] // 2, 2,
                              current.shape[1] // 2, 2).sum(axis=(1, 3))
        current = np.where(counts > 0, sums / np.maximum(counts, 1),
                           np.nan).astype(np.float32)
    finite = np.isfinite(current)
    if not finite.all():
        mean = float(current[finite].mean()) if finite.any() else 0.0
        current = np.where(finite, current, mean).astype(np.float32)
    for level in reversed(levels):
        upscaled = cv2.resize(current, (level.shape[1], level.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        current = np.where(np.isfinite(level), level, upscaled).astype(np.float32)
    return current


def _convex_hull_mask(valid: np.ndarray, max_points: int = 1_500_000) -> np.ndarray:
    """Маска выпуклой оболочки занятых ячеек — область, которую заполняет DEM."""
    height, width = valid.shape
    flat = np.flatnonzero(valid.ravel())
    if flat.size < 3:
        return valid.copy()
    if flat.size > max_points:
        flat = flat[::int(math.ceil(flat.size / max_points))]
    rows, cols = np.divmod(flat, width)
    points = np.column_stack([cols, rows]).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    return mask.astype(bool)


def _fill_by_delaunay(dem: np.ndarray, valid: np.ndarray, to_fill: np.ndarray,
                      max_vertices: int) -> Optional[np.ndarray]:
    """Линейная интерполяция по Делоне (как `fill_holes` в хиллшейд-проекте)."""
    if int(np.count_nonzero(valid)) > max_vertices:
        return None
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
    except Exception:
        return None
    try:
        rows, cols = np.nonzero(valid)
        triangulation = Delaunay(np.column_stack([cols, rows]).astype(np.float64))
        interpolator = LinearNDInterpolator(triangulation, dem[rows, cols])
        filled = dem.copy()
        fill_rows, fill_cols = np.nonzero(to_fill)
        chunk = 2_000_000
        for start in range(0, fill_rows.size, chunk):
            window = slice(start, start + chunk)
            values = interpolator(
                np.column_stack([fill_cols[window], fill_rows[window]]).astype(np.float64)
            )
            known = np.isfinite(values)
            filled[fill_rows[window][known], fill_cols[window][known]] = values[known]
        return filled
    except Exception:
        return None


def fill_interior_holes(dem: np.ndarray, max_vertices: int = 4_000_000) -> np.ndarray:
    """Заполняет пустоты DEM внутри контура съёмки; вне контура — остаётся nodata.

    Контур — выпуклая оболочка занятых ячеек: ровно та область, которую
    заполняет триангуляция Делоне в хиллшейд-проекте. Основной способ —
    линейная интерполяция по Делоне (scipy); если scipy недоступен или ячеек
    слишком много — многомасштабное заполнение без внешних зависимостей.
    """
    dem = np.asarray(dem)
    if dem.ndim != 2:
        raise ValueError(f"DEM должен быть двумерным, получено {dem.shape}")
    valid = np.isfinite(dem)
    if valid.all():
        return dem
    to_fill = _convex_hull_mask(valid) & ~valid
    if not to_fill.any():
        return dem
    filled = _fill_by_delaunay(dem, valid, to_fill, max_vertices)
    if filled is None:
        filled = np.where(to_fill, _fill_nan_pyramid(dem), dem)
    # страховка: у границы оболочки интерполяция может не дать значения
    unresolved = to_fill & ~np.isfinite(filled)
    if unresolved.any():
        filled = np.where(unresolved, _fill_nan_pyramid(filled), filled)
    return np.asarray(filled, dtype=np.float32)


def build_dem_from_las(las_path: Path, res: float = 0.1, interp: str = "mean",
                       ground_class: int = 2, use_all: bool = False,
                       max_pixels: int = 60_000_000, chunk_points: int = 2_000_000
                       ) -> Tuple[np.ndarray, Affine, Dict[str, Any]]:
    """Строит регулярную модель рельефа из LAS потоково (память ~сетка DEM).

    Повторяет логику `make_dem` из хиллшейд-проекта: агрегация mean/max по
    ячейкам, заполнение внутренних пустот, nodata вне контура съёмки.
    """
    from rasterio.transform import from_origin

    if interp not in ("mean", "max"):
        raise ValueError(f"interp={interp!r}: поддерживаются mean и max")
    header = _las_header_info(las_path)
    min_x, min_y, max_x, max_y = header["bounds"]

    def _extent(step: float) -> Tuple[float, float, int, int]:
        left = math.floor(min_x / step) * step
        bottom = math.floor(min_y / step) * step
        width = int(math.ceil((max_x - left) / step)) + 1
        height = int(math.ceil((max_y - bottom) / step)) + 1
        return left, bottom, width, height

    left, bottom, width, height = _extent(res)
    if max_pixels and width * height > max_pixels:
        factor = int(math.ceil(math.sqrt(width * height / max_pixels)))
        res = res * factor
        left, bottom, width, height = _extent(res)
        if factor > 1:
            logger.info(
                f"{Colors.DIM}      габариты дают слишком большой растр → "
                f"шаг увеличен в {factor}× до {res:g} м{Colors.RESET}"
            )
    transform = from_origin(left, bottom + height * res, res, res)
    top = transform.f
    cells = width * height

    def _accumulate(use_all_points: bool):
        sums = np.zeros(cells, dtype=np.float64) if interp == "mean" else None
        peaks = np.full(cells, -np.inf, dtype=np.float32) if interp == "max" else None
        counts = np.zeros(cells, dtype=np.int32)
        classes: Dict[int, int] = {}
        selected = 0
        for x, y, z, classification in _iter_las_chunks(las_path, header, chunk_points):
            unique, freq = np.unique(classification, return_counts=True)
            for value, amount in zip(unique, freq):
                classes[int(value)] = classes.get(int(value), 0) + int(amount)
            mask = np.ones(classification.shape, dtype=bool) if use_all_points \
                else (classification == ground_class)
            if not mask.any():
                continue
            x, y, z = x[mask], y[mask], z[mask]
            selected += int(x.size)
            col = np.floor((x - left) / res).astype(np.int64, copy=False)
            row = np.floor((top - y) / res).astype(np.int64, copy=False)
            inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            if not inside.any():
                continue
            flat = row[inside] * width + col[inside]
            values = z[inside]
            order = np.argsort(flat, kind="stable")
            flat, values = flat[order], values[order]
            starts = np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1])))
            uniq = flat[starts]
            if interp == "max":
                peaks[uniq] = np.maximum(peaks[uniq], np.maximum.reduceat(values, starts))
            else:
                sums[uniq] += np.add.reduceat(values, starts)
            counts[uniq] += np.diff(np.concatenate((starts, [flat.size])))
        return sums, counts, peaks, classes, selected

    sums, counts, peaks, classes, selected = _accumulate(use_all)
    if selected == 0 and not use_all:
        logger.warning("      класса %s в файле нет (есть %s) → берём все точки",
                       ground_class, classes)
        del sums, counts, peaks
        sums, counts, peaks, classes, selected = _accumulate(True)
    if selected == 0:
        raise RuntimeError(f"{las_path.name}: не удалось выбрать ни одной точки")

    occupied = counts > 0
    if interp == "max":
        dem = np.where(occupied, peaks, np.nan)
        del peaks
    else:
        dem = np.where(occupied, sums / np.maximum(counts, 1), np.nan)
        del sums
    del counts
    dem = dem.reshape(height, width).astype(np.float32)
    holes = int(np.count_nonzero(~occupied))
    dem = fill_interior_holes(dem)
    stats = {
        "res": res,
        "points": selected,
        "classes": classes,
        "holes": holes,
        "transform": transform,
    }
    return dem, transform, stats


def hillshade_esri(dem: np.ndarray, res_x: float, res_y: float,
                   azimuth: float = 315.0, altitude: float = 45.0,
                   z_factor: float = 1.0) -> np.ndarray:
    """Классическая формула ESRI/ArcGIS «Отмывка» (Hillshade)."""
    gy_north = -np.gradient(dem, res_y, axis=0)   # ось 0 — строки, вниз = юг
    gx_east = np.gradient(dem, res_x, axis=1)     # ось 1 — столбцы, вправо = восток
    slope = np.arctan(np.hypot(gx_east, gy_north) * z_factor)
    aspect = np.arctan2(-gx_east, -gy_north)
    zenith = math.radians(90.0 - altitude)
    azimuth_rad = math.radians(azimuth)
    shade = 255.0 * (math.cos(zenith) * np.cos(slope)
                     + math.sin(zenith) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return np.clip(shade, 0.0, 255.0)


def hsv_to_rgb_u8(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Векторный HSV→RGB (uint8, поканально) без зависимости от matplotlib."""
    h = (np.asarray(hue) % 1.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [val, q, p, p, t, val])
    g = np.select(conds, [t, val, val, q, p, p])
    b = np.select(conds, [p, p, t, val, val, q])
    rgb = np.stack([r, g, b]).astype(np.float32)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _open_writer(path: Path, height: int, width: int, count: int,
                 transform: Affine, crs) -> "rasterio.io.DatasetWriter":
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": count,
        "dtype": "uint8", "transform": transform, "crs": crs,
    }
    return rasterio.open(path, "w", **profile)


def _write_gray_pair(out_dir: Path, name: str, filled: np.ndarray, valid: np.ndarray,
                     step: float, transform: Affine, crs, tags: Iterable[str],
                     block_rows: int = 4096) -> List[Path]:
    """Серая и/или инвертированная отмывка (ArcGIS 315°/45°, Z = 1) по блокам строк."""
    tags = [tag for tag in ("g", "i") if tag in set(tags)]
    if not tags:
        return []
    height, width = filled.shape
    paths = {tag: out_dir / f"{name}_{tag}.tif" for tag in tags}
    writers = {tag: _open_writer(paths[tag], height, width, 1, transform, crs) for tag in tags}
    try:
        for top in range(0, height, block_rows):
            bottom = min(top + block_rows, height)
            lo, hi = max(0, top - 1), min(height, bottom + 1)   # halo для градиента
            shade = np.rint(hillshade_esri(filled[lo:hi], step, step)).astype(np.uint8)
            shade = shade[top - lo:top - lo + (bottom - top)]
            mask = ~valid[top:bottom]
            window = ((top, bottom), (0, width))
            if "g" in writers:
                gray = shade.copy()
                gray[mask] = 255
                writers["g"].write(gray, 1, window=window)
            if "i" in writers:
                inverted = (255 - shade.astype(np.int16)).astype(np.uint8)
                inverted[mask] = 255
                writers["i"].write(inverted, 1, window=window)
    finally:
        for writer in writers.values():
            writer.close()
    return [paths[tag] for tag in tags]


def _write_palette(out_dir: Path, name: str, tag: str, filled: np.ndarray,
                   valid: np.ndarray, z_min: float, z_range: float, cycles: float,
                   transform: Affine, crs, block_rows: int = 4096) -> Path:
    """Цветная отмывка «Hot to cold» (TerraModeler): hue = frac(0.75 − N·t)."""
    height, width = filled.shape
    path = out_dir / f"{name}_{tag}.tif"
    with _open_writer(path, height, width, 3, transform, crs) as writer:
        for top in range(0, height, block_rows):
            bottom = min(top + block_rows, height)
            block = filled[top:bottom]
            t = (block - z_min) / z_range
            hue = (HOT_TO_COLD_PHASE - cycles * t) % 1.0
            value = np.full(t.shape, HOT_TO_COLD_VALUE, dtype=np.float32)
            rgb = hsv_to_rgb_u8(hue, np.ones_like(t), value)    # (3, h, w)
            rgb[:, ~valid[top:bottom]] = 0
            writer.write(rgb, window=((top, bottom), (0, width)))
    return path


def make_hillshade_set(las_path: Path, out_dir: Path, res: float = 0.1,
                       modalities: Iterable[str] = HILLSHADE_MODALITIES,
                       interp: str = "mean", ground_class: int = 2,
                       use_all: bool = False, max_pixels: int = 60_000_000,
                       region_crs: Optional[str] = None,
                       cycles_c: float = DEFAULT_CYCLES_C,
                       cycles_ch: Optional[float] = None) -> List[Path]:
    """Строит из LAS стандартный набор растров и возвращает пути к ним."""
    modalities = [tag for tag in HILLSHADE_MODALITIES if tag in set(modalities)]
    if not modalities:
        raise ValueError("не выбрано ни одной модальности хиллшейдов")
    out_dir.mkdir(parents=True, exist_ok=True)
    if region_crs is None:
        region_crs = load_las_crs(las_path)

    dem, transform, stats = build_dem_from_las(
        las_path, res=res, interp=interp, ground_class=ground_class,
        use_all=use_all, max_pixels=max_pixels,
    )
    valid = np.isfinite(dem)
    if not valid.any():
        raise RuntimeError(f"{las_path.name}: DEM пуст")
    z_min = float(np.nanmin(dem))
    z_max = float(np.nanmax(dem))
    z_range = max(z_max - z_min, 1e-9)
    step = float(transform.a)
    filled = np.where(valid, dem, z_min)

    name = las_path.stem.replace("_La_точки", "").replace("_La", "").replace("_ground", "")
    written: List[Path] = []

    written.extend(_write_gray_pair(out_dir, name, filled, valid, step, transform,
                                    region_crs, modalities))

    cycles = {
        "c": float(cycles_c),
        # `c` — из инструкции TerraModeler; `ch` — «разницу высот умножаем на 2».
        # В датасете число циклов у каждой площадки своё, поэтому оба значения
        # можно задать явно (--cycles-c / --cycles-ch).
        "ch": float(cycles_ch) if cycles_ch else float(max(1, round(2.0 * z_range))),
    }
    for tag in ("c", "ch"):
        if tag not in modalities:
            continue
        written.append(_write_palette(out_dir, name, tag, filled, valid, z_min, z_range,
                                      cycles[tag], transform, region_crs))

    written.sort(key=lambda path: HILLSHADE_MODALITIES.index(path.stem.rsplit("_", 1)[-1]))
    log_table_row("DEM", f"{dem.shape[1]}×{dem.shape[0]} @ {step:g} м, "
                          f"z {z_min:.1f}…{z_max:.1f} м")
    log_table_row("Ground Points", f"{stats['points']:,} "
                                   f"(пустых ячеек {stats['holes']:,})")
    return written


def mask_to_polygons(mask: np.ndarray,
                     max_polygons: int = MAX_CONTOURS_PER_MASK) -> Iterable[Polygon]:
    """Бинарная маска -> полигоны через контуры (быстрее, чем rasterio.features.shapes)."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        if count >= max_polygons:
            break
        
        simplified = cv2.approxPolyDP(contour, CONTOUR_EPSILON_PX, True)
        
        if len(simplified) < 3:
            continue
        
        coords = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]
        
        try:
            poly = Polygon(coords)
            poly = make_valid(poly)
            
            if poly.is_empty or poly.area < MIN_CONTOUR_AREA:
                continue
            
            if isinstance(poly, MultiPolygon):
                for sub in poly.geoms:
                    if not sub.is_empty:
                        yield sub
                        count += 1
            else:
                yield poly
                count += 1
        except Exception as exc:
            logger.debug("пропущен контур маски: %s", exc)
            continue


def detections_to_features(detections, class_names: Dict[int, str],
                           transform: Affine, source_crs: str, scale: float, angle: float,
                           crop_offset: Tuple[int, int], rotation_center: Tuple[float, float],
                           region_name: str, markup_type: str = "li") -> List[Dict[str, Any]]:
    """Переводит найденные маски в GeoJSON-фичи с координатами целевой CRS."""
    affine_params = (transform.a, transform.b, transform.d, transform.e, transform.c, transform.f)
    transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    source_urn = f"urn:ogc:def:crs:{source_crs.replace(':', '::')}"
    target_urn = "urn:ogc:def:crs:EPSG::3857"
    features: List[Dict[str, Any]] = []
    rotate_origin = (float(rotation_center[0]), float(rotation_center[1]))
    
    need_rotate = abs(angle) > 1e-3
    need_scale = abs(scale - 1.0) > 1e-6
    
    def apply_transforms(poly: Polygon) -> Optional[Polygon]:
        """Apply all geometric transformations efficiently."""
        try:
            if crop_offset[0] != 0 or crop_offset[1] != 0:
                poly = shapely_translate(poly, xoff=float(crop_offset[0]), yoff=float(crop_offset[1]))
            
            if need_rotate:
                poly = shapely_rotate(poly, angle=-angle, origin=rotate_origin, use_radians=False)
            
            if need_scale:
                poly = shapely_scale(poly, xfact=1.0 / scale, yfact=1.0 / scale, origin=(0.0, 0.0))
            
            poly = shapely_affine_transform(poly, affine_params)
            poly = shapely_transform(transformer.transform, poly)
            
            poly = make_valid(poly)
            if poly.is_empty:
                return None
            # make_valid иногда возвращает GeometryCollection — оставляем только
            # полигональные части, иначе в GeoJSON попадёт геометрия без coordinates
            # (сервис метрики такую фичу отклоняет).
            if poly.geom_type not in ("Polygon", "MultiPolygon"):
                parts = [g for g in getattr(poly, "geoms", [])
                         if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty]
                if not parts:
                    return None
                poly = unary_union(parts)
                if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
                    return None
            return poly
        except Exception:
            return None
    
    for det in detections:
        if det.mask is None or not det.mask.any():
            continue

        polygons = mask_to_polygons(det.mask.astype(np.uint8))
        for poly in polygons:
            poly = apply_transforms(poly)
            if poly is None or poly.area < MIN_FEATURE_AREA:
                continue
            
            poly_simplified = poly.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
            if poly_simplified.is_empty or poly_simplified.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            geometry = mapping(poly_simplified)
            if not geometry.get("coordinates"):
                continue

            features.append({
                "type": "Feature",
                "properties": {
                    "class_name": normalize_class_name(class_names.get(det.class_id, det.class_name)),
                    "region_name": region_name,
                    "sub_region_name": "",
                    "markup_type": markup_type,
                    "original_crs": source_urn,
                    "crs": target_urn,
                    "fid": 0,
                    "confidence": round(det.confidence, 3),
                },
                "geometry": geometry,
            })
    
    return features


def deduplicate_polygons(features: List[Dict[str, Any]], iou_threshold: float = 0.95) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for feature in features:
        grouped[feature["properties"]["class_name"]].append(feature)
    kept: List[Dict[str, Any]] = []
    for items in grouped.values():
        unique: List[Tuple[Polygon, Dict[str, Any]]] = []
        for feat in items:
            poly = make_valid(shape(feat["geometry"]))
            if poly.is_empty:
                continue
            duplicate = False
            for stored_poly, _ in unique:
                intersection = poly.intersection(stored_poly).area
                union = poly.union(stored_poly).area
                if union and intersection / union > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                unique.append((poly, feat))
        kept.extend(feat for _, feat in unique)
    return kept


def merge_features(features: List[Dict[str, Any]], buffer_distance: float = 0.5,
                   min_area: float = 1.0) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for feat in features:
        props = feat["properties"]
        key = (
            props["class_name"],
            props["region_name"],
            props["sub_region_name"],
            props["markup_type"],
        )
        groups[key].append(feat)
    merged: List[Dict[str, Any]] = []
    for key, feats in groups.items():
        polygons: List[Polygon] = []
        confidences: List[float] = []
        for feat in feats:
            poly = make_valid(shape(feat["geometry"]))
            if poly.is_empty:
                continue
            polygons.append(poly)
            conf = feat["properties"].get("confidence")
            if conf is not None:
                confidences.append(conf)
        if not polygons:
            continue
        buffered = [poly.buffer(buffer_distance) for poly in polygons]
        union = unary_union(buffered).buffer(-buffer_distance)
        geometries = union.geoms if isinstance(union, MultiPolygon) else [union]
        template = feats[0]["properties"].copy()
        for geom in geometries:
            geom = make_valid(geom)
            if not isinstance(geom, Polygon) or geom.is_empty or geom.area < min_area:
                continue
            props = template.copy()
            if confidences:
                props["confidence"] = round(sum(confidences) / len(confidences), 3)
            merged.append({
                "type": "Feature",
                "properties": props,
                "geometry": mapping(geom),
            })
    return merged


def _infer_image(
    model,
    class_names: Dict[int, str],
    image: np.ndarray,
    transform: Affine,
    source_crs: Optional[str],
    region_name: str,
    markup_type: str,
    infer: InferConfig,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """prepare → CLAHE → (downscale) → (align) → тайловый инференс → GeoJSON-фичи."""
    prep_start = time.time()
    working_image = prepare_image(image)
    if verbose:
        log_table_row("Original Image Size", f"{working_image.shape[1]}×{working_image.shape[0]}")
        log_timing("Prepare Time", time.time() - prep_start)

    scale = 1.0
    if infer.enable_clahe:
        clahe_start = time.time()
        working_image = apply_clahe(working_image, infer.clahe_clip_limit,
                                    (infer.clahe_tile_size, infer.clahe_tile_size))
        if verbose:
            log_timing("CLAHE Time", time.time() - clahe_start)

    if infer.enable_downscale:
        scale_start = time.time()
        working_image, scale = downscale_image(working_image)
        if verbose:
            log_table_row("Downscaled To", f"{working_image.shape[1]}×{working_image.shape[0]} (scale: {scale:.3f})")
            log_timing("Downscale Time", time.time() - scale_start)

    if infer.enable_alignment:
        align_start = time.time()
        aligned_image, angle, crop_offset, rotation_center = align_and_crop_image(working_image)
        if verbose:
            log_table_row("Aligned & Cropped", f"{aligned_image.shape[1]}×{aligned_image.shape[0]} (rotated {angle:.2f}°)")
            log_timing("Alignment Time", time.time() - align_start)
    else:
        aligned_image = working_image
        angle = 0.0
        crop_offset = (0, 0)
        rotation_center = (working_image.shape[1] / 2.0, working_image.shape[0] / 2.0)

    if verbose:
        log_metric("Processing Size", f"{aligned_image.shape[1]}×{aligned_image.shape[0]}")

    infer_start = time.time()
    detections = run_tiled_inference(model, aligned_image, infer, class_names)
    if verbose:
        log_timing("Inference Time", time.time() - infer_start)

    if not detections:
        if verbose:
            log_table_row("Detections", f"{Colors.DIM}0 (no objects found){Colors.RESET}")
        return []

    crs = source_crs or "EPSG:32636"
    if "LOCAL_CS" in str(crs).upper():
        logger.warning("  ⚠️  LOCAL_CS detected, falling back to EPSG:32636")
        crs = "EPSG:32636"

    feat_start = time.time()
    features = detections_to_features(
        detections,
        class_names,
        transform,
        crs,
        scale,
        angle,
        crop_offset,
        rotation_center,
        region_name,
        markup_type=markup_type
    )
    if verbose:
        log_metric("Features Generated", len(features), "✓")
        log_timing("Feature Conversion Time", time.time() - feat_start)
    return features


def process_hillshade(
    model,
    class_names: Dict[int, str],
    tif_path: Path,
    idx: int,
    total: int,
    region_crs: Optional[str],
    region_name: str,
    infer: InferConfig
) -> List[Dict[str, Any]]:
    """Прогоняет модель по одному растру и возвращает GeoJSON-фичи."""
    file_start_time = time.time()
    file_size_mb = tif_path.stat().st_size / (1024 * 1024)
    log_file_header(f"{idx}/{total}: {tif_path.name}")
    log_table_row("File Size", f"{file_size_mb:.2f} MB")
    features: List[Dict[str, Any]] = []
    try:
        load_start = time.time()
        with rasterio.open(tif_path) as src:
            data = src.read()
            transform = src.transform
            file_crs = src.crs.to_string() if src.crs else None
        log_timing("Load Time", time.time() - load_start)

        features = _infer_image(model, class_names, data, transform,
                                region_crs or file_crs, region_name, "li", infer)
    except Exception as exc:
        logger.error("  ✗ Failed: %s", exc)

    log_timing("Total File Time", time.time() - file_start_time)
    log_file_footer()
    return features


# ─────────────────────────────────────────────────────────────────────────────
#  Не-лидарные модальности (ортофото / спутник / аэрофото)
#
#  Готовые хиллшейды (`*_Lidar_*.tif`, `*_Li_карты*`) по-прежнему не читаются:
#  лидарный набор строится из LAS. Остальные снимки площадки можно подать в
#  модель тем же пайплайном — они читаются окнами, потому что орто/аэро бывают
#  в сотни Мпикс и целиком в память не влезают.
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_MODALITIES = ("or", "sp_or", "ae")

#: Какие не-лидарные снимки обрабатываем по умолчанию (ортофото + спутник).
DEFAULT_EXTRA_MODALITIES = ("or", "sp_or")


def modality_of(path: Path) -> Optional[str]:
    """Модальность растра по пути; None — если это готовый хиллшейд или не снимок."""
    low = str(path).lower()
    if any(mark in low for mark in ("lidar", "li_карт", "li_карты", "hillshade", "отмывк")):
        return None
    if "spor" in low or "sp_or" in low or "спутник" in low:
        return "sp_or"
    if "_or_" in low or "ортофото" in low:
        return "or"
    if "_ae_" in low or "немецк" in low or "luft" in low:
        return "ae"
    return None


def find_extra_rasters(region_dir: Path, modalities: Iterable[str]) -> List[Tuple[Path, str]]:
    """Пары (растр, модальность) для не-лидарных снимков площадки."""
    wanted = {m.lower() for m in modalities}
    found: List[Tuple[Path, str]] = []
    for path in sorted(region_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        modality = modality_of(path)
        if modality and modality in wanted:
            found.append((path, modality))
    return found


def process_extra_raster(
    model,
    class_names: Dict[int, str],
    raster_path: Path,
    modality: str,
    idx: int,
    total: int,
    region_crs: Optional[str],
    region_name: str,
    infer: InferConfig,
    window_pixels: int = 48_000_000
) -> List[Dict[str, Any]]:
    """Не-лидарный снимок: чтение окнами + инференс по каждому окну."""
    from rasterio.windows import Window, transform as window_transform

    features: List[Dict[str, Any]] = []
    file_start_time = time.time()
    try:
        with rasterio.open(raster_path) as src:
            width, height, bands = src.width, src.height, src.count
            file_crs = src.crs.to_string() if src.crs else None
            crs = region_crs or file_crs
            side = int(max(1024, math.floor(math.sqrt(max(window_pixels, 1_000_000)))))
            cols = math.ceil(width / side)
            rows = math.ceil(height / side)
            log_file_header(f"{idx}/{total}: {raster_path.name}")
            log_table_row("Modality", modality)
            log_table_row("Raster", f"{width}×{height}, {bands} band(s)")
            log_table_row("Windows", f"{cols}×{rows} = {cols * rows} по {side}px")
            for row in range(rows):
                for col in range(cols):
                    x, y = col * side, row * side
                    w, h = min(side, width - x), min(side, height - y)
                    window = Window(x, y, w, h)
                    data = src.read(window=window)
                    transform = window_transform(window, src.transform)
                    try:
                        batch = _infer_image(model, class_names, data, transform, crs,
                                             region_name, modality, infer, verbose=False)
                    except Exception as exc:
                        logger.error("    окно %d,%d: %s", x, y, exc)
                        batch = []
                    if batch:
                        logger.info(
                            f"{Colors.DIM}    окно {row + 1}/{rows}×{col + 1}/{cols} "
                            f"({w}×{h}): +{len(batch)} фич{Colors.RESET}"
                        )
                    features.extend(batch)
                    del data
    except Exception as exc:
        logger.error("  ✗ Failed: %s", exc)
    log_timing("Total File Time", time.time() - file_start_time)
    log_file_footer()
    return features


def _filter_min_area(features: List[Dict[str, Any]], min_area: float) -> List[Dict[str, Any]]:
    """Отбрасывает полигоны мельче min_area (м²) — как это делает merge_features."""
    kept: List[Dict[str, Any]] = []
    skipped = 0
    for feature in features:
        try:
            if shape(feature["geometry"]).area >= min_area:
                kept.append(feature)
        except Exception:
            skipped += 1
    if skipped:
        logger.debug("отброшено полигонов с нечитаемой геометрией: %d", skipped)
    return kept


def process_region(
    model,
    class_names: Dict[int, str],
    region_dir: Path,
    infer: InferConfig,
    hillshade_res: float = 0.1,
    hillshade_max_pixels: int = 60_000_000,
    hillshade_modalities: Iterable[str] = HILLSHADE_MODALITIES,
    hillshade_interp: str = "mean",
    hillshade_class: int = 2,
    hillshade_cycles_c: float = DEFAULT_CYCLES_C,
    hillshade_cycles_ch: Optional[float] = None,
    keep_hillshades: Optional[str] = None,
    extra_modalities: Iterable[str] = (),
    window_pixels: int = 48_000_000
) -> List[Dict[str, Any]]:
    region_start_time = time.time()
    log_section(f"🏗️  Processing Region: {region_dir.name}")
    las_files = find_las_files(region_dir)
    extra_rasters = find_extra_rasters(region_dir, extra_modalities)
    if not las_files and not extra_rasters:
        logger.warning("  ⚠️  No LAS files found")
        return []
    region_crs = load_region_crs(region_dir)
    if region_crs:
        log_table_row("Region CRS", f"{region_crs} (from UTM.json)")
    elif las_files:
        region_crs = load_las_crs(las_files[0])
        if region_crs:
            log_table_row("Region CRS", f"{region_crs} (from LAS header)")
        else:
            log_table_row("Region CRS", "Auto-detect")
    else:
        log_table_row("Region CRS", "Auto-detect (from rasters)")
    log_table_row("LAS Files Found", len(las_files))
    if extra_modalities:
        log_table_row("Extra Rasters", f"{len(extra_rasters)} ({', '.join(sorted({m for _, m in extra_rasters})) or '—'})")

    # Готовые хиллшейды строим сами из LAS во временный каталог (или сохраняем,
    # если задан --keep-hillshades) — входные растры больше не ищутся.
    temp_root: Optional[Path] = None
    if keep_hillshades:
        hillshade_dir = Path(keep_hillshades) / region_dir.name
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="koz3_hillshades_"))
        hillshade_dir = temp_root
    try:
        hillshades: List[Path] = []
        build_start = time.time()
        for las_path in las_files:
            log_table_row("LAS File", f"{las_path.name} ({las_path.stat().st_size / (1024 ** 2):.0f} MB)")
            try:
                hillshades.extend(
                    make_hillshade_set(
                        las_path,
                        hillshade_dir / las_path.stem,
                        res=hillshade_res,
                        modalities=hillshade_modalities,
                        interp=hillshade_interp,
                        ground_class=hillshade_class,
                        max_pixels=hillshade_max_pixels,
                        region_crs=region_crs,
                        cycles_c=hillshade_cycles_c,
                        cycles_ch=hillshade_cycles_ch,
                    )
                )
            except Exception as exc:
                logger.error("  ✗ %s: %s", las_path.name, exc)
        hillshades = list(dict.fromkeys(hillshades))
        if las_files:
            log_timing("Hillshade Build Time", time.time() - build_start)
        if not hillshades and not extra_rasters:
            logger.warning("  ⚠️  No hillshades built")
            return []
        log_table_row("Hillshades Built", len(hillshades))

        features_all: List[Dict[str, Any]] = []
        for idx, tif_path in enumerate(hillshades, 1):
            features_all.extend(
                process_hillshade(model, class_names, tif_path, idx,
                                  len(hillshades), region_crs, region_dir.name, infer)
            )

        for idx, (raster_path, modality) in enumerate(extra_rasters, 1):
            features_all.extend(
                process_extra_raster(model, class_names, raster_path, modality, idx,
                                     len(extra_rasters), region_crs, region_dir.name, infer,
                                     window_pixels=window_pixels)
            )

        region_elapsed = time.time() - region_start_time
        log_timing("Total Region Time", region_elapsed)
        return features_all
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


def assign_fids(features: List[Dict[str, Any]]) -> None:
    for idx, feat in enumerate(features):
        feat["properties"]["fid"] = idx


@dataclass(frozen=True)
class RunConfig:
    """Настройки прогона: инференс, хиллшейды из LAS, постобработка, модальности."""

    infer: InferConfig = field(default_factory=InferConfig)
    merge_buffer: float = 0.5
    min_area: float = MIN_FEATURE_AREA
    hillshade_res: float = 0.1
    hillshade_max_pixels: int = 60_000_000
    hillshade_modalities: Tuple[str, ...] = tuple(HILLSHADE_MODALITIES)
    hillshade_interp: str = "mean"
    hillshade_class: int = 2
    hillshade_cycles_c: float = DEFAULT_CYCLES_C
    hillshade_cycles_ch: Optional[float] = None
    keep_hillshades: Optional[str] = None
    enable_merge: bool = False
    extra_modalities: Tuple[str, ...] = tuple(DEFAULT_EXTRA_MODALITIES)
    window_pixels: int = 48_000_000
    seed: int = DEFAULT_SEED


def _resolve_paths(input_dir: str, output_dir: str,
                   model_path: Optional[str]) -> Tuple[Path, Path, Path]:
    """Проверяет вход/выход/веса и создаёт каталог результата.

    Если output_dir оканчивается на «.geojson» — это итоговый файл результата,
    создаётся его родительский каталог. Иначе output_dir — каталог результата,
    он создаётся целиком.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    output_path = Path(output_dir)
    if output_path.suffix == ".geojson":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    weights = Path(model_path) if model_path else Path(__file__).parent / "model.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    return input_path, output_path, weights



def _log_config(cfg: RunConfig, input_dir: str, output_dir: str, weights: Path) -> None:
    log_section("📋 Configuration")
    log_table_row("Input Directory", input_dir)
    log_table_row("Output Directory", output_dir)
    log_table_row("Model Weights", weights.name)

    log_subsection("Model Parameters")
    log_table_row("Confidence Threshold", f"{cfg.infer.conf_threshold:.2f}")
    log_table_row("IOU Threshold", f"{cfg.infer.iou_threshold:.2f}")

    log_subsection("Tiling Parameters")
    log_table_row("Tile Size", f"{cfg.infer.tile_width}×{cfg.infer.tile_height}")
    log_table_row("Tile Overlap", f"{cfg.infer.overlap_ratio:.1%}")

    log_subsection("Preprocessing Options")
    log_table_row("CLAHE Enhancement", "✓ Enabled" if cfg.infer.enable_clahe else "✗ Disabled")
    if cfg.infer.enable_clahe:
        log_table_row("  └─ Clip Limit", f"{cfg.infer.clahe_clip_limit:.1f}")
        log_table_row("  └─ Tile Grid Size", f"{cfg.infer.clahe_tile_size}×{cfg.infer.clahe_tile_size}")
    log_table_row("Downscaling", "✓ Enabled" if cfg.infer.enable_downscale else "✗ Disabled")
    log_table_row("Alignment & Rotation", "✓ Enabled" if cfg.infer.enable_alignment else "✗ Disabled")

    log_subsection("Hillshades (built from LAS)")
    cycles_note = f"{cfg.hillshade_cycles_c:g} / {cfg.hillshade_cycles_ch:g}" if cfg.hillshade_cycles_ch \
        else f"{cfg.hillshade_cycles_c:g} / 2·Δz"
    log_table_row("Modalities", ", ".join(cfg.hillshade_modalities))
    log_table_row("DEM Step", f"{cfg.hillshade_res:g} m (max {cfg.hillshade_max_pixels:,} px)")
    log_table_row("DEM Interpolation", cfg.hillshade_interp)
    log_table_row("Ground Class", cfg.hillshade_class)
    log_table_row("Palette Cycles c/ch", cycles_note)
    log_table_row("Keep Hillshades", cfg.keep_hillshades or "✗ (temp dir)")

    log_subsection("Post-Processing")
    log_table_row("Polygon Merging", "✓ Enabled (--merge)" if cfg.enable_merge else "✗ Disabled (dedup only)")
    if cfg.enable_merge:
        log_table_row("Merge Buffer", f"{cfg.merge_buffer:.1f} м")
    log_table_row("Minimum Area", f"{cfg.min_area:.1f} м²")

    log_subsection("Extra modalities (non-LiDAR)")
    log_table_row("Modalities", ", ".join(cfg.extra_modalities) or "✗ Disabled")
    log_table_row("Window Size", f"{cfg.window_pixels / 1e6:.0f} Mpx per window")

    log_subsection("Determinism")
    log_table_row("Random Seed", cfg.seed)
    log_table_row("cuDNN Deterministic", "✓ Yes (benchmark off)")

    log_subsection("GPU Optimization")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log_table_row("GPU Device", f"{gpu_name}")
        log_table_row("GPU Memory", f"{gpu_memory:.1f} GB")
        log_table_row("CUDA Enabled", "✓ Yes")
        log_table_row("Mixed Precision", "✓ FP32 (Disabled)")
    else:
        log_table_row("GPU Available", "✗ No (CPU Mode)")


def _find_region_dirs(input_path: Path) -> List[Path]:
    """Площадки: сначала папки `*_FINAL` (раскладка исходного датасета), иначе —
    каждая подпапка входа (на платформе площадки анонимизированы как
    `region_XXXX`, и имя папки идёт в `region_name` результата), и только если
    подпапок нет — сам вход как единственная площадка.
    """
    region_dirs = sorted(d for d in input_path.iterdir() if d.is_dir() and d.name.endswith("_FINAL"))
    if not region_dirs:
        region_dirs = sorted(d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith("."))
    return region_dirs or [input_path]


def _collect_features(model, class_names: Dict[int, str], region_dirs: List[Path],
                      cfg: RunConfig) -> List[Dict[str, Any]]:
    """Обходит площадки и собирает сырые фичи (до фильтрации классов)."""
    collected: List[Dict[str, Any]] = []
    for region_dir in region_dirs:
        collected.extend(
            process_region(
                model,
                class_names,
                region_dir,
                cfg.infer,
                hillshade_res=cfg.hillshade_res,
                hillshade_max_pixels=cfg.hillshade_max_pixels,
                hillshade_modalities=cfg.hillshade_modalities,
                hillshade_interp=cfg.hillshade_interp,
                hillshade_class=cfg.hillshade_class,
                hillshade_cycles_c=cfg.hillshade_cycles_c,
                hillshade_cycles_ch=cfg.hillshade_cycles_ch,
                keep_hillshades=cfg.keep_hillshades,
                extra_modalities=cfg.extra_modalities,
                window_pixels=cfg.window_pixels,
            )
        )
    return collected


def _postprocess(collected: List[Dict[str, Any]],
                 cfg: RunConfig) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Класс-фильтр, чистка геометрий, дедупликация, (слияние | фильтр площади), FID."""
    log_section("📊 Post-Processing Statistics")
    log_metric("Raw Features", len(collected))
    collected = drop_unsupported_classes(collected)
    log_metric("After Class Filter", len(collected))
    collected = drop_empty_geometries(collected)

    dedup_elapsed = merge_elapsed = 0.0
    if collected:
        dedup_start = time.time()
        unique = deduplicate_polygons(collected)
        dedup_elapsed = time.time() - dedup_start
        log_metric("After Deduplication", len(unique), f"(-{len(collected) - len(unique)})")
        log_timing("Deduplication Time", dedup_elapsed)

        if cfg.enable_merge:
            merge_start = time.time()
            merged = merge_features(unique, buffer_distance=cfg.merge_buffer, min_area=cfg.min_area)
            merge_elapsed = time.time() - merge_start
            log_metric("After Merging", len(merged), f"(-{len(unique) - len(merged)})")
            log_timing("Merge Time", merge_elapsed)
        else:
            # Без слияния: матчинг в метрике жадно один-к-одному, поэтому
            # склейка масок соседних объектов режет recall (TP ≤ число полигонов).
            merged = _filter_min_area(unique, cfg.min_area)
            log_metric("After Area Filter", len(merged), f"(-{len(unique) - len(merged)})")
    else:
        merged = []

    fid_start = time.time()
    assign_fids(merged)
    fid_elapsed = time.time() - fid_start
    log_timing("Assign FID Time", fid_elapsed)

    return merged, {"dedup": dedup_elapsed, "merge": merge_elapsed, "fid": fid_elapsed}


def _save_geojson(output_path: Path,
                  features: List[Dict[str, Any]]) -> Tuple[Path, float]:
    """Пишет GeoJSON в заданный файл .geojson или в <output_path>/result.geojson.

    Возвращает путь к записанному файлу и время записи.
    """
    save_start = time.time()
    output_file = (output_path if output_path.suffix == ".geojson"
                   else output_path / "result.geojson")
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features},
                  handle, ensure_ascii=False, indent=2)
    return output_file, time.time() - save_start



def predict(
    input_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    merge_buffer: float = 0.5,
    min_area: float = MIN_FEATURE_AREA,
    enable_downscale: bool = True,
    enable_alignment: bool = True,
    enable_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    tile_height: int = 1024,
    tile_width: int = 1024,
    overlap_ratio: float = 0.3,
    iou_threshold: float = 0.5,
    hillshade_res: float = 0.1,
    hillshade_max_pixels: int = 60_000_000,
    hillshade_modalities: Iterable[str] = HILLSHADE_MODALITIES,
    hillshade_interp: str = "mean",
    hillshade_class: int = 2,
    hillshade_cycles_c: float = DEFAULT_CYCLES_C,
    hillshade_cycles_ch: Optional[float] = None,
    keep_hillshades: Optional[str] = None,
    enable_merge: bool = False,
    extra_modalities: Iterable[str] = DEFAULT_EXTRA_MODALITIES,
    window_pixels: int = 48_000_000,
    seed: int = DEFAULT_SEED
) -> Dict[str, Any]:
    """Оркестратор: пути → конфиг → модель → сбор фич → постобработка → GeoJSON."""
    pipeline_start_time = time.time()
    log_header("🗿🔎 Archeology segmentation pipeline 🔎🗿", "═", 80)
    logger.info("⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺⛏️🕵️‍♀️🏺")

    input_path, output_path, weights = _resolve_paths(input_dir, output_dir, model_path)
    cfg = RunConfig(
        infer=InferConfig(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            tile_height=tile_height,
            tile_width=tile_width,
            overlap_ratio=overlap_ratio,
            enable_downscale=enable_downscale,
            enable_alignment=enable_alignment,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size,
        ),
        merge_buffer=merge_buffer,
        min_area=min_area,
        hillshade_res=hillshade_res,
        hillshade_max_pixels=hillshade_max_pixels,
        hillshade_modalities=tuple(hillshade_modalities),
        hillshade_interp=hillshade_interp,
        hillshade_class=hillshade_class,
        hillshade_cycles_c=hillshade_cycles_c,
        hillshade_cycles_ch=hillshade_cycles_ch,
        keep_hillshades=keep_hillshades,
        enable_merge=enable_merge,
        extra_modalities=tuple(extra_modalities),
        window_pixels=window_pixels,
        seed=seed,
    )
    _log_config(cfg, input_dir, output_dir, weights)

    configure_determinism(seed)

    model_load_start = time.time()
    model, class_names = load_model(str(weights))
    model_load_elapsed = time.time() - model_load_start
    log_timing("Model Load Time", model_load_elapsed)

    region_dirs = _find_region_dirs(input_path)
    log_table_row("Regions to Process", len(region_dirs))

    processing_start = time.time()
    collected = _collect_features(model, class_names, region_dirs, cfg)
    processing_elapsed = time.time() - processing_start
    log_timing("Total Processing Time", processing_elapsed)

    merged, timings = _postprocess(collected, cfg)
    output_file, save_elapsed = _save_geojson(output_path, merged)
    log_timing("Save GeoJSON Time", save_elapsed)

    log_section("✅ Processing Complete")
    log_metric("Total Features", len(merged), "🎯")
    log_metric("Output File", str(output_file), "💾")

    pipeline_elapsed = time.time() - pipeline_start_time
    log_section("⏱️  Timing Summary")
    log_timing("Model Load", model_load_elapsed)
    log_timing("Region Processing", processing_elapsed)
    log_timing("Post-Processing",
               sum(timings.values()) + save_elapsed)
    log_timing("Total Pipeline Time", pipeline_elapsed)

    log_header("Done!", "═", 80)

    return {
        "total_features": len(merged),
        "output_file": str(output_file),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solution.py",
        description="Archeology segmentation: hillshades from LAS + optional non-LiDAR rasters.",
    )
    parser.add_argument("input_dir", help="каталог с площадками (например public_dataset)")
    parser.add_argument("output_dir", help="каталог для result.geojson")
    parser.add_argument("model", nargs="?", default=None, help="путь к весам (по умолчанию model.pt)")
    parser.add_argument("--model", dest="model_flag", default=None, help="путь к весам")

    prep = parser.add_argument_group("препроцессинг")
    prep.add_argument("--downscale", action="store_true", help="уменьшать изображение перед инференсом")
    prep.add_argument("--align", action="store_true", help="выравнивать и обрезать изображение")
    prep.add_argument("--clahe", action="store_true", default=True, help=argparse.SUPPRESS)
    prep.add_argument("--no-clahe", dest="clahe", action="store_false", help="отключить CLAHE")
    prep.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit (2.0)")
    prep.add_argument("--clahe-tile", type=int, default=8, help="CLAHE tile grid size (8)")

    hill = parser.add_argument_group("хиллшейды из LAS")
    hill.add_argument("--res", type=float, default=0.1, help="шаг сетки DEM, м (0.1)")
    hill.add_argument("--max-pixels", type=int, default=60_000_000, help="лимит пикселей растра (60e6)")
    hill.add_argument("--modalities", default=",".join(HILLSHADE_MODALITIES),
                      help="какие растры строить: c,ch,g,i (все)")
    hill.add_argument("--interp", choices=("mean", "max"), default="mean", help="агрегация DEM (mean)")
    hill.add_argument("--hillshade-class", type=int, default=2, help="класс LAS для модели рельефа (2)")
    hill.add_argument("--cycles-c", type=float, default=DEFAULT_CYCLES_C, help="циклов палитры у c (3)")
    hill.add_argument("--cycles-ch", type=float, default=None, help="циклов палитры у ch (2·Δz)")
    hill.add_argument("--keep-hillshades", default=None, metavar="DIR",
                      help="сохранить построенные растры в DIR (по умолчанию временный каталог)")

    post = parser.add_argument_group("постобработка и модальности")
    post.add_argument("--merge", action="store_true", help="склеивать пересекающиеся полигоны")
    post.add_argument("--extra-modalities", default=",".join(DEFAULT_EXTRA_MODALITIES),
                      metavar="LIST", help=f"не-лидарные снимки: {','.join(EXTRA_MODALITIES)} "
                                           f"(по умолчанию {','.join(DEFAULT_EXTRA_MODALITIES)}; none — выключить)")
    post.add_argument("--window-pixels", type=int, default=48_000_000,
                      help="размер окна для больших снимков, пикселей (48e6)")
    post.add_argument("--seed", type=int, default=DEFAULT_SEED,
                      help=f"зерно ГПСЧ для воспроизводимости прогона ({DEFAULT_SEED})")
    return parser


def parse_modalities(raw: str, allowed: Iterable[str], option: str) -> Tuple[str, ...]:
    """'c, ch' -> ('c', 'ch'); 'none' -> пустой кортеж."""
    tags = tuple(
        tag.strip().lower()
        for tag in raw.replace(" ", ",").split(",")
        if tag.strip() and tag.strip().lower() not in ("none", "off")
    )
    unknown = [tag for tag in tags if tag not in allowed]
    if unknown:
        raise ValueError(
            f"{option}: недопустимые значения {unknown} "
            f"(ожидается подмножество {','.join(allowed)} или 'none')"
        )
    return tags


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        modalities = parse_modalities(args.modalities, HILLSHADE_MODALITIES, "--modalities")
        if not modalities:
            raise ValueError("--modalities: нужен хотя бы один растр (c, ch, g или i)")
        extras = parse_modalities(args.extra_modalities, EXTRA_MODALITIES, "--extra-modalities")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    try:
        stats = predict(
            args.input_dir,
            args.output_dir,
            model_path=args.model_flag or args.model,
            enable_downscale=args.downscale,
            enable_alignment=args.align,
            enable_clahe=args.clahe,
            clahe_clip_limit=args.clahe_clip,
            clahe_tile_size=args.clahe_tile,
            hillshade_res=args.res,
            hillshade_max_pixels=args.max_pixels,
            hillshade_modalities=modalities,
            hillshade_interp=args.interp,
            hillshade_class=args.hillshade_class,
            hillshade_cycles_c=args.cycles_c,
            hillshade_cycles_ch=args.cycles_ch,
            keep_hillshades=args.keep_hillshades,
            enable_merge=args.merge,
            extra_modalities=extras,
            window_pixels=args.window_pixels,
            seed=args.seed,
        )
        print(f"Features: {stats['total_features']}")
        print(f"Result: {stats['output_file']}")
    except Exception as exc:
        logger.error("Processing failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
