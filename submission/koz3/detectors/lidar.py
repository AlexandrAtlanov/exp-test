"""Детектор лидар: YOLO-seg на отмывках рельефа из LAS -> полигоны в UTM.

Для каждого LAS строит отмывки бейзлайна (lidar_dem) во временной папке,
режет каждую инференс-отмывку на тайлы с перекрытием, предсказывает маски
и переводит их в полигоны в UTM региона. Дубли от перекрытия тайлов и от
нескольких отмывок здесь не подавляются — это делает postprocess по парам
(class_name, markup_type).
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import rasterio
from pyproj import CRS as ProjCRS
from rasterio.crs import CRS as RasterCRS
from rasterio.transform import Affine
from shapely.geometry import Polygon

from .. import config, geometry, yolo_runtime
from ..datatypes import Detection, Region
from ..timing import Budget
from . import lidar_dem

logger = logging.getLogger(__name__)

PRIORITY: int = 1
"""Порядок запуска детекторов: меньше — раньше (раздел 5 пакета)."""

MARKUP_TYPE: str = "li"
"""Тип данных детектора для свойства markup_type."""

_BYTES_PER_GB: float = 1e9
"""ГБ = 1e9 байт; в этих единицах калибруется seconds_per_gb."""

_model: yolo_runtime.LoadedModel | None = None
"""Кэш модели на прогон: загрузка один раз (раздел 5 пакета)."""

_model_error: str = ""
"""Причина неудачной загрузки модели: повторы в этом прогоне не делаются."""


def is_available(region: Region) -> bool:
    """Детектор применим: есть LAS в регионе и веса (.pt или .engine)."""
    if not region.files.get("li"):
        return False
    weights_dir = config.WEIGHTS_DIR
    has_pt = (weights_dir / str(config.YOLO["weights_pt"])).is_file()
    has_engine = (weights_dir / str(config.YOLO["weights_engine"])).is_file()
    return has_pt or has_engine


def estimate_seconds(region: Region) -> float:
    """Оценка времени на регион, с: суммарный размер LAS * seconds_per_gb."""
    try:
        las_files = lidar_dem.find_las_files(region.path)
    except Exception:  # noqa: BLE001  # оценка не должна ронять планирование
        logger.exception("li %s: поиск LAS упал: %s", region.name, region.path)
        return 0.0
    total_bytes = 0
    for las_path in las_files:
        try:
            total_bytes += las_path.stat().st_size
        except OSError as exc:
            logger.warning(
                "li %s: нет размера %s: %s",
                region.name,
                las_path.name,
                exc,
            )
    return total_bytes / _BYTES_PER_GB * float(config.LIDAR["seconds_per_gb"])


def run(region: Region, budget: Budget, cfg: ModuleType) -> list[Detection]:
    """Найти объекты региона по лидару; результат — полигоны в UTM региона.

    Ошибка одного LAS (или одного файла внутри него) не прерывает остальные.
    При нехватке бюджета пропускаются целые LAS и отдельные отмывки; уже
    найденные детекции возвращаются всегда.
    """
    try:
        las_files = lidar_dem.find_las_files(region.path)
    except Exception:  # noqa: BLE001  # список LAS не должен ронять регион
        logger.exception("li %s: поиск LAS упал: %s", region.name, region.path)
        return []
    if not las_files:
        logger.warning("li %s: LAS-файлы не найдены", region.name)
        return []

    model: yolo_runtime.LoadedModel | None = None
    detections: list[Detection] = []
    for las_path in las_files:
        try:
            las_estimate_s = _las_seconds(las_path, cfg)
        except OSError as exc:
            logger.warning(
                "li %s: %s недоступен: %s",
                region.name,
                las_path.name,
                exc,
            )
            continue
        if not budget.can_afford(las_estimate_s):
            logger.warning(
                "li %s: пропуск %s (бюджет %.0f < %.0f с)",
                region.name,
                las_path.name,
                budget.remaining(),
                las_estimate_s,
            )
            continue
        if model is None:
            model = _load_model_once()
            if model is None:
                return []
        try:
            detections.extend(
                _process_las(region, las_path, model, budget, cfg, las_estimate_s),
            )
        except Exception:  # noqa: BLE001  # один LAS не роняет регион
            logger.exception(
                "li %s: ошибка %s, следующий LAS",
                region.name,
                las_path.name,
            )
    return detections


def _las_seconds(las_path: Path, cfg: ModuleType) -> float:
    """Оценка времени обработки одного LAS, с: размер в ГБ * seconds_per_gb."""
    size_gb = las_path.stat().st_size / _BYTES_PER_GB
    return size_gb * float(cfg.LIDAR["seconds_per_gb"])


def _load_model_once() -> yolo_runtime.LoadedModel | None:
    """Загрузить модель один раз на прогон (кэш на уровне модуля).

    При неудаче возвращает None и запоминает причину: повторные вызовы в
    этом прогоне не пытаются загрузить модель снова.
    """
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error:
        logger.warning("li: модель уже не загрузилась ранее (%s)", _model_error)
        return None
    try:
        _model = yolo_runtime.load_model()
    except Exception as exc:  # noqa: BLE001  # без модели детектор пропускается
        _model_error = str(exc)
        logger.exception("li: модель не загрузилась, детектор пропущен")
        return None
    return _model


def _process_las(
    region: Region,
    las_path: Path,
    model: yolo_runtime.LoadedModel,
    budget: Budget,
    cfg: ModuleType,
    las_estimate_s: float,
) -> list[Detection]:
    """Обработать один LAS: отмывки во временной папке -> инференс -> Detection.

    Временная папка живёт только внутри вызова. При нехватке бюджета между
    отмывками возвращается уже найденное; ошибка одной отмывки не отменяет
    остальные.
    """
    started_s = time.monotonic()
    variants: tuple[str, ...] = tuple(cfg.LIDAR["infer_hillshades"])
    if not variants:
        logger.warning("li %s: infer_hillshades пуст: %s", region.name, las_path.name)
        return []
    per_variant_estimate_s = las_estimate_s / len(variants)
    found: list[Detection] = []
    per_variant: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="koz3_li_") as tmp_dir:
        tif_paths = lidar_dem.hillshades_for_las(
            las_path,
            region.utm_crs,
            Path(tmp_dir),
        )
        by_variant = _match_hillshades(tif_paths, cfg)
        for variant in variants:
            tif_path = by_variant.get(variant)
            if tif_path is None:
                logger.warning(
                    "li %s: отмывка %s не построена (%s)",
                    region.name,
                    variant,
                    las_path.name,
                )
                continue
            if not budget.can_afford(per_variant_estimate_s):
                logger.warning(
                    "li %s: бюджет %.0f с не покрывает отмывку %s (%s), "
                    "возвращаю найденное (%d)",
                    region.name,
                    budget.remaining(),
                    variant,
                    las_path.name,
                    len(found),
                )
                break
            try:
                new_detections = _predict_hillshade(
                    region, las_path, tif_path, variant, model, cfg,
                )
            except Exception:  # noqa: BLE001  # одна отмывка не роняет весь LAS
                logger.exception(
                    "li %s: ошибка отмывки %s (%s)",
                    region.name,
                    variant,
                    las_path.name,
                )
                continue
            found.extend(new_detections)
            per_variant[variant] = len(new_detections)
    summary = ", ".join(f"{name}={count}" for name, count in per_variant.items())
    logger.info(
        "li %s: %s — %d детекций за %.1f с; по отмывкам: %s",
        region.name,
        las_path.name,
        len(found),
        time.monotonic() - started_s,
        summary or "нет",
    )
    return found


def _match_hillshades(tif_paths: list[Path], cfg: ModuleType) -> dict[str, Path]:
    """Сопоставить tif-файлы отмывок вариантам: порядок = cfg.LIDAR["hillshades"]."""
    variants: tuple[str, ...] = tuple(cfg.LIDAR["hillshades"])
    if len(tif_paths) != len(variants):
        logger.warning(
            "li: hillshades_for_las вернула %d файлов, ожидалось %d",
            len(tif_paths),
            len(variants),
        )
    matched = dict(zip(variants, tif_paths, strict=False))
    logger.debug("li: отмывки по вариантам: %s", matched)
    return matched


@dataclass
class _Tile:
    """Тайл для инференса и его положение в растре отмывки."""

    image: np.ndarray  # uint8 (tile_px, tile_px, 3), край дополнен нулями
    row0: int  # верхняя строка тайла в растре, px
    col0: int  # левый столбец тайла в растре, px
    height: int  # реальная высота фрагмента без дополнения, px
    width: int  # реальная ширина фрагмента без дополнения, px


def _make_tiles(image: np.ndarray, tile_px: int, overlap: float) -> list[_Tile]:
    """Нарезать отмывку (H, W, 3) на тайлы с перекрытием.

    Положения — yolo_runtime.tile_positions (последний тайл прижат к краю);
    короткий край дополняется нулями до tile_px x tile_px.
    """
    height, width = image.shape[:2]
    rows = yolo_runtime.tile_positions(height, tile_px, overlap)
    cols = yolo_runtime.tile_positions(width, tile_px, overlap)
    tiles: list[_Tile] = []
    for row0 in rows:
        for col0 in cols:
            fragment = image[row0 : row0 + tile_px, col0 : col0 + tile_px]
            tiles.append(
                _Tile(
                    image=_pad_to_tile(fragment, tile_px),
                    row0=row0,
                    col0=col0,
                    height=int(fragment.shape[0]),
                    width=int(fragment.shape[1]),
                )
            )
    return tiles


def _pad_to_tile(fragment: np.ndarray, tile_px: int) -> np.ndarray:
    """Дополнить фрагмент нулями до полного тайла tile_px x tile_px."""
    if fragment.shape[0] == tile_px and fragment.shape[1] == tile_px:
        return fragment
    padded = np.zeros((tile_px, tile_px, fragment.shape[2]), dtype=fragment.dtype)
    padded[: fragment.shape[0], : fragment.shape[1]] = fragment
    return padded


def _predict_hillshade(
    region: Region,
    las_path: Path,
    tif_path: Path,
    variant: str,
    model: yolo_runtime.LoadedModel,
    cfg: ModuleType,
) -> list[Detection]:
    """Предсказать объекты на одной отмывке: растр -> тайлы -> маски -> UTM.

    Маски обрезаются до реального края тайла (зона дополнения нулями
    отбрасывается), полигоны переводятся из CRS растра в UTM региона.
    """
    started_s = time.monotonic()
    with rasterio.open(tif_path) as src:
        raster_transform = src.transform
        raster_crs = src.crs
        raster = src.read()
    if raster_crs is None:
        logger.warning(
            "li %s: %s без CRS, отмывка пропущена",
            region.name,
            tif_path.name,
        )
        return []
    image = yolo_runtime.to_model_image(raster)
    tiles = _make_tiles(image, int(cfg.LIDAR["tile_px"]), float(cfg.LIDAR["overlap"]))
    detections: list[Detection] = []
    predicted = yolo_runtime.predict_tiles(model, [tile.image for tile in tiles])
    for tile, instances in zip(tiles, predicted, strict=True):
        tile_transform = raster_transform * Affine.translation(tile.col0, tile.row0)
        for inst in instances:
            mask = inst.mask[: tile.height, : tile.width]
            if not mask.any():
                continue
            for geom in geometry.mask_to_polygons(mask, tile_transform):
                detections.append(
                    Detection(
                        geom=_to_region_crs(geom, raster_crs, region),
                        class_name=inst.class_name,
                        confidence=inst.confidence,
                        markup_type=MARKUP_TYPE,
                        region_name=region.name,
                        source=f"{las_path.name}:{variant}",
                    )
                )
    logger.info(
        "li %s: %s, отмывка %s — %d тайлов, %d детекций за %.1f с",
        region.name,
        las_path.name,
        variant,
        len(tiles),
        len(detections),
        time.monotonic() - started_s,
    )
    return detections


def _to_region_crs(geom: Polygon, raster_crs: RasterCRS, region: Region) -> Polygon:
    """Перевести полигон из CRS растра отмывки в UTM региона (при различии)."""
    raster_proj = ProjCRS.from_user_input(raster_crs.to_string())
    region_proj = ProjCRS.from_user_input(region.utm_crs)
    if raster_proj == region_proj:
        return geom
    return geometry.reproject(geom, raster_crs.to_string(), region.utm_crs)
