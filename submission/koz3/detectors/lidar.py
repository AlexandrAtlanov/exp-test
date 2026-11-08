"""Детектор лидар: YOLO-seg на отмывках рельефа из LAS -> полигоны в UTM.

Для каждого LAS строит отмывки бейзлайна (lidar_dem) во временной папке и
подаёт их модели одним из двух способов (LIDAR["infer_mode"]):
- "whole" — как бейзлайн: вся отмывка сжата до LIDAR["whole_px"] и выровнена,
  один кадр на отмывку (lidar_whole); так обучен model.pt бейзлайна;
- "tiles" — нарезка на тайлы с перекрытием (как training/build_dataset.py);
  дальше описан этот режим.
Маски переводятся в полигоны в UTM региона. Список LAS берётся только из
region.files["li"] (discovery) — один источник для is_available,
estimate_seconds и run: рассинхрон паттернов поиска исключён (В1).
Тайлы с долей валидных пикселей
ниже LIDAR["min_valid_share"] (вне охвата данных) пропускаются — тот же
фильтр, что при сборке датасета. Инференс идёт чанками по batch с проверкой
бюджета перед каждым чанком (оценка — по фактической скорости): при
исчерпании остаток отмывки пропускается, найденное возвращается. Ошибка
одного чанка (OOM, несовпадение числа результатов у engine со статическим
batch) не роняет отмывку: чанк пропускается, найденное копится. Дубли от
перекрытия тайлов и от нескольких отмывок здесь не подавляются — это делает
postprocess по парам (class_name, markup_type). Маски, обрезанные внутренним
краем тайла, помечаются extra["edge_cut"] = True: NMS предпочитает им целые
копии того же объекта из соседнего тайла. Известное ограничение (склейка
отложена, не к первой отправке): объект крупнее тайла обрезан во всех
своих тайлах — NMS может не собрать его из фрагментов; краткосрочно
tile_px выбирается с запасом против максимального объекта крупных классов
(поселения, фортификации).
"""

import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import rasterio
from pyproj import CRS as ProjCRS
from rasterio.crs import CRS as RasterCRS
from rasterio.transform import Affine
from shapely.geometry import Polygon

from .. import config, geometry, yolo_runtime
from ..datatypes import Detection, Region
from ..timing import Budget
from . import lidar_dem, lidar_whole

logger = logging.getLogger(__name__)

PRIORITY: int = 1
"""Порядок запуска детекторов: меньше — раньше (раздел 5 пакета)."""

MARKUP_TYPE: str = "li"
"""Тип данных детектора для свойства markup_type."""

_GROUND_SUFFIX = re.compile(r"[_\- ]ground$", re.IGNORECASE)
"""Суффикс отфильтрованного облака: <имя>_ground.las — ground-версия <имя>.las."""

_BYTES_PER_GB: float = 1e9
"""ГБ = 1e9 байт; в этих единицах калибруется seconds_per_gb."""

_model: yolo_runtime.LoadedModel | None = None
"""Кэш модели на прогон: загрузка один раз (раздел 5 пакета)."""

_model_error: str = ""
"""Причина неудачной загрузки модели: повторы в этом прогоне не делаются."""


def is_available(region: Region) -> bool:
    """Детектор применим: есть LAS в регионе и веса (.pt или .engine).

    Единственный источник списка LAS — region.files["li"] (discovery);
    его же читают estimate_seconds и run (В1), поэтому is_available=True
    при пустом run исключён самой структурой, без сверки паттернов.
    """
    if not _las_files(region):
        return False
    weights_dir = config.WEIGHTS_DIR
    has_pt = (weights_dir / str(config.YOLO["weights_pt"])).is_file()
    has_engine = (weights_dir / str(config.YOLO["weights_engine"])).is_file()
    return has_pt or has_engine


def estimate_seconds(region: Region) -> float:
    """Оценка времени на регион, с: суммарный размер LAS * seconds_per_gb.

    LAS — из region.files["li"] (тот же список, что в is_available и run,
    В1): оценка и запуск видят одни и те же файлы.
    """
    total_bytes = 0
    for las_path in _las_files(region):
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
    найденные детекции возвращаются всегда. Список LAS — region.files["li"],
    как в is_available/estimate_seconds (единственный источник, В1).
    """
    las_files = _las_files(region)
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


def _las_files(region: Region) -> list[Path]:
    """LAS региона из region.files["li"] без исходников, у которых рядом есть ground-версия.

    X.las убирается, если в той же папке есть X_ground.las (или X-ground, X ground): DEM
    иначе строится дважды, а детекции дублируются. В отличие от бейзлайна (один LAS на
    папку) несколько разных LAS в одной папке сохраняются — данные не теряются.
    """
    files = list(region.files.get("li", []))
    ground_twins = {
        (path.parent, _GROUND_SUFFIX.sub("", path.stem).lower())
        for path in files
        if _GROUND_SUFFIX.search(path.stem)
    }
    return [
        path
        for path in files
        if _GROUND_SUFFIX.search(path.stem) or (path.parent, path.stem.lower()) not in ground_twins
    ]


def _las_seconds(las_path: Path, cfg: ModuleType) -> float:
    """Оценка времени обработки одного LAS, с: размер в ГБ * seconds_per_gb."""
    size_gb = las_path.stat().st_size / _BYTES_PER_GB
    return size_gb * float(cfg.LIDAR["seconds_per_gb"])


def _load_model_once() -> yolo_runtime.LoadedModel | None:
    """Загрузить модель один раз на прогон (кэш на уровне модуля).

    При неудаче возвращает None и запоминает причину: повторные вызовы в
    этом прогоне не пытаются загрузить модель снова. Время загрузки (10–30 с
    вне оценки seconds_per_gb, В5) пишется в лог — при калибровке оно
    закладывается в запас TIME_LIMIT_S, а не в seconds_per_gb.
    """
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error:
        logger.warning("li: модель уже не загрузилась ранее (%s)", _model_error)
        return None
    started_s = time.monotonic()
    try:
        _model = yolo_runtime.load_model()
    except Exception as exc:  # noqa: BLE001  # без модели детектор пропускается
        _model_error = str(exc)
        logger.exception("li: модель не загрузилась, детектор пропущен")
        return None
    logger.info(
        "li: модель загружена (%s) за %.1f с — вне оценки seconds_per_gb",
        _model.kind,
        time.monotonic() - started_s,
    )
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
            variants=variants,  # только нужные инференсу отмывки
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
                    "li %s: бюджет %.0f с не покрывает отмывку %s (%s), возвращаю найденное (%d)",
                    region.name,
                    budget.remaining(),
                    variant,
                    las_path.name,
                    len(found),
                )
                break
            try:
                new_detections = _predict_hillshade(
                    region,
                    las_path,
                    tif_path,
                    variant,
                    model,
                    budget,
                    cfg,
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
    """Сопоставить tif-файлы отмывок вариантам по суффиксу имени файла.

    Симметрично обучению (training/hillshades.tag_of): вариант — суффикс
    имени после последнего «_» (…_c.tif → «c»). Позиционное сопоставление
    dict(zip) молча путало варианты при другом порядке возврата lidar_dem;
    по тегу порядок не важен. Неизвестный тег и повтор варианта —
    предупреждение в лог, отсутствующие варианты логирует _process_las.
    """
    variants: tuple[str, ...] = tuple(cfg.LIDAR["hillshades"])
    matched: dict[str, Path] = {}
    for path in tif_paths:
        tag = path.stem.rsplit("_", 1)[-1].lower()
        if tag not in variants:
            logger.warning(
                "li: суффикс %r файла %s не является вариантом отмывки %s",
                tag,
                path.name,
                list(variants),
            )
            continue
        if tag in matched:
            logger.warning(
                "li: вариант %s повторяется (%s), оставлен первый: %s",
                tag,
                path.name,
                matched[tag].name,
            )
            continue
        matched[tag] = path
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


@dataclass(frozen=True)
class _Raster:
    """Растр отмывки: привязка, CRS и размер, общие для всех его тайлов."""

    transform: Affine
    crs: RasterCRS
    height: int
    width: int


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


def _valid_tiles(
    tiles: list[_Tile], valid: np.ndarray, min_share: float
) -> tuple[list[_Tile], int]:
    """Тайлы с долей валидных пикселей не ниже min_share; (оставленные, пропущенные).

    Тот же фильтр, что при сборке датасета (build_dataset): тайлы целиком
    вне охвата данных не подаются модели, частично валидные — обрабатываются.
    """
    kept: list[_Tile] = []
    for tile in tiles:
        rows = valid[tile.row0 : tile.row0 + tile.height]
        window = rows[:, tile.col0 : tile.col0 + tile.width]
        if float(window.mean()) >= min_share:
            kept.append(tile)
    return kept, len(tiles) - len(kept)


def _predict_hillshade(
    region: Region,
    las_path: Path,
    tif_path: Path,
    variant: str,
    model: yolo_runtime.LoadedModel,
    budget: Budget,
    cfg: ModuleType,
) -> list[Detection]:
    """Предсказать объекты на одной отмывке: растр -> тайлы -> маски -> UTM.

    Тайлы с долей валидных пикселей ниже LIDAR["min_valid_share"] (вне охвата
    данных) пропускаются — тот же фильтр, что при сборке датасета; частично
    валидные тайлы обрабатываются. При исчерпании бюджета внутри отмывки
    (см. _predict_chunks) найденное возвращается. Полигоны переводятся из
    CRS растра в UTM региона.
    """
    started_s = time.monotonic()
    with rasterio.open(tif_path) as src:
        # CRS проверяется до чтения растра: битые файлы не тянут весь массив
        if src.crs is None:
            logger.warning(
                "li %s: %s без CRS, отмывка пропущена",
                region.name,
                tif_path.name,
            )
            return []
        raster = _Raster(transform=src.transform, crs=src.crs, height=src.height, width=src.width)
        bands = src.read()
    image = yolo_runtime.to_model_image(bands)
    if str(cfg.LIDAR.get("infer_mode", "tiles")) == "whole":
        return _predict_whole(image, raster, region, las_path, variant, model, cfg)
    detections: list[Detection] = []
    for factor in model.scales or cfg.LIDAR.get("scales", (1,)):
        scaled, scaled_raster = _downscale(image, raster, float(factor))
        found = _predict_tiles_at_scale(
            scaled, scaled_raster, region, las_path, variant, model, budget, cfg
        )
        logger.info(
            "li %s: %s, отмывка %s, масштаб ×%g (%dx%d) — детекций %d",
            region.name,
            las_path.name,
            variant,
            float(factor),
            scaled_raster.width,
            scaled_raster.height,
            len(found),
        )
        detections.extend(found)
    logger.info(
        "li %s: %s, отмывка %s — детекций %d за %.1f с",
        region.name,
        las_path.name,
        variant,
        len(detections),
        time.monotonic() - started_s,
    )
    return detections


def _downscale(image: np.ndarray, raster: _Raster, factor: float) -> tuple[np.ndarray, _Raster]:
    """Сжать отмывку в factor раз (INTER_AREA) и пересчитать привязку; 1 — без изменений."""
    if factor <= 1.0:
        return image, raster
    width = max(1, int(round(raster.width / factor)))
    height = max(1, int(round(raster.height / factor)))
    scaled = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    transform = raster.transform * Affine.scale(raster.width / width, raster.height / height)
    return np.ascontiguousarray(scaled), _Raster(transform, raster.crs, height, width)


def _predict_tiles_at_scale(
    image: np.ndarray,
    raster: _Raster,
    region: Region,
    las_path: Path,
    variant: str,
    model: yolo_runtime.LoadedModel,
    budget: Budget,
    cfg: ModuleType,
) -> list[Detection]:
    """Тайлы одной (возможно сжатой) отмывки → детекции в UTM региона."""
    tile_px = int(model.tile_px or cfg.LIDAR["tile_px"])
    tiles = _make_tiles(image, tile_px, float(cfg.LIDAR["overlap"]))
    tiles, nodata_tiles = _valid_tiles(
        tiles,
        yolo_runtime.valid_pixel_mask(image),
        float(cfg.LIDAR["min_valid_share"]),
    )
    if not tiles:
        logger.warning(
            "li %s: %s, отмывка %s — все тайлы вне данных, инференс пропущен",
            region.name,
            las_path.name,
            variant,
        )
        return []
    if nodata_tiles:
        logger.info(
            "li %s: %s, отмывка %s — тайлов %d, вне данных пропущено %d",
            region.name,
            las_path.name,
            variant,
            len(tiles),
            nodata_tiles,
        )
    return _predict_chunks(tiles, raster, region, las_path, variant, model, budget)


def _predict_whole(
    image: np.ndarray,
    raster: _Raster,
    region: Region,
    las_path: Path,
    variant: str,
    model: yolo_runtime.LoadedModel,
    cfg: ModuleType,
) -> list[Detection]:
    """Режим бейзлайна: вся отмывка → сжатие до whole_px → выравнивание → один кадр.

    Маски кадра переводятся в CRS отмывки одним аффинным преобразованием
    (lidar_whole.frame_to_raster), затем в UTM региона. Отмывка без валидных
    пикселей модели не подаётся.
    """
    started_s = time.monotonic()
    valid_share = float(yolo_runtime.valid_pixel_mask(image).mean())
    if valid_share < float(cfg.LIDAR["min_valid_share"]):
        logger.warning(
            "li %s: %s, отмывка %s — нет данных (%.3f), инференс пропущен",
            region.name,
            las_path.name,
            variant,
            valid_share,
        )
        return []
    whole_px = int(cfg.LIDAR["whole_px"])
    frame, scale = lidar_whole.downscale_image(image, whole_px)
    angle, crop_offset = 0.0, (0, 0)
    center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
    if bool(cfg.LIDAR["whole_align"]):
        frame, angle, crop_offset, center = lidar_whole.align_and_crop_image(frame)
    frame = np.ascontiguousarray(frame)
    pixel_to_raster = lidar_whole.frame_to_raster(
        raster.transform, scale, angle, crop_offset, center
    )
    [instances] = list(yolo_runtime.predict_tiles(model, [frame], imgsz=whole_px))
    detections: list[Detection] = []
    for inst in instances:
        for geom in geometry.mask_to_polygons(inst.mask, pixel_to_raster):
            detections.append(
                Detection(
                    geom=_to_region_crs(geom, raster.crs, region),
                    class_name=inst.class_name,
                    confidence=inst.confidence,
                    markup_type=MARKUP_TYPE,
                    region_name=region.name,
                    source=f"{las_path.name}:{variant}",
                    extra={"edge_cut": False},
                )
            )
    logger.info(
        "li %s: %s, отмывка %s — целиком %dx%d → кадр %dx%d (масштаб %.3f, поворот %.1f°), "
        "детекций %d за %.1f с",
        region.name,
        las_path.name,
        variant,
        raster.width,
        raster.height,
        frame.shape[1],
        frame.shape[0],
        scale,
        angle,
        len(detections),
        time.monotonic() - started_s,
    )
    return detections


def _predict_chunks(
    tiles: list[_Tile],
    raster: _Raster,
    region: Region,
    las_path: Path,
    variant: str,
    model: yolo_runtime.LoadedModel,
    budget: Budget,
) -> list[Detection]:
    """Инференс по тайлам чанками по batch; перед каждым чанком — проверка бюджета.

    Оценка следующего чанка — по фактической скорости уже обработанных
    тайлов. Первый чанк запускается без проверки: он прикрыт оценкой отмывки
    в _process_las, а перерасход при заниженной оценке ограничен одним
    чанком. При исчерпании бюджета остаток отмывки пропускается, найденное
    возвращается. Ошибка одного чанка (OOM, несовпадение числа результатов
    у engine со статическим batch) не роняет отмывку: чанк пропускается с
    logger.exception, найденное копится; время неудачной попытки входит в
    оценку, её тайлы — нет (оценка консервативнее). Порядок и состав
    батчей совпадают со сплошным проходом, поэтому детекции полного
    прохода не меняются.
    """
    detections: list[Detection] = []
    batch = max(1, int(model.batch))
    done_tiles = 0
    spent_s = 0.0
    for start in range(0, len(tiles), batch):
        chunk = tiles[start : start + batch]
        if done_tiles:
            chunk_estimate_s = spent_s / done_tiles * len(chunk)
            if not budget.can_afford(chunk_estimate_s):
                logger.warning(
                    "li %s: %s, отмывка %s — бюджет %.0f с не покрывает чанк "
                    "(оценка %.0f с): остаток пропущен, не обработано тайлов %d из %d",
                    region.name,
                    las_path.name,
                    variant,
                    budget.remaining(),
                    chunk_estimate_s,
                    len(tiles) - done_tiles,
                    len(tiles),
                )
                break
        chunk_started_s = time.monotonic()
        try:
            predicted = yolo_runtime.predict_tiles(model, [tile.image for tile in chunk])
            for tile, instances in zip(chunk, predicted, strict=True):
                detections.extend(
                    _tile_detections(tile, instances, raster, region, las_path.name, variant)
                )
            done_tiles += len(chunk)
        except Exception:  # noqa: BLE001  # чанк не роняет отмывку: найденное копится
            logger.exception(
                "li %s: %s, отмывка %s — ошибка чанка тайлов %d–%d из %d, "
                "чанк пропущен, найденное сохраняется (%d)",
                region.name,
                las_path.name,
                variant,
                start,
                start + len(chunk) - 1,
                len(tiles),
                len(detections),
            )
        spent_s += time.monotonic() - chunk_started_s
    return detections


def _tile_detections(
    tile: _Tile,
    instances: list[yolo_runtime.TileInstance],
    raster: _Raster,
    region: Region,
    las_name: str,
    variant: str,
) -> list[Detection]:
    """Детекции одного тайла: маски -> полигоны в UTM региона.

    Маски обрезаются до реального края тайла (зона дополнения нулями
    отбрасывается); маски, обрезанные внутренним краем тайла, помечаются
    extra["edge_cut"] = True.
    """
    tile_transform = raster.transform * Affine.translation(tile.col0, tile.row0)
    detections: list[Detection] = []
    for inst in instances:
        mask = inst.mask[: tile.height, : tile.width]
        if not mask.any():
            continue
        edge_cut = _touches_inner_edge(mask, tile, raster.height, raster.width)
        for geom in geometry.mask_to_polygons(mask, tile_transform):
            detections.append(
                Detection(
                    geom=_to_region_crs(geom, raster.crs, region),
                    class_name=inst.class_name,
                    confidence=inst.confidence,
                    markup_type=MARKUP_TYPE,
                    region_name=region.name,
                    source=f"{las_name}:{variant}",
                    extra={"edge_cut": edge_cut},
                )
            )
    return detections


def _touches_inner_edge(mask: np.ndarray, tile: _Tile, height: int, width: int) -> bool:
    """Маска касается края тайла, который не совпадает с краем растра (объект обрезан).

    mask уже обрезана до реального фрагмента (tile.height x tile.width);
    height, width — размер всего растра отмывки, px.
    """
    return bool(
        (tile.row0 > 0 and mask[0].any())
        or (tile.col0 > 0 and mask[:, 0].any())
        or (tile.row0 + tile.height < height and mask[-1].any())
        or (tile.col0 + tile.width < width and mask[:, -1].any())
    )


def _to_region_crs(geom: Polygon, raster_crs: RasterCRS, region: Region) -> Polygon:
    """Перевести полигон из CRS растра отмывки в UTM региона (при различии)."""
    raster_proj = ProjCRS.from_user_input(raster_crs.to_string())
    region_proj = ProjCRS.from_user_input(region.utm_crs)
    if raster_proj == region_proj:
        return geom
    return geometry.reproject(geom, raster_crs.to_string(), region.utm_crs)
