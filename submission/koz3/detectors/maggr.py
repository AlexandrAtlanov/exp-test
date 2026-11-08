"""Детектор магниторазведки (MagGr): находки-точки и аномальные полигоны.

Признаки, кандидаты и NMS — общий код обучения и инференса
``koz3.geophys.maggr_features``; детектор грузит две модели из
``weights/maggr_models.joblib`` (контракт 13b), считает вероятности
и переводит геометрию из CRS серой карты в UTM региона: точки →
полигоны-круги ``finds_points``, полигоны → ``object_poly``.
Без joblib/sklearn в образе или при любой ошибке загрузки детектор
выключается с предупреждением в лог — молча никогда.
"""

import logging
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from .. import config
from ..datatypes import Detection, Region
from ..geometry import point_to_circle, reproject
from ..geophys.maggr_features import (
    POINT_FEATURE_NAMES,
    POLY_FEATURE_NAMES,
    MaggrField,
    gray_raster_paths,
    load_field,
    nms_points,
    nms_polys,
    point_candidates,
    poly_candidates,
)
from ..timing import Budget

logger = logging.getLogger(__name__)

PRIORITY: int = 2
"""Порядок запуска детектора: после лидара (PRIORITY = 1)."""

_MARKUP_TYPE = "maggr"
"""Тип разметки всех детекций детектора (инвариант 4.5)."""

_REQUIRED_KEYS: tuple[str, ...] = (
    "points",
    "polys",
    "thr_points",
    "thr_polys",
    "feature_names",
    "sklearn",
)
"""Обязательные ключи weights/maggr_models.joblib (контракт 13b)."""


@dataclass
class _MaggrModels:
    """Модели MagGr и пороги из файла весов."""

    points: Any
    polys: Any
    thr_points: float
    thr_polys: float


_MODELS_CACHE: dict[Path, _MaggrModels | None] = {}
"""Кеш моделей по пути файла весов: загрузка один раз на прогон."""


def is_available(region: Region) -> bool:
    """True, если в регионе есть серая карта MagGr и модели грузятся."""
    if not gray_raster_paths(region.files.get("maggr", []), config.MAGGR):
        return False
    return _load_models(config) is not None


def estimate_seconds(region: Region) -> float:
    """Линейная оценка времени: мегапиксели рабочих сеток × seconds_per_mpx."""
    res_m = float(config.MAGGR["res_m"])
    gray_paths = gray_raster_paths(region.files.get("maggr", []), config.MAGGR)
    total_mpx = sum(_raster_mpx(path, res_m) for path in gray_paths)
    return total_mpx * float(config.MAGGR["seconds_per_mpx"])


def run(region: Region, budget: Budget, cfg: ModuleType) -> list[Detection]:
    """Детекции региона: круги finds_points и полигоны object_poly в UTM.

    Модели грузятся один раз (кеш по пути весов); бюджет проверяется
    перед каждым файлом; ошибка файла или ветки не роняет прогон.
    """
    models = _load_models(cfg)
    if models is None:
        logger.warning("maggr: регион %s без моделей — детектор пропущен", region.name)
        return []
    res_m = float(cfg.MAGGR["res_m"])
    per_mpx = float(cfg.MAGGR["seconds_per_mpx"])
    detections: list[Detection] = []
    for gray_path in gray_raster_paths(region.files.get("maggr", []), cfg.MAGGR):
        est_s = _raster_mpx(gray_path, res_m) * per_mpx
        if not budget.can_afford(est_s):
            logger.warning(
                "maggr: %s пропущен — не хватает бюджета (оценка %.1f с)", gray_path.name, est_s
            )
            continue
        detections.extend(_run_file(gray_path, region, models, cfg))
    return detections


def _run_file(
    gray_path: Path, region: Region, models: _MaggrModels, cfg: ModuleType
) -> list[Detection]:
    """Детекции одной серой карты; ошибки файла и веток изолированы."""
    try:
        field = load_field(gray_path, cfg.MAGGR)
    except Exception as exc:
        logger.warning("maggr: %s пропущен (%s)", gray_path.name, exc)
        return []
    detections: list[Detection] = []
    for branch in (_point_detections, _poly_detections):
        try:
            detections.extend(branch(field, gray_path.name, region, models, cfg))
        except Exception as exc:
            logger.warning(
                "maggr: %s: ветка %s пропущена (%s)", gray_path.name, branch.__name__, exc
            )
    return detections


def _point_detections(
    field: MaggrField, gray_name: str, region: Region, models: _MaggrModels, cfg: ModuleType
) -> list[Detection]:
    """finds_points: кандидаты → вероятности → NMS → круг в UTM региона."""
    xy, features = point_candidates(field, cfg.MAGGR)
    if len(xy) == 0:
        return []
    scores = _predict(models.points, features)
    keep = nms_points(xy, scores, models.thr_points, cfg.MAGGR)
    same_crs = field.crs == CRS.from_user_input(region.utm_crs)
    detections: list[Detection] = []
    for i in keep:
        point = Point(float(xy[i, 0]), float(xy[i, 1]))
        if not same_crs:
            point = reproject(point, field.crs, region.utm_crs)
        detections.append(
            Detection(
                geom=point_to_circle(point.x, point.y),
                class_name="finds_points",
                confidence=float(scores[i]),
                markup_type=_MARKUP_TYPE,
                region_name=region.name,
                source=gray_name,
            )
        )
    return detections


def _poly_detections(
    field: MaggrField, gray_name: str, region: Region, models: _MaggrModels, cfg: ModuleType
) -> list[Detection]:
    """object_poly: кандидаты → вероятности → NMS → полигон в UTM региона."""
    polys, features = poly_candidates(field, cfg.MAGGR)
    if len(polys) == 0:
        return []
    scores = _predict(models.polys, features)
    keep = nms_polys(polys, scores, models.thr_polys, cfg.MAGGR)
    same_crs = field.crs == CRS.from_user_input(region.utm_crs)
    detections: list[Detection] = []
    for i in keep:
        geom = polys[i] if same_crs else reproject(polys[i], field.crs, region.utm_crs)
        poly = _largest_polygon(geom)
        if poly is None:
            logger.warning("maggr: %s: кандидат без полигона пропущен", gray_name)
            continue
        detections.append(
            Detection(
                geom=poly,
                class_name="object_poly",
                confidence=float(scores[i]),
                markup_type=_MARKUP_TYPE,
                region_name=region.name,
                source=gray_name,
            )
        )
    return detections


def _load_models(cfg: ModuleType) -> _MaggrModels | None:
    """Модели из кеша или из файла весов; None — детектор выключен."""
    weights_path = Path(cfg.WEIGHTS_DIR) / str(cfg.MAGGR["models_file"])
    if weights_path in _MODELS_CACHE:
        return _MODELS_CACHE[weights_path]
    models: _MaggrModels | None = None
    if weights_path.is_file():
        try:
            models = _read_models(weights_path)
        except Exception as exc:
            logger.warning("maggr: детектор выключен — %s не загружен (%s)", weights_path, exc)
    else:
        logger.debug("maggr: нет файла весов %s — детектор выключен", weights_path)
    _MODELS_CACHE[weights_path] = models
    return models


def _read_models(weights_path: Path) -> _MaggrModels:
    """Читает weights/maggr_models.joblib и проверяет контракт 13b."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("нет joblib/sklearn в образе") from exc
    raw = joblib.load(weights_path)
    if not isinstance(raw, dict):
        raise TypeError(f"в файле {type(raw).__name__}, ожидался dict")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError("нет ключей: " + ", ".join(missing))
    _check_feature_names(raw["feature_names"])
    for name in ("points", "polys"):
        if not callable(getattr(raw[name], "predict_proba", None)):
            raise TypeError(f"модель {name!r} без predict_proba")
    _warn_sklearn_mismatch(raw["sklearn"])
    return _MaggrModels(
        points=raw["points"],
        polys=raw["polys"],
        thr_points=_as_threshold(raw["thr_points"], "thr_points"),
        thr_polys=_as_threshold(raw["thr_polys"], "thr_polys"),
    )


def _check_feature_names(names: Any) -> None:
    """Имена признаков из файла весов должны совпадать с maggr_features."""
    if not isinstance(names, dict):
        raise TypeError("feature_names должен быть dict")
    pairs = (("points", POINT_FEATURE_NAMES), ("polys", POLY_FEATURE_NAMES))
    for kind, expected in pairs:
        actual = list(names.get(kind) or [])
        if actual != list(expected):
            raise ValueError(
                f"feature_names[{kind!r}] не совпадает с maggr_features: "
                f"{len(actual)} имён вместо {len(expected)} — модель на других признаках"
            )


def _as_threshold(value: Any, name: str) -> float:
    """Порог из файла весов: число в [0, 1], иначе исключение."""
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name}={threshold} вне диапазона [0, 1]")
    return threshold


def _warn_sklearn_mismatch(trained_version: Any) -> None:
    """Предупреждение о несовпадении версии sklearn (не отказ)."""
    try:
        installed = metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        return
    if str(trained_version) != installed:
        logger.warning(
            "maggr: sklearn %s при обучении, в образе %s — возможны расхождения",
            trained_version,
            installed,
        )


def _raster_mpx(path: Path, res_m: float) -> float:
    """Мегапиксели рабочей сетки res_m; 0.0, если растр не открывается."""
    try:
        with rasterio.open(path) as src:
            height = int(round(src.height * src.res[1] / res_m))
            width = int(round(src.width * src.res[0] / res_m))
    except Exception as exc:
        logger.warning("maggr: %s не открывается для оценки времени (%s)", path.name, exc)
        return 0.0
    return height * width / 1e6


def _predict(model: Any, features: np.ndarray) -> np.ndarray:
    """Вероятности положительного класса (N,); NaN/Inf -> -1 (ниже порога)."""
    proba = np.asarray(model.predict_proba(features), dtype=np.float64)
    if proba.ndim != 2 or proba.shape != (len(features), 2):
        raise ValueError(
            f"predict_proba вернул форму {proba.shape}, ожидалась ({len(features)}, 2)"
        )
    positive = proba[:, 1]
    return np.where(np.isfinite(positive), np.clip(positive, 0.0, 1.0), -1.0)


def _largest_polygon(geom: BaseGeometry) -> Polygon | None:
    """Одиночный Polygon; MultiPolygon -> часть с максимальной площадью."""
    if geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda part: part.area)
    return None