"""Детектор электроразведки (EtGr): аномальные полигоны на картах-срезах.

Конвейер карты — общий код обучения и инференса
``koz3.geophys.etgr_features``: обращение шкалы jet, кандидаты-изолинии,
окна CNN; детектор грузит ансамбль CNN из ``weights/etgr_maps.pt``
(контракт training/train_etgr.py), скорит окна усреднением по
отражениям (внутри predict_cnn) и сетям ансамбля, режет слившиеся
пятна (predict_polys) и переводит полигоны из CRS растра в UTM региона:
``object_poly``, markup_type «etgr». Профили не обрабатываются
(``map_raster_paths`` их отфильтровывает). Без библиотек или при ошибке
загрузки детектор выключается с предупреждением в лог — молча никогда.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .. import config
from ..datatypes import Detection, Region
from ..geometry import reproject
from ..timing import Budget

try:
    from ..geophys import etgr_features as ef
except ImportError as exc:  # skimage в образе платформы не подтверждён (env.md)
    ef = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)

PRIORITY: int = 3
"""Порядок запуска детектора: после лидара (1) и магниторазведки (2)."""

_MARKUP_TYPE = "etgr"
"""Тип разметки всех детекций детектора (инвариант 4.5)."""

_REQUIRED_KEYS: tuple[str, ...] = (
    "arch",
    "state_dicts",
    "thr",
    "crop",
    "feature_names",
    "timm",
)
"""Обязательные ключи weights/etgr_maps.pt (контракт training/train_etgr.py).

Ключи «scorer» (None — оставлен для профилей) и «torch» (версия при
обучении; torch может быть новее образа) детектор карт не использует.
"""


@dataclass
class _EtGrNets:
    """Ансамбль CNN карт EtGr и порог из файла весов."""

    nets: list[Any]  # torch.nn.Module; Any — torch не импортируется на уровне модуля
    thr: float


_NETS_CACHE: dict[Path, _EtGrNets | None] = {}
"""Кеш ансамбля по пути файла весов: загрузка один раз на прогон."""


def is_available(region: Region) -> bool:
    """True, если в регионе есть карты EtGr и ансамбль CNN грузится."""
    if _load_nets(config) is None:
        return False
    assert ef is not None  # недостижимо при ef = None: _load_nets уже вернул None
    return bool(ef.map_raster_paths(region.files.get("etgr", [])))


def estimate_seconds(region: Region) -> float:
    """Оценка времени: карты региона × seconds_per_map (сетка <= work_side px)."""
    if ef is None:
        return 0.0  # детектор выключен — бюджет не резервируем
    maps = ef.map_raster_paths(region.files.get("etgr", []))
    return len(maps) * float(config.ETGR["seconds_per_map"])


def run(region: Region, budget: Budget, cfg: ModuleType) -> list[Detection]:
    """Детекции региона: полигоны object_poly в UTM (markup_type «etgr»).

    Ансамбль грузится один раз (кеш по пути весов); бюджет проверяется
    перед каждой картой; ошибка карты не роняет прогон.
    """
    nets = _load_nets(cfg)
    if nets is None:
        logger.warning("etgr: регион %s без моделей — детектор пропущен", region.name)
        return []
    assert ef is not None  # недостижимо при ef = None: _load_nets уже вернул None
    per_map_s = float(cfg.ETGR["seconds_per_map"])
    detections: list[Detection] = []
    for raster_path in ef.map_raster_paths(region.files.get("etgr", [])):
        if not budget.can_afford(per_map_s):
            logger.warning(
                "etgr: %s пропущен — не хватает бюджета (оценка %.1f с)",
                raster_path.name,
                per_map_s,
            )
            continue
        detections.extend(_run_raster(raster_path, region, nets, cfg))
    return detections


def _run_raster(
    raster_path: Path, region: Region, nets: _EtGrNets, cfg: ModuleType
) -> list[Detection]:
    """Детекции одной карты; ошибки чтения и конвейера изолированы."""
    try:
        field = ef.load_field(raster_path, cfg.ETGR)
    except Exception as exc:
        logger.warning("etgr: %s пропущен (%s)", raster_path.name, exc)
        return []
    try:
        return _map_detections(field, raster_path.name, region, nets, cfg)
    except Exception as exc:
        logger.warning("etgr: %s: карта пропущена (%s)", raster_path.name, exc)
        return []


def _map_detections(
    field: Any, raster_name: str, region: Region, nets: _EtGrNets, cfg: ModuleType
) -> list[Detection]:
    """Конвейер карты: кандидаты → окна → скоры → NMS/разрезание → UTM.

    field — EtGrField; Any, потому что модуль признаков импортируется
    не всегда (см. try в начале файла).
    """
    cand_polys, cand_x = ef.candidates(field, cfg.ETGR)
    if not cand_polys:
        return []
    kids = ef.nested_children(cand_polys, cand_x)
    crops = ef.candidate_crops(field, cand_polys, cfg.ETGR)
    scores = _score(nets.nets, crops)
    out_polys, confs = ef.predict_polys(cand_polys, kids, scores, nets.thr, cfg.ETGR)
    return _to_detections(field, out_polys, confs, raster_name, region)


def _score(nets: list[Any], crops: np.ndarray) -> np.ndarray:
    """Средние вероятности ансамбля (усреднение по сидам; TTA — в predict_cnn)."""
    return np.mean([ef.predict_cnn(net, crops) for net in nets], axis=0)


def _to_detections(
    field: Any,
    polys: Sequence[BaseGeometry],
    confs: Sequence[float],
    raster_name: str,
    region: Region,
) -> list[Detection]:
    """Полигоны карты в детекции: CRS растра → UTM региона → object_poly."""
    same_crs = field.crs == CRS.from_user_input(region.utm_crs)
    detections: list[Detection] = []
    for geom_src, conf in zip(polys, confs, strict=True):
        geom = (
            geom_src
            if same_crs
            else reproject(geom_src, field.crs.to_string(), region.utm_crs)
        )
        poly = _largest_polygon(geom)
        if poly is None:
            logger.warning("etgr: %s: кандидат без полигона пропущен", raster_name)
            continue
        detections.append(
            Detection(
                geom=poly,
                class_name="object_poly",
                confidence=float(conf),
                markup_type=_MARKUP_TYPE,
                region_name=region.name,
                source=raster_name,
            )
        )
    return detections


def _load_nets(cfg: ModuleType) -> _EtGrNets | None:
    """Ансамбль из кеша или из файла весов; None — детектор выключен."""
    weights_path = Path(cfg.WEIGHTS_DIR) / str(cfg.ETGR["models_file"])
    if weights_path in _NETS_CACHE:
        return _NETS_CACHE[weights_path]
    nets: _EtGrNets | None = None
    if ef is None:
        logger.warning(
            "etgr: нет библиотек модуля признаков (%s) — детектор выключен", _IMPORT_ERROR
        )
    elif not weights_path.is_file():
        logger.debug("etgr: нет файла весов %s — детектор выключен", weights_path)
    else:
        try:
            nets = _read_nets(weights_path, cfg)
        except Exception as exc:
            logger.warning("etgr: детектор выключен — %s не загружен (%s)", weights_path, exc)
    _NETS_CACHE[weights_path] = nets
    return nets


def _read_nets(weights_path: Path, cfg: ModuleType) -> _EtGrNets:
    """Читает weights/etgr_maps.pt и проверяет контракт train_etgr.py.

    Формат: dict с ключами arch (имя архитектуры timm), state_dicts
    (непустой список state_dict — ансамбль по сидам), thr (порог OOF в
    [0, 1]), crop (параметры окна), feature_names (имена признаков),
    timm (версия при обучении); scorer и torch картами не используются.
    Файл из безопасных типов — грузится с weights_only=True. Все проверки
    контракта идут до создания сетей: ошибка не тянет за собой timm.
    """
    import torch

    if ef is None:  # недостижимо: _load_nets зовёт _read_nets только при ef не None
        raise RuntimeError("модуль признаков etgr_features не импортирован")
    raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"в файле {type(raw).__name__}, ожидался dict")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError("нет ключей: " + ", ".join(missing))
    arch = raw["arch"]
    if not isinstance(arch, str) or not arch:
        raise TypeError(f"arch={arch!r}, ожидалась непустая строка")
    state_dicts = raw["state_dicts"]
    if not isinstance(state_dicts, list) or not state_dicts:
        raise TypeError("state_dicts должен быть непустым списком")
    if not all(isinstance(sd, dict) for sd in state_dicts):
        raise TypeError("элементы state_dicts должны быть словарями тензоров")
    thr = _as_threshold(raw["thr"], "thr")
    _check_feature_names(raw["feature_names"])
    _check_crop(raw["crop"], cfg)
    _warn_timm_mismatch(raw["timm"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets: list[Any] = []
    for state_dict in state_dicts:
        net = ef.make_cnn(arch, pretrained=False)
        net.load_state_dict(state_dict)
        nets.append(net.to(device).eval())
    return _EtGrNets(nets=nets, thr=thr)


def _check_feature_names(names: Any) -> None:
    """Имена признаков из файла весов должны совпадать с etgr_features.

    Признаки нужны детектору для nested_children (разрезание пятен);
    расхождение означает модель на другой подготовке.
    """
    actual = list(names) if names is not None else []
    if actual != list(ef.FEATURE_NAMES):
        raise ValueError(
            f"feature_names не совпадает с etgr_features: {len(actual)} имён "
            f"вместо {len(ef.FEATURE_NAMES)} — модель на других признаках"
        )


def _check_crop(crop: Any, cfg: ModuleType) -> None:
    """Параметры окна из файла весов должны совпадать с окном инференса.

    candidate_crops строит окна crop_px из ETGR с захардкоженными в
    crop_box pad=0.5 и min_side=16: модель, обученная на других окнах,
    даст неверные скоры.
    """
    expected = {"crop_px": int(cfg.ETGR["crop_px"]), "pad": 0.5, "min_side": 16}
    if crop != expected:
        raise ValueError(
            f"crop={crop!r}, окно инференса {expected!r} — модель на других окнах"
        )


def _as_threshold(value: Any, name: str) -> float:
    """Порог из файла весов: число в [0, 1], иначе исключение."""
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name}={threshold} вне диапазона [0, 1]")
    return threshold


def _warn_timm_mismatch(trained_version: Any) -> None:
    """Предупреждение о несовпадении версии timm (не отказ)."""
    try:
        installed = metadata.version("timm")
    except metadata.PackageNotFoundError:
        return
    if str(trained_version) != installed:
        logger.warning(
            "etgr: timm %s при обучении, в образе %s — возможны расхождения",
            trained_version,
            installed,
        )


def _largest_polygon(geom: BaseGeometry) -> Polygon | None:
    """Одиночный Polygon; MultiPolygon -> часть с максимальной площадью."""
    if geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda part: part.area)
    return None