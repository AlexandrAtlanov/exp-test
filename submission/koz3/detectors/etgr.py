"""Детектор электроразведки (EtGr): аномальные полигоны на картах и профилях.

Конвейер карт — общий код обучения и инференса
``koz3.geophys.etgr_features``: обращение шкалы jet, кандидаты-изолинии,
окна CNN; детектор грузит ансамбль CNN из ``weights/etgr_maps.pt``
(контракт training/train_etgr.py), скорит окна усреднением по
отражениям (внутри predict_cnn) и сетям ансамбля, режет слившиеся
пятна (predict_polys) и переводит полигоны из CRS растра в UTM региона:
``object_poly``, markup_type «etgr».

Профили — вторая, независимая ветка: те же кандидаты, профильные
признаки (глубины, карты под полигоном — ``profile_extra``) и HGB из
``weights/etgr_profiles.joblib`` (контракт training/train_etgr.py
--kind profiles); NMS без разрезания пятен. Полигоны профиля лежат в
вертикальном разрезе и на выход идут как есть — GT размечен так же.

Отсутствие весов или ошибка одной ветки не отключает другую; без
библиотек или при ошибке загрузки ветка выключается с предупреждением
в лог — молча никогда.
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

_PROFILE_REQUIRED_KEYS: tuple[str, ...] = (
    "clf",
    "thr",
    "feature_names",
    "sklearn",
)
"""Обязательные ключи weights/etgr_profiles.joblib (train_etgr.py --kind profiles).

clf — HGB с predict_proba на признаки PROFILE_FEATURE_NAMES; thr — порог
OOF в [0, 1] (детектор применяет его в NMS, postprocess не режет повторно);
sklearn — версия при обучении (расхождение — предупреждение).
"""


@dataclass
class _EtGrNets:
    """Ансамбль CNN карт EtGr и порог из файла весов."""

    nets: list[Any]  # torch.nn.Module; Any — torch не импортируется на уровне модуля
    thr: float


_NETS_CACHE: dict[Path, _EtGrNets | None] = {}
"""Кеш ансамбля по пути файла весов: загрузка один раз на прогон."""


@dataclass
class _EtGrProfiles:
    """HGB профилей EtGr и порог из файла весов."""

    clf: Any  # sklearn-классификатор; Any — sklearn импортируется лениво
    thr: float


_PROFILES_CACHE: dict[Path, _EtGrProfiles | None] = {}
"""Кеш HGB профилей по пути файла весов: загрузка один раз на прогон."""


def is_available(region: Region) -> bool:
    """True, если в регионе есть EtGr и готова хотя бы одна ветка.

    Карты: есть карты-срезы и грузится ансамбль CNN. Профили: есть
    профили и грузится HGB. Ветки независимы: нет весей одной — вторая
    работает.
    """
    if ef is None:
        _warn_features_missing()
        return False
    files = region.files.get("etgr", [])
    if ef.map_raster_paths(files) and _load_nets(config) is not None:
        return True
    return bool(ef.profile_raster_paths(files)) and _load_profiles(config) is not None


_features_warned = False
"""Предупреждение «нет библиотек модуля признаков» уже выдано в этом прогоне."""


def _warn_features_missing() -> None:
    """Сообщить один раз за прогон, что детектор выключен из-за импорта признаков."""
    global _features_warned
    if not _features_warned:
        _features_warned = True
        logger.warning(
            "etgr: нет библиотек модуля признаков (%s) — детектор выключен", _IMPORT_ERROR
        )


def estimate_seconds(region: Region) -> float:
    """Оценка времени = сумме проверок бюджета внутри run.

    Карты: карты × seconds_per_map (ветка CNN). Профили: профили ×
    seconds_per_profile + карты × seconds_per_map — поля карт для
    признаков профилей грузятся отдельно от ветки карт (конвейер карт
    не меняется); оценки-заглушки до замера на платформе.
    """
    if ef is None:
        return 0.0  # детектор выключен — бюджет не резервируем
    files = region.files.get("etgr", [])
    maps = ef.map_raster_paths(files)
    profiles = ef.profile_raster_paths(files)
    per_map_s = float(config.ETGR["seconds_per_map"])
    est_s = len(maps) * per_map_s
    if profiles:
        est_s += len(profiles) * float(config.ETGR["seconds_per_profile"]) + len(maps) * per_map_s
    return est_s


def run(region: Region, budget: Budget, cfg: ModuleType) -> list[Detection]:
    """Детекции региона: карты (CNN) и профили (HGB) в UTM, markup_type «etgr».

    Карты важнее: при нехватке бюджета профили пропускаются первыми.
    Модели грузятся один раз (кеш по пути весов); бюджет проверяется
    перед каждым файлом; ошибка файла или ветки не роняет прогон.
    """
    detections: list[Detection] = []
    detections.extend(_run_maps(region, budget, cfg))
    detections.extend(_run_profiles(region, budget, cfg))
    return detections


def _run_maps(region: Region, budget: Budget, cfg: ModuleType) -> list[Detection]:
    """Детекции карт региона (ансамбль CNN); без модели — пусто с предупреждением."""
    nets = _load_nets(cfg)
    if nets is None:
        logger.warning("etgr: регион %s без модели карт — ветка карт пропущена", region.name)
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


def _run_profiles(region: Region, budget: Budget, cfg: ModuleType) -> list[Detection]:
    """Детекции профилей региона (HGB); без модели — пусто с предупреждением.

    Признакам профилей нужны поля карт региона («карты под полигоном»):
    они грузятся здесь отдельно — конвейер карт не меняется. Карта не
    читается или не лезет в бюджет — её признаки остаются MISSING.
    """
    profiles = _load_profiles(cfg)
    if profiles is None:
        logger.warning(
            "etgr: регион %s без модели профилей — ветка профилей пропущена", region.name
        )
        return []
    assert ef is not None  # недостижимо при ef = None: _load_profiles уже вернул None
    files = region.files.get("etgr", [])
    per_map_s = float(cfg.ETGR["seconds_per_map"])
    map_fields: list[Any] = []
    for map_path in ef.map_raster_paths(files):
        if not budget.can_afford(per_map_s):
            logger.warning(
                "etgr: %s пропущен для признаков профилей — не хватает бюджета (оценка %.1f с)",
                map_path.name,
                per_map_s,
            )
            continue
        try:
            map_fields.append(ef.load_field(map_path, cfg.ETGR))
        except Exception as exc:
            logger.warning("etgr: %s не читается для признаков профилей (%s)", map_path.name, exc)
    per_profile_s = float(cfg.ETGR["seconds_per_profile"])
    detections: list[Detection] = []
    for raster_path in ef.profile_raster_paths(files):
        if not budget.can_afford(per_profile_s):
            logger.warning(
                "etgr: профиль %s пропущен — не хватает бюджета (оценка %.1f с)",
                raster_path.name,
                per_profile_s,
            )
            continue
        detections.extend(_run_profile_raster(raster_path, region, profiles, map_fields, cfg))
    return detections


def _run_profile_raster(
    raster_path: Path,
    region: Region,
    profiles: _EtGrProfiles,
    map_fields: list[Any],
    cfg: ModuleType,
) -> list[Detection]:
    """Детекции одного профиля; ошибки чтения и конвейера изолированы."""
    try:
        field = ef.load_field(raster_path, cfg.ETGR)
    except Exception as exc:
        logger.warning("etgr: профиль %s пропущен (%s)", raster_path.name, exc)
        return []
    try:
        return _profile_detections(field, raster_path.name, region, profiles, map_fields, cfg)
    except Exception as exc:
        logger.warning("etgr: %s: профиль пропущен (%s)", raster_path.name, exc)
        return []


def _profile_detections(
    field: Any,
    raster_name: str,
    region: Region,
    profiles: _EtGrProfiles,
    map_fields: list[Any],
    cfg: ModuleType,
) -> list[Detection]:
    """Конвейей профиля: кандидаты → признаки → HGB → NMS → UTM.

    field — EtGrField профиля; map_fields — поля карт региона для
    признаков «карты под полигоном». Полигоны лежат в вертикальном
    разрезе и на выход идут как есть (GT размечен так же); NMS — без
    разрезания пятен (ноутбук 02, predict_polys для профилей).
    """
    cand_polys, cand_x = ef.candidates(field, cfg.ETGR)
    if not cand_polys:
        return []
    extra = ef.profile_extra(field, cand_polys, cand_x, map_fields)
    features = np.hstack([cand_x, extra])
    scores = _predict_clf(profiles.clf, features)
    keep = ef.nms(cand_polys, scores, profiles.thr, cfg.ETGR)
    polys = [cand_polys[i] for i in keep]
    confs = [float(scores[i]) for i in keep]
    return _to_detections(field, polys, confs, raster_name, region)


def _predict_clf(clf: Any, features: np.ndarray) -> np.ndarray:
    """Вероятности HGB (N,); NaN/Inf -> -1 (ниже любого порога)."""
    proba = np.asarray(clf.predict_proba(features), dtype=np.float64)
    if proba.ndim != 2 or proba.shape != (len(features), 2):
        raise ValueError(
            f"predict_proba вернул форму {proba.shape}, ожидалась ({len(features)}, 2)"
        )
    positive = proba[:, 1]
    return np.where(np.isfinite(positive), np.clip(positive, 0.0, 1.0), -1.0)


def _to_detections(
    field: Any,
    polys: Sequence[BaseGeometry],
    confs: Sequence[float],
    raster_name: str,
    region: Region,
) -> list[Detection]:
    """Полигоны растра (карта или профиль) в детекции: CRS → UTM, object_poly."""
    same_crs = field.crs == CRS.from_user_input(region.utm_crs)
    detections: list[Detection] = []
    for geom_src, conf in zip(polys, confs, strict=True):
        geom = geom_src if same_crs else reproject(geom_src, field.crs.to_string(), region.utm_crs)
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
        raise ValueError(f"crop={crop!r}, окно инференса {expected!r} — модель на других окнах")


def _load_profiles(cfg: ModuleType) -> _EtGrProfiles | None:
    """HGB профилей из кеша или из файла весов; None — ветка выключена."""
    weights_path = Path(cfg.WEIGHTS_DIR) / str(cfg.ETGR["profiles_models_file"])
    if weights_path in _PROFILES_CACHE:
        return _PROFILES_CACHE[weights_path]
    profiles: _EtGrProfiles | None = None
    if ef is None:
        logger.warning(
            "etgr: нет библиотек модуля признаков (%s) — профили выключены", _IMPORT_ERROR
        )
    elif not weights_path.is_file():
        logger.debug("etgr: нет файла весов %s — профили выключены", weights_path)
    else:
        try:
            profiles = _read_profiles(weights_path)
        except Exception as exc:
            logger.warning("etgr: профили выключены — %s не загружен (%s)", weights_path, exc)
    _PROFILES_CACHE[weights_path] = profiles
    return profiles


def _read_profiles(weights_path: Path) -> _EtGrProfiles:
    """Читает weights/etgr_profiles.joblib и проверяет контракт обучения.

    Формат: dict с ключами _PROFILE_REQUIRED_KEYS: clf (HGB с
    predict_proba), thr (порог OOF в [0, 1]), feature_names
    (PROFILE_FEATURE_NAMES — модель на других признаках = ошибка),
    sklearn (версия при обучении). Без joblib/sklearn в образе —
    исключение, ветка выключается с предупреждением.
    """
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("нет joblib в образе") from exc
    raw = joblib.load(weights_path)
    if not isinstance(raw, dict):
        raise TypeError(f"в файле {type(raw).__name__}, ожидался dict")
    missing = [key for key in _PROFILE_REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError("нет ключей: " + ", ".join(missing))
    if not callable(getattr(raw["clf"], "predict_proba", None)):
        raise TypeError("clf без predict_proba")
    _check_profile_feature_names(raw["feature_names"])
    _warn_sklearn_mismatch(raw["sklearn"])
    return _EtGrProfiles(clf=raw["clf"], thr=_as_threshold(raw["thr"], "thr"))


def _check_profile_feature_names(names: Any) -> None:
    """Имена признаков из файла весов = PROFILE_FEATURE_NAMES etgr_features."""
    actual = list(names) if names is not None else []
    if actual != list(ef.PROFILE_FEATURE_NAMES):
        raise ValueError(
            f"feature_names не совпадает с etgr_features: {len(actual)} имён "
            f"вместо {len(ef.PROFILE_FEATURE_NAMES)} — модель на других признаках"
        )


def _warn_sklearn_mismatch(trained_version: Any) -> None:
    """Предупреждение о несовпадении версии sklearn (не отказ)."""
    try:
        installed = metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        return
    if str(trained_version) != installed:
        logger.warning(
            "etgr: sklearn %s при обучении, в образе %s — возможны расхождения",
            trained_version,
            installed,
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
