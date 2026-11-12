"""Признаки магниторазведки (MagGr): общий код обучения и инференса.

Загрузка серой карты в рабочую сетку, маска съёмки по самому растру
(контур съёмки на платформе недоступен), нормировка z по MAD,
кандидаты точек и полигонов с признаками и NMS обеих веток.
Формулы повторяют исследовательский ноутбук MagGr; константы формул,
менять которые можно только вместе с переобучением моделей, живут
здесь (``_PT_*``, ``_POLY_*``), калибруемые параметры — в config.MAGGR.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, xy
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

POINT_FEATURE_NAMES: tuple[str, ...] = (
    "sign",
    "dy_m",
    "dx_m",
    "dist_m",
    "edge_m",
    "density",
    "rank_amp",
    "rank_amp3",
    "rank_dog1",
    "rank_dog2",
    "rank_opp",
    "amp_over_noise",
    "dog1_over_noise",
    "opp_over_noise",
    "h1_over_noise",
    "h2_over_noise",
)
"""Имена признаков кандидата точки; порядок колонок point_candidates."""

POLY_FEATURE_NAMES: tuple[str, ...] = (
    "знак",
    "фон σ",
    "уровень",
    "log площади",
    "доля площади",
    "z среднее",
    "z макс",
    "контраст с кольцом",
    "z без фона",
    "solidity",
    "eccentricity",
    "компактность",
    "удал. от края ср.",
    "удал. от края мин.",
    "длина, м",
)
"""Имена признаков полигонального кандидата (как в ноутбуке)."""

_PT_AUX_SIGMA: float = 3.0
"""Сигма второй шкалы сглаживания точек (zs[3] ноутбука)."""
_PT_DOG_SIGMAS: tuple[float, float] = (10.0, 30.0)
"""Сигмы вычитания фона для dog1/dog2 точек, px."""
_PT_NOISE_WIN: int = 31
"""Окно локального шума точек, px."""
_PT_HESS_SIGMA: float = 3.0
"""Сигма сглаживания перед гессианом, px."""
_PT_DENS_R_M: float = 3.0
"""Радиус оценки плотности кандидатов точки, м."""
_POLY_RING_DILATE: int = 5
"""Толщина кольца «контраст с кольцом», px грубой сетки."""
_POLY_PAD_PX: int = 6
"""Отступ окна кандидата от bbox региона, px грубой сетки."""


@dataclass
class MaggrField:
    """Серая карта MagGr в рабочей сетке: z, маска съёмки, геопривязка."""

    z: np.ndarray  # (H, W) float32: (v - med) / MAD, вне маски заполнено ближайшим
    mask: np.ndarray  # (H, W) bool: маска съёмки по самому растру
    transform: Affine  # привязка рабочей сетки (row/col -> x/y)
    crs: CRS  # CRS растра (метры, UTM)


def survey_mask(values: np.ndarray, erode_px: int) -> np.ndarray:
    """Маска съёмки по самому растру: фон 0/255, связанный с краем.

    Изолированные внутри поля 0/255 остаются валидными данными (не
    связаны с краем растра); кромка данных подтачивается эрозией.
    """
    lab, _ = ndimage.label((values == 0) | (values == 255))
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, edge[edge > 0])
    return ndimage.binary_erosion(~bg, iterations=int(erode_px))


def load_field(raster_path: Path, cfg: Mapping[str, Any]) -> MaggrField:
    """Серая карта в рабочей сетке cfg["res_m"] с маской съёмки и z.

    Вне маски съёмки значение заменяется ближайшим валидным (чтобы
    фильтры не «видели» фон); z = (v - медиана) / MAD по маске.
    """
    res_m = float(cfg["res_m"])
    with rasterio.open(raster_path) as src:
        height = int(round(src.height * src.res[1] / res_m))
        width = int(round(src.width * src.res[0] / res_m))
        v = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        crs = src.crs
    mask = survey_mask(v, int(cfg["mask_erode_px"]))
    if not mask.any():
        logger.warning("%s: маска съёмки пуста — поле нулей", raster_path.name)
        return MaggrField(z=np.zeros(v.shape, np.float32), mask=mask, transform=transform, crs=crs)
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    vf = v.astype(np.float32)[iy, ix]
    med = np.median(vf[mask])
    mad = 1.4826 * np.median(np.abs(vf[mask] - med)) + 1e-6
    z = ((vf - med) / mad).astype(np.float32)
    return MaggrField(z=z, mask=mask, transform=transform, crs=crs)


def gray_raster_paths(maggr_files: Sequence[Path], cfg: Mapping[str, Any]) -> list[Path]:
    """Серые карты MagGr из файлов типа maggr (glob cfg["gray_glob"] по имени)."""
    pattern = str(cfg["gray_glob"])
    named = [path for path in maggr_files if fnmatch(path.name, pattern)]
    if named:
        return named
    # имена на платформе могут отличаться от трейна: тогда серая карта — одноканальный
    # GeoTIFF (цветные карты — 3–4 канала). Нет совпадений и по каналам — пусто.
    return [path for path in maggr_files if _is_single_band_tif(path)]


def _is_single_band_tif(path: Path) -> bool:
    """Одноканальный GeoTIFF, не помеченный как цветная карта; ошибка чтения — False."""
    if path.suffix.lower() not in (".tif", ".tiff") or "цвет" in path.name.lower():
        return False
    try:
        with rasterio.open(path) as src:
            return src.count == 1
    except Exception:  # noqa: BLE001  # нечитаемый файл — не серая карта
        return False


def point_candidates(field: MaggrField, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Кандидаты точек: экстремумы обоих знаков и их признаки.

    Возвращает (xy (N, 2) в CRS растра, X (N, 16) float32); колонки X
    соответствуют ``POINT_FEATURE_NAMES``.
    """
    z, m = field.z, field.mask
    res_m = float(cfg["res_m"])
    zs_s = ndimage.gaussian_filter(z, float(cfg["pt_smooth"]))
    zs3 = ndimage.gaussian_filter(z, _PT_AUX_SIGMA)
    dog1 = zs_s - ndimage.gaussian_filter(z, _PT_DOG_SIGMAS[0])
    dog2 = zs3 - ndimage.gaussian_filter(z, _PT_DOG_SIGMAS[1])
    mu = ndimage.uniform_filter(z, _PT_NOISE_WIN)
    noise = np.sqrt(np.maximum(ndimage.uniform_filter(z * z, _PT_NOISE_WIN) - mu**2, 0))
    e1, e2 = _hessian_eigvals(z, _PT_HESS_SIGMA)
    edge = ndimage.distance_transform_edt(m) * res_m
    half = int(cfg["pt_dipole_win_px"])
    height, width = z.shape
    rows: list[tuple[int, int]] = []
    raw: list[list[float]] = []
    for sign in (-1.0, 1.0):
        a = sign * zs_s
        peaks = _peaks(np.where(m, a, -np.inf), int(cfg["pt_min_dist"]), float(cfg["pt_min_z"]))
        for r, c in peaks:
            r0, r1 = max(int(r) - half, 0), min(int(r) + half + 1, height)
            c0, c1 = max(int(c) - half, 0), min(int(c) + half + 1, width)
            w = -a[r0:r1, c0:c1]
            k0, k1 = np.unravel_index(np.argmax(w), w.shape)
            dy, dx = (k0 + r0 - r) * res_m, (k1 + c0 - c) * res_m
            rows.append((int(r), int(c)))
            raw.append(
                [
                    sign,
                    a[r, c],
                    sign * zs3[r, c],
                    sign * dog1[r, c],
                    sign * dog2[r, c],
                    w.max(),
                    noise[r, c],
                    dy,
                    dx,
                    np.hypot(dy, dx),
                    sign * e1[r, c],
                    sign * e2[r, c],
                    edge[r, c],
                ]
            )
    if not rows:
        return np.zeros((0, 2), np.float64), np.zeros((0, len(POINT_FEATURE_NAMES)), np.float32)
    rc = np.array(rows, dtype=np.intp)
    data = np.array(raw, dtype=np.float32)
    xs, ys = xy(field.transform, rc[:, 0], rc[:, 1])
    xy_pts = np.c_[xs, ys]
    sign, amp, amp3, dg1, dg2, opp, nz, dy, dx, dd, h1, h2, edg = data.T

    def _rank(values: np.ndarray) -> np.ndarray:
        return rankdata(values) / len(values)

    dens = np.array([len(nb) - 1 for nb in cKDTree(xy_pts).query_ball_point(xy_pts, _PT_DENS_R_M)])
    feats = np.c_[
        sign,
        dy,
        dx,
        dd,
        edg,
        dens,
        _rank(amp),
        _rank(amp3),
        _rank(dg1),
        _rank(dg2),
        _rank(opp),
        amp / (nz + 1e-3),
        dg1 / (nz + 1e-3),
        opp / (nz + 1e-3),
        h1 / (nz + 1e-3),
        h2 / (nz + 1e-3),
    ]
    return xy_pts, feats.astype(np.float32)


def poly_candidates(
    field: MaggrField, cfg: Mapping[str, Any]
) -> tuple[list[BaseGeometry], np.ndarray]:
    """Полигональные кандидаты: нарезка укрупнённого поля уровнями по std.

    Возвращает (полигоны в CRS растра, X (N, 15) float32); колонки X
    соответствуют ``POLY_FEATURE_NAMES``.
    """
    z0, m, tr = _coarse(field, cfg)
    if z0.size == 0 or not m.any():
        return [], np.zeros((0, len(POLY_FEATURE_NAMES)), np.float32)
    poly_res = float(cfg["poly_res"])
    edge = ndimage.distance_transform_edt(m) * poly_res
    valid_area = int(m.sum())
    zs = ndimage.gaussian_filter(z0, float(cfg["smooth"]))
    z_std = np.std(zs[m])
    if not np.isfinite(z_std) or z_std <= 0:
        logger.warning("poly_candidates: нулевой разброс сглаженного поля — без кандидатов")
        return [], np.zeros((0, len(POLY_FEATURE_NAMES)), np.float32)
    min_px = float(cfg["min_cand_m2"]) / poly_res**2
    levels = [float(level) for level in cfg["levels"]]
    structure = np.ones((3, 3), dtype=bool)
    polys: list[BaseGeometry] = []
    feats: list[list[float]] = []
    for bg in cfg["bg_sigmas"]:
        z = zs - ndimage.gaussian_filter(z0, float(bg)) if bg else zs
        std = np.std(z[m])
        if not np.isfinite(std) or std <= 0:
            logger.warning("poly_candidates: std=0 при фоне σ=%s — уровень пропущен", bg)
            continue
        z = z / std
        for sign in (1.0, -1.0):
            for lv in levels:
                lab, _ = ndimage.label((sign * z > lv) & m, structure=structure)
                for idx, sl in enumerate(ndimage.find_objects(lab), start=1):
                    if sl is None:
                        continue
                    area = int((lab[sl] == idx).sum())
                    if area < min_px or area > 0.5 * valid_area:
                        continue
                    y0 = max(sl[0].start - _POLY_PAD_PX, 0)
                    x0 = max(sl[1].start - _POLY_PAD_PX, 0)
                    y1 = min(sl[0].stop + _POLY_PAD_PX, lab.shape[0])
                    x1 = min(sl[1].stop + _POLY_PAD_PX, lab.shape[1])
                    reg = lab[y0:y1, x0:x1] == idx
                    ring = ndimage.binary_dilation(reg, iterations=_POLY_RING_DILATE)
                    ring = ring & ~reg & m[y0:y1, x0:x1]
                    z_win = sign * z[y0:y1, x0:x1]
                    zin = z_win[reg]
                    zring = z_win[ring] if ring.any() else np.array([lv], dtype=z_win.dtype)
                    cnts, _ = cv2.findContours(
                        reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not cnts:
                        continue
                    cnt = max(cnts, key=cv2.contourArea)
                    if len(cnt) < 4:
                        continue
                    pts = cnt.squeeze(1)
                    xs, ys = xy(tr, (pts[:, 1] + y0).tolist(), (pts[:, 0] + x0).tolist())
                    poly = Polygon(list(zip(xs, ys))).buffer(0)
                    if poly.is_empty:
                        continue
                    rr, cc = np.nonzero(reg)
                    perimeter = _perimeter(reg)
                    ecc, axis_px = _ellipse_feats(rr, cc)
                    e = edge[y0:y1, x0:x1][reg]
                    feats.append(
                        [
                            sign,
                            float(bg),
                            lv,
                            np.log(area),
                            area / valid_area,
                            float(zin.mean()),
                            float(zin.max()),
                            float(zin.mean() - zring.mean()),
                            float((sign * zs[y0:y1, x0:x1][reg]).mean() / z_std),
                            _solidity(reg, rr, cc),
                            ecc,
                            4 * np.pi * area / max(perimeter, 1.0) ** 2,
                            float(e.mean()),
                            float(e.min()),
                            axis_px * poly_res,
                        ]
                    )
                    polys.append(poly)
    if not polys:
        return [], np.zeros((0, len(POLY_FEATURE_NAMES)), np.float32)
    return polys, np.array(feats, dtype=np.float32).reshape(-1, len(POLY_FEATURE_NAMES))


def nms_points(
    xy: np.ndarray, scores: np.ndarray, thr: float, cfg: Mapping[str, Any]
) -> list[int]:
    """NMS точек: жадно по убыванию score, подавление в радиусе pt_nms_m.

    Возвращает индексы оставшихся кандидатов (исходная нумерация).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(xy) == 0:
        return []
    order = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] >= thr]
    if not order:
        return []
    tree = cKDTree(xy[order])
    taken = np.zeros(len(order), dtype=bool)
    keep: list[int] = []
    for k, i in enumerate(order):
        if taken[k]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(xy[i], float(cfg["pt_nms_m"])):
            if j > k:
                taken[j] = True
    return keep


def nms_polys(
    polys: Sequence[BaseGeometry], scores: np.ndarray, thr: float, cfg: Mapping[str, Any]
) -> list[int]:
    """NMS полигонов: IoU >= nms_iou или покрытие >= nms_cover меньшей фигуры."""
    scores = np.asarray(scores, dtype=np.float64)
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] < thr:
            break
        p = polys[i]
        ok = True
        for k in keep:
            inter = p.intersection(polys[k]).area
            iou = inter / max(p.union(polys[k]).area, 1e-12)
            if iou >= float(cfg["nms_iou"]) or inter >= float(cfg["nms_cover"]) * min(
                p.area, polys[k].area
            ):
                ok = False
                break
        if ok:
            keep.append(int(i))
    return keep


def _peaks(a: np.ndarray, min_dist: int, thr: float) -> np.ndarray:
    """Локальные максимумы окна (2*min_dist+1) с жадным прореживанием.

    Замена ``skimage.feature.peak_local_max`` (min_distance,
    threshold_abs, exclude_border=False): кандидаты — точки, равные
    максимуму окна и >= thr; затем по убыванию амплитуды убираются
    соседи в шахматном радиусе < min_dist.
    """
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.intp)
    window = 2 * int(min_dist) + 1
    amax = ndimage.maximum_filter(a, size=window, mode="nearest")
    rows, cols = np.nonzero((a == amax) & (a >= thr))
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.intp)
    order = np.argsort(-a[rows, cols], kind="stable")
    pts = np.column_stack([rows, cols]).astype(np.float64)
    tree = cKDTree(pts)
    active = np.ones(rows.size, dtype=bool)
    kept: list[tuple[int, int]] = []
    radius = max(min_dist - 0.5, 0.0)  # целые координаты: chebyshev < min_dist
    for k in order:
        if not active[k]:
            continue
        kept.append((int(rows[k]), int(cols[k])))
        for j in tree.query_ball_point(pts[k], radius, p=np.inf):
            active[j] = False
    return np.array(kept, dtype=np.intp).reshape(-1, 2)


def _hessian_eigvals(z: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Собственные значения гессиана (e1 >= e2): гаусс (constant, 0) + np.gradient."""
    smoothed = ndimage.gaussian_filter(z, sigma, mode="constant", cval=0)
    gz_r, gz_c = np.gradient(smoothed)
    h_rr = np.gradient(gz_r, axis=0)
    h_rc = np.gradient(gz_r, axis=1)
    h_cc = np.gradient(gz_c, axis=1)
    mean = (h_rr + h_cc) / 2
    root = np.sqrt(((h_rr - h_cc) / 2) ** 2 + h_rc**2)
    return mean + root, mean - root


def _coarse(field: MaggrField, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Укрупнение блоками k×k (k = round(poly_res / res_m)): среднее z, «и» масок."""
    k = max(int(round(float(cfg["poly_res"]) / float(cfg["res_m"]))), 1)
    height, width = field.z.shape
    h2, w2 = (height // k) * k, (width // k) * k
    if h2 == 0 or w2 == 0:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), bool), field.transform
    z = field.z[:h2, :w2].reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))
    m = field.mask[:h2, :w2].reshape(h2 // k, k, w2 // k, k).all(axis=(1, 3))
    return z, m, field.transform * field.transform.scale(k, k)


def _perimeter(reg: np.ndarray) -> float:
    """Периметр региона по формуле skimage.measure.perimeter (связность n4)."""
    strel = ndimage.generate_binary_structure(2, 1)
    mask_u8 = reg.astype(np.uint8)
    eroded = ndimage.binary_erosion(mask_u8, strel, border_value=0)
    border = mask_u8 - eroded.astype(np.uint8)
    kernel = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]], dtype=np.float64)
    conv = ndimage.convolve(border, kernel, mode="constant", cval=0)
    hist = np.bincount(conv.astype(np.intp).ravel(), minlength=50)
    weights = np.zeros(50, dtype=np.float64)
    weights[[5, 7, 15, 17, 25, 27]] = 1.0
    weights[[21, 33]] = np.sqrt(2)
    weights[[13, 23]] = (1 + np.sqrt(2)) / 2
    return float(hist @ weights)


def _ellipse_feats(rr: np.ndarray, cc: np.ndarray) -> tuple[float, float]:
    """(eccentricity, axis_major_length в px) по центральным моментам пикселей."""
    n = rr.size
    if n == 0:
        return 0.0, 0.0
    cy, cx = float(rr.mean()), float(cc.mean())
    mu_rr = float(((rr - cy) ** 2).sum())
    mu_cc = float(((cc - cx) ** 2).sum())
    mu_rc = float(((rr - cy) * (cc - cx)).sum())
    inertia = np.array([[mu_rr, -mu_rc], [-mu_rc, mu_cc]]) / n
    eig = np.linalg.eigvalsh(inertia)
    l_max, l_min = float(eig.max()), float(eig.min())
    ecc = 0.0 if l_max <= 0.0 else float(np.sqrt(max(0.0, 1.0 - l_min / l_max)))
    return ecc, float(4.0 * np.sqrt(max(l_max, 0.0)))


def _solidity(reg: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> float:
    """solidity региона: площадь / площадь выпуклой оболочки (cv2)."""
    pts = np.column_stack([cc, rr]).astype(np.int32)
    hull = cv2.convexHull(pts)
    canvas = np.zeros(reg.shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [hull], 1)
    convex_px = int(np.count_nonzero(canvas))
    if convex_px <= 0:
        return 1.0
    return float(rr.size) / convex_px
