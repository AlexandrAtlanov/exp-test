"""Признаки электроразведки (EtGr): обращение шкалы jet, кандидаты, окна CNN.

Дословный перенос формул ноутбука организаторов 02_baseline_etgr_anomalies
(ячейки 7, 18, 21, 23) в общий код обучения и инференса (правило 5а.1: такой
код живёт в submission/koz3 и импортируется в training/).

Отличия от ноутбука:
- маска съёмки строится только по растру (dist < palette_max_dist): на платформе
  нет контура съёмки (extra/*_границы.geojson) и разметки;
- конвейер карт работает только с картами-срезами: map_raster_paths отфильтровывает
  профили (признак «профиль?» у карт всегда False); профили — отдельная ветка
  (profile_raster_paths, profile_extra): те же кандидаты + признаки глубины и карт
  под полигоном;
- torch и timm импортируются внутри функций: импорт модуля признаков не тянет их.

Геометрии кандидатов остаются в CRS растра; в UTM их переводит детектор.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from skimage.measure import label, regionprops

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
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
    "длина / размер растра",
    "профиль?",
)
"""Имена признаков кандидата (ноутбук 02, ячейка 18); порядок = столбцы X."""

PROFILE_EXTRA: tuple[str, ...] = (
    "глубина верха, м",
    "глубина низа, м",
    "карты под полигоном: среднее",
    "карты под полигоном: макс.",
    "доля карт с аномалией",
)
"""Имена профильных признаков кандидата (ноутбук 02, ячейка 21); порядок = столбцы extra."""

PROFILE_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + PROFILE_EXTRA
"""Признаки кандидата профиля: 16 базовых + 5 профильных; порядок = столбцы X HGB."""

MISSING: float = -9.0
"""Значение отсутствующего признака (нет линии поверхности, нет карт, нет пикселей)."""

NORM_MEAN = (0.485, 0.456, 0.406, 0.5)
NORM_STD = (0.229, 0.224, 0.225, 0.5)
"""Нормировка каналов окна CNN (RGB + маска), общая для обучения и инференса."""

_JET_TREE: cKDTree | None = None
"""Кэш KD-дерева по 256 цветам jet (ноутбук 02, ячейка 7)."""


@dataclass(frozen=True)
class EtGrField:
    """Рабочая сетка карты EtGr: RGB, значение z, маска шкалы, привязка."""

    rgb: np.ndarray  # (3, H, W) uint8 — исходные цвета
    z: np.ndarray  # (H, W) float — значение, нормированное на MAD
    mask: np.ndarray  # (H, W) bool — пиксели шкалы (маска съёмки)
    transform: Affine  # привязка рабочей сетки
    crs: CRS  # CRS растра
    is_profile: bool  # True у профилей; карты-срезы всегда False


def jet_palette() -> np.ndarray:
    """Палитра jet (256, 3), 0..255 — как цветовая шкала растров EtGr."""
    import matplotlib

    return matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255


def _jet_tree() -> cKDTree:
    """KD-дерево по палитре jet; строится один раз и кэшируется."""
    global _JET_TREE
    if _JET_TREE is None:
        _JET_TREE = cKDTree(jet_palette())
    return _JET_TREE


def rgb_to_scalar(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Обращение шкалы jet: (значение 0..1, расстояние до палитры), обе (H, W).

    rgb — (3, H, W), как читает rasterio; значение даёт ближайший цвет палитры.
    """
    dist, idx = _jet_tree().query(np.asarray(rgb).reshape(3, -1).T.astype(np.float32))
    return (idx / 255.0).reshape(rgb.shape[1:]), dist.reshape(rgb.shape[1:])


def raster_stem(raster: Path) -> str:
    """stem растра EtGr без порядкового префикса «N_» (ноутбук 02, ячейка 3)."""
    parts = raster.stem.split("_", 1)
    return parts[1] if len(parts) > 1 else raster.stem


def is_profile_raster(raster: Path) -> bool:
    """Профиль или карта-срез: маркер «_профиль_» в stem (ноутбук 02, ячейка 3)."""
    return "_профиль_" in raster_stem(raster)


def map_raster_paths(files: Sequence[Path]) -> list[Path]:
    """Карты-срезы EtGr из растров региона; профили отфильтрованы."""
    return [f for f in files if not is_profile_raster(f)]


def profile_raster_paths(files: Sequence[Path]) -> list[Path]:
    """Профили EtGr из растров региона (ветка HGB, отдельный пайплайн)."""
    return [f for f in files if is_profile_raster(f)]


def load_field(raster: Path, cfg: Mapping[str, object]) -> EtGrField:
    """Карту EtGr в рабочую сетку: downscale до work_side, обращение шкалы, z.

    Маска съёмки — только по растру: пиксели с расстоянием до палитры
    < palette_max_dist (контура съёмки на платформе нет). Невалидные пиксели
    заполняются ближайшим валидным, значение нормируется на MAD (ноутбук 02,
    ячейка 7). Если валидных пикселей нет — z из нулей и предупреждение.
    """
    with rasterio.open(raster) as src:
        scale = max(1.0, max(src.width, src.height) / float(cfg["work_side"]))
        height, width = int(round(src.height / scale)), int(round(src.width / scale))
        rgb = src.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.nearest)
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        crs = src.crs
    t, dist = rgb_to_scalar(rgb)
    mask = dist < float(cfg["palette_max_dist"])
    if not mask.any():
        logger.warning("%s: нет пикселей шкалы jet — пустая маска съёмки", raster.name)
        return EtGrField(rgb, np.zeros(t.shape), mask, transform, crs, is_profile_raster(raster))
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    t = t[iy, ix]
    v = t[mask]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
    return EtGrField(rgb, (t - med) / mad, mask, transform, crs, is_profile_raster(raster))


def candidates(
    field: EtGrField, cfg: Mapping[str, object]
) -> tuple[list[BaseGeometry], np.ndarray]:
    """Кандидаты-изолинии и их признаки (ноутбук 02, ячейка 18), дословно.

    Каждый фон из bg_sigmas × знак ±1 × уровень из levels даёт связные области;
    область крупнее min_cand_px и не более половины сетки становится кандидатом.
    Возвращает (полигоны в CRS растра, X (N, 16) float32 — столбцы FEATURE_NAMES).
    """
    m = field.mask
    if not m.any():
        return [], np.zeros((0, len(FEATURE_NAMES)), np.float32)
    edt = ndimage.distance_transform_edt(m)
    edt_n = edt / (edt.max() + 1e-6)
    valid_area = m.sum()
    zs = ndimage.gaussian_filter(field.z, float(cfg["smooth_px"]))
    z_std = np.std(zs[m])
    polys: list[BaseGeometry] = []
    feats: list[list[float]] = []
    for bg in cfg["bg_sigmas"]:
        z = zs - ndimage.gaussian_filter(field.z, bg) if bg else zs
        z = z / np.std(z[m])
        for sign in (1, -1):
            for lv in cfg["levels"]:
                lab = label((sign * z > lv) & m, connectivity=2)
                for rp in regionprops(lab):
                    if rp.area < float(cfg["min_cand_px"]) or rp.area > 0.5 * valid_area:
                        continue
                    sl, pad = rp.slice, 6
                    y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
                    y1, x1 = min(sl[0].stop + pad, z.shape[0]), min(sl[1].stop + pad, z.shape[1])
                    reg = np.zeros((y1 - y0, x1 - x0), bool)
                    reg[sl[0].start - y0 : sl[0].stop - y0, sl[1].start - x0 : sl[1].stop - x0] = (
                        lab[sl] == rp.label
                    )
                    ring = ndimage.binary_dilation(reg, iterations=5) & ~reg & m[y0:y1, x0:x1]
                    zin = sign * z[y0:y1, x0:x1][reg]
                    zring = sign * z[y0:y1, x0:x1][ring] if ring.any() else np.array([lv])
                    cnts = cv2.findContours(
                        reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )[0]
                    cnt = max(cnts, key=cv2.contourArea)
                    if len(cnt) < 4:
                        continue
                    poly = Polygon(
                        [
                            rasterio.transform.xy(field.transform, p[1] + y0, p[0] + x0)
                            for p in cnt.squeeze(1)
                        ]
                    ).buffer(0)
                    if poly.is_empty:
                        continue
                    e = edt_n[y0:y1, x0:x1][reg]
                    polys.append(poly)
                    feats.append(
                        [
                            sign,
                            bg,
                            lv,
                            np.log(rp.area),
                            rp.area / valid_area,
                            zin.mean(),
                            zin.max(),
                            zin.mean() - zring.mean(),
                            (sign * zs[y0:y1, x0:x1][reg]).mean() / z_std,
                            rp.solidity,
                            rp.eccentricity,
                            4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2,
                            e.mean(),
                            e.min(),
                            rp.axis_major_length / max(z.shape),
                            field.is_profile,
                        ]
                    )
    return polys, np.array(feats, np.float32).reshape(-1, len(FEATURE_NAMES))


def nested_children(polys: Sequence[BaseGeometry], x: np.ndarray) -> list[list[int]]:
    """Дети кандидата: тот же знак и фон, уровень на 0.5 выше, внутри родителя.

    Дети нужны для разрезания слившихся пятен (ноутбук 02, ячейка 21).
    """
    kids: list[list[int]] = [[] for _ in polys]
    if not polys:
        return kids
    isg, ibg, ilv = (FEATURE_NAMES.index(k) for k in ("знак", "фон σ", "уровень"))
    tree = STRtree([p.representative_point() for p in polys])
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if (
                j != i
                and x[j, isg] == x[i, isg]
                and x[j, ibg] == x[i, ibg]
                and abs(x[j, ilv] - x[i, ilv] - 0.5) < 1e-6
                and p.contains(polys[j].representative_point())
            ):
                kids[i].append(int(j))
    return kids


def _profile_surface(field: EtGrField) -> tuple[np.ndarray, np.ndarray] | None:
    """Линия поверхности профиля: (x, y) верхних валидных пикселей по столбцам.

    Ноутбук 02, ячейка 21: линия есть только у растров шире, чем выше
    (профили лежат горизонтально, глубина — вниз от линии); иначе None.
    """
    m = field.mask
    if m.shape[1] <= m.shape[0]:
        return None
    cols = np.where(m.any(0))[0]
    top = np.array([np.argmax(m[:, c]) for c in cols])
    x, y = rasterio.transform.xy(field.transform, top, cols)
    return np.asarray(x), np.asarray(y)


def mean_under(poly: BaseGeometry, field: EtGrField, sign: float) -> float:
    """Среднее sign * z карты field под полигоном (ноутбук 02, ячейка 21).

    Сетки карт одного участка немного различаются — окно берётся по
    координатам полигона в сетке карты; полигон вне сетки или под маской
    меньше 3 пикселей — NaN.
    """
    height, width = field.z.shape
    win = from_bounds(*poly.bounds, transform=field.transform).round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1 = min(int(win.row_off + win.height) + 1, height)
    c1 = min(int(win.col_off + win.width) + 1, width)
    if r1 <= r0 or c1 <= c0:
        return np.nan
    tr = field.transform * field.transform.translation(c0, r0)
    m = rasterize(
        [(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, all_touched=True
    ).astype(bool)
    m &= field.mask[r0:r1, c0:c1]
    return float(sign * field.z[r0:r1, c0:c1][m].mean()) if m.sum() >= 3 else np.nan


def profile_extra(
    field: EtGrField,
    cand_polys: Sequence[BaseGeometry],
    cand_x: np.ndarray,
    maps: Sequence[EtGrField],
) -> np.ndarray:
    """Профильные признаки кандидатов (N, 5), столбцы PROFILE_EXTRA (ноутбук 02, ячейка 21).

    Глубины верха/низа — от линии поверхности профиля в координатах растра
    (метры UTM); нет линии — MISSING. Карты под полигоном: среднее и
    максимум sign * z по картам региона и доля карт с аномалией
    (среднее > 0.5 уровня кандидата); нет карт или нет пикселей — MISSING.
    """
    out = np.full((len(cand_polys), len(PROFILE_EXTRA)), MISSING, np.float32)
    surface = _profile_surface(field)
    isign, ilvl = FEATURE_NAMES.index("знак"), FEATURE_NAMES.index("уровень")
    for i, poly in enumerate(cand_polys):
        if surface is not None:
            y_surf = float(np.interp(poly.centroid.x, surface[0], surface[1]))
            out[i, :2] = (y_surf - poly.bounds[3], y_surf - poly.bounds[1])
        v = np.asarray([mean_under(poly, m, cand_x[i, isign]) for m in maps])
        v = v[~np.isnan(v)]
        if len(v):
            out[i, 2:] = (v.mean(), v.max(), (v > 0.5 * cand_x[i, ilvl]).mean())
    return out


def crop_box(
    img: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: float = 0.5, min_side: int = 16
) -> np.ndarray:
    """Квадратное окно вокруг bbox (r0:r1, c0:c1) с полями pad; вне растра нули.

    Сторона = max(высота, ширина, min_side) * (1 + 2 * pad) (ноутбук 02, ячейка 23).
    """
    height, width = img.shape[:2]
    side = max(r1 - r0, c1 - c0, min_side) * (1 + 2 * pad)
    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
    a0, a1 = int(cy - side / 2), int(cy + side / 2)
    b0, b1 = int(cx - side / 2), int(cx + side / 2)
    out = np.zeros((a1 - a0, b1 - b0, img.shape[2]), np.float32)
    y0, y1 = max(a0, 0), min(a1, height)
    x0, x1 = max(b0, 0), min(b1, width)
    if y1 > y0 and x1 > x0:
        out[y0 - a0 : y1 - a0, x0 - b0 : x1 - b0] = img[y0:y1, x0:x1]
    return out


def _rasterize_polys(
    polys: Sequence[BaseGeometry], shape: tuple[int, int], transform: Affine
) -> np.ndarray:
    """Маска полигонов на сетке растра (замена rasterize_polys из utils ноутбука)."""
    if not polys:
        return np.zeros(shape, np.uint8)
    return rasterize([(p, 1) for p in polys], out_shape=shape, transform=transform, fill=0)


def candidate_crops(
    field: EtGrField, polys: Sequence[BaseGeometry], cfg: Mapping[str, object]
) -> np.ndarray:
    """Окна кандидатов для CNN: (N, 4, crop_px, crop_px) uint8 (ноутбук 02, ячейка 23).

    Каналы: RGB карты и маска кандидата; окно квадратное, приводится билинейно
    к crop_px.
    """
    import torch
    import torch.nn.functional as F

    crop = int(cfg["crop_px"])
    img = field.rgb.transpose(1, 2, 0).astype(np.float32) / 255.0
    inv = ~field.transform
    out = np.zeros((len(polys), 4, crop, crop), np.uint8)
    for i, p in enumerate(polys):
        minx, miny, maxx, maxy = p.bounds
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        box = (int(r0), int(r1) + 1, int(c0), int(c1) + 1)
        m = _rasterize_polys([p], field.z.shape, field.transform).astype(np.float32)[..., None]
        x = np.concatenate([crop_box(img, *box), crop_box(m, *box)], 2)
        x = F.interpolate(
            torch.from_numpy(x).permute(2, 0, 1)[None],
            size=(crop, crop),
            mode="bilinear",
            align_corners=False,
        )[0]
        out[i] = (x.numpy() * 255).round().astype(np.uint8)
    return out


def make_cnn(arch: str, pretrained: bool) -> "nn.Module":
    """Сконструировать CNN классификатор окон (timm): 4 входных канала, 1 логит.

    pretrained=True скачивает веса (только обучение, машина с интернетом);
    детектор создаёт модель с False и грузит state_dict из файла весов.
    """
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=1, in_chans=4)


def predict_cnn(net: "nn.Module", x: np.ndarray, bs: int = 512) -> np.ndarray:
    """Вероятности CNN по окнам: нормировка + среднее по 4 отражениям/поворотам.

    Ноутбук 02, ячейка 23 (predict_cnn); устройство берётся из модели.
    """
    import torch

    device = next(net.parameters()).device
    mean = torch.tensor(NORM_MEAN).view(1, 4, 1, 1).to(device)
    std = torch.tensor(NORM_STD).view(1, 4, 1, 1).to(device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = (torch.from_numpy(x[i : i + bs]).to(device).float() / 255.0 - mean) / std
            views = [xb, xb.flip(3), xb.flip(2), torch.rot90(xb, 1, (2, 3))]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                probs = torch.stack(
                    [torch.sigmoid(net(v).squeeze(1).float()) for v in views]
                ).mean(0)
            out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def nms(
    polys: Sequence[BaseGeometry], scores: np.ndarray, thr: float, cfg: Mapping[str, object]
) -> list[int]:
    """Индексы принятых кандидатов (ноутбук 02, ячейка 23).

    Порядок по убыванию score; подавление при IoU >= nms_iou или пересечении
    >= 0.5 площади меньшего полигона.
    """
    keep: list[int] = []
    for i in np.argsort(-np.asarray(scores)):
        if scores[i] < thr:
            break
        p = polys[i]
        ok = True
        for k in keep:
            inter = p.intersection(polys[k]).area
            if inter / max(p.union(polys[k]).area, 1e-12) >= float(
                cfg["nms_iou"]
            ) or inter >= 0.5 * min(p.area, polys[k].area):
                ok = False
                break
        if ok:
            keep.append(int(i))
    return keep


def predict_polys(
    polys: Sequence[BaseGeometry],
    kids: Sequence[Sequence[int]],
    scores: np.ndarray,
    thr: float,
    cfg: Mapping[str, object],
) -> tuple[list[BaseGeometry], list[float]]:
    """Итоговые полигоны карты: NMS, затем разрезание пятен на детей (ноутбук 02).

    Слившееся пятно заменяется вложенными кандидатами, если их не меньше двух
    со score >= split_ratio * thr (ноутбук 02, predict_polys, только для карт).
    Возвращает (полигоны, confidence).
    """
    keep = nms(polys, scores, thr, cfg)
    out: list[int] = []
    for i in keep:
        children = [j for j in kids[i] if scores[j] >= float(cfg["split_ratio"]) * thr]
        out += children if len(children) >= 2 else [i]
    keep = sorted(set(out))
    return [polys[i] for i in keep], [float(scores[i]) for i in keep]
