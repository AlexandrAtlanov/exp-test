"""Вариант отмывки «r» (C23): визуализация рельефа, принятая в археологическом лидаре.

Три канала uint8 из того же DEM, что и отмывки бейзлайна (Relief Visualization Toolbox,
Kokalj и др.: на них курганы и рвы контрастнее, чем на отмывке с одним источником света):
  1 — локальный рельеф (LRM): DEM − гаусс(DEM, σ = lrm_sigma_m), клип ±lrm_clip_m;
  2 — уклон, градусы, клип 0..slope_max_deg;
  3 — Sky-View Factor (упрощённо): доля открытого неба по svf_dirs направлениям в радиусе
      svf_radius_m, клип svf_min..1 (ровное место — светлое, яма/ров — темнее).
Гладкие части (гаусс LRM, SVF) считаются на грубой сетке work_res_m и интерполируются
обратно: на 0.1 м они иначе дороже всех отмывок вместе. Вне данных — (0, 0, 0), как у
цветных отмывок; внутри данных каналы в 1..254, чтобы valid_pixel_mask не принял пиксель
за nodata. Параметры — config.LIDAR["relief"]; пишется по блокам строк (память).
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from affine import Affine
from scipy import ndimage

DEFAULTS: dict[str, float] = {
    "work_res_m": 0.5,
    "lrm_sigma_m": 10.0,
    "lrm_clip_m": 1.0,
    "slope_max_deg": 45.0,
    "svf_radius_m": 10.0,
    "svf_dirs": 8,
    "svf_min": 0.6,
}
"""Параметры по умолчанию (перекрываются config.LIDAR["relief"])."""


def _coarse(filled: np.ndarray, k: int) -> np.ndarray:
    """Среднее блоками k×k; DEM дополнен краевыми значениями до кратного k размера."""
    height, width = filled.shape
    ph, pw = -height % k, -width % k
    padded = np.pad(filled, ((0, ph), (0, pw)), mode="edge") if ph or pw else filled
    h2, w2 = padded.shape
    return padded.reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3), dtype=np.float64)


def sky_view_factor(z: np.ndarray, step: float, radius_m: float, n_dirs: int) -> np.ndarray:
    """SVF по n_dirs направлениям: 1 − среднее sin(угла горизонта⁺); сдвигами массива."""
    radius_px = max(1, int(round(radius_m / step)))
    pad = radius_px + 1
    zp = np.pad(z, pad, mode="edge")
    height, width = z.shape
    total = np.zeros(z.shape, dtype=np.float32)
    for d in range(n_dirs):
        angle = 2.0 * math.pi * d / n_dirs
        dy, dx = math.sin(angle), math.cos(angle)
        best = np.zeros(z.shape, dtype=np.float32)  # tan угла горизонта, не ниже 0
        for r in range(1, radius_px + 1):
            oy, ox = int(round(r * dy)), int(round(r * dx))
            dist = math.hypot(oy, ox) * step
            if dist <= 0:
                continue
            shifted = zp[pad + oy : pad + oy + height, pad + ox : pad + ox + width]
            np.maximum(best, (shifted - z) * np.float32(1.0 / dist), out=best)
        total += best / np.sqrt(1.0 + best * best)  # sin(arctan t)
    return 1.0 - total / n_dirs


def _to_u8(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Линейно low..high → 1..254 (0 и 255 — признаки nodata у valid_pixel_mask); float32."""
    scaled = np.asarray(values, dtype=np.float32) - np.float32(low)
    scaled *= np.float32(253.0 / (high - low))
    scaled += np.float32(1.5)  # +0.5 — округление при отбрасывании дробной части
    np.clip(scaled, 1.0, 254.5, out=scaled)
    return scaled.astype(np.uint8)


_SLOPE_LUT_N = 4096
"""Ступеней таблицы уклона: квадрат тангенса 0..tan²(max) → градусы → 1..254."""


def _slope_u8(block: np.ndarray, step: float, max_deg: float) -> np.ndarray:
    """Уклон строк блока (с halo в 1 строку сверху и снизу) → uint8 через таблицу.

    Центральные разности как np.gradient (на краях — односторонние); arctan не считается
    поэлементно: квадрат тангенса квантуется в таблицу из _SLOPE_LUT_N значений.
    """
    b = block.astype(np.float32)
    gy = np.empty_like(b)
    gx = np.empty_like(b)
    gy[1:-1] = b[2:] - b[:-2]
    gy[1:-1] *= np.float32(0.5)
    gy[0], gy[-1] = b[1] - b[0], b[-1] - b[-2]
    gx[:, 1:-1] = b[:, 2:] - b[:, :-2]
    gx[:, 1:-1] *= np.float32(0.5)
    gx[:, 0], gx[:, -1] = b[:, 1] - b[:, 0], b[:, -1] - b[:, -2]
    tan2 = gx * gx
    tan2 += gy * gy
    tan2 *= np.float32(1.0 / step**2)
    t2_max = math.tan(math.radians(max_deg)) ** 2
    idx = np.minimum(tan2 * np.float32((_SLOPE_LUT_N - 1) / t2_max), _SLOPE_LUT_N - 1)
    deg = np.degrees(np.arctan(np.sqrt(np.linspace(0.0, t2_max, _SLOPE_LUT_N))))
    lut = _to_u8(deg, 0.0, max_deg)
    return lut[idx.astype(np.int32)]


def _upsample_rows(coarse: np.ndarray, k: int, top: int, bottom: int, width: int) -> np.ndarray:
    """Строки top..bottom полного разрешения из грубой сетки (билинейно, как cv2.resize ×k)."""
    c0 = max(top // k - 1, 0)
    c1 = min(-(-bottom // k) + 1, coarse.shape[0])
    part = cv2.resize(
        coarse[c0:c1].astype(np.float32),
        (coarse.shape[1] * k, (c1 - c0) * k),
        interpolation=cv2.INTER_LINEAR,
    )
    return part[top - c0 * k : bottom - c0 * k, :width]


def relief_params(overrides: Mapping[str, Any] | None) -> dict[str, float]:
    """DEFAULTS, перекрытые config.LIDAR["relief"]."""
    return {**DEFAULTS, **dict(overrides or {})}


def write_relief(
    path: Path,
    filled: np.ndarray,
    valid: np.ndarray,
    step: float,
    transform: Affine,
    crs,
    params: Mapping[str, Any] | None = None,
    block_rows: int = 2048,
) -> Path:
    """Трёхканальный tif «r»: LRM, уклон, SVF (см. модуль); пишется по блокам строк."""
    p = relief_params(params)
    height, width = filled.shape
    k = max(1, int(round(float(p["work_res_m"]) / step)))
    work_step = step * k
    # сглаживание только по данным (нормированная свёртка): заливка z_min вне съёмки
    # иначе тянет гаусс вниз, и вдоль края данных появляется ложный «вал» в LRM
    num = _coarse(filled.astype(np.float32) * valid, k)
    den = _coarse(valid.astype(np.float32), k)
    sigma = float(p["lrm_sigma_m"]) / work_step
    smooth = ndimage.gaussian_filter(num, sigma, mode="nearest") / np.maximum(
        ndimage.gaussian_filter(den, sigma, mode="nearest"), 1e-6
    )
    coarse = np.where(den > 0, num / np.maximum(den, 1e-12), smooth).astype(np.float32)
    svf = sky_view_factor(coarse, work_step, float(p["svf_radius_m"]), int(p["svf_dirs"]))
    lrm_clip = float(p["lrm_clip_m"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    )  # без сжатия, как отмывки бейзлайна (lidar_dem._open_writer): файл временный
    with rasterio.open(path, "w", **profile) as dst:
        for top in range(0, height, block_rows):
            bottom = min(top + block_rows, height)
            lo, hi = max(0, top - 1), min(height, bottom + 1)  # halo для градиента
            slope = _slope_u8(filled[lo:hi], step, float(p["slope_max_deg"]))
            lrm = filled[top:bottom].astype(np.float32)
            lrm -= _upsample_rows(smooth, k, top, bottom, width)
            svf_rows = _upsample_rows(svf, k, top, bottom, width)
            rgb = np.stack(
                [
                    _to_u8(lrm, -lrm_clip, lrm_clip),
                    slope[top - lo : top - lo + bottom - top],
                    _to_u8(svf_rows, float(p["svf_min"]), 1.0),
                ]
            )
            rgb *= valid[top:bottom]  # вне данных — 0 (быстрее булевой индексации)
            dst.write(rgb, window=((top, bottom), (0, width)))
    return path
