"""Метрика платформы и метки попаданий для обучения.

Дословный перенос формул из общего кода бейзлайнов (utils): жадное
сопоставление по убыванию confidence один к одному, Precision со
штрафом FP_WEIGHT, радиус точки R_POINT, покрытие COVERAGE от меньшей
фигуры; ``hit_flags`` — метки кандидатов из ноутбука MagGr.
"""

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

R_POINT: float = 2.0
"""Точка засчитана, если центр предсказания ближе, м (метрика платформы)."""
COVERAGE: float = 0.5
"""Полигон засчитан при перекрытии от меньшей фигуры."""
FP_WEIGHT: float = 0.5
"""Вес лишнего срабатывания в Precision."""


def platform_prf(tp: int, fp: int, fn: int, eps: float = 1e-6) -> tuple[float, float, float]:
    """(precision, recall, F1) платформы: FP со штрафом FP_WEIGHT."""
    p = tp / (tp + FP_WEIGHT * fp + eps)
    r = tp / (tp + fn + eps)
    return p, r, 2 * p * r / (p + r + eps)


def point_counts(xy: np.ndarray, confs: np.ndarray, gxy: np.ndarray) -> tuple[int, int, int]:
    """TP/FP/FN точек: жадно по убыванию confidence, ближайшая свободная точка в R_POINT."""
    if len(xy) == 0 or len(gxy) == 0:
        return 0, len(xy), len(gxy)
    tree = cKDTree(gxy)
    used: set[int] = set()
    tp = 0
    for i in np.argsort(-np.asarray(confs), kind="stable"):
        best, best_d = -1, np.inf
        for j in tree.query_ball_point(xy[i], R_POINT):
            dist = np.hypot(*(xy[i] - gxy[j]))
            if j not in used and dist < best_d - 1e-12:
                best, best_d = j, dist
        if best >= 0:
            used.add(best)
            tp += 1
    return tp, len(xy) - tp, len(gxy) - tp


def largest_part(g: BaseGeometry) -> BaseGeometry:
    """Наибольшая часть MultiPolygon (для метрики платформы)."""
    return max(g.geoms, key=lambda p: p.area) if isinstance(g, MultiPolygon) else g


def is_hit(p: BaseGeometry, g: BaseGeometry) -> tuple[bool, float]:
    """Полигон засчитан: покрытие >= COVERAGE от меньшей фигуры и rep-точка внутри."""
    m = min(p.area, g.area)
    if m <= 0:
        return False, 0.0
    cov = p.intersection(g).area / m
    pt = p.representative_point()
    return cov >= COVERAGE and (g.contains(pt) or g.boundary.distance(pt) < 1e-10), cov


def platform_counts(
    preds: Sequence[BaseGeometry], confs: Sequence[float], gts: Sequence[BaseGeometry]
) -> tuple[int, int, int]:
    """TP/FP/FN полигонов: жадное сопоставление по убыванию confidence."""
    preds = [largest_part(p) for p in preds]
    gts = [largest_part(g) for g in gts]
    if not preds or not gts:
        return 0, len(preds), len(gts)
    tree = STRtree(gts)
    used: set[int] = set()
    tp = 0
    for i in sorted(range(len(preds)), key=lambda i: -confs[i]):
        best, best_cov = -1, -np.inf
        for j in sorted(tree.query(preds[i])):
            if j in used:
                continue
            hit, cov = is_hit(preds[i], gts[j])
            if hit and cov > best_cov + 1e-9:
                best, best_cov = j, cov
        if best >= 0:
            used.add(best)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def hit_flags(
    polys: Sequence[BaseGeometry], gts: Sequence[BaseGeometry]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Флаги кандидатов и объектов разметки + максимальный IoU кандидата.

    Возвращает (cand — кандидат дотягивается до какого-то объекта по
    is_hit, gt — объект кем-то «дотянут», iou — max IoU кандидата).
    """
    cand = np.zeros(len(polys), bool)
    gt = np.zeros(len(gts), bool)
    iou = np.zeros(len(polys))
    if not polys or not gts:
        return cand, gt, iou
    gts_ = [largest_part(g) for g in gts]
    tree = STRtree(gts)
    for i, p in enumerate(polys):
        for j in tree.query(p):
            iou[i] = max(iou[i], p.intersection(gts[j]).area / max(p.union(gts[j]).area, 1e-12))
            if is_hit(largest_part(p), gts_[j])[0]:
                cand[i] = gt[j] = True
    return cand, gt, iou
