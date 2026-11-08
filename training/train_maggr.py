"""Обучение моделей магниторазведки (MagGr): точки finds_points и полигоны object_poly.

Повторяет ноутбук организаторов 01_baseline_maggr_anomalies (те же метки, модель,
выбор порога по out-of-fold), но признаки считает общим кодом инференса
koz3.geophys.maggr_features — обучение и детектор видят одно и то же. Маска съёмки
строится только по растру (контура съёмки на платформе нет), как в детекторе.

    python training/train_maggr.py --data <трейн> --out weights_out
    python training/train_maggr.py --data <трейн> --out weights_out --holdout 207

Шаги:
1) регионы трейна с серой картой MagGr (*_серый.tif) и разметкой MagGr
   (FindsPoints — точки, ObjectPoly — полигоны), через training/annotations;
2) кандидаты и признаки; метки: точка — ближе MAGGR["pt_label_m"] к точке разметки,
   полигон — засчитан метрикой платформы и IoU ≥ MAGGR["label_iou"];
3) leave-one-region-out: OOF-вероятности → порог с лучшим F1 метрики платформы
   (FP с весом 0.5), отдельно для точек и полигонов; печать P/R/F1 по регионам;
4) итоговые модели на всех регионах трейна (кроме --holdout) → <out>/maggr_models.joblib
   в формате, который читает koz3/detectors/maggr.py; для --holdout — честная оценка.

Скопируйте maggr_models.joblib в папку весов архива (рядом с lidar_yolo.pt) и
запишите sha256 в журнал. Версия scikit-learn при обучении должна совпадать с
образом платформы, иначе файл может не загрузиться.
"""

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from sklearn.ensemble import HistGradientBoostingClassifier

import settings
from annotations import find_annotation_files, load_objects
from koz3 import config
from koz3.discovery import find_regions
from koz3.geophys import maggr_features as mf
from matching import hit_flags, platform_counts, platform_prf, point_counts

logger = logging.getLogger("train_maggr")

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
"""Сетка порогов вероятности для выбора по OOF (как в ноутбуке)."""


@dataclass
class Unit:
    """Одна серая карта MagGr с кандидатами, признаками и разметкой региона."""

    region: str
    raster: Path
    pt_xy: np.ndarray  # (N, 2) кандидаты точек, CRS растра
    pt_x: np.ndarray  # (N, 16) признаки
    pt_y: np.ndarray  # (N,) метки
    gt_xy: np.ndarray  # (M, 2) точки разметки
    poly_cands: list[BaseGeometry] = field(default_factory=list)
    poly_x: np.ndarray = field(default_factory=lambda: np.zeros((0, 15), np.float32))
    poly_y: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    gt_polys: list[BaseGeometry] = field(default_factory=list)


def make_model() -> HistGradientBoostingClassifier:
    """Модель ноутбука: градиентный бустинг с балансом классов."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=settings.SEED
    )


def load_markup(region_dir: Path, crs: str) -> tuple[np.ndarray, list[BaseGeometry]]:
    """Разметка MagGr региона в CRS растра: (точки (M, 2), полигоны)."""
    points: list[tuple[float, float]] = []
    polys: list[BaseGeometry] = []
    for ann in find_annotation_files(region_dir):
        if ann.markup_type != "maggr":
            continue
        for class_name, geom in load_objects(ann, crs, settings.ALL_CLASSES):
            if class_name == "finds_points":
                center = geom if isinstance(geom, Point) else geom.centroid
                points.append((center.x, center.y))
            elif class_name == "object_poly" and geom.area > 0:
                polys.append(geom.buffer(0))
    return np.array(points, dtype=np.float64).reshape(-1, 2), polys


def build_unit(region: str, raster: Path, region_dir: Path) -> Unit | None:
    """Кандидаты, признаки и метки одной серой карты; None — нет разметки MagGr."""
    cfg = config.MAGGR
    fld = mf.load_field(raster, cfg)
    gt_xy, gt_polys = load_markup(region_dir, fld.crs.to_string())
    if len(gt_xy) == 0 and not gt_polys:
        logger.warning("%s: %s — нет разметки MagGr, пропуск", region, raster.name)
        return None
    pt_xy, pt_x = mf.point_candidates(fld, cfg)
    if len(gt_xy) and len(pt_xy):
        dist = cKDTree(gt_xy).query(pt_xy)[0]
    else:
        dist = np.full(len(pt_xy), np.inf)
    poly_cands, poly_x = mf.poly_candidates(fld, cfg)
    hit, _, iou = hit_flags(poly_cands, gt_polys)
    unit = Unit(
        region=region,
        raster=raster,
        pt_xy=pt_xy,
        pt_x=pt_x,
        pt_y=dist <= float(cfg["pt_label_m"]),
        gt_xy=gt_xy,
        poly_cands=poly_cands,
        poly_x=poly_x,
        poly_y=hit & (iou >= float(cfg["label_iou"])),
        gt_polys=gt_polys,
    )
    logger.info(
        "%s: точек %d (кандидатов %d, полож. %d); полигонов %d (кандидатов %d, полож. %d)",
        region,
        len(gt_xy),
        len(pt_xy),
        int(unit.pt_y.sum()),
        len(gt_polys),
        len(poly_cands),
        int(unit.poly_y.sum()),
    )
    return unit


def collect_units(data_dir: Path) -> list[Unit]:
    """Все серые карты MagGr трейна с разметкой."""
    units: list[Unit] = []
    for region in find_regions(data_dir):
        for raster in mf.gray_raster_paths(region.files.get("maggr", []), config.MAGGR):
            started = time.monotonic()
            unit = build_unit(region.name.split("_", 1)[0], raster, region.path)
            if unit is not None:
                units.append(unit)
                logger.info("  %s: %.1f с", raster.name, time.monotonic() - started)
    return units


def fit(x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier | None:
    """Обучить модель; None, если в метках один класс (обучать нечему)."""
    if len(np.unique(y)) < 2:
        return None
    return make_model().fit(x, y.astype(int))


def proba(model: HistGradientBoostingClassifier | None, x: np.ndarray) -> np.ndarray:
    """Вероятность положительного класса; без модели — нули."""
    if model is None or len(x) == 0:
        return np.zeros(len(x))
    return model.predict_proba(x)[:, 1]


def eval_points(units: list[Unit], scores: list[np.ndarray], thr: float) -> tuple[int, int, int]:
    """TP/FP/FN точек по метрике платформы после NMS (сумма по картам)."""
    total = np.zeros(3, int)
    for unit, score in zip(units, scores, strict=True):
        keep = mf.nms_points(unit.pt_xy, score, thr, config.MAGGR)
        total += point_counts(unit.pt_xy[keep], score[keep], unit.gt_xy)
    return int(total[0]), int(total[1]), int(total[2])


def eval_polys(units: list[Unit], scores: list[np.ndarray], thr: float) -> tuple[int, int, int]:
    """TP/FP/FN полигонов по метрике платформы после NMS (объединение карт, как в ноутбуке)."""
    preds: list[BaseGeometry] = []
    confs: list[float] = []
    gts: list[BaseGeometry] = []
    for unit, score in zip(units, scores, strict=True):
        keep = mf.nms_polys(unit.poly_cands, score, thr, config.MAGGR)
        preds += [unit.poly_cands[i] for i in keep]
        confs += [float(score[i]) for i in keep]
        gts += unit.gt_polys
    return platform_counts(preds, confs, gts)


def oof_scores(units: list[Unit], kind: str) -> list[np.ndarray]:
    """Leave-one-region-out: вероятности каждой карты от модели без её региона."""
    scores = []
    for unit in units:
        rest = [u for u in units if u.region != unit.region]
        x = np.concatenate([getattr(u, f"{kind}_x") for u in rest])
        y = np.concatenate([getattr(u, f"{kind}_y") for u in rest])
        scores.append(proba(fit(x, y), getattr(unit, f"{kind}_x")))
    return scores


def choose_threshold(units, scores, evaluate) -> tuple[float, tuple[float, float, float]]:
    """Порог с лучшим F1 платформы по OOF; (порог, (P, R, F1))."""
    best = max(THRESHOLDS, key=lambda t: platform_prf(*evaluate(units, scores, t))[2])
    return float(best), platform_prf(*evaluate(units, scores, best))


def report(name: str, units, scores, evaluate, thr: float) -> None:
    """Печать P/R/F1 по регионам при пороге thr."""
    logger.info("%s: порог %.2f", name, thr)
    for unit, score in zip(units, scores, strict=True):
        p, r, f1 = platform_prf(*evaluate([unit], [score], thr))
        logger.info("  %-6s P=%.3f R=%.3f F1=%.3f", unit.region, p, r, f1)


def train(data_dir: Path, out_dir: Path, holdout: list[str]) -> int:
    """Полный цикл; 0 — файл весов записан, 1 — нечего обучать."""
    units = collect_units(data_dir)
    train_units = [u for u in units if u.region not in holdout]
    test_units = [u for u in units if u.region in holdout]
    if len({u.region for u in train_units}) < 2:
        logger.error("Нужно минимум 2 региона трейна с MagGr, найдено: %d", len(train_units))
        return 1
    result: dict[str, object] = {}
    for kind, evaluate, label in (
        ("pt", eval_points, "точки"),
        ("poly", eval_polys, "полигоны"),
    ):
        scores = oof_scores(train_units, kind)
        thr, (p, r, f1) = choose_threshold(train_units, scores, evaluate)
        logger.info("OOF %s: порог %.2f P=%.3f R=%.3f F1=%.3f", label, thr, p, r, f1)
        report(f"OOF {label}", train_units, scores, evaluate, thr)
        x = np.concatenate([getattr(u, f"{kind}_x") for u in train_units])
        y = np.concatenate([getattr(u, f"{kind}_y") for u in train_units])
        model = fit(x, y)
        if model is None:
            logger.error("%s: в метках один класс — модель не обучена", label)
            return 1
        if test_units:
            test_scores = [proba(model, getattr(u, f"{kind}_x")) for u in test_units]
            p, r, f1 = platform_prf(*evaluate(test_units, test_scores, thr))
            logger.info("HOLDOUT %s: P=%.3f R=%.3f F1=%.3f", label, p, r, f1)
        key = "points" if kind == "pt" else "polys"
        result[key] = model
        result[f"thr_{key}"] = thr
    result["feature_names"] = {
        "points": list(mf.POINT_FEATURE_NAMES),
        "polys": list(mf.POLY_FEATURE_NAMES),
    }
    result["sklearn"] = metadata.version("scikit-learn")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / str(config.MAGGR["models_file"])
    joblib.dump(result, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    logger.info("Записано %s (sklearn %s), sha256 %s", out, result["sklearn"], digest)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Обучение моделей MagGr (точки и полигоны)")
    parser.add_argument("--data", type=Path, required=True, help="папка трейна с регионами")
    parser.add_argument("--out", type=Path, required=True, help="папка для maggr_models.joblib")
    parser.add_argument(
        "--holdout",
        nargs="*",
        default=[],
        help="номера регионов (207 …), не входящих в обучение — для честной оценки",
    )
    args = parser.parse_args(argv)
    return train(args.data, args.out, list(args.holdout))


if __name__ == "__main__":
    sys.exit(main())
