"""Обучение CNN классификатора кандидатов карт EtGr (object_poly).

Повторяет ноутбук организаторов 02_baseline_etgr_anomalies для карт-срезов
(те же кандидаты, метки, CNN, выбор порога по out-of-fold), но признаки и
окна считает общий код инференса koz3.geophys.etgr_features — обучение и
детектор видят одно и то же. Профили не затрагиваются (отдельный пайплайн).
Маска съёмки строится только по растру, как в детекторе.

    python training/train_etgr.py --data <трейн> --out weights_out
    python training/train_etgr.py --data <трейн> --out weights_out --holdout 216

Шаги:
1) регионы трейна с картами EtGr (без «_профиль_» в имени) и разметкой EtGr —
   файл <stem>_ObjectPoly (ноутбук 02, ячейка 3), через training/annotations;
   класс object_poly, геометрии в CRS растра;
2) кандидаты и признаки; метка — засчитан метрикой платформы и IoU с разметкой
   >= ETGR["label_iou"] (ноутбук, LABEL_IOU);
3) leave-one-region-out: OOF-вероятности ансамбля CNN → порог с лучшим F1
   метрики платформы (FP с весом 0.5); печать P/R/F1 по регионам;
4) итоговый ансамбль на всех регионах трейна (кроме --holdout) →
   <out>/etgr_maps.pt: state_dicts (без объекта модели, грузится с
   weights_only=True), порог, параметры окна, имена признаков, версии.

Скопируйте etgr_maps.pt в папку весов архива (рядом с lidar_yolo.pt и
maggr_models.joblib) и запишите sha256 в журнал. Версия timm при обучении
должна совпадать с образом платформы (state_dict переносим, но загрузку
проверить в образе); torch может быть новее — как в training/train.py.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry.base import BaseGeometry

import settings
from annotations import find_annotation_files, load_objects
from koz3 import config
from koz3.discovery import find_regions
from koz3.geophys import etgr_features as ef
from matching import hit_flags, platform_counts, platform_prf

logger = logging.getLogger("train_etgr")

THRESHOLDS = np.round(np.arange(0.05, 0.91, 0.05), 2)
"""Сетка порогов вероятности для выбора по OOF (ноутбук 02, ячейка 23)."""

ARCH = "resnet18"
"""Архитектура CNN (timm): 4 входных канала, 1 логит (ноутбук 02, ячейка 23)."""

CNN_EPOCHS = 8
"""Число эпох обучения CNN (ноутбук 02, ячейка 2)."""

CNN_SEEDS = (0, 1, 2)
"""Сида ансамбля CNN (ноутбук 02, ячейка 2)."""

CNN_LR = 3e-4
CNN_BS = 128
CNN_WD = 1e-4
"""Скорость обучения, размер батча, weight decay (ноутбук 02, ячейка 23)."""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Устройство обучения (ноутбук 02, ячейка 2)."""


@dataclass
class Unit:
    """Одна карта EtGr с кандидатами, признаками, метками и окнами CNN."""

    region: str
    raster: Path
    cand_polys: list[BaseGeometry]
    cand_x: np.ndarray  # (N, 16) признаки
    cand_y: np.ndarray  # (N,) bool — метки
    kids: list[list[int]]  # вложенные кандидаты (для разрезания пятен)
    crops: np.ndarray  # (N, 4, crop_px, crop_px) uint8 — окна CNN
    gt_polys: list[BaseGeometry]  # разметка ObjectPoly в CRS растра


def _determinism() -> None:
    """Детерминизм torch, как в ноутбуке (ячейка 2); вызывается до обучения."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def augment(x: torch.Tensor) -> torch.Tensor:
    """Случайные отражения и поворот на 90° (ноутбук 02, ячейка 23)."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


def train_cnn(x: np.ndarray, y: np.ndarray, seed: int) -> torch.nn.Module:
    """Обучить один CNN на окнах кандидатов (ноутбук 02, ячейка 23).

    Сбалансированные батчи (половина положительных), AdamW + OneCycle,
    bf16-autocast. Если положительных меньше половины батча — добор с
    повторением (ноутбук здесь упал бы).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ef.make_cnn(ARCH, pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=CNN_LR, weight_decay=CNN_WD)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    steps = CNN_EPOCHS * max(1, len(neg) // CNN_BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CNN_LR, total_steps=steps)
    mean = torch.tensor(ef.NORM_MEAN).view(1, 4, 1, 1).to(DEVICE)
    std = torch.tensor(ef.NORM_STD).view(1, 4, 1, 1).to(DEVICE)
    half = CNN_BS // 2
    net.train()
    for _ in range(steps):
        pos_idx = np.random.choice(pos, half, replace=len(pos) < half)
        neg_idx = np.random.choice(neg, half, replace=len(neg) < half)
        idx = np.r_[pos_idx, neg_idx]
        xb = torch.from_numpy(x[idx]).to(DEVICE).float() / 255.0
        xb = torch.cat([augment(xb[i : i + 1]) for i in range(len(idx))])
        yb = torch.from_numpy(y[idx].astype(np.float32)).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(
                net((xb - mean) / std).squeeze(1).float(), yb
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def fit_scorer(units: list[Unit]) -> list[torch.nn.Module] | None:
    """Ансамбль CNN на картах юнитов; None — в метках один класс."""
    y = np.concatenate([u.cand_y for u in units])
    if len(np.unique(y)) < 2:
        logger.warning("В метках один класс — CNN не обучается")
        return None
    x = np.concatenate([u.crops for u in units])
    nets = [train_cnn(x, y, seed) for seed in CNN_SEEDS]
    logger.info("Обучен ансамбль из %d CNN (%d окон)", len(nets), len(x))
    return nets


def predict_scores(nets: list[torch.nn.Module] | None, crops: np.ndarray) -> np.ndarray:
    """Средние вероятности ансамбля; без модели или окон — нули."""
    if nets is None or len(crops) == 0:
        return np.zeros(len(crops))
    return np.mean([ef.predict_cnn(net, crops) for net in nets], axis=0)


def load_markup(region_dir: Path, crs: str, stem: str) -> tuple[list[BaseGeometry], bool]:
    """Разметка ObjectPoly карты с данным stem: (полигоны в CRS, найден ли файл).

    Файл разметки называется <stem>_ObjectPoly (ноутбук 02, ячейка 3); ищется
    среди всех файлов разметки EtGr региона. Пустой файл — found=True без
    полигонов: карта остаётся в обучении отрицательной, как в ноутбуке.
    """
    polys: list[BaseGeometry] = []
    found = False
    for ann in find_annotation_files(region_dir):
        if ann.markup_type != "etgr" or ann.path.stem != f"{stem}_ObjectPoly":
            continue
        found = True
        for class_name, geom in load_objects(ann, crs, settings.ALL_CLASSES):
            if class_name == "object_poly" and geom.area > 0:
                polys.append(geom.buffer(0))
    return polys, found


def build_unit(region: str, raster: Path, region_dir: Path) -> Unit | None:
    """Кандидаты, признаки, метки и окна одной карты; None — нет файла разметки."""
    field = ef.load_field(raster, config.ETGR)
    stem = ef.raster_stem(raster)
    gt_polys, found = load_markup(region_dir, field.crs.to_string(), stem)
    if not found:
        logger.warning(
            "%s: %s — нет файла разметки %s_ObjectPoly, пропуск", region, raster.name, stem
        )
        return None
    cand_polys, cand_x = ef.candidates(field, config.ETGR)
    hit, _, iou = hit_flags(cand_polys, gt_polys)
    unit = Unit(
        region=region,
        raster=raster,
        cand_polys=cand_polys,
        cand_x=cand_x,
        cand_y=hit & (iou >= float(config.ETGR["label_iou"])),
        kids=ef.nested_children(cand_polys, cand_x),
        crops=ef.candidate_crops(field, cand_polys, config.ETGR),
        gt_polys=gt_polys,
    )
    logger.info(
        "%s: %s — полигонов %d, кандидатов %d, положительных %d",
        region,
        raster.name,
        len(gt_polys),
        len(cand_polys),
        int(unit.cand_y.sum()),
    )
    return unit


def collect_units(data_dir: Path) -> list[Unit]:
    """Все карты EtGr трейна; ошибка одной карты не останавливает сбор."""
    units: list[Unit] = []
    for region in find_regions(data_dir):
        for raster in ef.map_raster_paths(region.files.get("etgr", [])):
            started = time.monotonic()
            try:
                unit = build_unit(region.name.split("_", 1)[0], raster, region.path)
            except Exception:
                logger.warning(
                    "%s: %s — ошибка обработки, пропуск", region.name, raster.name, exc_info=True
                )
                continue
            if unit is not None:
                units.append(unit)
                logger.info("  %s: %.1f с", raster.name, time.monotonic() - started)
    return units


def pool_counts(units: list[Unit], scores: list[np.ndarray], thr: float) -> tuple[int, int, int]:
    """TP/FP/FN по метрике платформы: все карты одним пулом (ноутбук 02, pool)."""
    preds: list[BaseGeometry] = []
    confs: list[float] = []
    gts: list[BaseGeometry] = []
    for unit, score in zip(units, scores, strict=True):
        polys, conf = ef.predict_polys(unit.cand_polys, unit.kids, score, thr, config.ETGR)
        preds += polys
        confs += conf
        gts += unit.gt_polys
    return platform_counts(preds, confs, gts)


def choose_threshold(
    units: list[Unit], scores: list[np.ndarray]
) -> tuple[float, tuple[float, float, float]]:
    """Порог из сетки с лучшим F1 платформы по OOF; (порог, (P, R, F1))."""
    curve = [(t, platform_prf(*pool_counts(units, scores, t))) for t in THRESHOLDS]
    thr, (p, r, f1) = max(curve, key=lambda c: c[1][2])
    return float(thr), (p, r, f1)


def oof_scores(units: list[Unit]) -> list[np.ndarray]:
    """OOF-вероятности leave-one-region-out (ноутбук 02, ячейка 23).

    Для каждого региона обучается ансамбль без него; порядок ответа = порядок
    units.
    """
    nets_by_region: dict[str, list[torch.nn.Module] | None] = {}
    for region in sorted({u.region for u in units}):
        started = time.monotonic()
        rest = [u for u in units if u.region != region]
        nets_by_region[region] = fit_scorer(rest)
        logger.info("OOF: ансамбль без региона %s — %.1f с", region, time.monotonic() - started)
    return [predict_scores(nets_by_region[u.region], u.crops) for u in units]


def report(units: list[Unit], scores: list[np.ndarray], thr: float) -> None:
    """P/R/F1 по регионам при пороге thr — строками лога, без LOSO-таблиц."""
    for region in sorted({u.region for u in units}):
        chosen = [u for u in units if u.region == region]
        theirs = [s for u, s in zip(units, scores, strict=True) if u.region == region]
        p, r, f1 = platform_prf(*pool_counts(chosen, theirs, thr))
        logger.info("  %-6s P=%.3f R=%.3f F1=%.3f", region, p, r, f1)


def train(data_dir: Path, out_dir: Path, holdout: list[str]) -> int:
    """Полный цикл обучения; 0 — файл весов записан, 1 — нечего обучать."""
    if DEVICE == "cpu":
        logger.warning("CUDA недоступна — обучение CNN на CPU будет медленным")
    units = collect_units(data_dir)
    train_units = [u for u in units if u.region not in holdout]
    test_units = [u for u in units if u.region in holdout]
    if len({u.region for u in train_units}) < 2:
        logger.error(
            "Нужно минимум 2 региона трейна с картами EtGr, найдено: %d",
            len({u.region for u in train_units}),
        )
        return 1
    scores = oof_scores(train_units)
    thr, (p, r, f1) = choose_threshold(train_units, scores)
    logger.info("OOF карты: порог %.2f P=%.3f R=%.3f F1=%.3f", thr, p, r, f1)
    report(train_units, scores, thr)
    nets = fit_scorer(train_units)
    if nets is None:
        logger.error("карты: в метках один класс — модель не обучена")
        return 1
    if test_units:
        test_scores = [predict_scores(nets, u.crops) for u in test_units]
        p, r, f1 = platform_prf(*pool_counts(test_units, test_scores, thr))
        logger.info("HOLDOUT карты: порог %.2f P=%.3f R=%.3f F1=%.3f", thr, p, r, f1)
    result = {
        "arch": ARCH,
        "state_dicts": [
            {k: v.detach().cpu().float() for k, v in net.state_dict().items()} for net in nets
        ],
        "scorer": None,  # карты скорит CNN-ансамбль; ключ оставлен для профилей
        "thr": thr,
        "crop": {"crop_px": int(config.ETGR["crop_px"]), "pad": 0.5, "min_side": 16},
        "feature_names": list(ef.FEATURE_NAMES),
        "torch": str(torch.__version__),
        "timm": metadata.version("timm"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / str(config.ETGR["models_file"])
    torch.save(result, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    logger.info(
        "Записано %s (torch %s, timm %s), sha256 %s", out, result["torch"], result["timm"], digest
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Обучение CNN карт EtGr (object_poly)")
    parser.add_argument("--data", type=Path, required=True, help="папка трейна с регионами")
    parser.add_argument("--out", type=Path, required=True, help="папка для etgr_maps.pt")
    parser.add_argument(
        "--holdout",
        nargs="*",
        default=[],
        help="номера регионов (216 ...), не входящих в обучение — для честной оценки",
    )
    args = parser.parse_args(argv)
    _determinism()
    return train(args.data, args.out, list(args.holdout))


if __name__ == "__main__":
    sys.exit(main())