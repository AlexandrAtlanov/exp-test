"""Автоподбор параметров постобработки на holdout — детекторы запускаются один раз.

Детекторы прогоняются один раз с низким порогом модели (config.YOLO["conf"]), сырые
детекции сохраняются в --raw. Дальше для каждой точки сетки — только postprocess и метрика
организаторов (секунды), детекторы не перезапускаются.

Режимы (--search):
- thr  — порог уверенности по классам лидара (сетка --grid);
- full — то же плюс минимальная площадь по классам (множители --area-mult к текущему
  MIN_AREA_M2) и параметры NMS (--nms-iou × --nms-cover, общие для всех классов).

Почему полный перебор, а не Optuna: F1 классов независимы (пороги, площадь и NMS — внутри
пары (class_name, markup_type)), поэтому одна оценка на точку сетки даёт F1 сразу всех
классов, а лучшее значение для каждого класса выбирается отдельно. Сетка маленькая (сотни
точек) — перебор находит точный оптимум, воспроизводим и не требует новых библиотек.

    python tools/tune_thresholds.py --input work/holdout --gt work/gt.geojson \
        --raw work/raw.pkl [--search full]

Повторный запуск с тем же --raw детекторы не запускает (удалите файл после смены весов или
настроек детектора, например TTA). Пороги геофизики не трогаются: их уже применил детектор.

--apply: лучшие параметры записываются в submission/weights/postprocess.json (koz3/tuning.py),
решение применяет их при старте, архив (build_zip) берёт файл вместе с весами — config.py
руками не правится. Файл пишется, только если прирост на holdout не меньше --min-gain
(по умолчанию 0.005: меньший прирост — подгонка под 2–3 региона). Подбор всегда идёт от
значений config.py, а не от прошлого postprocess.json.
"""

import argparse
import copy
import itertools
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_SUBMISSION_DIR = Path(__file__).resolve().parents[1] / "submission"
if str(_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_DIR))

import evaluate  # noqa: E402
import solution  # noqa: E402
from koz3 import config, postprocess, tuning  # noqa: E402
from koz3.datatypes import Detection, Region  # noqa: E402
from koz3.discovery import find_regions  # noqa: E402
from koz3.timing import Budget  # noqa: E402

logger = logging.getLogger("tune_thresholds")

GRID: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
"""Сетка порогов уверенности."""

AREA_MULT: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
"""Множители к текущему MIN_AREA_M2 класса (режим full)."""

NMS_IOU_GRID: tuple[float, ...] = (0.3, 0.5, 0.7)
NMS_COVER_GRID: tuple[float, ...] = (0.5, 0.7, 0.9)
"""Сетка параметров NMS (режим full)."""

TUNED_MARKUP = "li"
"""Подбираются параметры детекций этого типа данных (лидар)."""


@dataclass
class Params:
    """Точка поиска: пороги и минимальные площади по классам, параметры NMS."""

    conf: dict[str, float] = field(default_factory=dict)
    min_area: dict[str, float] = field(default_factory=dict)
    nms_iou: float | None = None
    nms_cover: float | None = None


def collect(input_dir: Path) -> tuple[list[Detection], dict[str, Region]]:
    """Сырые детекции всех включённых детекторов по регионам (без postprocess)."""
    regions = find_regions(input_dir)
    detections: list[Detection] = []
    for detector in solution._load_detectors():
        name = detector.__name__.rsplit(".", 1)[-1]
        for region in regions:
            if not detector.is_available(region):
                continue
            found = detector.run(region, Budget(10**9), config)
            logger.info("%s × %s: сырых детекций %d", name, region.name, len(found))
            detections.extend(found)
    return detections, {region.name: region for region in regions}


def score(
    detections: list[Detection], regions: dict[str, Region], gt: dict, metric, params: Params
) -> dict:
    """Метрика организаторов после postprocess с параметрами params (config восстанавливается)."""
    saved = (
        dict(config.CONF_THRESHOLD),
        dict(config.MIN_AREA_M2),
        config.NMS_IOU,
        config.NMS_COVER,
    )
    try:
        config.CONF_THRESHOLD.update({(c, TUNED_MARKUP): t for c, t in params.conf.items()})
        config.MIN_AREA_M2.update(params.min_area)
        if params.nms_iou is not None:
            config.NMS_IOU = params.nms_iou
        if params.nms_cover is not None:
            config.NMS_COVER = params.nms_cover
        kept = postprocess.run(list(detections))
    finally:
        config.CONF_THRESHOLD.clear()
        config.CONF_THRESHOLD.update(saved[0])
        config.MIN_AREA_M2.clear()
        config.MIN_AREA_M2.update(saved[1])
        config.NMS_IOU, config.NMS_COVER = saved[2], saved[3]
    pred = {"type": "FeatureCollection", "features": solution._to_features(kept, regions)}
    return evaluate.calculate(metric, pred, copy.deepcopy(gt))


def _f1(result: dict, cls: str) -> float:
    return float(result["per_class"].get(cls, {}).get("f1", 0.0))


def tuned_classes(detections: list[Detection]) -> list[str]:
    """Классы лидара, для которых есть детекции."""
    return sorted({d.class_name for d in detections if d.markup_type == TUNED_MARKUP})


def best_per_class(
    detections, regions, gt, metric, grid, area_mult, nms_iou=None, nms_cover=None
) -> tuple[Params, dict]:
    """Перебор (порог × множитель площади) при заданном NMS; лучшее значение по каждому классу.

    Одна оценка на точку сетки — все классы сразу (F1 классов независимы).
    Возвращает (лучшие параметры, результат метрики с ними).
    """
    classes = tuned_classes(detections)
    best: dict[str, tuple[float, float, float]] = {}  # класс → (F1, порог, площадь)
    for thr, mult in itertools.product(grid, area_mult):
        params = Params(
            conf={c: thr for c in classes},
            min_area={c: config.MIN_AREA_M2[c] * mult for c in classes},
            nms_iou=nms_iou,
            nms_cover=nms_cover,
        )
        result = score(detections, regions, gt, metric, params)
        for cls in classes:
            candidate = (_f1(result, cls), thr, config.MIN_AREA_M2[cls] * mult)
            # при равном F1 — более высокий порог и площадь (меньше FP вне holdout)
            if cls not in best or candidate > best[cls]:
                best[cls] = candidate
    params = Params(
        conf={c: v[1] for c, v in best.items()},
        min_area={c: v[2] for c, v in best.items()},
        nms_iou=nms_iou,
        nms_cover=nms_cover,
    )
    return params, score(detections, regions, gt, metric, params)


def search(
    detections,
    regions,
    gt,
    metric,
    mode: str = "thr",
    grid: tuple[float, ...] = GRID,
    area_mult: tuple[float, ...] = AREA_MULT,
    nms_iou_grid: tuple[float, ...] = NMS_IOU_GRID,
    nms_cover_grid: tuple[float, ...] = NMS_COVER_GRID,
) -> tuple[Params, dict]:
    """Полный перебор: thr — только пороги; full — пороги, площади и NMS."""
    if mode == "thr":
        return best_per_class(detections, regions, gt, metric, grid, (1.0,))
    best: tuple[Params, dict] | None = None
    for nms_iou, nms_cover in itertools.product(nms_iou_grid, nms_cover_grid):
        started = time.monotonic()
        params, result = best_per_class(
            detections, regions, gt, metric, grid, area_mult, nms_iou, nms_cover
        )
        logger.info(
            "NMS iou %.2f cover %.2f: final %.4f (%.0f с)",
            nms_iou,
            nms_cover,
            result["final_score"],
            time.monotonic() - started,
        )
        if best is None or result["final_score"] > best[1]["final_score"]:
            best = (params, result)
    if best is None:
        raise ValueError("пустая сетка NMS")
    return best


def config_block(params: Params, mode: str) -> str:
    """Текст для вставки в submission/koz3/config.py."""
    lines = ["# CONF_THRESHOLD — добавить пары:"]
    lines += [f'    ("{c}", "{TUNED_MARKUP}"): {t},' for c, t in sorted(params.conf.items())]
    if mode == "full":
        lines.append("# MIN_AREA_M2 — заменить значения:")
        lines += [f'    "{c}": {a:g},' for c, a in sorted(params.min_area.items())]
        lines.append(f"NMS_IOU: float = {params.nms_iou}")
        lines.append(f"NMS_COVER: float = {params.nms_cover}")
    return "\n".join(lines)


def load_or_collect(input_dir: Path, raw: Path | None) -> tuple[list[Detection], dict]:
    """Сырые детекции из кэша --raw или запуском детекторов (и сохранение в кэш)."""
    if raw is not None and raw.is_file():
        with raw.open("rb") as handle:
            detections, regions = pickle.load(handle)
        logger.info("сырые детекции из %s: %d", raw, len(detections))
        return detections, regions
    detections, regions = collect(input_dir)
    if raw is not None:
        with raw.open("wb") as handle:
            pickle.dump((detections, regions), handle)
    return detections, regions


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Автоподбор постобработки на holdout")
    parser.add_argument("--input", type=Path, required=True, help="holdout (make_holdout.py)")
    parser.add_argument("--gt", type=Path, required=True, help="GT (make_gt.py)")
    parser.add_argument("--raw", type=Path, help="кэш сырых детекций (.pkl)")
    parser.add_argument("--search", choices=("thr", "full"), default="thr")
    parser.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    parser.add_argument("--area-mult", type=float, nargs="+", default=list(AREA_MULT))
    parser.add_argument("--nms-iou", type=float, nargs="+", default=list(NMS_IOU_GRID))
    parser.add_argument("--nms-cover", type=float, nargs="+", default=list(NMS_COVER_GRID))
    parser.add_argument(
        "--apply", action="store_true", help="записать лучшие параметры в weights/postprocess.json"
    )
    parser.add_argument("--min-gain", type=float, default=0.005, help="минимальный прирост")
    parser.add_argument(
        "--metric",
        type=Path,
        default=Path(__file__).resolve().parent / "compute_metrics.py",
        help="метрика организаторов",
    )
    args = parser.parse_args(argv)
    detections, regions = load_or_collect(args.input, args.raw)
    if not detections:
        logger.error("детекций нет — проверьте веса в submission/weights и вход --input")
        return 1
    metric = evaluate.load_metric(args.metric)
    gt = evaluate._load_geojson(args.gt)
    current = score(detections, regions, gt, metric, Params())
    params, result = search(
        detections,
        regions,
        gt,
        metric,
        args.search,
        tuple(args.grid),
        tuple(args.area_mult),
        tuple(args.nms_iou),
        tuple(args.nms_cover),
    )
    print("\nкласс                     сейчас  с подбором  порог  мин.площадь")
    for cls in tuned_classes(detections):
        area = params.min_area.get(cls, config.MIN_AREA_M2[cls])
        print(
            f"{cls:<25} {_f1(current, cls):6.3f}  {_f1(result, cls):10.3f}  "
            f"{params.conf[cls]:5.2f}  {area:g}"
        )
    print(
        f"\nfinal_score: сейчас {current['final_score']:.4f} → "
        f"с подбором {result['final_score']:.4f}"
    )
    print("\n" + config_block(params, args.search))
    gain = result["final_score"] - current["final_score"]
    if args.apply:
        if gain < args.min_gain:
            print(f"\n--apply: прирост {gain:.4f} < {args.min_gain} — файл не записан")
        else:
            path = tuning.write_tuned(
                {(c, TUNED_MARKUP): t for c, t in params.conf.items()},
                params.min_area if args.search == "full" else {},
                params.nms_iou,
                params.nms_cover,
                {
                    "search": args.search,
                    "holdout_final_score": round(result["final_score"], 6),
                    "config_final_score": round(current["final_score"], 6),
                },
            )
            print(f"\n--apply: записано {path} — решение применит его при запуске")
    return 0


if __name__ == "__main__":
    sys.exit(main())
