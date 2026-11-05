"""Оценка предсказаний метрикой организаторов: итог, таблицы, сравнение.

Обёртка над tools/compute_metrics.py (копия метрики КОЗ-3): модуль
загружается по пути через importlib, код не копируется. Таблицы: классы
(вес, TP/FP/FN, P/R/F1), группы данных, размерные категории GT-полигонов.
--save пишет полный результат в run.json, --compare печатает «было /
стало / Δ» относительно прошлого прогона. Расчёт — F1Calculator с
дефолтами: тот же итог, что у CLI метрики.

Пример: python tools/evaluate.py --pred result.geojson --gt gt.geojson --md
"""

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

_DEFAULT_METRIC = Path(__file__).resolve().parent / "compute_metrics.py"
"""Копия метрики организаторов рядом с этим скриптом (по умолчанию)."""

logger = logging.getLogger("evaluate")


def load_metric(metric_path: Path) -> ModuleType:
    """Загружает модуль метрики организаторов по пути.

    Поднимает RuntimeError, если файла нет, в нём нет ожидаемого API или
    он подхватил сторонний config.py: PLATFORM_CONFIG_LOADED обязан быть
    False (встроенные дефолты равны платформенным).
    """
    if not metric_path.is_file():
        raise RuntimeError(f"файл метрики не найден: {metric_path}")
    spec = importlib.util.spec_from_file_location("compute_metrics", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить модуль из {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "PLATFORM_CONFIG_LOADED", None) is not False:
        raise RuntimeError(
            "PLATFORM_CONFIG_LOADED is not False: метрика подхватила сторонний "
            "config.py из sys.path — итог может отличаться от платформенного"
        )
    for attr in ("F1Calculator", "CLASS_WEIGHTS"):
        if not hasattr(module, attr):
            raise RuntimeError(f"в {metric_path} нет {attr} — это compute_metrics.py?")
    return module


def calculate(metric: ModuleType, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    """Результат calculate_f1 с дефолтами (совпадает с CLI метрики)."""
    return metric.F1Calculator(coverage_threshold=None, r_tolerance=None).calculate_f1(pred, gt)


def _load_geojson(path: Path) -> dict[str, Any]:
    """Читает GeoJSON-файл; OSError/ValueError прокидываются наружу."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_run(path: Path) -> dict[str, Any]:
    """Читает run.json прошлого прогона и валидирует минимальную структуру."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    results = data["results"]
    if not isinstance(results, dict) or "final_score" not in results or "per_class" not in results:
        raise ValueError("нет results.final_score / results.per_class — это run.json из evaluate.py?")
    return data


def _run_metadata(args: argparse.Namespace) -> dict[str, str]:
    """Метаданные прогона для run.json (пути и время UTC)."""
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pred": str(args.pred),
        "gt": str(args.gt),
        "metric": str(args.metric),
    }


def _fmt_opt(value: float | None) -> str:
    """Доля (P/R/recall) или «—», если метрика вернула None."""
    return "—" if value is None else f"{value:.4f}"


def _render(header: list[str], rows: list[list[str]], md: bool) -> str:
    """Таблица: Markdown для PR (--md) или выровненный текст."""
    if md:
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|" + "|".join("---" for _ in header) + "|")
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)
    widths = [len(cell) for cell in header]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def _section(title: str, table: tuple[list[str], list[list[str]]], md: bool) -> str:
    """Заголовок секции + таблица (пустая строка перед заголовком)."""
    header, rows = table
    heading = f"### {title}" if md else f"{title}:"
    return f"\n{heading}\n{_render(header, rows, md)}"


def _class_rows(metric: ModuleType, results: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """Таблица по классам: все 7, включая отсутствующие в GT (пометка †)."""
    header = ["класс", "вес", "TP", "FP", "FN", "P", "R", "F1", "pred", "GT"]
    rows: list[list[str]] = []
    for cls, st in results["per_class"].items():
        absent = st["num_ground_truth"] == 0
        rows.append(
            [
                cls + (" †" if absent else ""),
                str(metric.CLASS_WEIGHTS.get(cls, 0)),
                str(st["tp"]),
                str(st["fp"]),
                str(st["fn"]),
                f"{st['precision']:.4f}",
                f"{st['recall']:.4f}",
                f"{st['f1']:.4f}",
                str(st["num_predictions"]),
                str(st["num_ground_truth"]),
            ]
        )
    return header, rows


def _dtype_rows(results: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """Таблица групп данных (лидар/аэрофото/…); ключи — из auxiliary метрики."""
    by_type: dict[str, dict[str, Any]] = results.get("auxiliary", {}).get("by_data_type", {})
    header = ["группа", "TP", "FP", "FN", "P", "R"]
    rows = [
        [
            group,
            str(stats.get("tp", 0)),
            str(stats.get("fp", 0)),
            str(stats.get("fn", 0)),
            _fmt_opt(stats.get("precision")),
            _fmt_opt(stats.get("recall")),
        ]
        for group, stats in sorted(by_type.items())
    ]
    return header, rows


def _size_rows(results: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """Таблица размерных категорий GT-полигонов (м², по верхней границе)."""
    cats: list[dict[str, Any]] = results.get("auxiliary", {}).get("size_categories", [])
    header = ["площадь, м²", "GT", "TP", "recall"]
    rows = []
    for cat in cats:
        max_area = cat.get("max_area")
        label = f"≤ {max_area:g}" if max_area is not None else "> 10000"
        rows.append([label, str(cat.get("gt", 0)), str(cat.get("tp", 0)), _fmt_opt(cat.get("recall"))])
    return header, rows


def _compare_rows(
    prev: dict[str, Any], cur: dict[str, Any]
) -> tuple[list[str], list[list[str]]]:
    """Таблица «было → стало» по классам: F1, TP, FP, FN."""
    header = ["класс", "F1 было", "F1 стало", "Δ F1", "TP", "FP", "FN"]
    rows: list[list[str]] = []
    for cls, cur_st in cur.items():
        prev_st = prev.get(cls, {})
        rows.append(
            [
                cls,
                f"{prev_st.get('f1', 0.0):.4f}",
                f"{cur_st['f1']:.4f}",
                f"{cur_st['f1'] - prev_st.get('f1', 0.0):+.4f}",
                f"{prev_st.get('tp', 0)} → {cur_st['tp']}",
                f"{prev_st.get('fp', 0)} → {cur_st['fp']}",
                f"{prev_st.get('fn', 0)} → {cur_st['fn']}",
            ]
        )
    return header, rows


def main(argv: list[str] | None = None) -> int:
    """Точка входа: 0 — успех; 1 — ошибка чтения файлов или структуры run.json."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Оценка pred по GT метрикой организаторов (compute_metrics.py)"
    )
    parser.add_argument("--pred", type=Path, required=True, help="GeoJSON предсказаний")
    parser.add_argument("--gt", type=Path, required=True, help="GeoJSON разметки (GT)")
    parser.add_argument(
        "--metric",
        type=Path,
        default=_DEFAULT_METRIC,
        help="путь к compute_metrics.py (по умолчанию — копия в tools/)",
    )
    parser.add_argument("--md", action="store_true", help="таблицы в Markdown (для PR)")
    parser.add_argument("--save", type=Path, help="сохранить полный результат в run.json")
    parser.add_argument("--compare", type=Path, help="run.json прошлого прогона: было / стало / Δ")
    args = parser.parse_args(argv)

    try:
        pred = _load_geojson(args.pred)
        gt = _load_geojson(args.gt)
        metric = load_metric(args.metric)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    results = calculate(metric, pred, gt)
    final = results["final_score"]
    md = args.md
    print(f"final_score: {final:.10f}")
    print(_section("Классы", _class_rows(metric, results), md))
    if any(st["num_ground_truth"] == 0 for st in results["per_class"].values()):
        print("† — класса нет в GT, в final_score не входит")
    dtype_table = _dtype_rows(results)
    if dtype_table[1]:
        print(_section("Типы данных", dtype_table, md))
    size_table = _size_rows(results)
    if size_table[1]:
        print(_section("Размерные категории GT-полигонов", size_table, md))

    if args.compare is not None:
        try:
            prev = _load_run(args.compare)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("не читается %s: %s", args.compare, exc)
            return 1
        prev_final = float(prev["results"]["final_score"])
        print(
            _section(
                f"Сравнение с {args.compare.name}",
                _compare_rows(prev["results"]["per_class"], results["per_class"]),
                md,
            )
        )
        print(f"final: {prev_final:.10f} → {final:.10f} (Δ {final - prev_final:+.10f})")

    if args.save is not None:
        payload = {"meta": _run_metadata(args), "results": results}
        try:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            args.save.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("не удалось записать %s: %s", args.save, exc)
            return 1
        logger.info("сохранено: %s", args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())