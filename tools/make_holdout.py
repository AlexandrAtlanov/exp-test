"""Отложенная выборка «как на платформе»: копии регионов трейна без хиллшейдов и разметки.

На платформе в регионе нет готовых хиллшейдов Li и разметки — решение строит
отмывки из LAS само. Инструмент собирает такие же папки из трейна, чтобы
прогнать на них solution.py и сравнить результат с GT (tools/make_gt.py по тем
же --regions):

    python tools/make_holdout.py --data <трейн> --regions 207_Регион 216_Регион --out holdout
    python submission/solution.py holdout pred.geojson
    python tools/make_gt.py --data <трейн> --regions 207_Регион 216_Регион --out gt.geojson
    python tools/evaluate.py --pred pred.geojson --gt gt.geojson

Исключаются папки верхнего уровня региона, в имени которых есть токен «Li»
(02_<Регион>_Li_карты — хиллшейды; «La» с LAS остаётся), и папки разметки
(«разметк» в имени, как в training/annotations.find_annotation_files). Остальное,
включая UTM.json и extra/, переносится с сохранением путей. Файлы — относительными
символическими ссылками (место на диске не тратится); если ОС их не даёт
(Windows без режима разработчика) или указан --copy — копиями.

Коды возврата: 0 — успех; 1 — нет региона, папка результата уже занята или
регион пуст после исключений.
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("make_holdout")

_TOKEN_SPLIT = re.compile(r"[_\-\s.]+")
"""Разделители токенов в именах папок организаторов (01_Регион_La_точки)."""

HILLSHADE_TOKEN = "li"
"""Токен папки готовых хиллшейдов (02_<Регион>_Li_карты)."""

MARKUP_MARK = "разметк"
"""Подстрока имени папки разметки (06_<Регион>_разметка)."""


def is_excluded(dir_name: str) -> bool:
    """Папка верхнего уровня региона, которой нет на платформе: хиллшейды Li или разметка."""
    lowered = dir_name.lower()
    tokens = {token for token in _TOKEN_SPLIT.split(lowered) if token}
    return HILLSHADE_TOKEN in tokens or MARKUP_MARK in lowered


def region_files(region_dir: Path) -> tuple[list[Path], list[str]]:
    """Файлы региона для holdout (пути относительно региона) и имена исключённых папок."""
    files: list[Path] = []
    excluded: list[str] = []
    for entry in sorted(region_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if is_excluded(entry.name):
                excluded.append(entry.name)
                continue
            files.extend(
                path.relative_to(region_dir)
                for path in sorted(entry.rglob("*"))
                if path.is_file() and not path.name.startswith(".")
            )
        elif entry.is_file():
            files.append(entry.relative_to(region_dir))
    return files, excluded


def place_file(src: Path, dst: Path, copy: bool) -> bool:
    """Положить файл в holdout: относительная ссылка или копия; True — если ссылка."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)
            return True
        except OSError:
            pass  # нет прав на ссылки (обычный Windows) — копируем
    shutil.copy2(src, dst)
    return False


def build_holdout(data_dir: Path, regions: list[str], out_dir: Path, copy: bool) -> int:
    """Собрать holdout; 0 — успех, 1 — ошибка входа (до начала записи ничего не создаётся)."""
    missing = [name for name in regions if not (data_dir / name).is_dir()]
    if missing:
        logger.error("Нет регионов в %s: %s", data_dir, ", ".join(missing))
        return 1
    busy = [name for name in regions if (out_dir / name).exists()]
    if busy:
        logger.error("В %s уже есть %s — удалите их или укажите другой --out", out_dir, busy)
        return 1
    plan = {name: region_files(data_dir / name) for name in regions}
    empty = [name for name, (files, _) in plan.items() if not files]
    if empty:
        logger.error("После исключения Li и разметки регионы пусты: %s", ", ".join(empty))
        return 1

    links = copies = 0
    for name, (files, excluded) in plan.items():
        for rel in files:
            if place_file(data_dir / name / rel, out_dir / name / rel, copy):
                links += 1
            else:
                copies += 1
        las = sum(rel.suffix.lower() in (".las", ".laz") for rel in files)
        logger.info(
            "%s: файлов %d (LAS %d); исключены: %s",
            name,
            len(files),
            las,
            ", ".join(excluded) or "—",
        )
    logger.info("Holdout: %s (ссылок %d, копий %d)", out_dir, links, copies)
    if links == 0 and copies and not copy:
        logger.info("Символические ссылки недоступны — файлы скопированы")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Регионы трейна → папка входа «как на платформе» (без Li и разметки)"
    )
    parser.add_argument("--data", type=Path, required=True, help="папка трейна с регионами")
    parser.add_argument("--regions", nargs="+", required=True, help="имена папок регионов")
    parser.add_argument("--out", type=Path, required=True, help="папка holdout (вход solution.py)")
    parser.add_argument("--copy", action="store_true", help="копировать файлы вместо ссылок")
    args = parser.parse_args(argv)
    return build_holdout(args.data, args.regions, args.out, args.copy)


if __name__ == "__main__":
    sys.exit(main())
