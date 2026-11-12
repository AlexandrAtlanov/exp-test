"""Сборка GT из разметки трейна: регионы → один GeoJSON формата платформы.

Выход — FeatureCollection в EPSG:3857 с properties class_name (одно из 7
канонических имён), markup_type (метка в имени файла разметки) и region_name
(имя папки региона). Точки разметки остаются точками: метрика сравнивает их
по расстоянию, круги она строит сама. Чтение разметки —
training/annotations.py, логика не дублируется. Инструмент разработки,
в сабмишн не входит.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

_REPO_DIR = Path(__file__).resolve().parents[1]
_TRAINING_DIR = _REPO_DIR / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

import settings  # noqa: E402
from annotations import find_annotation_files, load_objects  # noqa: E402

_DST_CRS = "EPSG:3857"
"""CRS выходного файла — формат платформы."""

logger = logging.getLogger("make_gt")


def _feature(
    class_name: str,
    markup_type: str | None,
    region_name: str,
    geom: BaseGeometry,
) -> dict[str, object]:
    """GeoJSON-фича GT: минимальные свойства платформы, геометрия как есть."""
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "class_name": class_name,
            "markup_type": markup_type,
            "region_name": region_name,
        },
    }


def _region_features(region_dir: Path) -> list[dict[str, object]]:
    """Фичи одного региона из всех файлов его разметки (формат платформы)."""
    features: list[dict[str, object]] = []
    for ann in find_annotation_files(region_dir):
        if ann.markup_type is None:
            logger.warning(
                "Тип данных не распознан в имени файла %s — markup_type=null",
                ann.path.name,
            )
        for class_name, geom in load_objects(ann, _DST_CRS, classes=settings.ALL_CLASSES):
            features.append(_feature(class_name, ann.markup_type, region_dir.name, geom))
    return features


def build_gt(data_dir: Path, region_names: list[str]) -> list[dict[str, object]]:
    """Фичи GT перечисленных регионов; отсутствующий регион — ошибка.

    Все имена проверяются заранее: опечатка не должна молча уменьшать GT.
    """
    missing = [name for name in region_names if not (data_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Нет регионов в {data_dir}: {', '.join(missing)}")
    features: list[dict[str, object]] = []
    for name in region_names:
        region_features = _region_features(data_dir / name)
        logger.info("Регион %s: %d объектов разметки", name, len(region_features))
        features.extend(region_features)
    return features


def write_gt(features: list[dict[str, object]], out_path: Path) -> None:
    """Пишет FeatureCollection с явным членом crs EPSG:3857."""
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": _DST_CRS}},
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(collection, handle, ensure_ascii=False, allow_nan=False)


def _log_summary(features: list[dict[str, object]]) -> None:
    """Сводка: всего объектов, распределение по классам и markup_type."""
    by_class: Counter[str] = Counter()
    by_markup: Counter[str] = Counter()
    for feature in features:
        props = feature["properties"]
        by_class[props["class_name"]] += 1
        by_markup[props["markup_type"] or "null"] += 1
    logger.info("GT: %d объектов", len(features))
    logger.info("По классам: %s", dict(sorted(by_class.items())))
    logger.info("По markup_type: %s", dict(sorted(by_markup.items())))


def main(argv: list[str] | None = None) -> int:
    """Точка входа: 0 — успех; 1 — пустой GT или отсутствующий регион."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Разметка регионов трейна → GT GeoJSON (EPSG:3857, 7 классов)"
    )
    parser.add_argument("--data", type=Path, required=True, help="папка трейна с регионами")
    parser.add_argument("--regions", nargs="+", required=True, help="имена папок регионов")
    parser.add_argument("--out", type=Path, required=True, help="выходной файл .geojson")
    args = parser.parse_args(argv)
    try:
        features = build_gt(args.data, args.regions)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    write_gt(features, args.out)
    _log_summary(features)
    logger.info("GT записан: %s", args.out)
    if not features:
        logger.warning(
            "GT пуст — проверьте --data/--regions и папки разметки "
            "(find_annotation_files ищет «разметк» в имени папки)"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
