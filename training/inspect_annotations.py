"""Шаг 1. Сводка по разметке трейна — БЕЗ координат (вывод можно пересылать).

Печатает по каждому региону: файлы разметки, распознанный тип данных,
CRS файла, ключи properties, значения полей-кандидатов в класс и во что они
переводятся. В конце — нераспознанные имена: их нужно добавить в
settings.CLASS_SYNONYMS. Также печатает, какие хиллшейды Li и LAS есть.

Запуск: python training/inspect_annotations.py --data /путь/к/трейну > inspect.txt
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import settings
from annotations import canonical_class, find_annotation_files


def summarize_region(region_dir: Path, unknown: Counter) -> list[str]:
    """Строки отчёта по одному региону."""
    lines = [f"\n=== {region_dir.name}"]
    las = sorted(p.name for p in region_dir.rglob("*") if p.suffix.lower() in (".las", ".laz"))
    li = sorted(p.name for p in region_dir.rglob("*.tif") if "_li_" in p.as_posix().lower())
    lines.append(f"  LAS: {len(las)} {las[:3]}")
    lines.append(f"  Li-хиллшейды: {len(li)} {li[:6]}")
    for ann in find_annotation_files(region_dir):
        data = json.loads(ann.path.read_text(encoding="utf-8"))
        features = data.get("features", [])
        keys = Counter(k for f in features for k in (f.get("properties") or {}))
        geom_types = Counter((f.get("geometry") or {}).get("type") for f in features)
        lines.append(
            f"  {ann.path.name}: тип={ann.markup_type} crs={ann.crs} объектов={len(features)} "
            f"геометрии={dict(geom_types)} ключи={dict(keys)}"
        )
        file_class = canonical_class(ann.path.stem)
        values = Counter()
        for feature in features:
            props = feature.get("properties") or {}
            for key in settings.CLASS_PROPERTY_KEYS:
                if props.get(key) is not None:
                    values[(key, str(props[key]))] += 1
        for (key, value), count in values.most_common(15):
            mapped = canonical_class(value)
            lines.append(f"      {key}={value!r} ×{count} → {mapped}")
            if mapped is None:
                unknown[value] += count
        lines.append(f"      по имени файла → {file_class}")
    return lines


def main() -> None:
    """Обойти регионы трейна и напечатать сводку."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="папка трейна (регионы NNN_*)")
    args = parser.parse_args()
    unknown: Counter = Counter()
    regions = sorted(p for p in args.data.iterdir() if p.is_dir())
    for region_dir in regions:
        print("\n".join(summarize_region(region_dir, unknown)))
    print(f"\nРегионов: {len(regions)}")
    print("Нераспознанные значения классов (добавить в settings.CLASS_SYNONYMS, если нужны):")
    for value, count in unknown.most_common():
        print(f"  {value!r}: {count}")


if __name__ == "__main__":
    main()
