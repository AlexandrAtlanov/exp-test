"""Сборка холдаута: детерминированное разбиение регионов на train и holdout.

Продолжает цепочку tools/make_gt.py: список регионов берётся из каталога
регионов (подпапки с «06_разметка») и/или из GT-файла, часть регионов
детерминированно уходит в holdout, остальные — в train. Результаты в --out:

- holdout_regions.json / train_regions.json — списки имён регионов;
- holdout_gt.geojson / train_gt.geojson — разрезанный GT (если передан --gt);
- holdout_manifest.json — сид, размер холдаута и счётчики классов.

Выбор детерминированный: имена регионов сортируются, ранжируются по
md5("<seed>|<имя>") и первые k уходят в holdout, поэтому при том же сиде
разбиение воспроизводится точно.

Коды возврата: 0 — успех; 1 — нет входных данных, пустой холдаут, ошибка
формата или пустой holdout-GT (в последнем случае файлы всё равно пишутся —
так же, как это делает make_gt.py). Сообщения пишутся в stderr.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

MARKUP_DIR = "06_разметка"


def _fail(message: str) -> int:
    """Пишет сообщение в stderr и возвращает код 1."""
    print(f"make_holdout: {message}", file=sys.stderr)
    return 1


def find_region_dirs(root: Path) -> list[str]:
    """Отсортированные имена регионов: подпапки с «06_разметка»."""
    return sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and (entry / MARKUP_DIR).is_dir()
    )


def region_names_from_gt(gt: dict, region_key: str) -> tuple[list[str], int]:
    """Имена регионов из properties объектов GT и число объектов без региона."""
    names: set[str] = set()
    without_region = 0
    for feature in gt.get("features", []):
        name = (feature.get("properties") or {}).get(region_key)
        if isinstance(name, str) and name:
            names.add(name)
        else:
            without_region += 1
    return sorted(names), without_region


def _rank(name: str, seed: int) -> str:
    """Ключ детерминированного ранжирования региона."""
    return hashlib.md5(f"{seed}|{name}".encode("utf-8")).hexdigest()


def pick_holdout(names: list[str], seed: int, count: int) -> list[str]:
    """Детерминированный выбор count регионов (ранжирование по md5, итог сортирован)."""
    ranked = sorted(names, key=lambda name: _rank(name, seed))
    return sorted(ranked[:count])


def load_gt(path: Path) -> dict:
    """Читает GT-файл и проверяет, что это GeoJSON FeatureCollection."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("файл не является GeoJSON FeatureCollection")
    return data


def split_gt(gt: dict, holdout: set[str], region_key: str) -> tuple[list[dict], list[dict]]:
    """Делит объекты GT на (holdout, train); объекты без региона остаются в train."""
    holdout_features: list[dict] = []
    train_features: list[dict] = []
    for feature in gt.get("features", []):
        name = (feature.get("properties") or {}).get(region_key)
        if isinstance(name, str) and name in holdout:
            holdout_features.append(feature)
        else:
            train_features.append(feature)
    return holdout_features, train_features


def class_counts(features: list[dict], class_key: str) -> dict[str, int]:
    """Счётчик объектов по классу; объекты без класса — под ключом «?»."""
    counts: Counter[str] = Counter()
    for feature in features:
        value = (feature.get("properties") or {}).get(class_key)
        counts["?" if value is None else str(value)] += 1
    return dict(sorted(counts.items()))


def _collection(gt: dict, features: list[dict]) -> dict:
    """FeatureCollection с crs из исходного GT."""
    collection: dict = {"type": "FeatureCollection"}
    if "crs" in gt:
        collection["crs"] = gt["crs"]
    collection["features"] = features
    return collection


def _write_json(path: Path, payload: object) -> None:
    """Пишет JSON в UTF-8 с отступами."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Точка входа: собирает холдаут, возвращает код 0 или 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt", type=Path, help="GT-файл tools/make_gt.py (GeoJSON FeatureCollection)"
    )
    parser.add_argument("--root", type=Path, help="каталог регионов, как у make_gt.py")
    parser.add_argument(
        "--out", type=Path, help="каталог результатов (по умолчанию holdout/ рядом с GT)"
    )
    parser.add_argument(
        "--frac", type=float, default=0.2, help="доля регионов в holdout (по умолчанию 0.2)"
    )
    parser.add_argument(
        "--count", type=int, help="точное число holdout-регионов (приоритетнее --frac)"
    )
    parser.add_argument("--seed", type=int, default=42, help="сид выбора (по умолчанию 42)")
    parser.add_argument(
        "--region-key", default="region", help="ключ региона в properties (по умолчанию region)"
    )
    parser.add_argument(
        "--class-key", default="class", help="ключ класса в properties (по умолчанию class)"
    )
    args = parser.parse_args(argv)

    if args.gt is None and args.root is None:
        return _fail("укажите --gt и/или --root")

    region_names: list[str] = []
    if args.root is not None:
        if not args.root.is_dir():
            return _fail(f"каталог регионов не найден: {args.root}")
        try:
            region_names = find_region_dirs(args.root)
        except OSError as exc:
            return _fail(f"не читается каталог регионов {args.root}: {exc}")
        if not region_names:
            return _fail(f"в {args.root} нет регионов (подпапок с «{MARKUP_DIR}»)")

    gt: dict | None = None
    gt_names: list[str] = []
    without_region = 0
    if args.gt is not None:
        try:
            gt = load_gt(args.gt)
        except (OSError, ValueError) as exc:
            return _fail(f"не читается GT {args.gt}: {exc}")
        if not gt.get("features"):
            return _fail(f"GT {args.gt} не содержит объектов")
        gt_names, without_region = region_names_from_gt(gt, args.region_key)
        if not gt_names:
            keys = sorted((gt["features"][0].get("properties") or {}).keys())
            hint = ", ".join(keys) if keys else "свойств нет"
            return _fail(
                f"в объектах GT нет строкового свойства «{args.region_key}»; "
                f"передайте --region-key (ключи первого объекта: {hint})"
            )
        if not region_names:
            region_names = gt_names

    total = len(region_names)
    if args.count is not None:
        if args.count < 1:
            return _fail(f"--count должен быть не меньше 1, получено {args.count}")
        size = args.count
    else:
        if not 0.0 < args.frac < 1.0:
            return _fail(f"--frac должен быть в (0, 1), получено {args.frac}")
        size = max(1, round(args.frac * total))
    if size >= total:
        return _fail(f"holdout размера {size} >= {total} регионов — для train ничего не остаётся")

    holdout = pick_holdout(region_names, args.seed, size)
    holdout_set = set(holdout)
    train = [name for name in region_names if name not in holdout_set]

    if args.out is not None:
        out_dir = args.out
    elif args.gt is not None:
        out_dir = args.gt.parent / "holdout"
    else:
        out_dir = Path("holdout")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "holdout_regions.json", holdout)
    _write_json(out_dir / "train_regions.json", train)

    not_in_root = sorted(set(gt_names) - set(region_names)) if args.root is not None else []
    manifest: dict = {
        "tool": "make_holdout",
        "seed": args.seed,
        "holdout_size": size,
        "n_regions": total,
        "n_holdout": len(holdout),
        "n_train": len(train),
        "regions": holdout,
        "gt_regions_not_in_root": not_in_root,
    }

    exit_code = 0
    holdout_features: list[dict] = []
    if gt is not None:
        holdout_features, train_features = split_gt(gt, holdout_set, args.region_key)
        _write_json(out_dir / "holdout_gt.geojson", _collection(gt, holdout_features))
        _write_json(out_dir / "train_gt.geojson", _collection(gt, train_features))
        manifest["holdout_classes"] = class_counts(holdout_features, args.class_key)
        manifest["train_classes"] = class_counts(train_features, args.class_key)
        manifest["n_holdout_features"] = len(holdout_features)
        manifest["n_train_features"] = len(train_features)
        manifest["features_without_region"] = without_region
        if not holdout_features:
            print(
                "make_holdout: в holdout-регионах нет объектов GT — код 1 (файлы записаны)",
                file=sys.stderr,
            )
            exit_code = 1

    _write_json(out_dir / "holdout_manifest.json", manifest)

    print(f"Регионов: {total} → holdout {len(holdout)}, train {len(train)} (seed {args.seed})")
    print(f"holdout: {', '.join(holdout)}")
    if not_in_root:
        print(f"регионы GT без папки в --root: {', '.join(not_in_root)}")
    if gt is not None:
        print(
            f"Объектов GT: holdout {manifest['n_holdout_features']},"
            f" train {manifest['n_train_features']}"
        )
        classes = ", ".join(
            f"{name}={count}" for name, count in manifest["holdout_classes"].items()
        )
        print(f"Классы в holdout: {classes or '—'}")
    print(f"Результаты: {out_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
