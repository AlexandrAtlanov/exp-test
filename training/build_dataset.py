"""Шаг 3. Датасет YOLO-seg: хиллшейды + разметка → тайлы, подписи, dataset.yaml.

Для каждого региона трейна:
1. хиллшейды (из LAS алгоритмом бейзлайна или готовые Li — флаг --source);
2. разметка лидара (тип li) переводится в CRS хиллшейда и в пиксели;
3. для каждого масштаба --scales отмывка сжимается в k раз (yolo_runtime.downscale — как
   на инференсе) и режется на тайлы --tile-px с перекрытием --overlap; тайл с объектами
   сохраняется всегда, пустой — с вероятностью NEGATIVE_TILE_SHARE;
4. подписи YOLO-seg: «класс x1 y1 x2 y2 …» (нормировано на размер тайла);
5. train (не val): к тайлу с объектами — копии, повёрнутые на 90/180/270° (--no-rot90 —
   без них); подписи поворачиваются вместе с картинкой;
6. train: тайл с редким классом (RARE_CLASSES) и его повороты сохраняются ещё
   --rare-repeat раз (больший вес в эпохе).
Параметры нарезки пишутся в <out>/dataset_meta.json: train.py переносит их в паспорт
весов, и детектор на платформе режет отмывку так же, как при обучении.
Разбиение train/val — по регионам целиком (VAL_REGIONS или хеш имени).
Изображения готовятся той же функцией, что на инференсе (koz3.yolo_runtime).
Формат тайлов — только png (settings.IMAGE_FORMAT, В3): jpg сжимает с
потерями и ломает пиксельное единство обучения и инференса (5а.1);
не-png значение останавливает сборку до нарезки.

Запуск:
  python training/build_dataset.py --data /трейн --out /датасет --cache /кэш_хиллшейдов
  [--source las|li] [--workers 8] [--regions 200_Судьбищи_2 ...]
  [--tile-px 512] [--overlap 0.1] [--scales 1 2 4] [--no-rot90] [--rare-repeat 2]
  [--exclude <регион> ...] [--negatives work/negatives.json]
"""

import argparse
import json
import logging
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box

import augment_data
import hillshades
import settings
from annotations import find_annotation_files, load_objects
from koz3.discovery import find_regions
from koz3.yolo_runtime import downscale, tile_positions, to_model_image, valid_pixel_mask

logger = logging.getLogger("build_dataset")


@dataclass(frozen=True)
class TileOptions:
    """Параметры нарезки датасета (пишутся в dataset_meta.json → паспорт весов)."""

    tile_px: int = settings.TILE_PX
    overlap: float = settings.TILE_OVERLAP
    scales: tuple[float, ...] = settings.TILE_SCALES
    rot90: bool = settings.ROT90_AUGMENT
    rare_repeat: int = settings.RARE_TILE_REPEAT  # доп. копий train-тайлов с редкими классами
    fix: bool = settings.AUTO_FIX  # CRS и сдвиг разметки (augment_data)
    extra: bool = settings.EXTRA_DATA  # доп. азимуты и синтетика в train


def split_of(region_name: str) -> str:
    """train или val для региона (детерминированно)."""
    if settings.VAL_REGIONS:
        return "val" if region_name in settings.VAL_REGIONS else "train"
    bucket = zlib.crc32(region_name.encode("utf-8")) % 100
    return "val" if bucket < settings.VAL_SHARE * 100 else "train"


def check_image_format() -> None:
    """Проверить settings.IMAGE_FORMAT (В3): допустим только png — без потерь.

    jpg/jpeg — сжатие с потерями: пиксели датасета разойдутся с инференсом
    (пиксельное единство, 5а.1), причём молча. Ошибка конфигурации — до
    старта нарезки, а не по факту испорченного датасета.
    """
    fmt = str(settings.IMAGE_FORMAT).strip().lower().lstrip(".")
    if fmt != "png":
        raise SystemExit(
            f"settings.IMAGE_FORMAT={settings.IMAGE_FORMAT!r}: допустим только 'png' — "
            "jpg сжимает с потерями и ломает пиксельное единство с инференсом (5а.1)"
        )


def to_pixels(geom, transform) -> shapely.Geometry:
    """Геометрия в CRS растра → пиксельные координаты (столбец, строка)."""
    inv = ~transform
    return affine_transform(geom, [inv.a, inv.b, inv.d, inv.e, inv.xoff, inv.yoff])


def as_area(geom, res_m: float):
    """Точки и линии разметки → полигоны (буфер 1 м); полигоны без изменений."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    return geom.buffer(max(1.0, res_m))


def _nodata_share(part: Polygon, valid: np.ndarray) -> float:
    """Доля пикселей объекта (пиксельные координаты растра) над «нет данных»."""
    minx, miny, maxx, maxy = (int(np.floor(v)) for v in part.bounds)
    r0, c0 = max(miny, 0), max(minx, 0)
    r1, c1 = min(maxy + 1, valid.shape[0]), min(maxx + 1, valid.shape[1])
    if r1 <= r0 or c1 <= c0:
        return 1.0
    inside = rasterize(
        [(part, 1)], out_shape=(r1 - r0, c1 - c0), transform=Affine.translation(c0, r0)
    ).astype(bool)
    if not inside.any():
        return 0.0
    return float(1.0 - valid[r0:r1, c0:c1][inside].mean())


def label_lines(
    objects_px, tree, tile: Polygon, x0: int, y0: int, size: int, valid=None, stats=None
) -> list[str]:
    """Строки подписи YOLO-seg для одного тайла.

    valid — маска данных растра: объект, который больше чем на
    settings.DROP_LABEL_NODATA_SHARE лежит над «нет данных», не подписывается (признак
    сдвига разметки или края съёмки) — счётчик stats["labels_over_nodata"].
    """
    lines = []
    for idx in tree.query(tile):
        class_id, geom = objects_px[int(idx)]
        clipped = geom.intersection(tile)
        parts = getattr(clipped, "geoms", [clipped])
        for part in parts:
            if not isinstance(part, Polygon) or part.area < settings.MIN_INSTANCE_PX:
                continue
            if valid is not None and _nodata_share(part, valid) > settings.DROP_LABEL_NODATA_SHARE:
                if stats is not None:
                    stats["labels_over_nodata"] += 1
                continue
            part = part.simplify(settings.LABEL_SIMPLIFY_PX, preserve_topology=True)
            coords = np.asarray(part.exterior.coords)[:-1]
            if len(coords) < 3:
                continue
            norm = np.clip((coords - (x0, y0)) / size, 0.0, 1.0)
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in norm.ravel()))
    return lines


def rotate_lines(lines: list[str], k: int) -> list[str]:
    """Подписи YOLO-seg для картинки, повёрнутой np.rot90(·, k) (против часовой стрелки).

    np.rot90 переносит пиксель (строка r, столбец c) картинки ширины W в (W-1-c, r):
    в нормированных координатах (x, y) → (y, 1 - x), k раз.
    """
    rotated = []
    for line in lines:
        class_id, *values = line.split()
        xy = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        for _ in range(k % 4):
            xy = np.column_stack([xy[:, 1], 1.0 - xy[:, 0]])
        rotated.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
    return rotated


def _save_tile(out_dir: Path, split: str, stem: str, tile: np.ndarray, lines: list[str]) -> None:
    """Картинка png и подпись YOLO-seg одного тайла."""
    Image.fromarray(tile).save(out_dir / "images" / split / f"{stem}.{settings.IMAGE_FORMAT}")
    (out_dir / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def tile_raster(
    raster_path: Path,
    objects,
    out_dir: Path,
    split: str,
    region: str,
    opts: TileOptions | None = None,
    focus: list | None = None,
    negatives: list | None = None,
) -> Counter:
    """Нарезать один хиллшейд на тайлы с подписями (все масштабы); вернуть счётчики.

    focus — геометрии в CRS растра: сохраняются только тайлы, задевающие их (синтетика).
    negatives — рамки ложных срабатываний (tools/mine_negatives.py) в CRS растра: пустой
    тайл, задевающий такую рамку, сохраняется всегда (hard negative), сверх
    NEGATIVE_TILE_SHARE.
    """
    opts = opts or TileOptions()
    stats: Counter = Counter()
    with rasterio.open(raster_path) as src:
        full = to_model_image(src.read())
        full_transform, crs = src.transform, src.crs.to_string()
        res_m = abs(full_transform.a)
    rng = np.random.default_rng(zlib.crc32(raster_path.name.encode()) + settings.SEED)
    for factor in opts.scales:
        image = downscale(full, float(factor))
        height, width = image.shape[:2]
        transform = full_transform * full_transform.scale(
            full.shape[1] / width, full.shape[0] / height
        )
        valid = valid_pixel_mask(image)
        objects_px = []
        for class_name, geom in objects.get(crs, []):
            px = to_pixels(as_area(geom, res_m), transform)
            if not px.is_empty:
                objects_px.append((settings.CLASSES.index(class_name), px))
        tree = shapely.STRtree([g for _, g in objects_px])
        focus_px = [to_pixels(g, transform) for g in focus] if focus is not None else None
        focus_tree = shapely.STRtree(focus_px) if focus_px else None
        neg_px = [to_pixels(g, transform) for g in negatives] if negatives else []
        neg_tree = shapely.STRtree(neg_px) if neg_px else None
        size = opts.tile_px
        for y0 in tile_positions(height, size, opts.overlap):
            for x0 in tile_positions(width, size, opts.overlap):
                if valid[y0 : y0 + size, x0 : x0 + size].mean() < settings.MIN_VALID_SHARE:
                    continue
                tile_box = box(x0, y0, x0 + size, y0 + size)
                if focus is not None and (
                    focus_tree is None or len(focus_tree.query(tile_box, "intersects")) == 0
                ):
                    continue
                lines = label_lines(objects_px, tree, tile_box, x0, y0, size, valid, stats)
                if focus is not None and not lines:
                    continue
                hard = bool(
                    not lines
                    and neg_tree is not None
                    and len(neg_tree.query(tile_box, "intersects"))
                )
                if not lines and not hard and rng.random() > settings.NEGATIVE_TILE_SHARE:
                    continue
                stats["tiles_hard_negative"] += hard
                tile = np.zeros((size, size, 3), dtype=np.uint8)
                crop = image[y0 : y0 + size, x0 : x0 + size]
                tile[: crop.shape[0], : crop.shape[1]] = crop  # край растра дополняем нулями
                stem = f"{region}__{raster_path.stem}__s{factor:g}__y{y0}_x{x0}"
                _save_tile(out_dir, split, stem, tile, lines)
                stats["tiles"] += 1
                stats[f"tiles_s{factor:g}"] += 1
                stats["tiles_empty"] += not lines
                for line in lines:
                    stats[settings.CLASSES[int(line.split()[0])]] += 1
                turns = (1, 2, 3) if opts.rot90 and split == "train" and lines else ()
                for k in turns:
                    rotated = np.ascontiguousarray(np.rot90(tile, k))
                    _save_tile(out_dir, split, f"{stem}__r{k}", rotated, rotate_lines(lines, k))
                    stats["tiles_rot90"] += 1
                if split == "train" and has_rare_class(lines):
                    for n in range(1, opts.rare_repeat + 1):
                        _save_tile(out_dir, split, f"{stem}__dup{n}", tile, lines)
                        stats["tiles_rare_repeat"] += 1
                        for k in turns:
                            rotated = np.ascontiguousarray(np.rot90(tile, k))
                            _save_tile(
                                out_dir,
                                split,
                                f"{stem}__r{k}__dup{n}",
                                rotated,
                                rotate_lines(lines, k),
                            )
                            stats["tiles_rare_repeat"] += 1
    return stats


def has_rare_class(lines: list[str]) -> bool:
    """Есть ли в подписи тайла объект редкого класса (settings.RARE_CLASSES)."""
    rare_ids = {settings.CLASSES.index(c) for c in settings.RARE_CLASSES if c in settings.CLASSES}
    return any(int(line.split()[0]) in rare_ids for line in lines)


def process_las(
    region,
    las_path: Path,
    ann_files,
    out_dir: Path,
    cache_dir: Path,
    split: str,
    opts,
    negatives: list | None = None,
) -> tuple[Counter, list[str]]:
    """Один LAS: отмывки → исправление разметки → тайлы; train — ещё доп. данные."""
    stats: Counter = Counter()
    notes: list[str] = []
    try:
        augment_data.las_bounds(las_path)
        rasters = hillshades.from_las(las_path, region.utm_crs, cache_dir / region.name)
    except Exception as exc:  # noqa: BLE001  # битый LAS — пропуск с записью в отчёт
        stats["las_skipped"] += 1
        return stats, [f"{las_path.name}: пропущен — {exc}"]
    with rasterio.open(rasters[0]) as src:
        crs = src.crs.to_string()
    objects, crs_source = augment_data.choose_objects(ann_files, las_path, crs, opts.fix)
    if crs_source != "регион":
        stats["las_crs_fixed"] += 1
        notes.append(f"{las_path.name}: разметка взята в CRS {crs_source}")
    dem = augment_data.LazyDem(las_path)
    if opts.fix:
        key = "las" if crs_source != "регион" else "region"
        try:
            result = augment_data.shift_for(objects, dem, cache_dir / region.name, key)
        except Exception as exc:  # noqa: BLE001  # без оценки сдвига — разметка как есть
            result = None
            notes.append(f"{las_path.name}: сдвиг не оценён — {type(exc).__name__}: {exc}")
        objects, applied = augment_data.apply_shift(objects, result, opts.fix)
        if applied:
            stats["las_shifted"] += 1
            notes.append(f"{las_path.name}: разметка сдвинута на (В, С) {applied} м")
        elif result and result.get("contrast_m", 1.0) <= 0:
            notes.append(f"{las_path.name}: под контурами курганов нет насыпей — проверить")
    hard = negatives if split == "train" else None
    for raster_path in rasters:
        stats += tile_raster(
            raster_path, {crs: objects}, out_dir, split, region.name, opts, negatives=hard
        )
    if split == "train" and opts.extra:
        seed = zlib.crc32(f"{region.name}/{las_path.name}".encode()) + settings.SEED
        try:
            extras = augment_data.extra_rasters(dem, crs, cache_dir / region.name, objects, seed)
        except Exception as exc:  # noqa: BLE001  # нет доп. данных — основные тайлы остаются
            extras = []
            notes.append(
                f"{las_path.name}: доп. данные не построены — {type(exc).__name__}: {exc}"
            )
        for path, extra_objects, focus in extras:
            extra = tile_raster(
                path, {crs: extra_objects}, out_dir, split, region.name, opts, focus
            )
            stats["tiles_synth" if focus is not None else "tiles_azimuth"] += extra["tiles"]
            stats += extra
    return stats, notes


def process_region(
    region,
    out_dir: Path,
    cache_dir: Path,
    source: str,
    opts: TileOptions | None = None,
    negatives: list | None = None,
) -> tuple[str, str, Counter, list[str]]:
    """Все хиллшейды региона → тайлы; возвращает (регион, split, счётчики, заметки).

    negatives — рамки ложных срабатываний региона (shapely) для hard negative тайлов.
    """
    opts = opts or TileOptions()
    split = split_of(region.name)
    ann_files = [a for a in find_annotation_files(region.path) if a.markup_type == "li"]
    stats: Counter = Counter()
    notes: list[str] = []
    if source == "las":
        for las_path in region.files.get("li", []):
            las_stats, las_notes = process_las(
                region, las_path, ann_files, out_dir, cache_dir, split, opts, negatives
            )
            stats += las_stats
            notes += las_notes
    else:
        objects: dict[str, list] = {}
        for raster_path in hillshades.provided_li(region.path):
            with rasterio.open(raster_path) as src:
                crs = src.crs.to_string()
            if crs not in objects:
                objects[crs] = [obj for ann in ann_files for obj in load_objects(ann, crs)]
            stats += tile_raster(raster_path, objects, out_dir, split, region.name, opts)
    stats["annotation_files"] = len(ann_files)
    return region.name, split, stats, notes


def load_negatives(path: Path) -> dict[str, list]:
    """negatives.json (tools/mine_negatives.py) → {регион: [рамки shapely в CRS отмывок]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [box(*b) for b in info.get("boxes", [])] for name, info in data.items()}


def write_yaml(out_dir: Path) -> Path:
    """dataset.yaml для ultralytics."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(settings.CLASSES))
    root = out_dir.resolve().as_posix()
    text = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    path = out_dir / "dataset.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_meta(out_dir: Path, opts: TileOptions, source: str = "las") -> Path:
    """dataset_meta.json: параметры нарезки — train.py переносит их в паспорт весов.

    hillshades — отмывки датасета: решение подаёт модели только их (готовые Li-отмывки
    организаторов — без «r»).
    """
    path = out_dir / "dataset_meta.json"
    hillshades = [
        h for h in settings.HILLSHADES if source == "las" or h in settings.LEGACY_HILLSHADES
    ]
    meta = {
        **asdict(opts),
        "scales": list(opts.scales),
        "res_m": 0.1,
        "classes": settings.CLASSES,
        "hillshades": hillshades,
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_report(out_dir: Path, summary: dict, per_region: dict) -> Path:
    """dataset_report.md: что исправлено, пропущено и добавлено (без координат)."""
    keys = (
        "tiles", "tiles_rot90", "tiles_rare_repeat", "tiles_azimuth", "tiles_synth",
        "labels_over_nodata",
        "tiles_hard_negative", "las_skipped", "las_crs_fixed", "las_shifted", "regions_failed",
    )  # fmt: skip
    lines = ["# Сборка датасета", "", "| split | " + " | ".join(keys) + " |"]
    lines.append("|---" * (len(keys) + 1) + "|")
    for split, stats in summary.items():
        lines.append(f"| {split} | " + " | ".join(str(stats.get(k, 0)) for k in keys) + " |")
    lines += ["", "## Объекты по классам", ""]
    for split, stats in summary.items():
        lines.append(
            f"- {split}: " + ", ".join(f"{c}={stats.get(c, 0)}" for c in settings.CLASSES)
        )
    lines += ["", "## Исправления и пропуски по регионам", ""]
    for name, info in sorted(per_region.items()):
        for note in info.get("notes", []):
            lines.append(f"- {name}: {note}")
    path = out_dir / "dataset_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """Собрать датасет и напечатать сводку по классам в train/val."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="папка трейна")
    parser.add_argument("--out", required=True, type=Path, help="куда писать датасет")
    parser.add_argument(
        "--cache", type=Path, default=Path("hillshade_cache"), help="кэш хиллшейдов из LAS"
    )
    parser.add_argument("--source", choices=("las", "li"), default="las")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--regions", nargs="*", help="только эти регионы (для пробы)")
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="регионы, которые не брать (битые по аудиту)"
    )
    parser.add_argument("--no-fix", action="store_true", help="не исправлять CRS и сдвиг разметки")
    parser.add_argument("--no-extra", action="store_true", help="без доп. азимутов и синтетики")
    parser.add_argument("--tile-px", type=int, default=settings.TILE_PX, help="размер тайла")
    parser.add_argument("--overlap", type=float, default=settings.TILE_OVERLAP)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(settings.TILE_SCALES),
        help="масштабы нарезки: 1 — 0.1 м/пикс, 2 — 0.2 м/пикс …",
    )
    parser.add_argument(
        "--no-rot90", action="store_true", help="без повёрнутых копий тайлов с объектами"
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        help="negatives.json (tools/mine_negatives.py): пустые тайлы на местах FP модели",
    )
    parser.add_argument(
        "--rare-repeat",
        type=int,
        default=settings.RARE_TILE_REPEAT,
        help="доп. копий train-тайлов с редкими классами (0 — выкл.)",
    )
    args = parser.parse_args(argv)
    opts = TileOptions(
        tile_px=args.tile_px,
        overlap=args.overlap,
        scales=tuple(args.scales),
        rot90=not args.no_rot90 and settings.ROT90_AUGMENT,
        rare_repeat=max(0, args.rare_repeat),
        fix=not args.no_fix and settings.AUTO_FIX,
        extra=not args.no_extra and settings.EXTRA_DATA,
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    check_image_format()

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    regions = find_regions(args.data)
    if args.regions:
        regions = [r for r in regions if r.name in set(args.regions)]
    excluded = set(args.exclude) | set(settings.EXCLUDE_REGIONS)
    if excluded:
        regions = [r for r in regions if r.name not in excluded]
        logger.info("исключены регионы: %s", sorted(excluded))
    negatives = load_negatives(args.negatives) if args.negatives else {}
    summary: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    per_region = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                process_region, r, args.out, args.cache, args.source, opts, negatives.get(r.name)
            )
            for r in regions
        ]
        notes_by_region: dict[str, list[str]] = {}
        for region, future in zip(regions, futures, strict=True):
            try:
                name, split, stats, notes = future.result()
            except Exception as exc:  # noqa: BLE001  # один регион не роняет всю сборку
                name, split = region.name, split_of(region.name)
                stats = Counter(regions_failed=1)
                notes = [f"регион пропущен — {type(exc).__name__}: {exc}"]
            summary[split] += stats
            per_region[name] = {"split": split, **stats, "notes": notes}
            notes_by_region[name] = notes
            logger.info("%s [%s]: %s", name, split, dict(stats))
            for note in notes:
                logger.warning("%s: %s", name, note)
    yaml_path = write_yaml(args.out)
    write_meta(args.out, opts, args.source)
    write_report(args.out, summary, per_region)
    (args.out / "stats.json").write_text(
        json.dumps({"summary": summary, "regions": per_region}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    for split, stats in summary.items():
        print(
            f"{split}: тайлов {stats['tiles']} (пустых {stats['tiles_empty']}, "
            f"повёрнутых копий {stats['tiles_rot90']}, повторов редких классов "
            f"{stats['tiles_rare_repeat']}); объекты: "
            + ", ".join(f"{c}={stats[c]}" for c in settings.CLASSES)
        )
    missing = [c for c in settings.CLASSES if summary["val"][c] == 0]
    if missing:
        print(f"ВНИМАНИЕ: в val нет классов {missing} — задайте settings.VAL_REGIONS вручную")
    print(f"готово: {yaml_path}")


if __name__ == "__main__":
    main()
