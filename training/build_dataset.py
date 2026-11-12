"""Шаг 3. Датасет YOLO-seg: хиллшейды + разметка → тайлы, подписи, dataset.yaml.

Для каждого региона трейна:
1. хиллшейды (из LAS алгоритмом бейзлайна или готовые Li — флаг --source);
2. разметка лидара (тип li) переводится в CRS хиллшейда и в пиксели;
3. для каждого масштаба --scales отмывка сжимается в k раз (yolo_runtime.downscale — как
   на инференсе) и режется на тайлы --tile-px с перекрытием --overlap; тайл с объектами
   сохраняется всегда, пустой — с вероятностью NEGATIVE_TILE_SHARE;
4. подписи YOLO-seg: «класс x1 y1 x2 y2 …» (нормировано на размер тайла);
5. train (не val): к тайлу с объектами — копии, повёрнутые на 90/180/270° (--no-rot90 —
   без них); подписи поворачиваются вместе с картинкой.
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
  [--tile-px 512] [--overlap 0.1] [--scales 1 2] [--no-rot90]
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
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box

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


def label_lines(objects_px, tree, tile: Polygon, x0: int, y0: int, size: int) -> list[str]:
    """Строки подписи YOLO-seg для одного тайла."""
    lines = []
    for idx in tree.query(tile):
        class_id, geom = objects_px[int(idx)]
        clipped = geom.intersection(tile)
        parts = getattr(clipped, "geoms", [clipped])
        for part in parts:
            if not isinstance(part, Polygon) or part.area < settings.MIN_INSTANCE_PX:
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
) -> Counter:
    """Нарезать один хиллшейд на тайлы с подписями (все масштабы); вернуть счётчики."""
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
        size = opts.tile_px
        for y0 in tile_positions(height, size, opts.overlap):
            for x0 in tile_positions(width, size, opts.overlap):
                if valid[y0 : y0 + size, x0 : x0 + size].mean() < settings.MIN_VALID_SHARE:
                    continue
                tile_box = box(x0, y0, x0 + size, y0 + size)
                lines = label_lines(objects_px, tree, tile_box, x0, y0, size)
                if not lines and rng.random() > settings.NEGATIVE_TILE_SHARE:
                    continue
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
                if opts.rot90 and split == "train" and lines:
                    for k in (1, 2, 3):
                        rotated = np.ascontiguousarray(np.rot90(tile, k))
                        _save_tile(
                            out_dir, split, f"{stem}__r{k}", rotated, rotate_lines(lines, k)
                        )
                        stats["tiles_rot90"] += 1
    return stats


def process_region(
    region, out_dir: Path, cache_dir: Path, source: str, opts: TileOptions | None = None
) -> tuple[str, str, Counter]:
    """Все хиллшейды региона → тайлы; возвращает (регион, split, счётчики)."""
    split = split_of(region.name)
    ann_files = [a for a in find_annotation_files(region.path) if a.markup_type == "li"]
    if source == "las":
        rasters = []
        for las_path in region.files.get("li", []):
            rasters += hillshades.from_las(las_path, region.utm_crs, cache_dir / region.name)
    else:
        rasters = hillshades.provided_li(region.path)
    objects: dict[str, list] = {}
    stats: Counter = Counter()
    for raster_path in rasters:
        with rasterio.open(raster_path) as src:
            crs = src.crs.to_string()
        if crs not in objects:
            objects[crs] = [obj for ann in ann_files for obj in load_objects(ann, crs)]
        stats += tile_raster(raster_path, objects, out_dir, split, region.name, opts)
    stats["annotation_files"] = len(ann_files)
    stats["rasters"] = len(rasters)
    return region.name, split, stats


def write_yaml(out_dir: Path) -> Path:
    """dataset.yaml для ultralytics."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(settings.CLASSES))
    root = out_dir.resolve().as_posix()
    text = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    path = out_dir / "dataset.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_meta(out_dir: Path, opts: TileOptions) -> Path:
    """dataset_meta.json: параметры нарезки — train.py переносит их в паспорт весов."""
    path = out_dir / "dataset_meta.json"
    meta = {**asdict(opts), "scales": list(opts.scales), "res_m": 0.1, "classes": settings.CLASSES}
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
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
    args = parser.parse_args(argv)
    opts = TileOptions(
        tile_px=args.tile_px,
        overlap=args.overlap,
        scales=tuple(args.scales),
        rot90=not args.no_rot90 and settings.ROT90_AUGMENT,
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
    summary: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    per_region = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_region, r, args.out, args.cache, args.source, opts)
            for r in regions
        ]
        for future in futures:
            name, split, stats = future.result()
            summary[split] += stats
            per_region[name] = {"split": split, **stats}
            logger.info("%s [%s]: %s", name, split, dict(stats))
    yaml_path = write_yaml(args.out)
    write_meta(args.out, opts)
    (args.out / "stats.json").write_text(
        json.dumps({"summary": summary, "regions": per_region}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    for split, stats in summary.items():
        print(
            f"{split}: тайлов {stats['tiles']} (пустых {stats['tiles_empty']}, "
            f"повёрнутых копий {stats['tiles_rot90']}); объекты: "
            + ", ".join(f"{c}={stats[c]}" for c in settings.CLASSES)
        )
    missing = [c for c in settings.CLASSES if summary["val"][c] == 0]
    if missing:
        print(f"ВНИМАНИЕ: в val нет классов {missing} — задайте settings.VAL_REGIONS вручную")
    print(f"готово: {yaml_path}")


if __name__ == "__main__":
    main()
