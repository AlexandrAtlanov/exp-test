"""Шаг 3. Датасет YOLO-seg: хиллшейды + разметка → тайлы, подписи, dataset.yaml.

Для каждого региона трейна:
1. хиллшейды (из LAS алгоритмом бейзлайна или готовые Li — флаг --source);
2. разметка лидара (тип li) переводится в CRS хиллшейда и в пиксели;
3. растр режется на тайлы settings.TILE_PX с перекрытием; тайл с объектами
   сохраняется всегда, пустой — с вероятностью NEGATIVE_TILE_SHARE;
4. подписи YOLO-seg: «класс x1 y1 x2 y2 …» (нормировано на размер тайла).
Разбиение train/val — по регионам целиком (VAL_REGIONS или хеш имени).
Изображения готовятся той же функцией, что на инференсе (koz3.yolo_runtime).

Запуск:
  python training/build_dataset.py --data /трейн --out /датасет --cache /кэш_хиллшейдов
  [--source las|li] [--workers 8] [--regions 200_Судьбищи_2 ...]
"""

import argparse
import json
import logging
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
from koz3.yolo_runtime import tile_positions, to_model_image

logger = logging.getLogger("build_dataset")


def split_of(region_name: str) -> str:
    """train или val для региона (детерминированно)."""
    if settings.VAL_REGIONS:
        return "val" if region_name in settings.VAL_REGIONS else "train"
    bucket = zlib.crc32(region_name.encode("utf-8")) % 100
    return "val" if bucket < settings.VAL_SHARE * 100 else "train"


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


def tile_raster(raster_path: Path, objects, out_dir: Path, split: str, region: str) -> Counter:
    """Нарезать один хиллшейд на тайлы с подписями; вернуть счётчики."""
    stats: Counter = Counter()
    with rasterio.open(raster_path) as src:
        image = to_model_image(src.read())
        transform, crs = src.transform, src.crs.to_string()
        res_m = abs(transform.a)
    valid = ~(np.all(image == 0, axis=-1) | np.all(image == 255, axis=-1))
    objects_px = []
    for class_name, geom in objects.get(crs, []):
        px = to_pixels(as_area(geom, res_m), transform)
        if not px.is_empty:
            objects_px.append((settings.CLASSES.index(class_name), px))
    tree = shapely.STRtree([g for _, g in objects_px])
    rng = np.random.default_rng(zlib.crc32(raster_path.name.encode()) + settings.SEED)
    height, width = image.shape[:2]
    size = settings.TILE_PX
    for y0 in tile_positions(height, size, settings.TILE_OVERLAP):
        for x0 in tile_positions(width, size, settings.TILE_OVERLAP):
            window_valid = valid[y0 : y0 + size, x0 : x0 + size]
            if window_valid.mean() < settings.MIN_VALID_SHARE:
                continue
            lines = label_lines(objects_px, tree, box(x0, y0, x0 + size, y0 + size), x0, y0, size)
            if not lines and rng.random() > settings.NEGATIVE_TILE_SHARE:
                continue
            tile = np.zeros((size, size, 3), dtype=np.uint8)
            crop = image[y0 : y0 + size, x0 : x0 + size]
            tile[: crop.shape[0], : crop.shape[1]] = crop  # край растра дополняем нулями
            stem = f"{region}__{raster_path.stem}__y{y0}_x{x0}"
            Image.fromarray(tile).save(
                out_dir / "images" / split / f"{stem}.{settings.IMAGE_FORMAT}"
            )
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            stats["tiles"] += 1
            stats["tiles_empty"] += not lines
            for line in lines:
                stats[settings.CLASSES[int(line.split()[0])]] += 1
    return stats


def process_region(
    region, out_dir: Path, cache_dir: Path, source: str
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
        stats += tile_raster(raster_path, objects, out_dir, split, region.name)
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


def main() -> None:
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
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

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
            pool.submit(process_region, r, args.out, args.cache, args.source) for r in regions
        ]
        for future in futures:
            name, split, stats = future.result()
            summary[split] += stats
            per_region[name] = {"split": split, **stats}
            logger.info("%s [%s]: %s", name, split, dict(stats))
    yaml_path = write_yaml(args.out)
    (args.out / "stats.json").write_text(
        json.dumps({"summary": summary, "regions": per_region}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    for split, stats in summary.items():
        print(
            f"{split}: тайлов {stats['tiles']} (пустых {stats['tiles_empty']}); объекты: "
            + ", ".join(f"{c}={stats[c]}" for c in settings.CLASSES)
        )
    missing = [c for c in settings.CLASSES if summary["val"][c] == 0]
    if missing:
        print(f"ВНИМАНИЕ: в val нет классов {missing} — задайте settings.VAL_REGIONS вручную")
    print(f"готово: {yaml_path}")


if __name__ == "__main__":
    main()
