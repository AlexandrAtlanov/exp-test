"""Аудит данных лидара и датасета YOLO: битые файлы, CRS, сдвиг разметки, подписи.

Отвечает на вопросы «какие данные битые» и «совпадает ли разметка с рельефом».
Для каждого региона трейна:
1. LAS: читается ли заголовок, сколько точек, есть ли класс «земля» (2), совпадает ли CRS
   из заголовка LAS с CRS региона (UTM.json). Расхождение = вся разметка мимо картинки:
   отмывка строится в координатах LAS, а подписывается CRS региона.
2. Разметка лидара (*_Li_*.geojson): сколько объектов в файле и сколько дошло до датасета
   (остальные — нераспознанный класс, пустая геометрия, Struct*); класс по имени файла
   против класса из свойств объекта (name/class/...) — расхождения печатаются; доля объектов
   вне охвата LAS и над «нет данных».
3. Совпадение разметки с рельефом (курганы — насыпи): по DEM строится локальный рельеф
   (DEM минус сглаженный DEM); внутри контура кургана он должен быть выше, чем в кольце
   вокруг. Перебором сдвигов ±--max-shift м ищется сдвиг, при котором курганы «садятся» на
   насыпи лучше всего: если он заметно лучше нулевого — разметка смещена (CRS, ось, полпикселя).
4. Картинка-проверка <out>/regions/<регион>.png: отмывка g + контуры разметки по классам.
С --dataset дополнительно проверяется готовый датасет YOLO: пары картинка–подпись, формат
строк, id классов, координаты в [0, 1], вырожденные полигоны, подписи над «нет данных»,
битые png; --samples картинок с нарисованными подписями — в <out>/samples/.

    python training/audit_data.py --data <трейн> --cache work/hs_cache --out work/audit
    python training/audit_data.py --data <трейн> --cache work/hs_cache --out work/audit \
        --dataset work/ds_v3
    python training/audit_data.py --dataset work/ds_v3 --out work/audit_ds   # только датасет

Итог — <out>/report.md (таблицы, без координат объектов — можно пересылать) и report.json.
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union

import hillshades
import settings
from annotations import canonical_class, find_annotation_files, load_objects
from koz3.detectors import lidar_dem
from koz3.discovery import _las_wkt_crs, find_regions
from koz3.yolo_runtime import valid_pixel_mask

logger = logging.getLogger("audit_data")

MOUND_CLASSES = ("kurgany_tselye", "kurgany_povrezhdennye")
"""Классы-насыпи: по ним проверяется совпадение разметки с рельефом."""

CLASS_COLORS = {
    "kurgany_tselye": (255, 40, 40),
    "kurgany_povrezhdennye": (255, 160, 0),
    "gorodishcha": (40, 120, 255),
    "fortifikatsii": (200, 0, 200),
    "arkhitektury": (0, 200, 80),
}

WORK_RES_M = 0.5
"""Шаг сетки для проверки сдвига, м (DEM 0.1 м усредняется блоками — быстрее, сдвиг до 0.5 м)."""


# ---------------------------------------------------------------- рельеф и сдвиг


def local_relief(dem: np.ndarray, res_m: float, sigma_m: float = 10.0) -> tuple[np.ndarray, float]:
    """Локальный рельеф на сетке WORK_RES_M: DEM минус гаусс(DEM); вне данных — NaN.

    Возвращает (рельеф, шаг сетки, м).
    """
    k = max(1, int(round(WORK_RES_M / res_m)))
    h, w = (dem.shape[0] // k) * k, (dem.shape[1] // k) * k
    blocks = dem[:h, :w].reshape(h // k, k, w // k, k)
    with np.errstate(invalid="ignore"):
        coarse = np.nanmean(blocks, axis=(1, 3)) if k > 1 else dem[:h, :w].astype(np.float64)
    valid = np.isfinite(coarse)
    if not valid.any():
        return np.full(coarse.shape, np.nan), res_m * k
    filled = np.where(valid, coarse, np.nanmean(coarse))
    smooth = ndimage.gaussian_filter(filled, sigma_m / (res_m * k))
    return np.where(valid, filled - smooth, np.nan), res_m * k


def mound_alignment(
    relief: np.ndarray,
    transform,
    polygons: list,
    max_shift_m: float = 8.0,
    ring_m: float = 3.0,
    min_rel_gain: float = 0.2,
    min_gain_m: float = 0.05,
) -> dict | None:
    """Насколько контуры курганов совпадают с насыпями; None — мало пикселей под контурами.

    contrast_m — средний локальный рельеф внутри контуров минус в кольце ring_m вокруг
    (у насыпи > 0). best_shift_m — сдвиг (восток, север) контуров, при котором рельеф под
    ними в среднем максимален; gain_m — насколько это лучше нулевого сдвига; rel_gain — gain
    как доля лучшего значения. Смещение разметки: сдвиг ≥ 2 м, rel_gain ≥ min_rel_gain
    (на месте контуры дают меньше 80 % того, что дали бы сдвинутые; у верной разметки ≈ 0)
    и gain_m ≥ min_gain_m. Без порога в метрах (13.11, синтетика) низкие распаханные курганы
    (0.2 м) и участки без видимых насыпей давали «сдвиг» верной разметки в 4–11 случаях из 12:
    gain там ≈ 0.01 м (шум), у настоящего сдвига лесных курганов ≈ 0.2 м.
    """
    shapes = [(g, 1) for g in polygons if not g.is_empty]
    if not shapes:
        return None
    step = abs(transform.a)
    inside = rasterize(shapes, out_shape=relief.shape, transform=transform, fill=0) > 0
    finite = np.isfinite(relief)
    inside &= finite
    if inside.sum() < 10:
        return None
    ring_px = max(1, int(round(ring_m / step)))
    ring = ndimage.binary_dilation(inside, iterations=ring_px) & ~inside & finite
    contrast = float(relief[inside].mean() - relief[ring].mean()) if ring.any() else float("nan")

    rows, cols = np.nonzero(inside)
    filled = np.where(finite, relief, np.nan)
    span = int(round(max_shift_m / step))
    scores = np.full((2 * span + 1, 2 * span + 1), np.nan)
    height, width = relief.shape
    for i, dr in enumerate(range(-span, span + 1)):
        rr = rows + dr
        ok_r = (rr >= 0) & (rr < height)
        for j, dc in enumerate(range(-span, span + 1)):
            cc = cols + dc
            ok = ok_r & (cc >= 0) & (cc < width)
            if ok.sum() < 0.5 * len(rows):
                continue
            scores[i, j] = np.nanmean(filled[rr[ok], cc[ok]])
    if not np.isfinite(scores).any():
        return None
    zero = scores[span, span]
    bi, bj = np.unravel_index(np.nanargmax(scores), scores.shape)
    best = scores[bi, bj]
    east_m, north_m = float((bj - span) * step), float(-(bi - span) * step)  # строка вниз = юг
    shift_m = float(np.hypot(east_m, north_m))
    gain = float(best - zero) if np.isfinite(zero) else float("nan")
    rel_gain = gain / best if best > 0 and np.isfinite(gain) else 0.0
    return {
        "objects": len(shapes),
        "contrast_m": round(contrast, 3),
        "best_shift_m": (round(east_m, 2), round(north_m, 2)),
        "gain_m": round(gain, 3),
        "rel_gain": round(float(rel_gain), 2),
        "shifted": bool(
            shift_m >= 2.0
            and rel_gain >= min_rel_gain
            and np.isfinite(gain)
            and gain >= min_gain_m
        ),
    }


# ---------------------------------------------------------------- регион


def las_report(las_path: Path, region_crs: str) -> dict:
    """Заголовок LAS: читается ли, точки, габариты, CRS против CRS региона."""
    report: dict = {"file": las_path.name}
    try:
        header = lidar_dem._las_header_info(las_path)
    except Exception as exc:  # noqa: BLE001  # битый файл — в отчёт, не падаем
        report["error"] = str(exc)
        return report
    min_x, min_y, max_x, max_y = header["bounds"]
    las_crs = _las_wkt_crs(las_path)
    report.update(
        points=header["point_count"],
        size_m=(round(max_x - min_x), round(max_y - min_y)),
        las_crs=las_crs,
        crs_mismatch=bool(las_crs and las_crs != region_crs),
    )
    return report


def annotation_report(ann, crs: str, coverage) -> tuple[dict, list]:
    """Файл разметки лидара: дошло ли до датасета, классы, охват. Возвращает (отчёт, объекты)."""
    data = json.loads(ann.path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    objects = load_objects(ann, crs)
    file_class = canonical_class(ann.path.stem)
    disagree = Counter()
    for feature in features:
        props = feature.get("properties") or {}
        for key in settings.CLASS_PROPERTY_KEYS:
            if props.get(key) is not None and canonical_class(props[key]):
                prop_class = canonical_class(props[key])
                if file_class and prop_class != file_class:
                    disagree[f"{key}: файл {file_class} → объект {prop_class}"] += 1
                break
    outside = sum(1 for _, g in objects if coverage is not None and not g.intersects(coverage))
    return {
        "file": ann.path.name,
        "crs": ann.crs,
        "features": len(features),
        "loaded": len(objects),
        "dropped": len(features) - len(objects),
        "classes": dict(Counter(c for c, _ in objects)),
        "file_vs_property": dict(disagree),
        "outside_las": outside,
        "invalid_source": sum(
            1 for f in features if not (f.get("geometry") or {}).get("coordinates")
        ),
    }, objects


def draw_overlay(hillshade: Path, objects: list, out_png: Path, max_side: int = 1600) -> None:
    """Отмывка + контуры разметки по классам (уменьшенная копия)."""
    with rasterio.open(hillshade) as src:
        scale = max(src.width, src.height) / max_side
        out_w = max(1, int(src.width / max(scale, 1)))
        out_h = max(1, int(src.height / max(scale, 1)))
        band = src.read(1, out_shape=(out_h, out_w))
        inv = ~src.transform
    img = Image.fromarray(band.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    factor = max(scale, 1)
    for cls, geom in objects:
        for part in getattr(geom, "geoms", [geom]):
            if part.geom_type != "Polygon":
                continue
            pts = [inv * xy for xy in part.exterior.coords]
            draw.line(
                [(c / factor, r / factor) for c, r in pts], fill=CLASS_COLORS.get(cls), width=2
            )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def audit_region(region, cache_dir: Path, out_dir: Path, max_shift_m: float, dem: bool) -> dict:
    """Отчёт по региону: LAS, разметка, совпадение с рельефом, картинка-проверка."""
    las_files = region.files.get("li", [])
    report: dict = {
        "region": region.name,
        "crs": region.utm_crs,
        "las": [las_report(p, region.utm_crs) for p in las_files],
    }
    good_las = [p for p, r in zip(las_files, report["las"], strict=True) if "error" not in r]
    coverage = None
    if good_las:
        boxes = [box(*lidar_dem._las_header_info(p)["bounds"]) for p in good_las]
        coverage = unary_union(boxes)
    anns = [a for a in find_annotation_files(region.path) if a.markup_type == "li"]
    objects: list = []
    report["annotations"] = []
    for ann in anns:
        ann_report, ann_objects = annotation_report(ann, region.utm_crs, coverage)
        report["annotations"].append(ann_report)
        objects += ann_objects
    report["alignment"] = []
    for las_path in good_las:
        try:
            rasters = hillshades.from_las(las_path, region.utm_crs, cache_dir / region.name)
        except Exception as exc:  # noqa: BLE001
            report.setdefault("hillshade_errors", []).append(f"{las_path.name}: {exc}")
            continue
        gray = next((p for p in rasters if hillshades.tag_of(p) == "g"), None)
        if gray is not None:
            with rasterio.open(gray) as src:
                image = src.read()
            nodata = 1.0 - float(
                valid_pixel_mask(np.repeat(image[:1], 3, 0).transpose(1, 2, 0)).mean()
            )
            report.setdefault("nodata_share", {})[las_path.name] = round(nodata, 3)
            draw_overlay(
                gray, objects, out_dir / "regions" / f"{region.name}__{las_path.stem}.png"
            )
        if not dem:
            continue
        mounds = [g for c, g in objects if c in MOUND_CLASSES]
        if not mounds:
            continue
        dem_array, transform, _ = lidar_dem.build_dem_from_las(las_path, res=0.1)
        relief, step = local_relief(dem_array, abs(transform.a))
        work_transform = transform * transform.scale(step / abs(transform.a))
        result = mound_alignment(relief, work_transform, mounds, max_shift_m)
        if result is not None:
            report["alignment"].append({"las": las_path.name, **result})
    return report


# ---------------------------------------------------------------- датасет YOLO


def _drawable(line: str, n_classes: int) -> bool:
    """Строка подписи разбирается и пригодна для рисования примера."""
    parts = line.split()
    try:
        cls = int(parts[0])
        xy = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
    except (ValueError, IndexError):
        return False
    return 0 <= cls < n_classes and len(xy) >= 3


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def audit_dataset(dataset: Path, out_dir: Path, samples: int) -> dict:
    """Проверка датасета YOLO-seg: пары, формат подписей, подписи над «нет данных», png."""
    n_classes = len(settings.CLASSES)
    report: dict = {"dataset": str(dataset)}
    meta_path = dataset / "dataset_meta.json"
    tile_px = json.loads(meta_path.read_text())["tile_px"] if meta_path.is_file() else None
    sampled: list[tuple[Path, list[str]]] = []
    for split in ("train", "val"):
        images = {p.stem: p for p in (dataset / "images" / split).glob("*.png")}
        labels = {p.stem: p for p in (dataset / "labels" / split).glob("*.txt")}
        problems: Counter = Counter()
        examples: dict[str, list[str]] = {}
        per_class: Counter = Counter()

        def note(kind: str, name: str) -> None:
            problems[kind] += 1
            examples.setdefault(kind, [])
            if len(examples[kind]) < 5:
                examples[kind].append(name)

        for stem in sorted(set(images) - set(labels)):
            note("картинка без подписи", stem)
        for stem in sorted(set(labels) - set(images)):
            note("подпись без картинки", stem)
        for stem in sorted(set(images) & set(labels)):
            try:
                with Image.open(images[stem]) as im:
                    im.load()
                    array = np.asarray(im.convert("RGB"))
            except Exception:  # noqa: BLE001
                note("битый png", stem)
                continue
            if tile_px and array.shape[:2] != (tile_px, tile_px):
                note("размер не равен tile_px", stem)
            nodata = ~valid_pixel_mask(array)
            lines = [
                ln for ln in labels[stem].read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            for line in lines:
                parts = line.split()
                try:
                    cls = int(parts[0])
                    xy = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
                except (ValueError, IndexError):
                    note("строка не разбирается", stem)
                    continue
                if not 0 <= cls < n_classes:
                    note("id класса вне диапазона", stem)
                    continue
                per_class[settings.CLASSES[cls]] += 1
                if len(xy) < 3:
                    note("полигон < 3 точек", stem)
                    continue
                if (xy < -1e-6).any() or (xy > 1 + 1e-6).any():
                    note("координаты вне [0, 1]", stem)
                if _polygon_area(xy) * array.shape[0] * array.shape[1] < 4:
                    note("полигон меньше 4 пикс²", stem)
                mask = Image.new("L", array.shape[1::-1], 0)
                pts = [(x * array.shape[1], y * array.shape[0]) for x, y in xy]
                ImageDraw.Draw(mask).polygon(pts, fill=1)
                inside = np.asarray(mask, dtype=bool)
                if inside.any() and nodata[inside].mean() > 0.5:
                    note("объект над «нет данных» (> 50 %)", stem)
            drawable = [ln for ln in lines if _drawable(ln, n_classes)]
            if drawable:
                sampled.append((images[stem], drawable))
        report[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": dict(per_class),
            "problems": dict(problems),
            "examples": examples,
        }
    rng = random.Random(settings.SEED)
    for image_path, lines in rng.sample(sampled, min(samples, len(sampled))):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for line in lines:
            parts = line.split()
            xy = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
            pts = [(x * img.width, y * img.height) for x, y in xy]
            color = CLASS_COLORS.get(settings.CLASSES[int(parts[0])], (255, 255, 0))
            draw.line(pts + pts[:1], fill=color, width=2)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        img.save(out_dir / "samples" / image_path.name)
    return report


# ---------------------------------------------------------------- отчёт


def render_markdown(regions: list[dict], dataset: dict | None) -> str:
    """report.md: без координат объектов — можно пересылать."""
    lines = ["# Аудит данных", ""]
    if regions:
        lines += [
            "## LAS",
            "",
            "| Регион | Файл | Точек | Размер, м | CRS LAS | Проблема |",
            "|---|---|---|---|---|---|",
        ]
        for reg in regions:
            for las in reg["las"]:
                problem = las.get("error") or (
                    "CRS LAS ≠ CRS региона" if las.get("crs_mismatch") else ""
                )
                lines.append(
                    f"| {reg['region']} | {las['file']} | {las.get('points', '—')} | "
                    f"{las.get('size_m', '—')} | {las.get('las_crs', '—')} | {problem} |"
                )
            for err in reg.get("hillshade_errors", []):
                lines.append(f"| {reg['region']} | {err} | | | | отмывка не построилась |")
        lines += [
            "",
            "## Разметка лидара",
            "",
            "| Регион | Файл | В файле | В датасет | Отброшено | Вне LAS | Классы "
            "| Класс файла ≠ свойства |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for reg in regions:
            for ann in reg["annotations"]:
                lines.append(
                    f"| {reg['region']} | {ann['file']} | {ann['features']} | {ann['loaded']} | "
                    f"{ann['dropped']} | {ann['outside_las']} | {ann['classes']} | "
                    f"{ann['file_vs_property'] or ''} |"
                )
        lines += [
            "",
            "## Совпадение курганов с насыпями",
            "",
            "Контраст > 0 — внутри контура выше, чем вокруг (норма для насыпи). «Сдвиг» — где",
            "контуры садятся на насыпи лучше всего; СМЕЩЕНО — сдвиг ≥ 2 м и доля выигрыша",
            "≥ 0.2.",
            "",
            "| Регион | LAS | Курганов | Контраст, м | Лучший сдвиг (В, С), м | Выигрыш, м "
            "| Доля | Вывод |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for reg in regions:
            for al in reg["alignment"]:
                verdict = (
                    "СМЕЩЕНО"
                    if al["shifted"]
                    else ("нет насыпи" if al["contrast_m"] <= 0 else "ok")
                )
                lines.append(
                    f"| {reg['region']} | {al['las']} | {al['objects']} | {al['contrast_m']} | "
                    f"{al['best_shift_m']} | {al['gain_m']} | {al['rel_gain']} | {verdict} |"
                )
    if dataset:
        lines += ["", f"## Датасет YOLO ({dataset['dataset']})", ""]
        for split in ("train", "val"):
            part = dataset[split]
            lines.append(
                f"**{split}:** картинок {part['images']}, подписей {part['labels']}, объекты "
                f"{part['objects']}"
            )
            for kind, count in part["problems"].items():
                lines.append(f"- {kind}: {count} (например: {', '.join(part['examples'][kind])})")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Аудит данных лидара и датасета YOLO")
    parser.add_argument("--data", type=Path, help="папка трейна с регионами")
    parser.add_argument("--cache", type=Path, default=Path("hillshade_cache"), help="кэш отмывок")
    parser.add_argument("--out", type=Path, required=True, help="папка отчёта")
    parser.add_argument("--regions", nargs="*", help="только эти регионы")
    parser.add_argument("--dataset", type=Path, help="готовый датасет YOLO (build_dataset.py)")
    parser.add_argument("--samples", type=int, default=40, help="картинок датасета с подписями")
    parser.add_argument("--max-shift", type=float, default=8.0, help="поиск сдвига ±м")
    parser.add_argument("--no-dem", action="store_true", help="без проверки сдвига (быстрее)")
    args = parser.parse_args(argv)
    if args.data is None and args.dataset is None:
        parser.error("нужен --data и/или --dataset")
    args.out.mkdir(parents=True, exist_ok=True)
    regions_report: list[dict] = []
    if args.data is not None:
        regions = find_regions(args.data)
        if args.regions:
            regions = [r for r in regions if r.name in set(args.regions)]
        for region in regions:
            logger.info("регион %s", region.name)
            try:
                regions_report.append(
                    audit_region(region, args.cache, args.out, args.max_shift, not args.no_dem)
                )
            except Exception as exc:  # noqa: BLE001  # один регион не роняет аудит
                logger.exception("регион %s: аудит упал", region.name)
                regions_report.append(
                    {"region": region.name, "las": [{"file": "—", "error": str(exc)}],
                     "annotations": [], "alignment": []}
                )  # fmt: skip
    dataset_report = audit_dataset(args.dataset, args.out, args.samples) if args.dataset else None
    (args.out / "report.json").write_text(
        json.dumps({"regions": regions_report, "dataset": dataset_report}, ensure_ascii=False,
                   indent=1, default=str),
        encoding="utf-8",
    )  # fmt: skip
    (args.out / "report.md").write_text(
        render_markdown(regions_report, dataset_report), encoding="utf-8"
    )
    print(
        f"готово: {args.out / 'report.md'}; картинки — "
        f"{args.out / 'regions'}, {args.out / 'samples'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
