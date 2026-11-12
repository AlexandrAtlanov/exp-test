"""Hard negatives (C24): уверенные ложные срабатывания модели на train-регионах → negatives.json.

Модель путает с курганами стога, деревья, современные насыпи (PDF «Вопрос-ответ»). Такие
места добавляются в датасет пустыми тайлами (build_dataset.py --negatives) — модель учится,
что это не объекты. Порядок (машина обучения):

    python tools/make_holdout.py --data <трейн> --regions <10–20 train-регионов> --out work/neg_src
    python submission/solution.py work/neg_src work/neg_pred.geojson
    python tools/mine_negatives.py --data <трейн> --pred work/neg_pred.geojson \
        --out work/negatives.json

Ложное срабатывание — предсказание лидара (markup_type li) с confidence ≥ --min-conf, которое
не задевает ни один объект разметки региона, расширенный на --buffer-m (рядом с настоящим
объектом — не «чистый» фон). Выход: {регион: {"crs": UTM региона, "boxes": [[minx, miny,
maxx, maxy], ...]}} — рамки в CRS отмывок датасета. Val-регионы указывать нельзя: они —
holdout для оценки.

Неразмеченные курганы: разметка археологов бывает неполной, и «ложное срабатывание» может
оказаться настоящим курганом — пустой тайл на нём научил бы модель обратному. Поэтому место,
где под предсказанием явная насыпь (локальный рельеф внутри выше кольца вокруг на
≥ --mound-contrast-m, как в audit_data), в negatives не идёт (счётчик «похоже на курган»).
DEM строится из LAS региона (~20 с на 160 МБ); --no-relief-check — без проверки.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box, shape

_REPO_DIR = Path(__file__).resolve().parents[1]
for _sub in ("training", "submission"):
    if str(_REPO_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(_REPO_DIR / _sub))

import settings  # noqa: E402
from annotations import find_annotation_files, load_objects  # noqa: E402
from audit_data import local_relief  # noqa: E402
from build_dataset import split_of  # noqa: E402
from koz3 import config  # noqa: E402
from koz3.detectors import lidar_dem  # noqa: E402
from koz3.discovery import find_regions  # noqa: E402

logger = logging.getLogger("mine_negatives")

_PRED_CRS = "EPSG:3857"
"""CRS выходного файла решения."""


def mound_contrasts(relief: np.ndarray, transform, polys: list, ring_m: float = 3.0) -> list:
    """Локальный рельеф внутри полигона минус в кольце ring_m вокруг, м; нет данных — None."""
    step = abs(transform.a)
    ring_px = max(1, int(round(ring_m / step)))
    finite = np.isfinite(relief)
    out: list = []
    for poly in polys:
        inside = rasterize([(poly, 1)], out_shape=relief.shape, transform=transform) > 0
        inside &= finite
        ring = ndimage.binary_dilation(inside, iterations=ring_px) & ~inside & finite
        if inside.sum() < 4 or not ring.any():
            out.append(None)
            continue
        out.append(float(relief[inside].mean() - relief[ring].mean()))
    return out


def _drop_mounds(region, polys: list, min_contrast_m: float) -> tuple[list, int]:
    """Убрать полигоны, под которыми явная насыпь (возможный неразмеченный курган)."""
    keep = [True] * len(polys)
    for las_path in region.files.get("li", []):
        try:
            bounds = box(*lidar_dem._las_header_info(las_path)["bounds"])
        except Exception as exc:  # noqa: BLE001  # битый LAS — без проверки по нему
            logger.warning("%s: %s не читается (%s)", region.name, las_path.name, exc)
            continue
        idx = [i for i, p in enumerate(polys) if keep[i] and p.intersects(bounds)]
        if not idx:
            continue
        dem, transform, _ = lidar_dem.build_dem_from_las(
            las_path, res=float(config.LIDAR["res_m"])
        )
        relief, step = local_relief(dem, abs(transform.a))
        work = transform * transform.scale(step / abs(transform.a))
        contrasts = mound_contrasts(relief, work, [polys[i] for i in idx])
        for i, contrast in zip(idx, contrasts, strict=True):
            if contrast is not None and contrast >= min_contrast_m:
                keep[i] = False
    kept = [p for p, k in zip(polys, keep, strict=True) if k]
    return kept, len(polys) - len(kept)


def mine(
    data_dir: Path,
    pred_path: Path,
    min_conf: float = 0.3,
    buffer_m: float = 10.0,
    mound_contrast_m: float | None = 0.15,
) -> dict[str, dict]:
    """{регион: {"crs", "boxes"}} — рамки ложных срабатываний лидара в UTM региона.

    mound_contrast_m — насыпь не ниже этого (м) под предсказанием → не negative (возможно,
    неразмеченный курган); None — без проверки рельефа.
    """
    features = json.loads(pred_path.read_text(encoding="utf-8")).get("features", [])
    by_region: dict[str, list] = {}
    for feature in features:
        props = feature.get("properties") or {}
        if props.get("markup_type") != "li" or float(props.get("confidence", 0)) < min_conf:
            continue
        by_region.setdefault(str(props.get("region_name")), []).append(shape(feature["geometry"]))
    regions = {r.name: r for r in find_regions(data_dir)}
    result: dict[str, dict] = {}
    for name, preds in sorted(by_region.items()):
        region = regions.get(name)
        if region is None:
            logger.warning("%s: региона нет в %s — пропуск", name, data_dir)
            continue
        if split_of(name) != "train":
            logger.warning("%s: val-регион (holdout) — пропуск", name)
            continue
        gt = [
            geom.buffer(buffer_m)
            for ann in find_annotation_files(region.path)
            for _, geom in load_objects(ann, region.utm_crs, classes=settings.ALL_CLASSES)
        ]
        tree = shapely.STRtree(gt) if gt else None
        to_utm = Transformer.from_crs(_PRED_CRS, region.utm_crs, always_xy=True)
        fps = []
        for pred in preds:
            utm = shapely.transform(pred, to_utm.transform, interleaved=False)
            if tree is not None and len(tree.query(utm, "intersects")):
                continue
            fps.append(utm)
        mounds = 0
        if mound_contrast_m is not None and fps:
            fps, mounds = _drop_mounds(region, fps, mound_contrast_m)
        boxes = [[round(v, 2) for v in fp.bounds] for fp in fps]
        result[name] = {"crs": region.utm_crs, "boxes": boxes, "skipped_mounds": mounds}
        logger.info(
            "%s: предсказаний %d, ложных %d, похоже на курган (не берём) %d",
            name,
            len(preds),
            len(boxes),
            mounds,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ложные срабатывания → negatives.json")
    parser.add_argument("--data", type=Path, required=True, help="папка трейна с разметкой")
    parser.add_argument("--pred", type=Path, required=True, help="выход solution.py (.geojson)")
    parser.add_argument("--out", type=Path, required=True, help="negatives.json")
    parser.add_argument("--min-conf", type=float, default=0.3, help="мин. confidence FP")
    parser.add_argument("--buffer-m", type=float, default=10.0, help="отступ от разметки, м")
    parser.add_argument(
        "--mound-contrast-m",
        type=float,
        default=0.15,
        help="насыпь не ниже (м) под FP — возможно неразмеченный курган, не берём",
    )
    parser.add_argument("--no-relief-check", action="store_true", help="без проверки рельефа")
    args = parser.parse_args(argv)
    contrast = None if args.no_relief_check else args.mound_contrast_m
    result = mine(args.data, args.pred, args.min_conf, args.buffer_m, contrast)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(v["boxes"]) for v in result.values())
    mounds = sum(v.get("skipped_mounds", 0) for v in result.values())
    print(
        f"Записано {args.out}: регионов {len(result)}, ложных срабатываний {total}; "
        f"похоже на неразмеченный курган (не взято) {mounds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
