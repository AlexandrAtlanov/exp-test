"""Автоисправление разметки и генерация данных для build_dataset.py (только датасет).

Всё здесь работает при сборке датасета; решение на платформе не затрагивается.

Исправления (по каждому LAS):
- choose_objects: CRS разметки. Отмывка строится в координатах LAS, а подписывается CRS
  региона (UTM.json). Если разметка, переведённая в CRS из заголовка LAS, ложится на облако
  лучше, чем в CRS региона, берётся она — иначе такой регион молча давал пустые тайлы.
- estimate_shift: сдвиг разметки. Курганы — насыпи: ищется сдвиг (±8 м), при котором
  контуры курганов садятся на локальные возвышения лучше всего (audit_data.mound_alignment).
  Применяется ко всей разметке LAS, только если он значим и курганов не меньше
  settings.MIN_OBJECTS_FOR_SHIFT.

Новые данные (только train):
- extra_rasters: серые отмывки из того же DEM при других азимутах солнца
  (settings.EXTRA_AZIMUTHS) — те же объекты с другим направлением тени;
- synthetic_mounds: синтетические курганы — гауссовы купола на пустых участках DEM
  (диаметр, высота — settings.SYNTH_*), подпись — контур на 10 % высоты (так организаторы
  определяют границу распаханного кургана). Синтетические тайлы — только те, где есть
  синтетический курган.
Растры и результаты сдвига кэшируются в папке кэша отмывок региона — DEM строится один раз.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import rasterio
import shapely
from shapely.affinity import translate
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union

import settings
from annotations import load_objects
from audit_data import MOUND_CLASSES, local_relief, mound_alignment
from koz3.detectors import lidar_dem
from koz3.discovery import _las_wkt_crs

logger = logging.getLogger("augment_data")


def las_bounds(las_path: Path):
    """Габариты облака из заголовка LAS (shapely box); битый файл — исключение."""
    return box(*lidar_dem._las_header_info(las_path)["bounds"])


def choose_objects(
    ann_files, las_path: Path, region_crs: str, fix: bool = True
) -> tuple[list, str]:
    """Разметка LAS в той CRS, в которой она ложится на облако; (объекты, источник CRS)."""
    region_objects = [obj for ann in ann_files for obj in load_objects(ann, region_crs)]
    las_crs = _las_wkt_crs(las_path)
    if not fix or not las_crs or las_crs == region_crs:
        return region_objects, "регион"
    bounds = las_bounds(las_path)
    las_objects = [obj for ann in ann_files for obj in load_objects(ann, las_crs)]

    def hits(objects) -> int:
        return sum(1 for _, geom in objects if geom.intersects(bounds))

    if hits(las_objects) > hits(region_objects):
        logger.warning(
            "%s: разметка ложится на облако в CRS LAS %s, а не региона %s — берём CRS LAS",
            las_path.name,
            las_crs,
            region_crs,
        )
        return las_objects, f"LAS {las_crs}"
    return region_objects, "регион"


class LazyDem:
    """DEM одного LAS: строится при первом обращении и один раз (сдвиг и доп. растры)."""

    def __init__(self, las_path: Path) -> None:
        self.las_path = las_path
        self._value: tuple[np.ndarray, object] | None = None

    def get(self) -> tuple[np.ndarray, object]:
        """(DEM, привязка) — как у отмывок (шаг config.LIDAR["res_m"])."""
        if self._value is None:
            dem, transform, _ = lidar_dem.build_dem_from_las(
                self.las_path, res=float(settings.runtime_config.LIDAR["res_m"])
            )
            self._value = (dem, transform)
        return self._value


def shift_for(objects, dem: LazyDem, cache_dir: Path, key: str) -> dict | None:
    """Результат estimate_shift с кэшем в cache_dir/extra/<las>/shift_<key>.json."""
    # v2 (13.11): порог сдвига в метрах — результаты старых сборок не переиспользуются
    path = cache_dir / "extra" / dem.las_path.stem / f"shift_v2_{key}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")) or None
    mounds = [g for c, g in objects if c in MOUND_CLASSES]
    result = (
        estimate_shift(objects, *dem.get())
        if len(mounds) >= settings.MIN_OBJECTS_FOR_SHIFT
        else None
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result or {}), encoding="utf-8")
    return result


def estimate_shift(objects, dem, transform) -> dict | None:
    """Сдвиг разметки по курганам; None — мало курганов или нет данных."""
    bounds = box(*rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], transform))
    mounds = [g for c, g in objects if c in MOUND_CLASSES and g.intersects(bounds)]
    if len(mounds) < settings.MIN_OBJECTS_FOR_SHIFT:
        return None
    relief, step = local_relief(dem, abs(transform.a))
    work = transform * transform.scale(step / abs(transform.a))
    return mound_alignment(relief, work, mounds, settings.MAX_SHIFT_M)


def apply_shift(
    objects, result: dict | None, fix: bool = True
) -> tuple[list, tuple[float, float] | None]:
    """Сдвинуть всю разметку, если сдвиг значим; (объекты, применённый сдвиг)."""
    if not fix or not result or not result.get("shifted"):
        return objects, None
    east, north = result["best_shift_m"]
    moved = [(cls, translate(geom, xoff=east, yoff=north)) for cls, geom in objects]
    return moved, (east, north)


def render_gray(
    dem: np.ndarray, transform, crs: str, path: Path, azimuth: float, altitude: float = 45.0
) -> Path:
    """Серая отмывка как вариант «g» (nodata 255), но с заданным азимутом; по блокам строк."""
    valid = np.isfinite(dem)
    z_min = float(np.nanmin(dem))
    step = float(transform.a)
    height, width = dem.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with lidar_dem._open_writer(path, height, width, 1, transform, crs) as dst:
        for top in range(0, height, 4096):
            bottom = min(top + 4096, height)
            lo, hi = max(0, top - 1), min(height, bottom + 1)
            block = np.where(valid[lo:hi], dem[lo:hi], z_min)
            shade = lidar_dem.hillshade_esri(block, step, step, azimuth=azimuth, altitude=altitude)
            gray = np.rint(shade[top - lo : top - lo + (bottom - top)]).astype(np.uint8)
            gray[~valid[top:bottom]] = 255
            dst.write(gray, 1, window=((top, bottom), (0, width)))
    return path


def synthetic_mounds(dem: np.ndarray, transform, avoid, seed: int) -> tuple[np.ndarray, list]:
    """DEM с синтетическими курганами и их контуры (на 10 % высоты) в CRS растра."""
    rng = np.random.default_rng(seed)
    step = float(transform.a)
    valid = np.isfinite(dem)
    area_km2 = valid.sum() * step * step / 1e6
    wanted = int(round(area_km2 * settings.SYNTH_PER_KM2))
    blocked = unary_union([g.buffer(settings.SYNTH_MIN_GAP_M) for g in avoid]) if avoid else None
    if blocked is not None:
        shapely.prepare(blocked)  # тысячи проверок contains по одной геометрии
    out = dem.copy()
    placed: list = []
    # случайный пиксель с данными — отбором, без np.nonzero по всей сетке (до 60 Мпикс:
    # два массива индексов int64 ≈ 1 ГБ на процесс при --workers 12)
    attempts = wanted * 20 * max(1, int(round(valid.size / max(valid.sum(), 1))))
    for _ in range(attempts):
        if len(placed) >= wanted:
            break
        row, col = int(rng.integers(dem.shape[0])), int(rng.integers(dem.shape[1]))
        if not valid[row, col]:
            continue
        x, y = transform * (col + 0.5, row + 0.5)
        diameter = rng.uniform(*settings.SYNTH_DIAMETER_M)
        center = Point(x, y)
        if blocked is not None and blocked.contains(center):
            continue
        if any(center.distance(p) < settings.SYNTH_MIN_GAP_M + diameter for p in placed):
            continue
        sigma = (diameter / 2) / math.sqrt(2 * math.log(10))  # 10 % высоты на радиусе
        reach = int(math.ceil(3 * sigma / step))
        r0, r1 = max(row - reach, 0), min(row + reach + 1, dem.shape[0])
        c0, c1 = max(col - reach, 0), min(col + reach + 1, dem.shape[1])
        window = valid[r0:r1, c0:c1]
        if window.mean() < 0.95:  # купол целиком на данных
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        dist2 = ((rr - row) ** 2 + (cc - col) ** 2) * step * step
        height = rng.uniform(*settings.SYNTH_HEIGHT_M)
        out[r0:r1, c0:c1] += np.where(window, height * np.exp(-dist2 / (2 * sigma**2)), 0.0)
        placed.append(center.buffer(diameter / 2, 32))
    return out, placed


def extra_rasters(
    dem: LazyDem, crs: str, cache_dir: Path, objects, region_seed: int
) -> list[tuple[Path, list, list | None]]:
    """Дополнительные растры train: [(tif, объекты, только тайлы с этими объектами | None)].

    Отмывки при азимутах settings.EXTRA_AZIMUTHS (те же объекты) и отмывка 315° с
    синтетическими курганами (свои объекты: реальные + синтетические; тайлы — только с
    синтетическими). Кэш — cache_dir/extra/<las>/; DEM строится, только если кэша нет.
    """
    out_dir = cache_dir / "extra" / dem.las_path.stem
    az_paths = [out_dir / f"az{int(a):03d}.tif" for a in settings.EXTRA_AZIMUTHS]
    synth_tif, synth_json = out_dir / "synth_g.tif", out_dir / "synth.json"
    need_synth = settings.SYNTH_PER_KM2 > 0
    missing = [p for p in az_paths if not p.is_file()]
    if need_synth and not (synth_tif.is_file() and synth_json.is_file()):
        missing.append(synth_tif)
    if missing:
        array, transform = dem.get()
        for az, path in zip(settings.EXTRA_AZIMUTHS, az_paths, strict=True):
            if not path.is_file():
                render_gray(array, transform, crs, path, az)
        if need_synth and synth_tif in missing:
            avoid = [g for _, g in objects]
            synth_dem, polys = synthetic_mounds(array, transform, avoid, region_seed)
            render_gray(synth_dem, transform, crs, synth_tif, 315.0)
            synth_json.write_text(json.dumps([mapping(p) for p in polys]), encoding="utf-8")
            del synth_dem
    result: list[tuple[Path, list, list | None]] = [(p, objects, None) for p in az_paths]
    if need_synth:
        polys = [shape(g) for g in json.loads(synth_json.read_text(encoding="utf-8"))]
        if polys:
            synth_objects = objects + [("kurgany_tselye", p) for p in polys]
            result.append((synth_tif, synth_objects, polys))
    return result
