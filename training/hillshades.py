"""Источник хиллшейдов для датасета: пересчёт из LAS или готовые Li трейна.

На платформе хиллшейдов нет — решение строит их из LAS модулем
koz3.detectors.lidar_dem (дословный перенос алгоритма бейзлайна). Датасет по
умолчанию строится тем же модулем (источник «las»), поэтому обучение и инференс
видят одинаковые пиксели. Источник «li» (готовые растры трейна) быстрее, но
использовать его стоит только после check_hillshades.py.
"""

import logging
from pathlib import Path

import settings
from koz3.detectors.lidar_dem import hillshades_for_las

logger = logging.getLogger(__name__)


def tag_of(path: Path) -> str:
    """Вариант отмывки по суффиксу имени файла: _c, _ch, _g, _i (иначе '?')."""
    suffix = path.stem.rsplit("_", 1)[-1].lower()
    return suffix if suffix in ("c", "ch", "g", "i") else "?"


def from_las(las_path: Path, region_crs: str, cache_dir: Path) -> list[Path]:
    """Хиллшейды для LAS (кэшируются в cache_dir) — тем же кодом, что на инференсе."""
    out_dir = cache_dir / las_path.stem
    existing = sorted(out_dir.glob("*.tif"))
    if len(existing) >= len(settings.HILLSHADES):
        return [p for p in existing if tag_of(p) in settings.HILLSHADES]
    return hillshades_for_las(las_path, region_crs, out_dir)


def provided_li(region_dir: Path) -> list[Path]:
    """Готовые Li-хиллшейды региона (папка с меткой Li)."""
    paths = [
        p
        for p in sorted(region_dir.rglob("*.tif"))
        if "_li_" in p.as_posix().lower() and tag_of(p) in settings.HILLSHADES
    ]
    if not paths:
        logger.warning("%s: Li-хиллшейды не найдены", region_dir.name)
    return paths
