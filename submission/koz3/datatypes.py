"""Базовые типы данных решения: markup_type, классы, регион, детекция.

Только константы и dataclass-контракты из раздела 5 пакета контекста;
никакой логики.
"""

from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon

MARKUP_TYPES: tuple[str, ...] = ("li", "ae", "sp_or", "or", "maggr", "etgr", "terfl")
"""Допустимые типы исходных данных (markup_type)."""

CLASSES: tuple[str, ...] = (
    "kurgany_tselye",
    "kurgany_povrezhdennye",
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
    "finds_points",
    "object_poly",
)
"""Семь классов объектов; имена фиксированы метрикой соревнования."""


@dataclass(frozen=True)
class Region:
    """Регион: папка входных данных и её файлы по markup_type."""

    name: str
    path: Path
    utm_crs: str
    files: dict[str, list[Path]]


@dataclass
class Detection:
    """Одно обнаружение: геометрия в UTM региона плюс атрибуты."""

    geom: Polygon
    class_name: str
    confidence: float
    markup_type: str
    region_name: str
    source: str = ""
    extra: dict = field(default_factory=dict)