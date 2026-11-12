"""Чтение разметки трейна: файлы, тип данных, класс, геометрии в нужной CRS.

Формат файлов разметки организаторов нам известен не полностью, поэтому класс
объекта (feature_class): явное поле класса (settings.CLASS_FIELD_KEYS) → имя файла →
свободный текст (settings.NAME_FIELD_KEYS: name, layer, type). Тип данных — по меткам
в имени файла.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import shapely
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

import settings

_TOKEN_SPLIT = re.compile(r"[_\-\s.]+")


@dataclass
class AnnotationFile:
    """Файл разметки и то, что удалось о нём понять."""

    path: Path
    markup_type: str | None
    crs: str


def normalize_name(raw: str) -> str:
    """Имя класса → ключ словаря синонимов (нижний регистр, «_» вместо пробелов)."""
    return re.sub(r"[\s\-]+", "_", str(raw).strip().lower())


def canonical_class(raw: str | None) -> str | None:
    """Каноническое имя класса или None, если имя не распознано."""
    if raw is None:
        return None
    key = normalize_name(raw)
    if key in settings.CLASS_SYNONYMS:
        return settings.CLASS_SYNONYMS[key]
    # Самый длинный синоним первым: «курганы_поврежденные» не должен
    # распознаться как «курганы» (целые).
    for synonym in sorted(settings.CLASS_SYNONYMS, key=len, reverse=True):
        if synonym in key:
            return settings.CLASS_SYNONYMS[synonym]
    return None


def feature_class(props: dict[str, object], file_class: str | None) -> str | None:
    """Класс объекта: явное поле класса → класс из имени файла → name / layer / type.

    Свободный текст (имя объекта «Курган №5», слой) распознаётся по подстроке и не
    должен перекрывать класс файла «Курганы_поврежденные».
    """
    for key in settings.CLASS_FIELD_KEYS:
        if props.get(key) is not None and canonical_class(str(props[key])):
            return canonical_class(str(props[key]))
    if file_class is not None:
        return file_class
    for key in settings.NAME_FIELD_KEYS:
        if props.get(key) is not None and canonical_class(str(props[key])):
            return canonical_class(str(props[key]))
    return None


def _props_ignored(props: dict[str, object]) -> bool:
    """True, если классоподобное свойство — класс из игнор-листа."""
    for key in settings.CLASS_PROPERTY_KEYS:
        raw = props.get(key)
        if raw is not None and normalize_name(str(raw)) in settings.IGNORED_CLASSES:
            return True
    return False


def markup_type_from_name(path: Path) -> str | None:
    """Тип данных по меткам в имени файла (Li/La/Ae/SpOr/Or)."""
    tokens = [t.lower() for t in _TOKEN_SPLIT.split(path.stem) if t]
    for token in tokens:
        if token in settings.MARKUP_LABELS:
            return settings.MARKUP_LABELS[token]
    return None


def find_annotation_files(region_dir: Path) -> list[AnnotationFile]:
    """Все *.geojson в папках «разметка» региона."""
    files = []
    for path in sorted(region_dir.rglob("*.geojson")):
        if "разметк" not in path.parent.name.lower():
            continue
        files.append(AnnotationFile(path, markup_type_from_name(path), _file_crs(path)))
    return files


def _file_crs(path: Path) -> str:
    """CRS из поля crs файла; по умолчанию EPSG:3857 (формат платформы)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    name = ((data.get("crs") or {}).get("properties") or {}).get("name")
    return CRS.from_user_input(name).to_string() if name else "EPSG:3857"


def load_objects(
    ann: AnnotationFile, dst_crs: str, classes: tuple[str, ...] = settings.CLASSES
) -> list[tuple[str, BaseGeometry]]:
    """[(класс, геометрия в dst_crs)] из файла; классы вне ``classes`` пропускаются.

    По умолчанию — 5 поверхностных классов (датасет YOLO); для геофизики и GT
    передавайте settings.ALL_CLASSES. Точки остаются точками.
    """
    data = json.loads(ann.path.read_text(encoding="utf-8"))
    file_class = canonical_class(ann.path.stem)
    stem_ignored = any(name in normalize_name(ann.path.stem) for name in settings.IGNORED_CLASSES)
    transformer = None

    if CRS.from_user_input(ann.crs) != CRS.from_user_input(dst_crs):
        transformer = Transformer.from_crs(ann.crs, dst_crs, always_xy=True)
    objects = []
    for feature in data.get("features", []):
        geom_dict = feature.get("geometry")
        if not geom_dict or not geom_dict.get("coordinates"):
            continue
        props = feature.get("properties") or {}
        if stem_ignored or _props_ignored(props):
            continue
        class_name = feature_class(props, file_class)
        if class_name not in classes:
            continue
        geom = shape(geom_dict)
        if transformer is not None:
            geom = shapely.transform(geom, transformer.transform, interleaved=False)
        objects.append((class_name, shapely.make_valid(geom)))
    return objects
