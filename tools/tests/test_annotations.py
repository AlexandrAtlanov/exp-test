"""Тесты training/annotations.py: игнор-лист Struct*, класс файла, CRS.

Синтетическая разметка без реальных данных: контрактные поведения
load_objects (правка п. 7 задачи C11) и markup_type_from_name.
"""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest
from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry

import settings
from annotations import find_annotation_files, load_objects, markup_type_from_name


def _write_collection(path: Path, features: list[dict], crs: str) -> None:
    """Пишет синтетический файл разметки (папки создаёт)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _feature(geometry: BaseGeometry, **properties: object) -> dict:
    """GeoJSON-фича: геометрия shapely + произвольные properties."""
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def _load_single(
    tmp_path: Path, file_name: str, features: list[dict], crs: str = "EPSG:3857"
) -> list[tuple[str, object]]:
    """Объекты load_objects из одного файла разметки (в EPSG:3857)."""
    region = tmp_path / "001_Тестовый"
    _write_collection(region / "06_разметка" / file_name, features, crs)
    anns = find_annotation_files(region)
    assert len(anns) == 1
    return load_objects(anns[0], "EPSG:3857", classes=settings.ALL_CLASSES)


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("01_La_точки.geojson", "li"),
        ("02_Li_карты.geojson", "li"),
        ("03_Ae_аэрофото.geojson", "ae"),
        ("04_SpOr_спутник.geojson", "sp_or"),
        ("05_Or_ортофото.geojson", "or"),
        ("07_MagGr_карты.geojson", "maggr"),
        ("08_EtGr_профили.geojson", "etgr"),
        ("09_TerFl_карты.geojson", "terfl"),
    ],
)
def test_markup_type_from_name(file_name: str, expected: str) -> None:
    """Метка типа данных распознаётся для всех markup_type."""
    assert markup_type_from_name(Path(file_name)) == expected


def test_structrpoly_feature_skipped(tmp_path: Path) -> None:
    """StructrPoly в свойстве — фича пропущена, соседние остаются."""
    objects = _load_single(
        tmp_path,
        "TerFl_ObjectsPoly.geojson",
        [
            _feature(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), class_name="StructrPoly"),
            _feature(Point(5, 5), class_name="FindsPoints"),
            _feature(Polygon([(6, 0), (9, 0), (9, 3), (6, 3)]), class_name="ObjectPoly"),
        ],
    )
    assert [cls for cls, _ in objects] == ["finds_points", "object_poly"]


def test_ignored_file_stem_skips_all(tmp_path: Path) -> None:
    """Имя файла содержит StructPoly — пропускается весь файл."""
    objects = _load_single(
        tmp_path,
        "TerFl_StructPoly.geojson",
        [_feature(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), class_name="ObjectPoly")],
    )
    assert objects == []


def test_non_class_name_property_falls_back_to_file(tmp_path: Path) -> None:
    """name=«Объект 12» — не класс: фича остаётся с классом файла."""
    objects = _load_single(
        tmp_path,
        "Li_курганы.geojson",
        [_feature(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), name="Объект 12")],
    )
    assert [cls for cls, _ in objects] == ["kurgany_tselye"]


def test_unknown_class_skipped(tmp_path: Path) -> None:
    """Нераспознанный класс и класс в имени файла нет — фича пропущена."""
    objects = _load_single(
        tmp_path,
        "Ae_снимок.geojson",
        [_feature(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), class_name="Неизвестно")],
    )
    assert objects == []


def test_reprojection_utm_to_3857(tmp_path: Path) -> None:
    """Геометрии пересчитываются в EPSG:3857, точки остаются точками."""
    tr = Transformer.from_crs("EPSG:32637", "EPSG:3857", always_xy=True)
    expected_x, expected_y = tr.transform(500000, 5000000)
    objects = _load_single(
        tmp_path,
        "Li_курганы.geojson",
        [
            _feature(Point(500000, 5000000), class_name="FindsPoints"),
            _feature(
                Polygon(
                    [(500000, 5000000), (500030, 5000000), (500030, 5000030),
                     (500000, 5000030)]
                ),
                class_name="курганы_целые",
            ),
        ],
        crs="EPSG:32637",
    )
    assert [cls for cls, _ in objects] == ["finds_points", "kurgany_tselye"]
    point = objects[0][1]
    assert point.geom_type == "Point"
    assert point.x == pytest.approx(expected_x, abs=1e-6)
    assert point.y == pytest.approx(expected_y, abs=1e-6)
    assert objects[1][1].geom_type == "Polygon"