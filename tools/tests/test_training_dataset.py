"""Тесты подготовки датасета (training/) на синтетическом растре и разметке."""

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
import annotations  # noqa: E402
import build_dataset  # noqa: E402
import settings  # noqa: E402

CRS = "EPSG:32637"
ORIGIN_X, ORIGIN_Y, RES = 500_000.0, 6_000_000.0, 0.1


def _raster(path: Path, size_px: int) -> Path:
    data = np.full((size_px, size_px), 120, dtype=np.uint8)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size_px,
        width=size_px,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _annotation(path: Path, class_value: str, square_m: tuple[float, float, float]) -> Path:
    x, y, side = square_m
    ring = [[x, y], [x + side, y], [x + side, y - side], [x, y - side], [x, y]]
    data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{CRS.split(':')[1]}"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"class": class_value},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Курганы_целые", "kurgany_tselye"),
        ("курганы повреждённые", "kurgany_povrezhdennye"),
        ("200_Судьбищи_Li_Курганы_поврежденные", "kurgany_povrezhdennye"),
        ("Городище", "gorodishcha"),
        ("Курганы распаханные", "kurgany_povrezhdennye"),
        ("дороги", None),
    ],
)
def test_canonical_class(raw: str, expected: str | None) -> None:
    assert annotations.canonical_class(raw) == expected


def test_markup_type_from_name() -> None:
    assert annotations.markup_type_from_name(Path("200_X_Li_Курганы.geojson")) == "li"
    assert annotations.markup_type_from_name(Path("200_X_SpOr_Курганы.geojson")) == "sp_or"
    assert annotations.markup_type_from_name(Path("200_X_MagGr_ObjectPoly.geojson")) == "maggr"
    assert annotations.markup_type_from_name(Path("200_X_разметка.geojson")) is None


def test_tile_raster_writes_yolo_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "TILE_PX", 256)
    monkeypatch.setattr(settings, "NEGATIVE_TILE_SHARE", 0.0)
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    raster = _raster(tmp_path / "hs_g.tif", 512)
    # квадрат 5×5 м (50×50 пикс) с левым верхним углом в пикселе (100, 100)
    ann_path = _annotation(
        tmp_path / "r_Li_Курганы.geojson",
        "Курганы_поврежденные",
        (ORIGIN_X + 10.0, ORIGIN_Y - 10.0, 5.0),
    )
    ann = annotations.AnnotationFile(ann_path, "li", annotations._file_crs(ann_path))
    objects = {CRS: annotations.load_objects(ann, CRS)}
    stats = build_dataset.tile_raster(raster, objects, tmp_path, "train", "r")
    assert stats["kurgany_povrezhdennye"] >= 1
    label = (tmp_path / "labels" / "train" / "r__hs_g__y0_x0.txt").read_text(encoding="utf-8")
    class_id, *coords = label.split()
    assert int(class_id) == settings.CLASSES.index("kurgany_povrezhdennye")
    xs = [float(v) for v in coords[0::2]]
    ys = [float(v) for v in coords[1::2]]
    assert min(xs) == pytest.approx(100 / 256, abs=0.01)
    assert max(ys) == pytest.approx(150 / 256, abs=0.01)
    image = tmp_path / "images" / "train" / f"r__hs_g__y0_x0.{settings.IMAGE_FORMAT}"
    assert image.is_file()


def test_split_is_deterministic() -> None:
    assert build_dataset.split_of("200_A") == build_dataset.split_of("200_A")


def test_geophysics_classes_opt_in(tmp_path: Path) -> None:
    """Геофизика не попадает в датасет YOLO, но читается для GT и обучения геофизики."""
    path = _annotation(
        tmp_path / "r_MagGr_ObjectPoly.geojson", "ObjectPoly", (ORIGIN_X, ORIGIN_Y, 2.0)
    )
    ann = annotations.AnnotationFile(path, annotations.markup_type_from_name(path), CRS)
    assert ann.markup_type == "maggr"
    assert annotations.load_objects(ann, CRS) == []
    loaded = annotations.load_objects(ann, CRS, settings.ALL_CLASSES)
    assert [name for name, _ in loaded] == ["object_poly"]
