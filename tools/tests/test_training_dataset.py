"""Тесты подготовки датасета (training/) на синтетическом растре и разметке."""

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import numpy as np
import pytest
import rasterio
from PIL import Image
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
    opts = build_dataset.TileOptions(tile_px=256, overlap=0.25, scales=(1,), rot90=False)
    stats = build_dataset.tile_raster(raster, objects, tmp_path, "train", "r", opts)
    assert stats["kurgany_povrezhdennye"] >= 1
    label_path = tmp_path / "labels" / "train" / "r__hs_g__s1__y0_x0.txt"
    label = label_path.read_text(encoding="utf-8")
    class_id, *coords = label.split()
    assert int(class_id) == settings.CLASSES.index("kurgany_povrezhdennye")
    xs = [float(v) for v in coords[0::2]]
    ys = [float(v) for v in coords[1::2]]
    assert min(xs) == pytest.approx(100 / 256, abs=0.01)
    assert max(ys) == pytest.approx(150 / 256, abs=0.01)
    image = tmp_path / "images" / "train" / f"r__hs_g__s1__y0_x0.{settings.IMAGE_FORMAT}"
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


def _dataset_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def _polygon_mask(line: str, size: int) -> np.ndarray:
    """Растеризовать подпись YOLO-seg (одна строка) в маску size × size."""
    from PIL import ImageDraw

    values = [float(v) for v in line.split()[1:]]
    points = [(values[i] * size, values[i + 1] * size) for i in range(0, len(values), 2)]
    canvas = Image.new("L", (size, size), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


def test_rotated_labels_match_rotated_image() -> None:
    """Подпись, повёрнутая rotate_lines, совпадает с np.rot90 исходной маски."""
    size = 64
    line = "1 0.100000 0.200000 0.500000 0.200000 0.500000 0.400000 0.100000 0.400000"
    mask = _polygon_mask(line, size)
    for k in (1, 2, 3):
        [rotated] = build_dataset.rotate_lines([line], k)
        assert rotated.split()[0] == "1"
        expected = np.rot90(mask, k)
        got = _polygon_mask(rotated, size)
        assert (expected ^ got).sum() <= 2 * size  # расхождение — только пиксели границы


def test_scales_rotations_and_meta(tmp_path: Path, monkeypatch) -> None:
    """Масштабы 1 и 2, повороты только в train, dataset_meta.json для паспорта весов."""
    monkeypatch.setattr(settings, "NEGATIVE_TILE_SHARE", 0.0)
    _dataset_dirs(tmp_path)
    raster = _raster(tmp_path / "hs_c.tif", 512)
    ann_path = _annotation(
        tmp_path / "r_Li_Курганы.geojson", "Курганы_целые", (ORIGIN_X + 10.0, ORIGIN_Y - 10.0, 8.0)
    )
    ann = annotations.AnnotationFile(ann_path, "li", annotations._file_crs(ann_path))
    objects = {CRS: annotations.load_objects(ann, CRS)}
    opts = build_dataset.TileOptions(tile_px=256, overlap=0.1, scales=(1, 2), rot90=True)

    stats = build_dataset.tile_raster(raster, objects, tmp_path, "train", "r", opts)
    assert stats["tiles_s1"] >= 1 and stats["tiles_s2"] == 1  # 512 → 256: один тайл
    assert stats["tiles_rot90"] == 3 * (stats["tiles"] - stats["tiles_empty"])
    names = {p.name for p in (tmp_path / "images" / "train").iterdir()}
    assert "r__hs_g__s2__y0_x0.png" not in names  # имя растра — hs_c
    assert {"r__hs_c__s2__y0_x0.png", "r__hs_c__s2__y0_x0__r1.png"} <= names
    # масштаб 2: квадрат 8 м = 40 пикс при 0.2 м/пикс, левый край на 50 пикс
    label = (tmp_path / "labels" / "train" / "r__hs_c__s2__y0_x0.txt").read_text()
    xs = [float(v) for v in label.split()[1::2]]
    assert min(xs) == pytest.approx(50 / 256, abs=0.01)
    assert max(xs) == pytest.approx(90 / 256, abs=0.01)

    val_stats = build_dataset.tile_raster(raster, objects, tmp_path, "val", "r", opts)
    assert val_stats["tiles_rot90"] == 0

    meta = json.loads(build_dataset.write_meta(tmp_path, opts).read_text(encoding="utf-8"))
    assert meta["tile_px"] == 256 and meta["scales"] == [1, 2]
    import train

    tiling = train.dataset_tiling(tmp_path / "dataset.yaml")
    assert tiling == {
        "tile_px": 256,
        "scales": [1.0, 2.0],
        "rot90": True,
        "hillshades": list(settings.HILLSHADES),
    }


def test_rare_class_tiles_repeated_in_train_only(tmp_path: Path, monkeypatch) -> None:
    """Тайл с городищем (редкий класс) и его повороты — ещё rare_repeat копий, только train."""
    monkeypatch.setattr(settings, "NEGATIVE_TILE_SHARE", 0.0)
    _dataset_dirs(tmp_path)
    raster = _raster(tmp_path / "hs_c.tif", 256)
    ann_path = _annotation(
        tmp_path / "r_Li_Городище.geojson", "Городище", (ORIGIN_X + 5.0, ORIGIN_Y - 5.0, 8.0)
    )
    ann = annotations.AnnotationFile(ann_path, "li", annotations._file_crs(ann_path))
    objects = {CRS: annotations.load_objects(ann, CRS)}
    opts = build_dataset.TileOptions(
        tile_px=256, overlap=0.1, scales=(1,), rot90=True, rare_repeat=2
    )

    stats = build_dataset.tile_raster(raster, objects, tmp_path, "train", "r", opts)
    assert stats["tiles"] == 1 and stats["tiles_rot90"] == 3
    assert stats["tiles_rare_repeat"] == 2 * 4  # 2 копии × (тайл + 3 поворота)
    names = {p.stem for p in (tmp_path / "labels" / "train").iterdir()}
    assert {"r__hs_c__s1__y0_x0__dup1", "r__hs_c__s1__y0_x0__r3__dup2"} <= names
    base = (tmp_path / "labels" / "train" / "r__hs_c__s1__y0_x0.txt").read_text()
    dup = (tmp_path / "labels" / "train" / "r__hs_c__s1__y0_x0__dup1.txt").read_text()
    assert base == dup

    val_stats = build_dataset.tile_raster(raster, objects, tmp_path, "val", "r", opts)
    assert val_stats["tiles_rare_repeat"] == 0


def test_common_class_not_repeated(tmp_path: Path, monkeypatch) -> None:
    """Курганы — не редкий класс: повторов нет; has_rare_class по id класса."""
    monkeypatch.setattr(settings, "NEGATIVE_TILE_SHARE", 0.0)
    _dataset_dirs(tmp_path)
    raster = _raster(tmp_path / "hs_c.tif", 256)
    ann_path = _annotation(
        tmp_path / "r_Li_Курганы.geojson", "Курганы_целые", (ORIGIN_X + 5.0, ORIGIN_Y - 5.0, 8.0)
    )
    ann = annotations.AnnotationFile(ann_path, "li", annotations._file_crs(ann_path))
    objects = {CRS: annotations.load_objects(ann, CRS)}
    opts = build_dataset.TileOptions(tile_px=256, overlap=0.1, scales=(1,), rot90=False)
    stats = build_dataset.tile_raster(raster, objects, tmp_path, "train", "r", opts)
    assert stats["tiles_rare_repeat"] == 0
    gid = settings.CLASSES.index("gorodishcha")
    assert build_dataset.has_rare_class([f"{gid} 0.1 0.1 0.2 0.1 0.2 0.2"])
    assert not build_dataset.has_rare_class(["0 0.1 0.1 0.2 0.1 0.2 0.2"])


@pytest.mark.parametrize(
    ("props", "file_class", "expected"),
    [
        # имя объекта не перекрывает класс файла (раньше — «целые» по подстроке «курган»)
        ({"name": "Курган №5"}, "kurgany_povrezhdennye", "kurgany_povrezhdennye"),
        ({"layer": "Курганы"}, "kurgany_povrezhdennye", "kurgany_povrezhdennye"),
        # явное поле класса главнее файла
        ({"class_name": "Городище"}, "kurgany_povrezhdennye", "gorodishcha"),
        ({"class": "Курганы_поврежденные", "name": "Курган"}, None, "kurgany_povrezhdennye"),
        # файл без класса — берётся свободный текст
        ({"name": "Курганы целые"}, None, "kurgany_tselye"),
        ({"name": "дорога"}, None, None),
    ],
)
def test_feature_class_priority(props: dict, file_class: str | None, expected) -> None:
    assert annotations.feature_class(props, file_class) == expected
