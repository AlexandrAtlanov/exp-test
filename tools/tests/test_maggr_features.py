"""Тесты koz3.geophys.maggr_features на синтетике (sklearn не нужен)."""

from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.transform import xy as tr_xy
from scipy.spatial import cKDTree
from shapely.geometry import Point, box

from koz3 import config
from koz3.geophys.maggr_features import (
    POINT_FEATURE_NAMES,
    POLY_FEATURE_NAMES,
    gray_raster_paths,
    load_field,
    nms_points,
    nms_polys,
    point_candidates,
    poly_candidates,
    survey_mask,
)

_CRS = "EPSG:32637"


def _write_gray_tif(path: Path, arr: np.ndarray) -> None:
    """Запись одноканального uint8 GeoTIFF с сеткой 0.1 м."""
    transform = from_origin(500000.0, 5600000.0, 0.1, 0.1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        crs=_CRS,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


@pytest.fixture
def maggr_tif(tmp_path):
    """Синтетическая серая карта: краевой фон, шум, диполь ±40 и блоб +60."""
    rng = np.random.default_rng(42)
    h = w = 260
    y, x = np.mgrid[0:h, 0:w]
    v = 100.0 + rng.normal(0.0, 3.0, (h, w))
    v += 40.0 * np.exp(-((y - 130) ** 2 + (x - 100) ** 2) / 50.0)
    v -= 40.0 * np.exp(-((y - 130) ** 2 + (x - 110) ** 2) / 50.0)
    v += 60.0 * np.exp(-((y - 190) ** 2 + (x - 190) ** 2) / 800.0)
    arr = np.clip(v, 1, 254).astype(np.uint8)  # данные не касаются 0/255 фона
    arr[:4, :] = 0  # краевой фон ставим после clip, иначе он станет 1 и маска его не увидит
    arr[-4:, :] = 0
    arr[:, :4] = 0
    arr[:, -4:] = 0
    path = tmp_path / "участок_серый.tif"
    _write_gray_tif(path, arr)
    return path


def _xy_of(field, row, col):
    """Координаты (x, y) центра пикселя в CRS растра."""
    x, y = tr_xy(field.transform, row, col)
    return np.array([x, y], dtype=np.float64)


def _has_candidate(xy, target, tol_m):
    """Есть ли кандидат ближе tol_m от target."""
    if len(xy) == 0:
        return False
    dist, _ = cKDTree(xy).query(target.reshape(1, 2))
    return bool(dist[0] <= tol_m)


def test_survey_mask_without_outline():
    """Маска по растру: краевой фон убран, изолированные 0/255 остаются данными."""
    v = np.full((50, 50), 100, np.uint8)
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    v[20:23, 30:33] = 255  # изолированная «дырка»: не связана с краем — валидные данные
    mask = survey_mask(v, erode_px=2)
    assert not mask[0, 0]
    assert not mask[2, 25] and mask[4, 25]  # эрозия кромки данных
    assert mask[25, 25]
    assert mask[21, 31]


def test_load_field(maggr_tif):
    """Поле: форма сетки, маска без краёв, z конечен и видит диполь."""
    field = load_field(maggr_tif, config.MAGGR)
    assert field.z.shape == (260, 260)
    assert field.mask.dtype == bool and field.mask.any()
    assert not field.mask[:6, :].any()
    assert np.isfinite(field.z).all()
    assert field.z[130, 100] > 5
    assert field.z[130, 110] < -5


def test_point_candidates(maggr_tif):
    """Кандидаты точек: оба полюса диполя найдены, признаки (N, 16) конечны."""
    field = load_field(maggr_tif, config.MAGGR)
    xy, feats = point_candidates(field, config.MAGGR)
    assert xy.ndim == 2 and xy.shape[1] == 2
    assert feats.shape == (len(xy), len(POINT_FEATURE_NAMES))
    assert np.isfinite(feats).all()
    assert _has_candidate(xy, _xy_of(field, 130, 100), 0.5)
    assert _has_candidate(xy, _xy_of(field, 130, 110), 0.5)


def test_poly_candidates(maggr_tif):
    """Полигон на блобе: правильные формы, валиден, разумная площадь."""
    field = load_field(maggr_tif, config.MAGGR)
    polys, feats = poly_candidates(field, config.MAGGR)
    assert feats.shape == (len(polys), len(POLY_FEATURE_NAMES))
    assert polys and np.isfinite(feats).all()
    assert all(p.is_valid for p in polys)
    center = _xy_of(field, 190, 190)
    near = [p for p in polys if p.distance(Point(*center)) < 2.0]
    assert near
    assert any(1.0 < p.area < 500.0 for p in near)


def test_empty_raster(tmp_path):
    """Растр без данных: маска пуста, кандидаты пустых форм без падения."""
    path = tmp_path / "пусто_серый.tif"
    _write_gray_tif(path, np.zeros((100, 100), np.uint8))
    field = load_field(path, config.MAGGR)
    assert not field.mask.any()
    assert np.isfinite(field.z).all()
    xy, feats = point_candidates(field, config.MAGGR)
    assert xy.shape == (0, 2) and feats.shape == (0, len(POINT_FEATURE_NAMES))
    polys, feats = poly_candidates(field, config.MAGGR)
    assert polys == [] and feats.shape == (0, len(POLY_FEATURE_NAMES))


def test_nms_points():
    """NMS точек: сосед в 0.2 м подавлен, ниже порога не берётся."""
    xy = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0]])
    assert nms_points(xy, np.array([0.9, 0.8, 0.7]), 0.5, config.MAGGR) == [0, 2]
    assert nms_points(xy, np.array([0.9, 0.4, 0.3]), 0.5, config.MAGGR) == [0]


def test_nms_polys():
    """NMS полигонов: перекрытие подавлено, далёкий оставлен, порог работает."""
    polys = [box(0, 0, 10, 10), box(1, 1, 9, 9), box(100, 100, 110, 110)]
    scores = np.array([0.9, 0.8, 0.7])
    assert nms_polys(polys, scores, 0.5, config.MAGGR) == [0, 2]
    assert nms_polys(polys, scores, 0.85, config.MAGGR) == [0]


def test_gray_raster_paths():
    """Отбор серых карт из файлов maggr по glob конфига."""
    files = [Path("region/MagGr/участок_серый.tif"), Path("region/MagGr/участок_карта.tif")]
    assert gray_raster_paths(files, config.MAGGR) == [files[0]]
    assert gray_raster_paths([files[1]], config.MAGGR) == []


def test_gray_raster_fallback_single_band(tmp_path):
    """Имя не по шаблону: серой считается одноканальная карта, цветная (3 канала) — нет."""
    gray = tmp_path / "region_0001_MagGr_1.tif"
    color = tmp_path / "region_0001_MagGr_2.tif"
    _write_gray_tif(gray, np.full((20, 20), 100, np.uint8))
    with rasterio.open(
        color,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=3,
        dtype="uint8",
        crs=_CRS,
        transform=from_origin(500000.0, 5600000.0, 0.1, 0.1),
    ) as dst:
        dst.write(np.zeros((3, 20, 20), np.uint8))
    assert gray_raster_paths([gray, color], config.MAGGR) == [gray]
