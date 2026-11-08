"""Тесты признаков EtGr на синтетике: без данных, разметки и весов."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import Polygon

import _bootstrap  # noqa: F401
from koz3 import config
from koz3.geophys import etgr_features as ef

TRANSFORM = Affine(0.25, 0, 100.0, 0, -0.25, 150.0)
"""Привязка синтетической карты: пиксель 0.25 м, центр (100, 100) → (125, 125)."""


def make_field() -> ef.EtGrField:
    """Карта 200×200 px с единственным гауссовым пятном в пикселе (100, 100)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:200, 0:200]
    d2 = (yy - 100.0) ** 2 + (xx - 100.0) ** 2
    z = 3.0 * np.exp(-d2 / (2 * 8.0**2))
    return ef.EtGrField(
        rgb=rng.integers(0, 256, (3, 200, 200), dtype=np.uint8),
        z=z,
        mask=np.ones((200, 200), dtype=bool),
        transform=TRANSFORM,
        crs=CRS.from_epsg(32637),
        is_profile=False,
    )


def test_rgb_to_scalar_on_jet_gradient():
    palette = ef.jet_palette()
    rgb = palette.T.reshape(3, 1, 256)
    t, dist = ef.rgb_to_scalar(rgb)
    assert t.shape == (1, 256) and dist.shape == (1, 256)
    # запрос идёт в float32 (как в ноутбуке): на плоских участках jet ближайший
    # индекс может сместиться на 1, расстояние при этом ничтожно
    assert np.abs(t * 255 - np.arange(256)).max() <= 1
    assert dist.max() < 1e-3
    uint8 = np.round(palette).astype(np.uint8).T.reshape(3, 1, 256)
    t8, dist8 = ef.rgb_to_scalar(uint8)
    assert dist8.max() < 1.5
    assert np.abs(t8 * 255 - np.arange(256)).max() <= 2


def test_candidates_on_synthetic_blob():
    field = make_field()
    polys, x = ef.candidates(field, config.ETGR)
    assert len(polys) >= 5
    assert x.shape == (len(polys), len(ef.FEATURE_NAMES))
    assert x.dtype == np.float32
    assert all(p.is_valid and not p.is_empty for p in polys)
    assert (x[:, 0] == 1).any()  # «знак»: есть кандидаты положительного знака
    i = max(range(len(polys)), key=lambda k: polys[k].area if x[k, 0] == 1 else -1)
    assert abs(polys[i].centroid.x - 125.0) < 2.0
    assert abs(polys[i].centroid.y - 125.0) < 2.0


def test_nested_children_levels():
    field = make_field()
    polys, x = ef.candidates(field, config.ETGR)
    kids = ef.nested_children(polys, x)
    sign, bg, lv = 0, 1, 2  # столбцы «знак», «фон σ», «уровень»
    is_pos, is_bg0 = x[:, sign] == 1, x[:, bg] == 0
    parent = np.flatnonzero(is_pos & is_bg0 & (x[:, lv] == 0.5))
    child = np.flatnonzero(is_pos & is_bg0 & (x[:, lv] == 1.0))
    assert len(parent) == 1 and len(child) == 1
    assert int(child[0]) in kids[int(parent[0])]


def test_crop_box_square_with_margin():
    img = np.arange(50 * 60, dtype=np.float32).reshape(50, 60, 1)
    out = ef.crop_box(img, 10, 30, 5, 25)
    assert out.shape == (40, 40, 1)
    assert out[20, 20, 0] == img[20, 15, 0]
    assert out[0, 0, 0] == 0  # поле слева за краем растра
    assert ef.crop_box(img, 0, 0, 0, 0).shape == (32, 32, 1)  # min_side


def test_candidate_crops_shape():
    pytest.importorskip("torch")
    field = make_field()
    polys, _ = ef.candidates(field, config.ETGR)
    crops = ef.candidate_crops(field, polys, config.ETGR)
    crop = int(config.ETGR["crop_px"])
    assert crops.shape == (len(polys), 4, crop, crop)
    assert crops.dtype == np.uint8
    assert crops[:, 3].max() == 255 and crops[:, 3].min() == 0  # канал маски


def test_predict_polys_splits_blob():
    outer = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    kid_a = Polygon([(1, 1), (1, 4), (4, 4), (4, 1)])
    kid_b = Polygon([(6, 6), (6, 9), (9, 9), (9, 6)])
    polys = [outer, kid_a, kid_b]
    scores = np.array([0.9, 0.5, 0.6])
    kept, conf = ef.predict_polys(polys, [[1, 2], [], []], scores, 0.4, config.ETGR)
    assert kept == [kid_a, kid_b]
    assert conf == [0.5, 0.6]
    kept2, conf2 = ef.predict_polys(polys, [[], [], []], scores, 0.4, config.ETGR)
    assert kept2 == [outer]
    assert conf2 == [0.9]