"""Тесты koz3.geometry на синтетике: критерии приёмки 1–5 из задачи.

Растровая синтетика: шаг 0.1 м, верхний левый угол (100, 200), строки — вниз
(стандартная привязка rasterio). Порог обрывков — config.MASK_MIN_AREA_M2.
"""

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Polygon

from koz3 import config
from koz3.geometry import (
    clean_polygon,
    mask_to_polygons,
    point_to_circle,
    reproject,
    simplify_m,
)

RES_M = 0.1
TRANSFORM = Affine(RES_M, 0.0, 100.0, 0.0, -RES_M, 200.0)  # (col, row) → (x, y)


def test_mask_square_area_and_bounds():
    """Критерий 1: квадрат 10×10 пикселей при шаге 0.1 м → полигон ~1 м²."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 5:15] = 1
    polygons = mask_to_polygons(mask, TRANSFORM)
    assert len(polygons) == 1
    polygon = polygons[0]
    assert polygon.area == pytest.approx(1.0, rel=0.05)
    minx, miny, maxx, maxy = polygon.bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((100.5, 198.0, 101.5, 199.0), abs=1e-6)


def test_mask_two_blobs_two_polygons():
    """Критерий 2: два отдельных пятна → два непересекающихся полигона."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[5:15, 5:15] = 1  # 10×10 пикселей = 1.0 м²
    mask[30:45, 30:45] = 1  # 15×15 пикселей = 2.25 м²
    polygons = mask_to_polygons(mask, TRANSFORM)
    assert len(polygons) == 2
    assert sorted(p.area for p in polygons) == pytest.approx([1.0, 2.25])
    assert polygons[0].disjoint(polygons[1])


def test_mask_scraps_dropped_and_empty_mask():
    """Обрывки ниже config.MASK_MIN_AREA_M2 отбрасываются; пустая маска → []."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[5:7, 5:7] = 1  # 4 пикселя = 0.04 м² < 0.5 м²
    assert mask_to_polygons(mask, TRANSFORM) == []
    empty = np.zeros((60, 60), dtype=np.uint8)
    assert mask_to_polygons(empty, TRANSFORM) == []


def test_mask_dtype_tolerant():
    """Бинаризация «!= 0»: bool- и float-маски дают тот же полигон."""
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    for converted in (mask.astype(bool), mask.astype(float)):
        polygons = mask_to_polygons(converted, TRANSFORM)
        assert len(polygons) == 1
        assert polygons[0].area == pytest.approx(1.0, rel=0.05)


def test_mask_hole_becomes_interior_ring():
    """Дыра в пятне становится внутренним кольцом; площадь 4 − 1 = 3 м²."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:25, 5:25] = 1
    mask[10:20, 10:20] = 0
    polygons = mask_to_polygons(mask, TRANSFORM)
    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1
    assert polygons[0].area == pytest.approx(3.0)


def test_point_to_circle_area_and_centroid():
    """Критерий 3: площадь ≈ π·r² (±2 %), центроид совпадает с точкой."""
    x_m, y_m = 123456.7, 6543210.9
    circle = point_to_circle(x_m, y_m)
    assert circle.is_valid
    assert circle.area == pytest.approx(np.pi * config.POINT_RADIUS_M**2, rel=0.02)
    assert circle.centroid.x == pytest.approx(x_m, abs=1e-6)
    assert circle.centroid.y == pytest.approx(y_m, abs=1e-6)


def test_clean_polygon_bowtie():
    """Критерий 4: невалидный «бантик» → валидный полигон (самая большая часть)."""
    bowtie = Polygon([(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)])
    assert not bowtie.is_valid
    cleaned = clean_polygon(bowtie)
    assert cleaned is not None
    assert cleaned.is_valid
    assert cleaned.area == pytest.approx(1.0)


def test_clean_polygon_none_and_empty():
    """None, пустой и вырожденный (в линию) полигон → None."""
    assert clean_polygon(None) is None
    assert clean_polygon(Polygon()) is None
    flat = Polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    assert clean_polygon(flat) is None


def test_reproject_roundtrip_submillimeter():
    """Критерий 5: 32637 → 3857 → 32637 — ошибка координат меньше 1 мм."""
    square = Polygon(
        [
            (400000.0, 6150000.0),
            (400100.0, 6150000.0),
            (400100.0, 6150100.0),
            (400000.0, 6150100.0),
        ]
    )
    web = reproject(square, "EPSG:32637", "EPSG:3857")
    assert isinstance(web, Polygon)
    back = reproject(web, "EPSG:3857", "EPSG:32637")
    assert isinstance(back, Polygon)
    assert square.hausdorff_distance(back) < 1e-3
    assert back.area == pytest.approx(square.area, rel=1e-9)


def test_simplify_removes_collinear_vertices():
    """Упрощение убирает коллинеарные вершины, площадь не меняется."""
    ring = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (0.5, 1.0),
        (0.0, 1.0),
        (0.0, 0.5),
    ]
    polygon = Polygon(ring)
    simplified = simplify_m(polygon, 0.05)
    assert isinstance(simplified, Polygon)
    assert len(simplified.exterior.coords) == 5  # 4 угла + замыкание
    assert simplified.area == pytest.approx(1.0)