"""Тесты детектора koz3.detectors.lidar без GPU, модели и реального лидара.

YOLO и построение отмывок подменяются стабами (monkeypatch), отмывки —
маленькие синтетические tif. Проверяются: геометрии в UTM региона,
markup_type «li», формат source «<las>:<отмывка>», устойчивость к ошибке
одного LAS и к исчерпанию бюджета. Блобы сделаны крупными, чтобы пройти
любые отсевы площади внутри geometry.mask_to_polygons.
"""

import logging
from pathlib import Path

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from koz3 import config, geometry, yolo_runtime
from koz3.datatypes import Detection, Region
from koz3.detectors import lidar, lidar_dem
from koz3.timing import Budget
from koz3.yolo_runtime import TileInstance

REGION_CRS = "EPSG:32637"
"""UTM-зона синтетического региона."""

TILE_PX = 16
"""Мелкий тайл вместо 1024, чтобы растры в тестах были маленькими."""

UTM_TRANSFORM = Affine(2.0, 0.0, 500000.0, 0.0, -2.0, 6000000.0)
"""Трансформ отмывок в метрах: пиксель 2 м, угол (500000, 6000000)."""

GEO_TRANSFORM = Affine(0.3, 0.0, 37.0, 0.0, -0.3, 55.0)
"""Трансформ отмывки в градусах (EPSG:4326) для теста перепроецирования."""


def _make_region(region_dir: Path, las_names: list[str]) -> tuple[Region, list[Path]]:
    """Регион с фиктивными LAS (по 1 КиБ) в заданном каталоге."""
    region_dir.mkdir(parents=True, exist_ok=True)
    las_paths = []
    for name in las_names:
        las_path = region_dir / name
        las_path.write_bytes(b"0" * 1024)
        las_paths.append(las_path)
    region = Region(
        name="test_region",
        path=region_dir,
        utm_crs=REGION_CRS,
        files={"li": las_paths},
    )
    return region, las_paths


def _write_tif(
    path: Path, transform: Affine, crs: str, width: int, height: int, fill: int = 128
) -> None:
    """Синтетический однополосный uint8 tif, залитый серым 128.

    Не нулями: тайлы из одних 0 (или 255) lidar считает «вне данных» и не
    подаёт модели (LIDAR["min_valid_share"]).
    """
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, height, width), fill, dtype=np.uint8))


def _hillshade_stub(
    monkeypatch,
    write_crs,
    transform,
    width=12,
    height=12,
    fail_for=(),
    variants=None,
    fill=128,
):
    forced = variants  # набор, который «вернёт» lidar_dem, независимо от запроса
    """Стаб lidar_dem.hillshades_for_las: пишет синтетические tif (или падает)."""

    def fake_hillshades(las_path, region_crs, out_dir, variants=None):
        if las_path.name in fail_for:
            raise RuntimeError(f"синтетическая ошибка: {las_path.name}")
        written = config.LIDAR["hillshades"] if forced is None else forced
        paths = []
        for variant in written:
            tif_path = Path(out_dir) / f"hillshade_{variant}.tif"
            _write_tif(tif_path, transform, write_crs, width, height, fill)
            paths.append(tif_path)
        return paths

    monkeypatch.setattr(lidar_dem, "hillshades_for_las", fake_hillshades)


def _predict_stub(monkeypatch, instances_by_tile):
    """Стаб predict_tiles: одинаковые списки масок для каждого вызова."""

    def fake_predict(loaded, tiles):
        for tile_index in range(len(tiles)):
            if tile_index < len(instances_by_tile):
                yield instances_by_tile[tile_index]
            else:
                yield []

    monkeypatch.setattr(yolo_runtime, "predict_tiles", fake_predict)


def _load_stub(monkeypatch, calls=None):
    """Стаб load_model: фиктивная LoadedModel, вызовы считаются.

    batch=64 — вся синтетическая отмывка уходит одним вызовом predict_tiles, и
    _predict_stub раздаёт маски по номеру тайла внутри отмывки.
    """

    def fake_load(weights_dir=None):
        if calls is not None:
            calls.append(1)
        return yolo_runtime.LoadedModel(model=object(), kind="stub", names={}, batch=64)

    monkeypatch.setattr(yolo_runtime, "load_model", fake_load)


def _find_las_stub(monkeypatch, las_paths):
    """Стаб find_las_files: вернуть заданный список LAS."""
    monkeypatch.setattr(
        lidar_dem,
        "find_las_files",
        lambda region_dir: list(las_paths),
    )


def _blob_mask(size_px: int, row0: int, col0: int) -> np.ndarray:
    """Маска size x size с прямоугольным пятном 4 x 4 в (row0, col0)."""
    mask = np.zeros((size_px, size_px), dtype=bool)
    mask[row0 : row0 + 4, col0 : col0 + 4] = True
    return mask


def _spy_reproject(monkeypatch) -> list[tuple[str, str]]:
    """Заменить geometry.reproject обёрткой, запоминающей вызовы."""
    calls: list[tuple[str, str]] = []
    real_reproject = geometry.reproject

    def spy_reproject(geom, src_crs, dst_crs):
        calls.append((src_crs, dst_crs))
        return real_reproject(geom, src_crs, dst_crs)

    monkeypatch.setattr(geometry, "reproject", spy_reproject)
    return calls


class _ScriptedBudget:
    """Фальшивый бюджет с запрограммированными ответами can_afford."""

    def __init__(self, answers: list[bool]) -> None:
        self._answers = list(answers)

    def can_afford(self, seconds: float) -> bool:
        """Очередной запрограммированный ответ; кончились — False."""
        return self._answers.pop(0) if self._answers else False

    def remaining(self) -> float:
        """Остаток бюджета для сообщений лога."""
        return 0.0


@pytest.fixture()
def fresh_detector(monkeypatch):
    """Сброс кэша модели, мелкие тайлы, инференс только отмывки «c»."""
    monkeypatch.setattr(lidar, "_model", None)
    monkeypatch.setattr(lidar, "_model_error", "")
    monkeypatch.setitem(config.LIDAR, "infer_mode", "tiles")  # эти тесты — про тайлы
    monkeypatch.setitem(config.LIDAR, "scales", (1,))  # масштабы — в test_lidar_scales
    monkeypatch.setitem(config.LIDAR, "tile_px", TILE_PX)
    monkeypatch.setitem(config.LIDAR, "overlap", 0.3)
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c",))


def test_is_available_needs_li_files_and_weights(tmp_path, monkeypatch):
    """Доступность: нужны и LAS-файлы, и хотя бы одни веса (.pt или .engine)."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region, _ = _make_region(tmp_path / "region", ["ground.las"])
    pt_path = weights_dir / str(config.YOLO["weights_pt"])
    engine_path = weights_dir / str(config.YOLO["weights_engine"])

    assert not lidar.is_available(region)  # весов нет

    pt_path.write_bytes(b"weights")
    assert lidar.is_available(region)  # есть .pt

    pt_path.unlink()
    assert not lidar.is_available(region)

    engine_path.write_bytes(b"weights")
    assert lidar.is_available(region)  # есть только .engine

    empty_region = Region(
        name="empty",
        path=tmp_path,
        utm_crs=REGION_CRS,
        files={"li": []},
    )
    assert not lidar.is_available(empty_region)  # нет LAS


def test_estimate_seconds_is_linear_in_las_size(tmp_path, monkeypatch):
    """Оценка времени: сумма размеров LAS * seconds_per_gb."""
    region, las_paths = _make_region(tmp_path, ["a.las", "b.las"])
    las_paths[0].write_bytes(b"x" * 1_000)
    las_paths[1].write_bytes(b"x" * 3_000)
    _find_las_stub(monkeypatch, las_paths)

    expected = 4_000 / 1e9 * float(config.LIDAR["seconds_per_gb"])
    assert lidar.estimate_seconds(region) == pytest.approx(expected)


def test_estimate_seconds_skips_missing_las(tmp_path):
    """LAS из region.files, которого нет на диске, не роняет оценку: он не учитывается."""
    region, las_paths = _make_region(tmp_path, ["a.las", "b.las"])
    las_paths[1].unlink()
    per_gb = float(config.LIDAR["seconds_per_gb"])
    assert lidar.estimate_seconds(region) == pytest.approx(1024 / 1e9 * per_gb)


def test_all_nodata_tiles_skip_inference(tmp_path, monkeypatch, fresh_detector):
    """Отмывка целиком вне данных (одни нули): модель не вызывается, результат пуст."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM, width=24, height=24, fill=0)
    _load_stub(monkeypatch)
    calls = []
    monkeypatch.setattr(
        yolo_runtime, "predict_tiles", lambda loaded, tiles: calls.append(tiles) or iter([])
    )
    assert lidar.run(region, Budget(600), config) == []
    assert calls == []


def test_run_happy_path_detections_in_utm(tmp_path, monkeypatch, fresh_detector):
    """Прогон: геометрия в UTM, markup_type «li», source «<las>:<отмывка>»."""
    region, las_paths = _make_region(tmp_path, ["site_ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM)
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3))]],
    )
    reproject_calls = _spy_reproject(monkeypatch)

    detections = lidar.run(region, Budget(600), config)

    assert reproject_calls == []  # CRS растра совпадает с UTM региона
    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.markup_type == "li"
    assert det.region_name == "test_region"
    assert det.class_name == "kurgany_tselye"
    assert det.confidence == pytest.approx(0.9)
    assert det.source == "site_ground.las:c"
    assert not det.geom.is_empty
    # пятно: строки 2..5, колонки 3..6 -> x 500006..500014, y 5999988..5999996
    assert det.geom.centroid.x == pytest.approx(500010.0)
    assert det.geom.centroid.y == pytest.approx(5999992.0)


def test_run_source_covers_all_hillshade_variants(tmp_path, monkeypatch, fresh_detector):
    """Source различает отмывки: «<las>:c», «:ch», «:g», «:i» (голосование T5)."""
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c", "ch", "g", "i"))
    region, las_paths = _make_region(tmp_path, ["site_ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM)
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3))]],
    )

    detections = lidar.run(region, Budget(600), config)

    assert {det.source for det in detections} == {
        "site_ground.las:c",
        "site_ground.las:ch",
        "site_ground.las:g",
        "site_ground.las:i",
    }
    assert all(det.markup_type == "li" for det in detections)


def test_run_keeps_duplicates_from_tile_overlap(tmp_path, monkeypatch, fresh_detector):
    """Дубли от перекрытия тайлов не подавляются — это дело postprocess."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM, width=24, height=24)
    _load_stub(monkeypatch)
    # растр 24x24 -> тайлы (0,0), (0,8), (8,0), (8,8); пятно в глобальных
    # колонках 10..13 попадает и в тайл (0,0), и в тайл (0,8)
    _predict_stub(
        monkeypatch,
        [
            [TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 10))],
            [TileInstance("kurgany_tselye", 0.85, _blob_mask(TILE_PX, 2, 2))],
        ],
    )

    detections = lidar.run(region, Budget(600), config)

    assert len(detections) == 2
    centroids = sorted(det.geom.centroid.x for det in detections)
    assert centroids[0] == pytest.approx(centroids[1])  # одно место в мире
    assert centroids[0] == pytest.approx(500000.0 + 2.0 * 12.0)


def test_run_marks_masks_cut_by_inner_tile_edge(tmp_path, monkeypatch, fresh_detector):
    """edge_cut = True только у маски на внутреннем крае тайла, не на крае растра."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM, width=24, height=24)
    _load_stub(monkeypatch)
    # тайлы 16 px: (0,0), (0,8), (8,0), (8,8); растр 24 x 24
    _predict_stub(
        monkeypatch,
        [
            [
                TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 2)),  # внутри
                TileInstance("gorodishcha", 0.9, _blob_mask(TILE_PX, 2, 12)),  # край 16 < 24
            ],
            [TileInstance("arkhitektury", 0.9, _blob_mask(TILE_PX, 2, 12))],  # край растра
        ],
    )

    detections = lidar.run(region, Budget(600), config)

    cut = {det.class_name: det.extra["edge_cut"] for det in detections}
    assert cut == {"kurgany_tselye": False, "gorodishcha": True, "arkhitektury": False}


def test_run_crops_mask_to_real_tile_area(tmp_path, monkeypatch, fresh_detector):
    """Пятно только в зоне дополнения нулями не даёт детекции."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    # растр 10x10 < тайла 16: единственный тайл дополнен нулями до 16x16
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM, width=10, height=10)
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [
            [
                TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3)),
                TileInstance("gorodishcha", 0.7, _blob_mask(TILE_PX, 12, 12)),
            ],
        ],
    )

    detections = lidar.run(region, Budget(600), config)

    assert len(detections) == 1
    assert detections[0].class_name == "kurgany_tselye"
    assert detections[0].geom.centroid.x == pytest.approx(500010.0)


def test_run_reprojects_from_foreign_crs(tmp_path, monkeypatch, fresh_detector):
    """Если CRS растра отличается от UTM региона, полигоны перепроецируются."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, "EPSG:4326", GEO_TRANSFORM, width=10, height=10)
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3))]],
    )
    reproject_calls = _spy_reproject(monkeypatch)

    detections = lidar.run(region, Budget(600), config)

    assert reproject_calls == [("EPSG:4326", REGION_CRS)]
    assert len(detections) == 1
    centroid = detections[0].geom.centroid
    assert 100_000 < centroid.x < 900_000  # метры UTM, а не градусы
    assert 5_000_000 < centroid.y < 7_000_000


def test_run_zero_budget_returns_empty_without_error(tmp_path, monkeypatch, fresh_detector):
    """Бюджет 0: пустой список, без исключений, модель не загружается."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM)
    load_calls: list[int] = []
    _load_stub(monkeypatch, load_calls)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3))]],
    )

    detections = lidar.run(region, Budget(0), config)

    assert detections == []
    assert load_calls == []


def test_run_stops_between_hillshades_when_budget_out(
    tmp_path, monkeypatch, fresh_detector, caplog
):
    """Бюджет кончился между отмывками: возвращается уже найденное."""
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c", "ch", "g", "i"))
    region, las_paths = _make_region(tmp_path, ["one.las", "two.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM)
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3))]],
    )
    budget = _ScriptedBudget([True, True, False, False])

    with caplog.at_level(logging.WARNING):
        detections = lidar.run(region, budget, config)

    assert [det.source for det in detections] == ["one.las:c"]
    assert "не покрывает отмывку ch" in caplog.text


def test_run_one_las_error_does_not_stop_others(tmp_path, monkeypatch, fresh_detector, caplog):
    """Ошибка одного LAS логируется, остальные LAS обрабатываются."""
    region, las_paths = _make_region(tmp_path, ["good.las", "bad.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM, fail_for=("bad.las",))
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.8, _blob_mask(TILE_PX, 2, 3))]],
    )

    with caplog.at_level(logging.ERROR):
        detections = lidar.run(region, Budget(600), config)

    assert [det.source for det in detections] == ["good.las:c"]
    assert "bad.las" in caplog.text


def test_run_model_load_failure_returns_empty(tmp_path, monkeypatch, fresh_detector, caplog):
    """Модель не загрузилась: пустой результат, прогон не падает."""
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM)

    def broken_load(weights_dir=None):
        raise FileNotFoundError("весов нет")

    monkeypatch.setattr(yolo_runtime, "load_model", broken_load)

    with caplog.at_level(logging.ERROR):
        detections = lidar.run(region, Budget(600), config)

    assert detections == []
    assert "модель не загрузилась" in caplog.text


def test_run_loads_model_once_for_all_las(tmp_path, monkeypatch, fresh_detector):
    """Модель грузится один раз на прогон, а не на каждый LAS."""
    region, las_paths = _make_region(tmp_path, ["a.las", "b.las"])
    _find_las_stub(monkeypatch, las_paths)
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM)
    load_calls: list[int] = []
    _load_stub(monkeypatch, load_calls)
    _predict_stub(monkeypatch, [])

    lidar.run(region, Budget(600), config)

    assert len(load_calls) == 1


def test_run_skips_missing_hillshade_variant(tmp_path, monkeypatch, fresh_detector, caplog):
    """Отмывка не построена или не сопоставлена: предупреждение, остальные идут."""
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("c", "g"))
    region, las_paths = _make_region(tmp_path, ["ground.las"])
    _find_las_stub(monkeypatch, las_paths)
    # построены только c и ch: g не попадёт в соответствие вариантов
    _hillshade_stub(monkeypatch, REGION_CRS, UTM_TRANSFORM, variants=("c", "ch"))
    _load_stub(monkeypatch)
    _predict_stub(
        monkeypatch,
        [[TileInstance("kurgany_tselye", 0.9, _blob_mask(TILE_PX, 2, 3))]],
    )

    with caplog.at_level(logging.WARNING):
        detections = lidar.run(region, Budget(600), config)

    assert [det.source for det in detections] == ["ground.las:c"]
    assert "не построена" in caplog.text


def test_ground_las_preferred_in_same_folder(tmp_path):
    """X.las и X_ground.las в одной папке — только ground; другие LAS папки остаются."""
    raw = tmp_path / "a" / "R_La.las"
    ground = tmp_path / "a" / "R_La_ground.las"
    other = tmp_path / "b" / "R2_La.las"
    for path in (raw, ground, other):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"0" * 1024)
    second = tmp_path / "a" / "R_part2_La.las"  # другой LAS в той же папке — не теряется
    second.write_bytes(b"0" * 1024)
    files = {"li": [raw, ground, other, second]}
    region = Region(name="r", path=tmp_path, utm_crs=REGION_CRS, files=files)
    assert sorted(lidar._las_files(region)) == sorted([ground, other, second])
    assert lidar.estimate_seconds(region) == pytest.approx(
        3 * 1024 / 1e9 * float(config.LIDAR["seconds_per_gb"])
    )
