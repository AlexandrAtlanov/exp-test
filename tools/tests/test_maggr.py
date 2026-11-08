"""Тесты детектора maggr: модели-заглушки, синтетика, изоляция ошибок.

Модели — заглушки с predict_proba из фиксированного массива; растр —
синтетическая серая карта UTM с диполем и пятном-аномалией. sklearn не
нужен: заглушки — чистый Python; joblib требуется только тестам, пишущим
файл весов (pytest.importorskip внутри таких тестов).
"""

import logging
import sys
from importlib import metadata
from pathlib import Path

import _bootstrap  # noqa: F401  добавляет submission/ и tools/ в sys.path
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from koz3 import config
from koz3.datatypes import Region
from koz3.detectors import maggr
from koz3.detectors.maggr import estimate_seconds, is_available, run
from koz3.geophys import maggr_features
from koz3.timing import Budget


class _StubModel:
    """Заглушка sklearn-оценщика: predict_proba из фиксированного массива."""

    def __init__(self, values: float | np.ndarray) -> None:
        self.values = np.atleast_1d(np.asarray(values, dtype=np.float64))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Вероятности (N, 2): фиксированные значения, размноженные до N."""
        positive = np.clip(np.resize(self.values, len(features)), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


def _sklearn_version() -> str:
    """Версия sklearn окружения для ключа «sklearn» файла весов."""
    try:
        return metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        return "sklearn нет"


def _models_dict(point_values: float = 0.9, poly_values: float = 0.9) -> dict[str, object]:
    """Валидное содержимое maggr_models.joblib с заглушками (контракт 13b)."""
    return {
        "points": _StubModel(point_values),
        "polys": _StubModel(poly_values),
        "thr_points": 0.5,
        "thr_polys": 0.5,
        "feature_names": {
            "points": list(maggr_features.POINT_FEATURE_NAMES),
            "polys": list(maggr_features.POLY_FEATURE_NAMES),
        },
        "sklearn": _sklearn_version(),
    }


def _write_models(weights_dir: Path, point_values: float = 0.9, poly_values: float = 0.9) -> Path:
    """Пишет weights/maggr_models.joblib; тест без joblib пропускается."""
    joblib = pytest.importorskip("joblib")
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / str(config.MAGGR["models_file"])
    joblib.dump(_models_dict(point_values, poly_values), path)
    return path


def _write_gray(path: Path, size_px: int = 200, utm: str = "EPSG:32635") -> Path:
    """Синтетическая серая карта: шум, диполь (2 лунки) и пятно-аномалия."""
    rng = np.random.default_rng(42)
    rows, cols = np.mgrid[0:size_px, 0:size_px]
    values = rng.normal(128.0, 4.0, (size_px, size_px))
    blobs = ((70, 90, 80.0, 3.0), (70, 110, -80.0, 3.0), (150, 150, 60.0, 8.0))
    for row, col, amp, sigma in blobs:
        values += amp * np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * sigma**2))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size_px,
        width=size_px,
        count=1,
        dtype="uint8",
        crs=utm,
        transform=from_origin(500000.0, 5500000.0, 0.1, 0.1),
    ) as dst:
        dst.write(np.clip(np.round(values), 0, 255).astype("uint8"), 1)
    return path


def _region(directory: Path, files: dict[str, list[Path]], utm_crs: str = "EPSG:32635") -> Region:
    """Регион с данным набором файлов (UTM 32635 по умолчанию)."""
    return Region(name="r1", path=directory, utm_crs=utm_crs, files=files)


@pytest.fixture(autouse=True)
def _clear_models_cache():
    """Кеш моделей детектора не протекает между тестами."""
    maggr._MODELS_CACHE.clear()
    yield
    maggr._MODELS_CACHE.clear()


@pytest.fixture()
def maggr_region(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Region:
    """Регион с валидными весами, серой и цветной картами (proba 0.9)."""
    weights_dir = tmp_path / "weights"
    _write_models(weights_dir)
    gray = _write_gray(tmp_path / "g_серый.tif")
    colored = _write_gray(tmp_path / "g_цвет.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    return _region(tmp_path, {"maggr": [colored, gray]})


def test_is_available_without_gray(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Нет серой карты — детектор недоступен, даже если есть веса."""
    weights_dir = tmp_path / "weights"
    _write_models(weights_dir)
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region = _region(tmp_path, {})
    assert is_available(region) is False


def test_is_available_without_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Нет файла весов — детектор недоступен, вызов не падает."""
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    region = _region(tmp_path, {"maggr": [gray]})
    assert is_available(region) is False


def test_is_available_with_models(maggr_region: Region) -> None:
    """Серая карта и валидные веса — детектор доступен."""
    assert is_available(maggr_region) is True


def test_is_available_warns_on_sklearn_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Несовпадение версии sklearn — предупреждение, но не отказ."""
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    data = _models_dict()
    data["sklearn"] = "999.999.999"
    joblib.dump(data, weights_dir / str(config.MAGGR["models_file"]))
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region = _region(tmp_path, {"maggr": [gray]})
    with caplog.at_level(logging.WARNING):
        assert is_available(region) is True
    assert "sklearn" in caplog.text


def _drop_points(data: dict) -> None:
    """Убирает ключ points."""
    del data["points"]


def _rename_point_features(data: dict) -> None:
    """Подменяет имена точечных признаков."""
    data["feature_names"]["points"] = ["другой"]


def _oversize_threshold(data: dict) -> None:
    """Ставит порог полигонов выше единицы."""
    data["thr_polys"] = 1.5


def _model_without_proba(data: dict) -> None:
    """Ставит модель без predict_proba."""
    data["polys"] = object()


@pytest.mark.parametrize(
    "mutate",
    [_drop_points, _rename_point_features, _oversize_threshold, _model_without_proba],
)
def test_is_available_rejects_bad_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mutate,
) -> None:
    """Ошибка файла весов — is_available False, причина в логе."""
    joblib = pytest.importorskip("joblib")
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    data = _models_dict()
    mutate(data)
    joblib.dump(data, weights_dir / str(config.MAGGR["models_file"]))
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region = _region(tmp_path, {"maggr": [gray]})
    with caplog.at_level(logging.WARNING):
        assert is_available(region) is False
    assert "выключен" in caplog.text


def test_is_available_rejects_corrupt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Битый файл весов — is_available False, причина в логе."""
    pytest.importorskip("joblib")
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / str(config.MAGGR["models_file"])).write_bytes(b"not a pickle")
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region = _region(tmp_path, {"maggr": [gray]})
    with caplog.at_level(logging.WARNING):
        assert is_available(region) is False
    assert "выключен" in caplog.text


def test_is_available_without_joblib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """joblib нет в образе — предупреждение и is_available False."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / str(config.MAGGR["models_file"])).write_bytes(b"pickle")
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    monkeypatch.setitem(sys.modules, "joblib", None)
    region = _region(tmp_path, {"maggr": [gray]})
    with caplog.at_level(logging.WARNING):
        assert is_available(region) is False
    assert "joblib" in caplog.text


def test_estimate_seconds_linear_by_megapixels(tmp_path: Path) -> None:
    """Оценка линейна по мегапикселям серых карт; цветная не считается."""
    gray = _write_gray(tmp_path / "g_серый.tif", size_px=200)
    colored = _write_gray(tmp_path / "g_цвет.tif", size_px=200)
    region = _region(tmp_path, {"maggr": [gray, colored]})
    work_px = round(200 * 0.1 / float(config.MAGGR["res_m"]))
    expected = work_px**2 / 1e6 * float(config.MAGGR["seconds_per_mpx"])
    assert estimate_seconds(region) == pytest.approx(expected)


def test_run_finds_points_and_polys(maggr_region: Region) -> None:
    """Обе ветки: классы, markup_type, source, круги и валидные полигоны."""
    detections = run(maggr_region, Budget(600.0), config)
    points = [d for d in detections if d.class_name == "finds_points"]
    polys = [d for d in detections if d.class_name == "object_poly"]
    assert points, "нет детекций finds_points"
    assert polys, "нет детекций object_poly"
    circle_area = np.pi * config.POINT_RADIUS_M**2
    for detection in detections:
        assert detection.markup_type == "maggr"
        assert detection.region_name == "r1"
        assert detection.source == "g_серый.tif"  # цветная карта пропущена
        assert detection.confidence == pytest.approx(0.9)
        assert 0.0 <= detection.confidence <= 1.0
        assert isinstance(detection.geom, Polygon)
        assert not detection.geom.is_empty
        assert detection.geom.area > 0.0
    for detection in points:
        assert detection.geom.area == pytest.approx(circle_area, rel=0.02)
    # диполь из двух лунок: у каждой лунки есть точка ближе 1 м
    for col in (90, 110):
        x = 500000.0 + (col + 0.5) * 0.1
        y = 5500000.0 - (70 + 0.5) * 0.1
        nearest = min(d.geom.centroid.distance(Point(x, y)) for d in points)
        assert nearest < 1.0


def test_run_threshold_from_model_file_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Порог из файла весов режет: proba 0.3 при пороге 0.5 — пусто."""
    weights_dir = tmp_path / "weights"
    _write_models(weights_dir, point_values=0.3, poly_values=0.3)
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region = _region(tmp_path, {"maggr": [gray]})
    assert run(region, Budget(600.0), config) == []


def test_run_skips_colored_rasters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Цветные карты не обрабатываются: серых нет — детекций нет."""
    weights_dir = tmp_path / "weights"
    _write_models(weights_dir)
    colored = _write_gray(tmp_path / "g_цвет.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    region = _region(tmp_path, {"maggr": [colored]})
    assert is_available(region) is False
    assert run(region, Budget(600.0), config) == []


def test_run_same_crs_skips_reprojection(
    maggr_region: Region, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRS растра = UTM региона — reproject не вызывается."""

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("reproject не должен вызываться при том же CRS")

    monkeypatch.setattr(maggr, "reproject", _fail)
    assert run(maggr_region, Budget(600.0), config)


def test_run_reprojects_points_to_region_utm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Точки переведены в UTM региона: сверка с независимым pyproj."""
    weights_dir = tmp_path / "weights"
    _write_models(weights_dir)
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    same = _region(tmp_path, {"maggr": [gray]}, utm_crs="EPSG:32635")
    other = _region(tmp_path, {"maggr": [gray]}, utm_crs="EPSG:32636")
    same_dets = run(same, Budget(600.0), config)
    other_dets = run(other, Budget(600.0), config)
    points_same = [d for d in same_dets if d.class_name == "finds_points"]
    points_other = [d for d in other_dets if d.class_name == "finds_points"]
    assert points_same and points_other
    transformer = Transformer.from_crs("EPSG:32635", "EPSG:32636", always_xy=True)
    for det_same, det_other in zip(points_same, points_other):
        center = det_same.geom.centroid
        expected_x, expected_y = transformer.transform(center.x, center.y)
        assert det_other.geom.centroid.x == pytest.approx(expected_x, abs=1e-6)
        assert det_other.geom.centroid.y == pytest.approx(expected_y, abs=1e-6)


def test_run_broken_raster_isolated(
    maggr_region: Region, caplog: pytest.LogCaptureFixture
) -> None:
    """Битый растр пропускается с предупреждением, регион считается."""
    bad = maggr_region.path / "битый_серый.tif"
    bad.write_text("не растр")
    maggr_region.files["maggr"].append(bad)
    with caplog.at_level(logging.WARNING):
        detections = run(maggr_region, Budget(600.0), config)
    assert detections
    assert all(d.source == "g_серый.tif" for d in detections)
    assert "битый_серый" in caplog.text


def test_run_skips_files_without_budget(
    maggr_region: Region, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Бюджета на файл нет — файл пропускается с предупреждением."""
    monkeypatch.setitem(config.MAGGR, "seconds_per_mpx", 1e9)
    with caplog.at_level(logging.WARNING):
        assert run(maggr_region, Budget(600.0), config) == []
    assert "бюджета" in caplog.text


def test_run_without_models_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """run без весов не падает: пустой список и предупреждение в логе."""
    gray = _write_gray(tmp_path / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    region = _region(tmp_path, {"maggr": [gray]})
    with caplog.at_level(logging.WARNING):
        assert run(region, Budget(600.0), config) == []
    assert "без моделей" in caplog.text


def test_models_are_loaded_once_per_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """joblib.load вызывается один раз: кеш моделей на прогон."""
    joblib = pytest.importorskip("joblib")
    weights_dir = tmp_path / "weights"
    _write_models(weights_dir)
    gray1 = _write_gray(tmp_path / "r1" / "g_серый.tif")
    gray2 = _write_gray(tmp_path / "r2" / "g_серый.tif")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    load_calls: list[str] = []
    real_load = joblib.load

    def counting_load(*args: object, **kwargs: object) -> object:
        load_calls.append(str(args[0]))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    region1 = _region(gray1.parent, {"maggr": [gray1]})
    region2 = _region(gray2.parent, {"maggr": [gray2]})
    assert is_available(region1) is True
    assert run(region1, Budget(600.0), config)
    assert run(region2, Budget(600.0), config)
    assert len(load_calls) == 1
