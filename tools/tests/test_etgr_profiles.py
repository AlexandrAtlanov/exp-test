"""Тесты профильной ветки EtGr: признаки profile_extra и детектор (HGB).

Синтетика без реальных данных: поля EtGrField строятся напрямую,
профильные растры — GeoTIFF с цветами jet. Проверяются формулы
признаков (глубины, карты под полигоном, MISSING), контракт
etgr_profiles.joblib, изоляция ошибок и выключение без весов.
"""

import _bootstrap  # noqa: F401  # добавляет submission/, tools/, training/ в sys.path
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box

from koz3 import config
from koz3.datatypes import Region
from koz3.detectors import etgr
from koz3.timing import Budget

ef = pytest.importorskip("koz3.geophys.etgr_features")

CRS_UTM = "EPSG:32637"
TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 40.0)  # x = col, y = 40 - row, пиксель 1 м
PROFILE_TRANSFORM = Affine(0.5, 0.0, 500000.0, 0.0, -0.5, 5400000.0)


class _StubClf:
    """Классификатор с фиксированной вероятностью (контракт predict_proba)."""

    def __init__(self, proba: float) -> None:
        self.proba = float(proba)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile([1.0 - self.proba, self.proba], (len(x), 1))


def _make_field(z, mask, transform=TRANSFORM, is_profile=True):
    """EtGrField напрямую (без растра): rgb признаками не используется."""
    rgb = np.zeros((3, *z.shape), np.uint8)
    return ef.EtGrField(rgb, z, mask, transform, CRS.from_user_input(CRS_UTM), is_profile)


def _cand_x(sign=1.0, level=1.0):
    """Строка признаков одного кандидата: заданы только знак и уровень."""
    x = np.zeros((1, len(ef.FEATURE_NAMES)), np.float32)
    x[0, ef.FEATURE_NAMES.index("знак")] = sign
    x[0, ef.FEATURE_NAMES.index("уровень")] = level
    return x


def _write_jet_tif(path, values, transform):
    """GeoTIFF с цветами jet по значениям 0..1 (как растры EtGr)."""
    palette = ef.jet_palette()
    rgb = palette[np.clip(np.rint(values * 255), 0, 255).astype(int)].astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=3,
        dtype="uint8",
        crs=CRS_UTM,
        transform=transform,
    ) as dst:
        dst.write(rgb.transpose(2, 0, 1))


def _profile_values():
    """Профиль 60×200 (шире, чем выше): фон 0.45, аномальный блоб 0.95."""
    values = np.full((60, 200), 0.45)
    values[20:30, 80:120] = 0.95
    return values


def _dump_profiles_weights(weights_dir, proba=0.9, thr=0.5, names=None):
    """weights/etgr_profiles.joblib с заглушкой clf (контракт _read_profiles)."""
    joblib = pytest.importorskip("joblib")
    weights_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clf": _StubClf(proba),
        "thr": thr,
        "feature_names": list(names if names is not None else ef.PROFILE_FEATURE_NAMES),
        "sklearn": "stub",
    }
    joblib.dump(payload, weights_dir / str(config.ETGR["profiles_models_file"]))


def _profile_region(tmp_path, weights_dir, monkeypatch, extra_files=()):
    """Регион с одним профилем (и опционально картой) и весами HGB в weights_dir."""
    _dump_profiles_weights(weights_dir)
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    profile = tmp_path / "1_уч_профиль_01.tif"
    _write_jet_tif(profile, _profile_values(), PROFILE_TRANSFORM)
    files = {"etgr": [profile, *extra_files]}
    return Region(name="test", path=tmp_path, utm_crs=CRS_UTM, files=files)


def test_profile_extra_depths_and_maps():
    """Глубины от линии поверхности; среднее/макс/доля по картам под полигоном."""
    mask = np.zeros((40, 100), bool)
    mask[5:36, :] = True  # поверхность на строке 5 -> y = 35 на всех столбцах
    field = _make_field(np.zeros((40, 100)), mask)
    full = np.ones((40, 40), bool)
    maps = [
        _make_field(np.full((40, 40), 2.0), full, is_profile=False),
        _make_field(np.full((40, 40), 0.1), full, is_profile=False),
    ]
    extra = ef.profile_extra(field, [box(20, 25, 30, 30)], _cand_x(), maps)
    assert extra.shape == (1, len(ef.PROFILE_EXTRA))
    # поверхность — ЦЕНТР пикселя строки 5 (rasterio xy, как в ноутбуке): y = 34.5
    assert extra[0, 0] == pytest.approx(4.5, abs=1e-5)  # глубина верха: 34.5 - 30
    assert extra[0, 1] == pytest.approx(9.5, abs=1e-5)  # глубина низа: 34.5 - 25
    assert extra[0, 2] == pytest.approx(1.05, abs=1e-5)  # среднее (2.0 + 0.1) / 2
    assert extra[0, 3] == pytest.approx(2.0, abs=1e-5)  # максимум
    assert extra[0, 4] == pytest.approx(0.5, abs=1e-5)  # доля карт с аномалией


def test_profile_extra_missing_without_maps_or_surface():
    """Нет карт — MISSING в столбцах карт; растр выше, чем шире — MISSING глубин."""
    wide_mask = np.zeros((40, 100), bool)
    wide_mask[5:36, :] = True
    tall_mask = np.zeros((100, 40), bool)  # профиль «стоит»: линии поверхности нет
    tall_mask[5:36, :] = True
    full = np.ones((40, 40), bool)
    maps = [_make_field(np.full((40, 40), 2.0), full, is_profile=False)]
    no_maps = ef.profile_extra(
        _make_field(np.zeros((40, 100)), wide_mask), [box(20, 25, 30, 30)], _cand_x(), []
    )
    assert np.allclose(no_maps[0, 2:], ef.MISSING)
    tall = ef.profile_extra(
        _make_field(np.zeros((100, 40)), tall_mask), [box(20, 25, 30, 30)], _cand_x(), maps
    )
    assert np.allclose(tall[0, :2], ef.MISSING)
    assert tall[0, 2] == pytest.approx(2.0, abs=1e-5)  # карты считаются и без линии


def test_mean_under_needs_three_pixels():
    """Под полигоном меньше 3 валидных пикселей карты — её вклад NaN (отброшен)."""
    mask = np.zeros((40, 100), bool)
    mask[5:36, :] = True
    field = _make_field(np.zeros((40, 100)), mask)
    sparse = np.ones((40, 40), bool)
    sparse[10:16, 20:31] = False
    sparse[10, 20] = True
    sparse[11, 21] = True  # под полигоном 2 валидных пикселя
    full = np.ones((40, 40), bool)
    maps = [
        _make_field(np.full((40, 40), 2.0), sparse, is_profile=False),
        _make_field(np.full((40, 40), 2.0), full, is_profile=False),
    ]
    extra = ef.profile_extra(field, [box(20, 25, 30, 30)], _cand_x(), maps)
    assert extra[0, 2] == pytest.approx(2.0, abs=1e-5)  # осталась одна полная карта
    assert extra[0, 4] == pytest.approx(1.0, abs=1e-5)


def test_run_profiles_detects_anomaly(tmp_path, monkeypatch):
    """Профиль с блобом и карта региона → детекции object_poly/etgr в UTM."""
    map_raster = tmp_path / "3_срез_01.tif"
    _write_jet_tif(map_raster, _profile_values(), PROFILE_TRANSFORM)
    region = _profile_region(
        tmp_path, tmp_path / "weights", monkeypatch, extra_files=(map_raster,)
    )
    assert etgr.is_available(region)  # профили готовы, модели карт нет
    detections = etgr.run(region, Budget(600), config)
    assert detections, "нужна хотя бы одна детекция блоба"
    for d in detections:
        assert d.class_name == "object_poly"
        assert d.markup_type == "etgr"
        assert d.confidence == pytest.approx(0.9)
        assert d.geom.is_valid and d.geom.area > 0
        assert d.region_name == "test"


def test_run_profiles_threshold_from_weights(tmp_path, monkeypatch):
    """Порог из файла весов: скор ниже порога — детекций нет."""
    region = _profile_region(tmp_path, tmp_path / "weights", monkeypatch)
    _dump_profiles_weights(tmp_path / "weights", proba=0.3, thr=0.5)
    assert etgr.run(region, Budget(600), config) == []


def test_run_profiles_isolates_bad_file(tmp_path, monkeypatch):
    """Битый профиль пропускается с предупреждением, хороший работает."""
    region = _profile_region(tmp_path, tmp_path / "weights", monkeypatch)
    bad = tmp_path / "2_уч_профиль_02.tif"
    bad.write_bytes("не GeoTIFF".encode())
    region = Region(
        name="test",
        path=tmp_path,
        utm_crs=CRS_UTM,
        files={"etgr": [tmp_path / "1_уч_профиль_01.tif", bad]},
    )
    assert etgr.run(region, Budget(600), config)


def test_profiles_bad_contract_disables_branch(tmp_path, monkeypatch):
    """feature_names не совпали — ветка профилей выключается с предупреждением."""
    _dump_profiles_weights(tmp_path / "weights", names=["wrong"])
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    profile = tmp_path / "1_уч_профиль_01.tif"
    _write_jet_tif(profile, _profile_values(), PROFILE_TRANSFORM)
    region = Region(name="test", path=tmp_path, utm_crs=CRS_UTM, files={"etgr": [profile]})
    assert etgr.run(region, Budget(600), config) == []
    assert not etgr.is_available(region)


def test_profiles_without_weights_off(tmp_path, monkeypatch):
    """Нет etgr_profiles.joblib (и etgr_maps.pt) — детектор недоступен, прогон пуст."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "empty_weights")
    profile = tmp_path / "1_уч_профиль_01.tif"
    _write_jet_tif(profile, _profile_values(), PROFILE_TRANSFORM)
    region = Region(name="test", path=tmp_path, utm_crs=CRS_UTM, files={"etgr": [profile]})
    assert not etgr.is_available(region)
    assert etgr.run(region, Budget(600), config) == []
