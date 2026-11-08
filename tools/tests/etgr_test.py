"""Тесты детектора EtGr: контракт весов, кеш, конвейер карты, бюджет.

Сеть заменена заглушкой (make_cnn монкипатчится) — timm не нужен;
torch нужен по-настоящему: state_dict, TTA и autocast predict_cnn.
Синтетическая карта — GeoTIFF в палитре jet: фон jet(0.5), пятно jet(0.95).
"""

import logging
from collections.abc import Callable
from pathlib import Path

import _bootstrap  # noqa: F401,I001 — конвенция tools/tests

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from koz3 import config
from koz3.datatypes import Region
from koz3.detectors import etgr
from koz3.geophys import etgr_features as ef
from koz3.timing import Budget


class _StubNet(torch.nn.Module):
    """Заглушка CNN: постоянный логит, вход игнорируется (устройство — из параметра)."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = float(logit)
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.logit, dtype=x.dtype, device=x.device)


class _ExhaustedBudget:
    """Budget-заглушка: ни один запрос не проходит."""

    def can_afford(self, seconds: float) -> bool:
        return False


def _stub_make_cnn(logit: float) -> Callable[[str, bool], _StubNet]:
    """make_cnn-заглушка: сеть с постоянным логитом (timm не нужен)."""

    def make(arch: str, pretrained: bool) -> _StubNet:
        return _StubNet(logit)

    return make


@pytest.fixture(autouse=True)
def _clear_nets_cache():
    """Кеш ансамбля не должен протекать между тестами."""
    etgr._NETS_CACHE.clear()
    yield
    etgr._NETS_CACHE.clear()


def _region(files: dict[str, list[Path]], name: str = "reg", utm: str = "EPSG:32637") -> Region:
    """Регион с данным набором файлов."""
    return Region(name=name, path=Path(name), utm_crs=utm, files=files)


def _write_weights(
    tmp_path: Path,
    *,
    n_nets: int = 1,
    thr: float = 0.5,
    overrides: dict | None = None,
    drop: tuple[str, ...] = (),
) -> Path:
    """Файл etgr_maps.pt по контракту train_etgr.py (state_dicts заглушки)."""
    payload: dict = {
        "arch": "stub",
        "state_dicts": [{"param": torch.zeros(1)} for _ in range(n_nets)],
        "scorer": None,
        "thr": thr,
        "crop": {"crop_px": int(config.ETGR["crop_px"]), "pad": 0.5, "min_side": 16},
        "feature_names": list(ef.FEATURE_NAMES),
        "torch": "test",
        "timm": "0.0.0",
    }
    if overrides:
        payload.update(overrides)
    for key in drop:
        payload.pop(key)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(exist_ok=True)
    path = weights_dir / str(config.ETGR["models_file"])
    torch.save(payload, path)
    return path


def _write_jet_map(
    path: Path,
    *,
    size: int = 64,
    spot: tuple[int, int, int, int] = (24, 40, 24, 40),
    crs: str = "EPSG:32637",
    res: float = 1.0,
    west: float = 500000.0,
    north: float = 6000000.0,
) -> Path:
    """GeoTIFF-карта EtGr: фон jet(0.5), квадратное пятно jet(0.95)."""
    jet = np.round(ef.jet_palette()).astype(np.uint8)
    rgb = np.empty((3, size, size), np.uint8)
    rgb[:] = jet[128][:, None, None]
    y0, y1, x0, x1 = spot
    rgb[:, y0:y1, x0:x1] = jet[242][:, None, None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=from_origin(west, north, res, res),
    ) as dst:
        dst.write(rgb)
    return path


def test_map_raster_paths_filters_profiles() -> None:
    """Профили отфильтровываются до открытия файлов (карты-срезы остаются)."""
    files = [Path("01_карта_A.tif"), Path("02_B_профиль_C.tif"), Path("просто.tif")]
    assert ef.map_raster_paths(files) == [Path("01_карта_A.tif"), Path("просто.tif")]


def test_is_available_false_without_rasters(tmp_path: Path, monkeypatch) -> None:
    """Регион без карт EtGr: детектор недоступен."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    assert etgr.is_available(_region({})) is False


def test_is_available_false_without_weights(tmp_path: Path, monkeypatch) -> None:
    """Карты есть, файла весов нет: детектор недоступен."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    (tmp_path / "weights").mkdir()
    region = _region({"etgr": [Path("01_карта_A.tif")]})
    assert etgr.is_available(region) is False


def test_features_import_failure_disables_detector(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Нет библиотек признаков (skimage): детектор выключен с предупреждением."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(etgr, "ef", None)
    monkeypatch.setattr(etgr, "_IMPORT_ERROR", ImportError("No module named 'skimage'"))
    region = _region({"etgr": [Path("01_карта_A.tif")]})
    with caplog.at_level(logging.WARNING):
        assert etgr.is_available(region) is False
    assert any("выключен" in rec.getMessage() for rec in caplog.records)
    assert etgr.estimate_seconds(region) == 0.0


@pytest.mark.parametrize(
    ("drop", "overrides"),
    [
        (("arch",), {}),
        (("state_dicts",), {}),
        (("thr",), {}),
        (("crop",), {}),
        (("feature_names",), {}),
        (("timm",), {}),
        ((), {"arch": ""}),
        ((), {"arch": 7}),
        ((), {"state_dicts": []}),
        ((), {"state_dicts": {"param": 0}}),
        ((), {"state_dicts": [{"param": torch.zeros(1)}, "junk"]}),
        ((), {"thr": 1.5}),
        ((), {"thr": -0.1}),
        ((), {"feature_names": ["не", "то"]}),
        ((), {"crop": {"crop_px": 64, "pad": 0.5, "min_side": 16}}),
        ((), {"crop": {"crop_px": 128, "pad": 0.25, "min_side": 16}}),
    ],
)
def test_read_nets_rejects_bad_contract(
    tmp_path: Path, drop: tuple[str, ...], overrides: dict
) -> None:
    """Нарушение контракта etgr_maps.pt — исключение (детектор выключится)."""
    path = _write_weights(tmp_path, overrides=overrides, drop=drop)
    with pytest.raises((KeyError, TypeError, ValueError)):
        etgr._read_nets(path, config)


def test_read_nets_rejects_non_dict(tmp_path: Path) -> None:
    """Файл с не-словарём — TypeError с именем фактического типа."""
    path = tmp_path / "weights" / str(config.ETGR["models_file"])
    path.parent.mkdir(parents=True)
    torch.save(torch.zeros(2), path)
    with pytest.raises(TypeError, match="ожидался dict"):
        etgr._read_nets(path, config)


def test_load_nets_none_when_make_cnn_import_error(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Нет timm (ImportError из make_cnn): детектор выключен с предупреждением."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")

    def _no_timm(arch: str, pretrained: bool) -> None:
        raise ImportError("нет timm в образе")

    monkeypatch.setattr(ef, "make_cnn", _no_timm)
    _write_weights(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert etgr._load_nets(config) is None
    assert any("не загружен" in rec.getMessage() for rec in caplog.records)


def test_load_nets_ok_and_cached(tmp_path: Path, monkeypatch) -> None:
    """Валидный файл: ансамбль, порог, режим eval; повторный вызов — из кеша."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    path = _write_weights(tmp_path, n_nets=2, thr=0.35)
    nets = etgr._load_nets(config)
    assert nets is not None
    assert nets.thr == pytest.approx(0.35)
    assert len(nets.nets) == 2
    assert all(not net.training for net in nets.nets)
    path.unlink()  # далее только кеш
    assert etgr._load_nets(config) is nets


def test_is_available_true_with_weights_and_maps(tmp_path: Path, monkeypatch) -> None:
    """Карты и валидные веса: детектор доступен."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    _write_weights(tmp_path)
    region = _region({"etgr": [Path("01_карта_A.tif")]})
    assert etgr.is_available(region) is True


def test_estimate_seconds_counts_maps_only() -> None:
    """Оценка: карты (не профили) × seconds_per_map; пустой регион — 0."""
    files = [Path("01_карта_A.tif"), Path("02_B_профиль_C.tif")]
    expected = float(config.ETGR["seconds_per_map"])
    assert etgr.estimate_seconds(_region({"etgr": files})) == pytest.approx(expected)
    assert etgr.estimate_seconds(_region({})) == 0.0


def test_run_full_pipeline_synthetic(tmp_path: Path, monkeypatch) -> None:
    """Конвейер карты: пятно jet → детекции object_poly в UTM региона."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    _write_weights(tmp_path, thr=0.5)
    raster = _write_jet_map(tmp_path / "01_карта_A.tif")
    detections = etgr.run(_region({"etgr": [raster]}), Budget(600), config)
    assert 1 <= len(detections) <= 10
    for det in detections:
        assert det.class_name == "object_poly"
        assert det.markup_type == "etgr"
        assert det.region_name == "reg"
        assert det.source == raster.name
        assert 0.5 <= det.confidence <= 1.0  # порог файла весов применён
        assert det.geom.is_valid and not det.geom.is_empty
        assert det.geom.area > 0.0
        minx, miny, maxx, maxy = det.geom.bounds
        assert 500000.0 <= minx <= maxx <= 500064.0
        assert 5999936.0 <= miny <= maxy <= 6000000.0


def test_score_averages_ensemble() -> None:
    """Скор — среднее вероятностей сетей ансамбля (усреднение по сидам)."""
    crop = int(config.ETGR["crop_px"])
    crops = np.zeros((3, 4, crop, crop), np.uint8)
    sig0 = float(torch.sigmoid(torch.tensor(0.0)))
    sig2 = float(torch.sigmoid(torch.tensor(2.0)))
    expected = 0.5 * (sig0 + sig2)
    scores = etgr._score([_StubNet(0.0), _StubNet(2.0)], crops)
    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_run_reprojects_to_utm(tmp_path: Path, monkeypatch) -> None:
    """Карта в EPSG:4326: детекции в метрах UTM региона (инвариант 4.8)."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    _write_weights(tmp_path, thr=0.5)
    raster = _write_jet_map(
        tmp_path / "01_карта_B.tif", crs="EPSG:4326", res=0.00001, west=33.0, north=55.0
    )
    region = _region({"etgr": [raster]}, utm="EPSG:32637")
    detections = etgr.run(region, Budget(600), config)
    assert detections, "пятно должно быть найдено"
    for det in detections:
        minx, miny, maxx, maxy = det.geom.bounds
        assert 1e4 < minx < 1e7  # метры UTM, не градусы
        assert 1e4 < miny < 1e7
        assert 1.0 < det.geom.area < 10000.0  # пятно ~16 px * 1.1 м


def test_broken_raster_does_not_kill_region(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Битый растр пропускается с предупреждением; хороший обрабатывается."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    _write_weights(tmp_path, thr=0.5)
    broken = tmp_path / "01_битый.tif"
    broken.write_bytes("не GeoTIFF")
    raster = _write_jet_map(tmp_path / "02_карта_C.tif")
    region = _region({"etgr": [broken, raster]})
    with caplog.at_level(logging.WARNING):
        detections = etgr.run(region, Budget(600), config)
    assert detections, "хорошая карта должна дать детекции"
    assert any("битый" in rec.getMessage() for rec in caplog.records)
    assert all(det.source == raster.name for det in detections)


def test_budget_skips_all_rasters(tmp_path: Path, monkeypatch, caplog) -> None:
    """Исчерпанный бюджет: карты пропущены с предупреждением, конвейер не зовётся."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    _write_weights(tmp_path, thr=0.5)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_field не должен вызываться при пустом бюджете")

    monkeypatch.setattr(ef, "load_field", _boom)
    raster = _write_jet_map(tmp_path / "01_карта_D.tif")
    region = _region({"etgr": [raster]})
    with caplog.at_level(logging.WARNING):
        assert etgr.run(region, _ExhaustedBudget(), config) == []
    assert any("бюджета" in rec.getMessage() for rec in caplog.records)


def test_profiles_skipped_without_opening(tmp_path: Path, monkeypatch) -> None:
    """Профили не открываются: run и estimate по именам их отсеивают."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(ef, "make_cnn", _stub_make_cnn(2.0))
    _write_weights(tmp_path, thr=0.5)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("профиль не должен открываться")

    monkeypatch.setattr(ef, "load_field", _boom)
    missing = tmp_path / "03_X_профиль_Y.tif"  # файла нет на диске
    region = _region({"etgr": [missing]})
    assert etgr.run(region, Budget(600), config) == []
    assert etgr.estimate_seconds(region) == 0.0