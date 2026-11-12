"""Запасные пути без timm и scikit-image (их наличие в образе платформы не подтверждено).

timm → ResNet torchvision с теми же весами; skimage label/regionprops → scipy + cv2.
Сравнение «с библиотекой» и «без» на одних данных: выход сети и детекции EtGr совпадают.
"""

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
timm = pytest.importorskip("timm")
skimage_measure = pytest.importorskip("skimage.measure")

from etgr_test import _region, _write_jet_map, _write_weights  # noqa: E402

from koz3 import config  # noqa: E402
from koz3.detectors import etgr  # noqa: E402
from koz3.geophys import etgr_features as ef  # noqa: E402
from koz3.timing import Budget  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_nets_cache():
    etgr._NETS_CACHE.clear()
    yield
    etgr._NETS_CACHE.clear()


def _timm_state_dict(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return timm.create_model("resnet18", pretrained=False, num_classes=1, in_chans=4).state_dict()


def test_torchvision_resnet_same_output_as_timm() -> None:
    """Веса timm resnet18 грузятся в torchvision строго, выход совпадает."""
    torch.manual_seed(0)
    ref = timm.create_model("resnet18", pretrained=False, num_classes=1, in_chans=4).eval()
    alt = ef._torchvision_resnet("resnet18")
    alt.load_state_dict(ref.state_dict(), strict=True)
    alt.eval()
    x = torch.randn(4, 4, 128, 128)
    with torch.no_grad():
        torch.testing.assert_close(alt(x), ref(x), rtol=1e-5, atol=1e-5)


def test_make_cnn_without_timm(monkeypatch) -> None:
    """timm нет — make_cnn даёт torchvision ResNet; неизвестная архитектура — ImportError."""
    monkeypatch.setitem(sys.modules, "timm", None)
    net = ef.make_cnn("resnet18", pretrained=False)
    net.load_state_dict(_timm_state_dict(), strict=True)
    with pytest.raises(ImportError):
        ef.make_cnn("efficientnet_b0", pretrained=False)


def test_regions_without_skimage_match(monkeypatch) -> None:
    """Метки, площадь, рамка, периметр, эллипс — как skimage; solidity — близко."""
    rng = np.random.default_rng(5)
    from scipy import ndimage

    mask = ndimage.gaussian_filter(rng.normal(size=(120, 160)), 3) > 0.15
    lab_ref, ref = ef._label_regions(mask)
    monkeypatch.setattr(ef, "label", None)
    monkeypatch.setattr(ef, "regionprops", None)
    lab_alt, alt = ef._label_regions(mask)
    np.testing.assert_array_equal(lab_alt, lab_ref)
    assert len(alt) == len(ref) > 3
    for a, b in zip(alt, ref, strict=True):
        assert a.label == b.label and a.slice == b.slice and a.area == b.area
        assert a.perimeter == pytest.approx(b.perimeter, rel=1e-9)
        assert a.eccentricity == pytest.approx(b.eccentricity, abs=1e-9)
        assert a.axis_major_length == pytest.approx(b.axis_major_length, rel=1e-9)
        assert a.solidity == pytest.approx(b.solidity, abs=0.15)


def test_detector_same_without_timm_and_skimage(tmp_path: Path, monkeypatch) -> None:
    """Детектор EtGr с настоящим resnet18: без timm, skimage и matplotlib — те же детекции."""
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path / "weights")
    _write_weights(
        tmp_path,
        thr=0.0,
        overrides={"arch": "resnet18", "state_dicts": [_timm_state_dict(1)]},
    )
    raster = _write_jet_map(tmp_path / "01_карта_A.tif", size=96, spot=(30, 60, 30, 60))
    region = _region({"etgr": [raster]})
    ref = etgr.run(region, Budget(600), config)
    assert ref

    etgr._NETS_CACHE.clear()
    monkeypatch.setitem(sys.modules, "timm", None)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setattr(ef, "_JET_TREE", None)
    monkeypatch.setattr(ef, "label", None)
    monkeypatch.setattr(ef, "regionprops", None)
    alt = etgr.run(region, Budget(600), config)
    assert len(alt) == len(ref)
    for a, b in zip(alt, ref, strict=True):
        assert a.confidence == pytest.approx(b.confidence, abs=1e-3)
        assert a.geom.equals_exact(b.geom, 1e-6)


def test_jet_palette_without_matplotlib(monkeypatch) -> None:
    """Палитра jet без matplotlib совпадает с matplotlib."""
    pytest.importorskip("matplotlib")
    ref = ef.jet_palette()
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    alt = ef.jet_palette()
    np.testing.assert_allclose(alt, ref, atol=1e-9)
