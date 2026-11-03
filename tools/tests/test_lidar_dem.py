"""Хиллшейды lidar_dem совпадают с бейзлайном организаторов (эталонные хеши).

Эталонные хеши получены функцией make_hillshade_set из solution.py организаторов на этом же
синтетическом LAS. Падение теста значит, что алгоритм хиллшейдов изменился: тогда модель,
обученная на старых хиллшейдах, увидит на платформе другие картинки.
"""

import hashlib
import struct
from pathlib import Path

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import numpy as np
import pytest
import rasterio

lidar_dem = pytest.importorskip("koz3.detectors.lidar_dem")  # нужен cv2

BASELINE_SHA256 = {
    "c": "f4bf6bf59b3a38b89a91f0a860920e0e171a7a62ccd97f42a0aeb9c60b511fbd",
    "ch": "b99150f3e4b4f792c5c10f17699da45aec8caf7686c5fc3703f46b35ac4000ab",
    "g": "be6b18d433b5841e1ea2616441be698cf104fd3c7721515622803820e3a8e4ff",
    "i": "d1049814a5894b2a384adac302444105c836c3caa2c23d50a7d7994bdc046e89",
}


def write_synthetic_las(path: Path, n: int = 40000, seed: int = 0) -> Path:
    """LAS 1.2, формат точек 0: участок 40×30 м с холмом; 80 % точек — класс «земля»."""
    rng = np.random.default_rng(seed)
    x = 500000 + rng.uniform(0, 40, n)
    y = 6000000 + rng.uniform(0, 30, n)
    z = 150 + 3.5 * np.exp(-((x - 500020) ** 2 + (y - 6000015) ** 2) / 40) + 0.2 * np.sin(x / 3)
    scale, ox, oy = 0.001, 500000.0, 6000000.0
    dtype = [
        ("X", "<i4"),
        ("Y", "<i4"),
        ("Z", "<i4"),
        ("i", "<u2"),
        ("b", "u1"),
        ("c", "u1"),
        ("a", "i1"),
        ("u", "u1"),
        ("p", "<u2"),
    ]
    rec = np.zeros(n, dtype=dtype)
    rec["X"] = np.round((x - ox) / scale)
    rec["Y"] = np.round((y - oy) / scale)
    rec["Z"] = np.round(z / scale)
    rec["c"] = np.where(rng.random(n) < 0.8, 2, 1)
    head = bytearray(227)
    head[0:4] = b"LASF"
    head[24], head[25] = 1, 2
    struct.pack_into("<H", head, 94, 227)
    struct.pack_into("<I", head, 96, 227)
    head[104] = 0
    struct.pack_into("<H", head, 105, 20)
    struct.pack_into("<I", head, 107, n)
    struct.pack_into("<3d", head, 131, scale, scale, scale)
    struct.pack_into("<3d", head, 155, ox, oy, 0.0)
    struct.pack_into("<6d", head, 179, x.max(), x.min(), y.max(), y.min(), z.max(), z.min())
    path.write_bytes(bytes(head) + rec.tobytes())
    return path


def test_hillshades_match_baseline(tmp_path: Path) -> None:
    las = write_synthetic_las(tmp_path / "synthetic_La.las")
    paths = lidar_dem.hillshades_for_las(las, "EPSG:32637", tmp_path / "out")
    assert [p.stem.rsplit("_", 1)[-1] for p in paths] == ["c", "ch", "g", "i"]
    for path in paths:
        tag = path.stem.rsplit("_", 1)[-1]
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32637"
            digest = hashlib.sha256(src.read().tobytes()).hexdigest()
        assert digest == BASELINE_SHA256[tag], f"хиллшейд {tag} отличается от бейзлайна"


def test_missing_las_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lidar_dem.hillshades_for_las(tmp_path / "nope.las", "EPSG:32637", tmp_path / "out")
