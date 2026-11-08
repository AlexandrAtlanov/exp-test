"""Тесты tools/make_holdout.py на структуре папок организаторов (синтетика, без данных)."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest

import make_holdout
from koz3 import discovery

REGION = "207_Альфа"


def _make_train(root: Path, name: str = REGION) -> Path:
    """Регион трейна с папками, как у организаторов."""
    region = root / name
    files = {
        f"01_{name}_La_точки/{name}_La.las": b"LASF",
        f"02_{name}_Li_карты/{name}_Li_c.tif": b"tif",
        f"05_{name}_Or_ортофото/{name}_Or.tif": b"tif",
        f"06_{name}_разметка/{name}_Li_курганы.geojson": b"{}",
        f"07_{name}_MagGr_карты/extra/{name}_границы.geojson": b"{}",
        "UTM.json": json.dumps({"crs": "EPSG:32637"}).encode(),
    }
    for rel, content in files.items():
        path = region / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.mark.parametrize(
    ("name", "excluded"),
    [
        ("02_207_Альфа_Li_карты", True),
        ("06_207_Альфа_разметка", True),
        ("01_207_Альфа_La_точки", False),
        ("05_207_Альфа_Or_ортофото", False),
        ("08_207_Альфа_EtGr_профили", False),
        ("03_Липецк_Ae_аэрофото", False),  # «Ли» внутри слова — не токен Li
    ],
)
def test_is_excluded(name: str, excluded: bool) -> None:
    assert make_holdout.is_excluded(name) is excluded


@pytest.mark.parametrize("copy", [False, True])
def test_holdout_like_platform(tmp_path: Path, copy: bool) -> None:
    data = _make_train(tmp_path / "train")
    out = tmp_path / "holdout"
    args = ["--data", str(data), "--regions", REGION, "--out", str(out)]
    assert make_holdout.main(args + (["--copy"] if copy else [])) == 0

    region = out / REGION
    names = {p.relative_to(region).as_posix() for p in region.rglob("*") if p.is_file()}
    assert names == {
        f"01_{REGION}_La_точки/{REGION}_La.las",
        f"05_{REGION}_Or_ортофото/{REGION}_Or.tif",
        f"07_{REGION}_MagGr_карты/extra/{REGION}_границы.geojson",
        "UTM.json",
    }
    las = region / f"01_{REGION}_La_точки" / f"{REGION}_La.las"
    assert las.read_bytes() == b"LASF"  # ссылка или копия читается как исходный файл

    [found] = discovery.find_regions(out)
    assert found.name == REGION
    assert [p.suffix for p in found.files["li"]] == [".las"]
    assert found.utm_crs == "EPSG:32637"


def test_missing_region_creates_nothing(tmp_path: Path) -> None:
    data = _make_train(tmp_path / "train")
    out = tmp_path / "holdout"
    args = ["--data", str(data), "--regions", REGION, "999_Нет", "--out", str(out)]
    assert make_holdout.main(args) == 1
    assert not out.exists()


def test_existing_output_region_refused(tmp_path: Path) -> None:
    data = _make_train(tmp_path / "train")
    out = tmp_path / "holdout"
    (out / REGION).mkdir(parents=True)
    assert make_holdout.main(["--data", str(data), "--regions", REGION, "--out", str(out)]) == 1


def test_region_with_only_excluded_folders_is_error(tmp_path: Path) -> None:
    region = tmp_path / "train" / REGION / f"06_{REGION}_разметка"
    region.mkdir(parents=True)
    (region / "a.geojson").write_text("{}")
    args = ["--data", str(tmp_path / "train"), "--regions", REGION, "--out", str(tmp_path / "o")]
    assert make_holdout.main(args) == 1
