"""Тесты tools/make_holdout.py: детерминированное разбиение регионов на train/holdout."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import make_holdout


def _make_regions(root: Path, names: list[str]) -> None:
    """Каталоги регионов с папкой «06_разметка»."""
    for name in names:
        (root / name / "06_разметка").mkdir(parents=True, exist_ok=True)


def _feature(region: str, cls: str = "defo") -> dict:
    """Объект GT с регионом и классом."""
    return {
        "type": "Feature",
        "properties": {"region": region, "class": cls},
        "geometry": {"type": "Point", "coordinates": [37.0, 55.0]},
    }


def _write_gt(path: Path, features: list[dict]) -> None:
    """GT-файл в формате tools/make_gt.py."""
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_make_holdout_full_run(tmp_path: Path) -> None:
    """Шесть регионов и GT → списки, разрезанный GT и манифест."""
    names = [f"00{n}" for n in range(1, 7)]
    _make_regions(tmp_path / "train", names)
    _write_gt(tmp_path / "gt.geojson", [_feature(name) for name in names])
    out = tmp_path / "holdout"

    code = make_holdout.main(
        [
            "--gt", str(tmp_path / "gt.geojson"),
            "--root", str(tmp_path / "train"),
            "--frac", "0.34",
            "--seed", "7",
            "--out", str(out),
        ]
    )
    assert code == 0

    holdout = json.loads((out / "holdout_regions.json").read_text(encoding="utf-8"))
    train = json.loads((out / "train_regions.json").read_text(encoding="utf-8"))
    assert len(holdout) == 2  # max(1, round(0.34 * 6))
    assert sorted(holdout + train) == names
    assert set(train).isdisjoint(holdout)

    holdout_gt = json.loads((out / "holdout_gt.geojson").read_text(encoding="utf-8"))
    train_gt = json.loads((out / "train_gt.geojson").read_text(encoding="utf-8"))
    assert holdout_gt["type"] == "FeatureCollection"
    assert holdout_gt["crs"] == train_gt["crs"]
    assert len(holdout_gt["features"]) + len(train_gt["features"]) == 6
    assert {f["properties"]["region"] for f in holdout_gt["features"]} == set(holdout)

    manifest = json.loads((out / "holdout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["n_holdout"] == 2
    assert manifest["n_train"] == 4
    assert manifest["gt_regions_not_in_root"] == []
    assert sum(manifest["holdout_classes"].values()) == 2


def test_make_holdout_deterministic(tmp_path: Path) -> None:
    """Тот же сид — то же разбиение; соседние сиды дают другой выбор."""
    names = [f"region_{i:02d}" for i in range(20)]
    _make_regions(tmp_path / "train", names)

    def run(seed: int, out: Path) -> list[str]:
        code = make_holdout.main(
            [
                "--root", str(tmp_path / "train"),
                "--count", "5",
                "--seed", str(seed),
                "--out", str(out),
            ]
        )
        assert code == 0
        return json.loads((out / "holdout_regions.json").read_text(encoding="utf-8"))

    first = run(42, tmp_path / "a")
    assert first == run(42, tmp_path / "b")
    assert len(first) == 5
    others = [run(seed, tmp_path / f"s{seed}") for seed in (43, 44, 45)]
    assert any(first != other for other in others)


def test_make_holdout_count_mode(tmp_path: Path) -> None:
    """--count задаёт точный размер холдаута."""
    _make_regions(tmp_path / "train", [f"r{i}" for i in range(10)])
    out = tmp_path / "holdout"
    code = make_holdout.main(
        ["--root", str(tmp_path / "train"), "--count", "3", "--out", str(out)]
    )
    assert code == 0
    holdout = json.loads((out / "holdout_regions.json").read_text(encoding="utf-8"))
    train = json.loads((out / "train_regions.json").read_text(encoding="utf-8"))
    assert len(holdout) == 3
    assert len(train) == 7


def test_make_holdout_gt_only(tmp_path: Path) -> None:
    """Без --root имена регионов берутся из GT."""
    _write_gt(
        tmp_path / "gt.geojson",
        [_feature("a"), _feature("b"), _feature("c"), _feature("d")],
    )
    out = tmp_path / "holdout"
    code = make_holdout.main(
        ["--gt", str(tmp_path / "gt.geojson"), "--count", "1", "--seed", "3", "--out", str(out)]
    )
    assert code == 0
    holdout = json.loads((out / "holdout_regions.json").read_text(encoding="utf-8"))
    assert len(holdout) == 1
    holdout_gt = json.loads((out / "holdout_gt.geojson").read_text(encoding="utf-8"))
    assert [f["properties"]["region"] for f in holdout_gt["features"]] == holdout


def test_make_holdout_missing_gt_fails(tmp_path: Path) -> None:
    """Нет GT-файла — код 1, результаты не пишутся."""
    out = tmp_path / "holdout"
    code = make_holdout.main(["--gt", str(tmp_path / "nope.geojson"), "--out", str(out)])
    assert code == 1
    assert not (out / "holdout_regions.json").exists()


def test_make_holdout_invalid_gt_fails(tmp_path: Path) -> None:
    """GT без FeatureCollection — код 1."""
    bad = tmp_path / "bad.geojson"
    bad.write_text('{"type": "Point", "coordinates": [0, 0]}', encoding="utf-8")
    code = make_holdout.main(["--gt", str(bad), "--out", str(tmp_path / "holdout")])
    assert code == 1


def test_make_holdout_no_region_key_fails(tmp_path: Path) -> None:
    """В properties нет региона — код 1 с подсказкой доступных ключей."""
    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"area": "x"}, "geometry": None}],
    }
    gt = tmp_path / "gt.geojson"
    gt.write_text(json.dumps(payload), encoding="utf-8")
    code = make_holdout.main(["--gt", str(gt), "--out", str(tmp_path / "holdout")])
    assert code == 1


def test_make_holdout_empty_root_fails(tmp_path: Path) -> None:
    """Каталог без регионов (папок с «06_разметка») — код 1."""
    (tmp_path / "train").mkdir()
    code = make_holdout.main(
        ["--root", str(tmp_path / "train"), "--out", str(tmp_path / "holdout")]
    )
    assert code == 1


def test_make_holdout_frac_too_big_fails(tmp_path: Path) -> None:
    """Холдаут съедает все регионы — код 1."""
    _make_regions(tmp_path / "train", ["a", "b"])
    code = make_holdout.main(
        ["--root", str(tmp_path / "train"), "--frac", "0.9", "--out", str(tmp_path / "holdout")]
    )
    assert code == 1


def test_make_holdout_empty_holdout_gt_code_1(tmp_path: Path) -> None:
    """Holdout-регионы без объектов в GT — файлы записаны, код 1 (как у make_gt)."""
    names = ["a", "b", "c", "d"]
    _make_regions(tmp_path / "train", names)
    _write_gt(tmp_path / "gt.geojson", [_feature("a")])

    holdout = make_holdout.pick_holdout(names, seed=5, count=3)
    expected = 0 if "a" in holdout else 1
    out = tmp_path / "holdout"
    code = make_holdout.main(
        [
            "--gt", str(tmp_path / "gt.geojson"),
            "--root", str(tmp_path / "train"),
            "--count", "3",
            "--seed", "5",
            "--out", str(out),
        ]
    )
    assert code == expected

    data = json.loads((out / "holdout_gt.geojson").read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    if expected == 1:
        assert data["features"] == []