"""koz3/tuning.py: weights/postprocess.json пишется, применяется и защищён sha256 весов."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest

from koz3 import config, tuning


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch) -> Path:
    """Копии изменяемых настроек (тест не портит config) и своя папка весов."""
    monkeypatch.setattr(config, "CONF_THRESHOLD", dict(config.CONF_THRESHOLD))
    monkeypatch.setattr(config, "MIN_AREA_M2", dict(config.MIN_AREA_M2))
    monkeypatch.setattr(config, "NMS_IOU", config.NMS_IOU)
    monkeypatch.setattr(config, "NMS_COVER", config.NMS_COVER)
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / str(config.YOLO["weights_pt"])).write_bytes(b"model-a")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights)
    return weights


def test_write_then_apply(isolated_config: Path) -> None:
    tuning.write_tuned(
        {("kurgany_tselye", "li"): 0.15}, {"kurgany_tselye": 10.0}, 0.3, 0.9, {"search": "full"}
    )
    assert tuning.apply_tuned() is True
    assert config.CONF_THRESHOLD[("kurgany_tselye", "li")] == 0.15
    assert config.CONF_THRESHOLD[("finds_points", "maggr")] == 0.0  # прочие пары не тронуты
    assert config.MIN_AREA_M2["kurgany_tselye"] == 10.0
    assert (config.NMS_IOU, config.NMS_COVER) == (0.3, 0.9)


def test_other_weights_not_applied(isolated_config: Path) -> None:
    tuning.write_tuned({("kurgany_tselye", "li"): 0.15}, {}, None, None, {})
    (isolated_config / str(config.YOLO["weights_pt"])).write_bytes(b"model-b")  # новые веса
    before = dict(config.CONF_THRESHOLD)
    assert tuning.apply_tuned() is False
    assert config.CONF_THRESHOLD == before


@pytest.mark.parametrize(
    "content",
    ["не json", json.dumps({"conf_threshold": {"нет_такого|li": 0.1}})],
)
def test_bad_file_ignored(isolated_config: Path, content: str) -> None:
    (isolated_config / tuning.FILE_NAME).write_text(content, encoding="utf-8")
    before = (dict(config.CONF_THRESHOLD), dict(config.MIN_AREA_M2))
    assert tuning.apply_tuned() is False
    assert (dict(config.CONF_THRESHOLD), dict(config.MIN_AREA_M2)) == before


def test_no_file_is_fine(isolated_config: Path) -> None:
    assert tuning.apply_tuned() is False
