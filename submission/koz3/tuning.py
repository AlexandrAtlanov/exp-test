"""Подобранные на holdout параметры постобработки: weights/postprocess.json.

Файл пишет tools/tune_thresholds.py --apply, решение читает его при старте
(solution.main → apply_tuned) и поверх значений config.py ставит:
- CONF_THRESHOLD для пар (класс, markup_type);
- MIN_AREA_M2 по классам;
- NMS_IOU и NMS_COVER.
Файл лежит рядом с весами и попадает в архив вместе с ними (build_zip берёт всю папку весов).
Параметры подбирались для конкретной модели лидара: в файле записан sha256
lidar_yolo.pt, и при другой модели файл не применяется (предупреждение в логе) — иначе
пороги одной модели молча достались бы другой. Нет файла или он битый — работают значения
config.py, решение не падает.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

FILE_NAME = "postprocess.json"
"""Имя файла с подобранными параметрами в WEIGHTS_DIR."""


def weights_sha256(weights_dir: Path | None = None) -> str | None:
    """sha256 весов лидара (config.YOLO["weights_pt"]); нет файла — None."""
    path = Path(weights_dir or config.WEIGHTS_DIR) / str(config.YOLO["weights_pt"])
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tuned(
    conf: dict[tuple[str, str], float],
    min_area: dict[str, float],
    nms_iou: float | None,
    nms_cover: float | None,
    meta: dict[str, Any],
    weights_dir: Path | None = None,
) -> Path:
    """Записать weights/postprocess.json; ключ порога — «класс|markup_type»."""
    path = Path(weights_dir or config.WEIGHTS_DIR) / FILE_NAME
    payload = {
        "conf_threshold": {f"{cls}|{markup}": float(t) for (cls, markup), t in conf.items()},
        "min_area_m2": {cls: float(a) for cls, a in min_area.items()},
        "nms_iou": nms_iou,
        "nms_cover": nms_cover,
        "weights_sha256": weights_sha256(weights_dir),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def apply_tuned(weights_dir: Path | None = None) -> bool:
    """Применить weights/postprocess.json к config; True — применён.

    Нет файла, битый файл, другой sha256 весов — False, значения config.py не меняются.
    """
    path = Path(weights_dir or config.WEIGHTS_DIR) / FILE_NAME
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        conf = {
            tuple(key.split("|", 1)): float(value)
            for key, value in data.get("conf_threshold", {}).items()
        }
        min_area = {cls: float(a) for cls, a in data.get("min_area_m2", {}).items()}
        nms_iou, nms_cover = data.get("nms_iou"), data.get("nms_cover")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("%s не читается (%s) — параметры из config.py", path.name, exc)
        return False
    expected = data.get("weights_sha256")
    if expected and expected != weights_sha256(weights_dir):
        logger.warning(
            "%s подобран для других весов лидара (sha256 %s…) — не применён, "
            "перезапустите tools/tune_thresholds.py --apply",
            path.name,
            str(expected)[:8],
        )
        return False
    unknown = [cls for cls, _ in conf if cls not in config.MIN_AREA_M2]
    unknown += [cls for cls in min_area if cls not in config.MIN_AREA_M2]
    if unknown:
        logger.warning("%s: неизвестные классы %s — не применён", path.name, sorted(set(unknown)))
        return False
    config.CONF_THRESHOLD.update(conf)
    config.MIN_AREA_M2.update(min_area)
    if nms_iou is not None:
        config.NMS_IOU = float(nms_iou)
    if nms_cover is not None:
        config.NMS_COVER = float(nms_cover)
    logger.info(
        "применены подобранные параметры %s: пороги %s, площади %s, NMS iou %s cover %s",
        path.name,
        {f"{c}|{m}": t for (c, m), t in conf.items()},
        min_area,
        config.NMS_IOU,
        config.NMS_COVER,
    )
    return True
