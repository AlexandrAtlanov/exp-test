"""Точка входа решения КОЗ-3 — «Поток solution.py» (раздел 5 пакета).

Оркестратор без ML-логики: регионы (discovery.find_regions) × детекторы
(config.ENABLED_DETECTORS, порядок по PRIORITY) → postprocess.run →
GeoJSON-фичи в EPSG:3857 (_to_features) → output.write_result (атомарно).
Тяжёлые зависимости (torch, ultralytics, cv2) импортируют только модули
детекторов; здесь — stdlib, numpy и пакет koz3.

Контракт детектора — модуль ``koz3.detectors.<имя>``::

    PRIORITY: int                                 # меньше — раньше
    is_available(region) -> bool                  # есть ли работа
    estimate_seconds(region) -> float             # оценка времени run
    run(region, budget, cfg) -> list[Detection]   # cfg — модуль koz3.config

Пары обходятся детекторами внешним циклом (по PRIORITY), внутри — регионы:
при нехватке бюджета обрезаются последние детекторы, а не последние регионы.
Широкий ``except Exception`` ровно один — вокруг пары; падение детектора не
роняет прогон. После каждой реально отработавшей пары — чекпоинт, в конце —
финальная запись в любом случае (даже при пустом результате).

Детекции живут в UTM региона (метры); перевод в EPSG:3857 — только в
_to_features перед записью. Пороги, NMS и площади выполняет postprocess
в метрах UTM.

Запуск: ``python solution.py <входная_папка> <выход>``, выход — файл
*.geojson или каталог (output.resolve_output_path). Коды возврата: 0 —
успех (в том числе пустой результат), 2 — неверный CLI.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
from shapely.geometry import mapping

from koz3 import config, discovery, geometry, output, postprocess
from koz3.datatypes import Detection, Region
from koz3.timing import Budget

logger = logging.getLogger("koz3.solution")

#: Пакет с модулями детекторов.
DETECTORS_PACKAGE = "koz3.detectors"

#: CRS результата. После правки 5 в config.py появится SRC_CRS="EPSG:3857";
#: до неё используется локальное значение того же смысла.
OUTPUT_CRS: str = getattr(config, "SRC_CRS", "EPSG:3857")

#: Обязательные атрибуты модуля детектора (контракт раздела 5).
_DETECTOR_API: tuple[str, ...] = (
    "PRIORITY",
    "is_available",
    "estimate_seconds",
    "run",
)


def _log_level() -> int:
    """Числовой уровень логирования из config.LOG_LEVEL (имя или число)."""
    raw = str(getattr(config, "LOG_LEVEL", "INFO")).strip().upper()
    if raw.isdigit():
        return int(raw)
    level = getattr(logging, raw, None)
    return level if isinstance(level, int) else logging.INFO


def _setup_logging() -> None:
    """Настроить логирование уровня config.LOG_LEVEL (идемпотентно)."""
    logging.basicConfig(
        level=_log_level(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _fix_seed() -> None:
    """Зафиксировать random и numpy; PYTHONHASHSEED — дочерним процессам.

    torch здесь сознательно не импортируется: детерминизм GPU настраивают
    сами модули детекторов (у них свои тяжёлые зависимости).
    """
    seed = int(config.SEED)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("seed зафиксирован: %s", seed)


def _load_detectors() -> list[ModuleType]:
    """Импортировать и проверить модули из config.ENABLED_DETECTORS.

    Порядок — по возрастанию PRIORITY, при равенстве — по имени модуля.
    Модуль без полного контракта — ошибка конфигурации: падаем до старта
    обработки данных, а не посреди прогона.
    """
    modules: list[ModuleType] = []
    for name in config.ENABLED_DETECTORS:
        full_name = f"{DETECTORS_PACKAGE}.{name}"
        try:
            module = importlib.import_module(full_name)
        except ImportError as exc:
            raise ValueError(f"детектор {full_name} не импортируется: {exc}") from exc
        missing = [attr for attr in _DETECTOR_API if not hasattr(module, attr)]
        if missing:
            raise ValueError(f"детектор {full_name} нарушает контракт, нет: {missing}")
        modules.append(module)
    modules.sort(key=lambda module: (int(module.PRIORITY), module.__name__))
    for module in modules:
        logger.info("детектор %s: PRIORITY=%s", module.__name__, module.PRIORITY)
    return modules


def _to_features(
    detections: list[Detection], regions: dict[str, Region]
) -> list[dict]:
    """Детекции (UTM региона) → GeoJSON-фичи в OUTPUT_CRS; fid=регион:тип:n.

    Детекция с region_name, которого нет среди регионов, пропускается с
    logger.error. confidence не клипуется и координаты не округляются:
    некорректную уверенность отбракует output.write_result (bad_confidence),
    округление координат output тоже делает сам (COORD_PRECISION).
    """
    features: list[dict] = []
    counters: dict[tuple[str, str], int] = {}
    for det in detections:
        region = regions.get(det.region_name)
        if region is None:
            logger.error("неизвестный регион %r — детекция пропущена", det.region_name)
            continue
        key = (det.region_name, det.markup_type)
        counters[key] = counters.get(key, 0) + 1
        geom = geometry.reproject(det.geom, region.utm_crs, OUTPUT_CRS)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "class_name": det.class_name,
                    "confidence": float(det.confidence),
                    "markup_type": det.markup_type,
                    "region_name": det.region_name,
                    "fid": f"{det.region_name}:{det.markup_type}:{counters[key]}",
                },
                "geometry": mapping(geom),
            }
        )
    return features


def _flush(
    detections: list[Detection], regions: dict[str, Region], out_path: Path
) -> int:
    """Конвейер записи: postprocess → фичи → атомарный write_result.

    Вызывается как чекпоинт после каждой отработавшей пары и как финальная
    запись. Возвращает число фич, отданных в запись.
    """
    processed = postprocess.run(list(detections))
    features = _to_features(processed, regions)
    output.write_result(features, out_path)
    return len(features)


def _verify_result(out_path: Path) -> int:
    """Перечитать файл результата: FeatureCollection и число фич."""
    with open(out_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("type") != "FeatureCollection":
        raise RuntimeError(f"{out_path}: ожидался FeatureCollection")
    count = len(data.get("features") or [])
    written = output.last_stats.get("written")
    if written is not None and written != count:
        logger.warning(
            "расхождение: write_result записал %s, в файле %d", written, count
        )
    logger.info("итог: %s — %d фич", out_path, count)
    return count


def _process_pairs(
    regions: list[Region],
    detectors: list[ModuleType],
    budget: Budget,
    regions_by_name: dict[str, Region],
    out_path: Path,
) -> list[Detection]:
    """Обойти пары (детектор, регион) и вернуть накопленные детекции.

    Чекпоинт — только после пар, где run реально выполнялся: пропущенные
    по is_available или бюджету пары данных не добавляют.
    """
    detections: list[Detection] = []
    for detector in detectors:
        short = detector.__name__.rsplit(".", 1)[-1]
        for region in regions:
            try:
                if not detector.is_available(region):
                    logger.info("%s × %s: недоступен, пропуск", short, region.name)
                    continue
                seconds = float(detector.estimate_seconds(region))
                if not budget.can_afford(seconds):
                    logger.warning(
                        "%s × %s: не влезает в бюджет "
                        "(оценка %.0fs, осталось %.0fs)",
                        short,
                        region.name,
                        seconds,
                        budget.remaining(),
                    )
                    continue
                logger.info(
                    "%s × %s: запуск (оценка %.0fs, осталось %.0fs)",
                    short,
                    region.name,
                    seconds,
                    budget.remaining(),
                )
                produced = detector.run(region, budget, config)
            except Exception:  # noqa: BLE001 — единственный широкий перехват
                logger.exception(
                    "детектор %s упал на регионе %s, продолжаю", short, region.name
                )
                continue
            detections.extend(produced)
            _flush(detections, regions_by_name, out_path)
            logger.info(
                "чекпоинт после %s × %s: детекций накоплено %d",
                short,
                region.name,
                len(detections),
            )
    return detections


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: main([<вход>, <выход>]) → 0 (успех) | 2 (неверный CLI)."""
    args = list(sys.argv[1:] if argv is None else argv)
    _setup_logging()
    if len(args) != 2:
        logger.error(
            "запуск: python solution.py <входная_папка> <выход: .geojson или каталог>"
        )
        return 2
    input_arg, output_arg = args
    logger.info("вход: %s; выход: %s", input_arg, output_arg)

    out_path = output.resolve_output_path(output_arg)
    _fix_seed()
    budget = Budget(config.TIME_LIMIT_S)
    regions = discovery.find_regions(Path(input_arg))
    detectors = _load_detectors()
    regions_by_name = {region.name: region for region in regions}
    logger.info("регионов: %d; детекторов: %d", len(regions), len(detectors))

    detections = _process_pairs(regions, detectors, budget, regions_by_name, out_path)
    _flush(detections, regions_by_name, out_path)
    _verify_result(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
