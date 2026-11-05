"""Постобработка детекций: пороги, чистка геометрии, NMS, упрощение, лимит.

Единственная публичная функция run() реализует конвейер постобработки из
контракта раздела 5. Все площади и допуски здесь — в метрах UTM региона
(инвариант 4.8); перевод в EPSG:3857 выполняет только output.py. Входной
список и детекции не мутируются: изменённые детекции возвращаются копиями
через dataclasses.replace.
"""

import logging
from dataclasses import replace

from shapely import STRtree
from shapely.geometry import Polygon

from . import config, geometry
from .datatypes import CLASSES, Detection

logger = logging.getLogger(__name__)

_CLASS_SET = frozenset(CLASSES)
"""Множество допустимых class_name для быстрой проверки принадлежности."""


def run(detections: list[Detection]) -> list[Detection]:
    """Прогнать детекции через конвейер постобработки.

    Шаги контракта: фильтр классов и порогов → чистка геометрии и минимальная
    площадь → NMS внутри (class_name, markup_type) → упрощение колец →
    лимит числа объектов по confidence. Возвращает новый список детекций;
    порядок выхода детерминирован: группы по возрастанию ключа
    (class_name, markup_type), внутри группы — целые раньше обрезанных краем
    тайла, затем убывание confidence.
    """
    n_input = len(detections)
    thresholded = _filter_threshold(detections)
    by_area = _filter_area(thresholded)
    deduplicated = _nms(by_area)
    simplified = _simplify(deduplicated)
    result = _limit_features(simplified)
    logger.info(
        "postprocess: вход %d, порог %d, площадь %d, NMS %d, итог %d",
        n_input,
        len(thresholded),
        len(by_area),
        len(deduplicated),
        len(result),
    )
    return result


def _threshold_for(class_name: str, markup_type: str) -> float:
    """Вернуть порог уверенности для пары (class_name, markup_type)."""
    return config.CONF_THRESHOLD.get((class_name, markup_type), config.DEFAULT_CONF_THRESHOLD)


def _filter_threshold(detections: list[Detection]) -> list[Detection]:
    """Оставить детекции класса из CLASSES с confidence не ниже порога пары."""
    kept: list[Detection] = []
    for det in detections:
        if det.class_name not in _CLASS_SET:
            continue
        if det.confidence < _threshold_for(det.class_name, det.markup_type):
            continue
        kept.append(det)
    return kept


def _filter_area(detections: list[Detection]) -> list[Detection]:
    """Почистить геометрию и отбраковать детекции меньше минимальной площади.

    Чистка — geometry.clean_polygon (make_valid, самая большая часть);
    детекции без полигональной части и с площадью меньше
    config.MIN_AREA_M2[class_name] (м², UTM региона) отбрасываются.
    Годные детекции возвращаются копиями с чистой геометрией.
    """
    kept: list[Detection] = []
    for det in detections:
        cleaned = geometry.clean_polygon(det.geom)
        if cleaned is None or cleaned.area < config.MIN_AREA_M2[det.class_name]:
            continue
        kept.append(replace(det, geom=cleaned))
    return kept


def _nms(detections: list[Detection]) -> list[Detection]:
    """Подавить дубли внутри групп (class_name, markup_type).

    Разные markup_type не склеиваются (инвариант 4.5). Группы обходятся
    в отсортированном порядке ключей — итоговый порядок детерминирован.
    """
    groups: dict[tuple[str, str], list[Detection]] = {}
    for det in detections:
        groups.setdefault((det.class_name, det.markup_type), []).append(det)
    result: list[Detection] = []
    for key in sorted(groups):
        result.extend(_nms_group(groups[key]))
    return result


def _nms_group(group: list[Detection]) -> list[Detection]:
    """Жадный NMS одной группы: сначала целые, затем по убыванию confidence.

    Детекции с extra["edge_cut"] (маска обрезана внутренним краем тайла)
    идут после целых: иначе обрезок объекта с соседнего тайла подавил бы
    целую копию. Обрезок выживает, только если целой копии нет (объект
    крупнее перекрытия тайлов). STRtree.query с предикатом intersects даёт
    кандидатов на подавление; подавляется идущая позже в этом порядке
    (при равных ключах — позже в исходном порядке).
    """
    ordered = sorted(group, key=_nms_order)
    geoms = [det.geom for det in ordered]
    areas_m2 = [geom.area for geom in geoms]
    tree = STRtree(geoms)
    suppressed = [False] * len(ordered)
    for i in range(len(ordered)):
        if suppressed[i]:
            continue
        for j in (int(idx) for idx in tree.query(geoms[i], predicate="intersects")):
            if j <= i or suppressed[j]:
                continue
            if _suppress_pair(geoms[i], geoms[j], areas_m2[i], areas_m2[j]):
                suppressed[j] = True
    return [det for det, flag in zip(ordered, suppressed) if not flag]


def _nms_order(det: Detection) -> tuple[bool, float]:
    """Ключ порядка NMS: целые раньше обрезанных краем тайла, затем confidence."""
    return bool(det.extra.get("edge_cut", False)), -det.confidence


def _suppress_pair(
    geom_a: Polygon,
    geom_b: Polygon,
    area_a_m2: float,
    area_b_m2: float,
) -> bool:
    """Вернуть True, если пара геометрий считается дублем (строгие неравенства).

    Дубль — IoU > config.NMS_IOU или пересечение / меньшая площадь >
    config.NMS_COVER. Площади передаются заранее, чтобы не пересчитывать их
    попарно. При положительной площади пересечения делители ненулевые:
    пересечение не больше каждой из площадей, объединение — не меньше
    большей из них.
    """
    inter_m2 = geom_a.intersection(geom_b).area
    if inter_m2 <= 0.0:
        return False
    union_m2 = area_a_m2 + area_b_m2 - inter_m2
    iou = inter_m2 / union_m2
    cover = inter_m2 / min(area_a_m2, area_b_m2)
    return iou > config.NMS_IOU or cover > config.NMS_COVER


def _simplify(detections: list[Detection]) -> list[Detection]:
    """Упростить кольца полигонов с допуском config.SIMPLIFY_M по классу.

    Возвращает копии детекций; поведение — geometry.simplify_m
    (preserve_topology=True, откат к исходной геометрии при пустом результате).
    """
    simplified: list[Detection] = []
    for det in detections:
        geom = geometry.simplify_m(det.geom, config.SIMPLIFY_M[det.class_name])
        simplified.append(replace(det, geom=geom))
    return simplified


def _limit_features(detections: list[Detection]) -> list[Detection]:
    """Оставить не более config.MAX_FEATURES самых уверенных детекций."""
    if len(detections) <= config.MAX_FEATURES:
        return detections
    ordered = sorted(detections, key=lambda det: -det.confidence)
    limit = config.MAX_FEATURES
    return ordered[:limit]
