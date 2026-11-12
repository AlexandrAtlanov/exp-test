"""Тесты koz3.postprocess: критерии приёмки 1–6 и дополнительные проверки.

Два способа запуска (из любого каталога):
    python -m pytest tools/tests/test_postprocess.py -v
    python tools/tests/test_postprocess.py
Файл сам добавляет submission/ в sys.path и ожидает структуру раздела 5:
submission/ и tools/ — соседние каталоги одного репозитория.
"""

import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "submission"))

from koz3 import config, postprocess  # noqa: E402
from koz3.datatypes import CLASSES, Detection  # noqa: E402

_REGION_NAME = "test_region"
"""Имя региона для всех тестовых детекций."""


def _run_no_copy(detections: list[Detection]) -> list[Detection]:
    """postprocess.run без копий курганов (KURGANY_CROSS_COPY = None) — старые проверки."""
    saved = config.KURGANY_CROSS_COPY
    config.KURGANY_CROSS_COPY = None
    try:
        return postprocess.run(detections)
    finally:
        config.KURGANY_CROSS_COPY = saved


def _run_with_copy(detections: list[Detection], factor: float = 0.5) -> list[Detection]:
    """postprocess.run с копиями курганов во второй класс (confidence × factor)."""
    saved = config.KURGANY_CROSS_COPY
    config.KURGANY_CROSS_COPY = factor
    try:
        return postprocess.run(detections)
    finally:
        config.KURGANY_CROSS_COPY = saved


def _square(x_m: float, y_m: float, size_m: float) -> Polygon:
    """Квадрат со стороной size_m; (x_m, y_m) — левый нижний угол (UTM, м)."""
    return Polygon(
        [
            (x_m, y_m),
            (x_m + size_m, y_m),
            (x_m + size_m, y_m + size_m),
            (x_m, y_m + size_m),
        ]
    )


def _det(
    geom: Polygon,
    class_name: str = "kurgany_tselye",
    conf: float = 0.9,
    markup_type: str = "li",
) -> Detection:
    """Тестовая детекция с геометрией в UTM-метрах и фиктивным регионом."""
    return Detection(
        geom=geom,
        class_name=class_name,
        confidence=conf,
        markup_type=markup_type,
        region_name=_REGION_NAME,
    )


def test_duplicate_same_class_and_type_suppressed() -> None:
    """Приёмка 1: почти одинаковые квадраты одного класса и типа → один."""
    dets = [
        _det(_square(0.0, 0.0, 10.0), conf=0.9),
        _det(_square(0.5, 0.0, 10.0), conf=0.8),
    ]
    result = _run_no_copy(dets)
    assert len(result) == 1
    assert result[0].confidence == 0.9
    assert result[0].class_name == "kurgany_tselye"
    assert result[0].markup_type == "li"


def test_duplicate_different_markup_type_kept() -> None:
    """Приёмка 2: те же квадраты с разными markup_type → остаются оба."""
    dets = [
        _det(_square(0.0, 0.0, 10.0), conf=0.9, markup_type="li"),
        _det(_square(0.5, 0.0, 10.0), conf=0.8, markup_type="ae"),
    ]
    result = _run_no_copy(dets)
    assert len(result) == 2
    assert {det.markup_type for det in result} == {"li", "ae"}


def test_small_inside_large_suppressed() -> None:
    """Приёмка 3: малый квадрат целиком в большом → подавлен покрытием.

    IoU пары ≈ 0.04 (меньше NMS_IOU), но покрытие = 1.0 > NMS_COVER.
    """
    dets = [
        _det(_square(0.0, 0.0, 10.0), conf=0.9),
        _det(_square(1.0, 1.0, 2.0), conf=0.6),
    ]
    result = _run_no_copy(dets)
    assert len(result) == 1
    assert abs(result[0].geom.area - 100.0) < 1e-9


def test_disjoint_kept() -> None:
    """Приёмка 4: соседние непересекающиеся квадраты → остаются оба."""
    dets = [
        _det(_square(0.0, 0.0, 10.0), conf=0.9),
        _det(_square(100.0, 0.0, 10.0), conf=0.5),
    ]
    result = _run_no_copy(dets)
    assert len(result) == 2


def test_unknown_class_dropped() -> None:
    """Приёмка 5: класс вне CLASSES → отброшен, несмотря на confidence."""
    dets = [_det(_square(0.0, 0.0, 10.0), class_name="unknown_klass")]
    assert _run_no_copy(dets) == []


def test_default_threshold_boundary() -> None:
    """Порог DEFAULT_CONF_THRESHOLD включительный: 0.25 остаётся, 0.24 нет."""
    dets = [
        _det(_square(0.0, 0.0, 10.0), conf=0.24),
        _det(_square(100.0, 0.0, 10.0), conf=0.25),
    ]
    result = _run_no_copy(dets)
    assert [det.confidence for det in result] == [0.25]


def test_pair_threshold_overrides_default() -> None:
    """Порог из CONF_THRESHOLD перекрывает DEFAULT_CONF_THRESHOLD."""
    pair = ("kurgany_tselye", "li")
    saved = config.CONF_THRESHOLD.get(pair)
    config.CONF_THRESHOLD[pair] = 0.5
    try:
        dets = [
            _det(_square(0.0, 0.0, 10.0), conf=0.4),
            _det(_square(0.0, 0.0, 10.0), conf=0.6),
        ]
        result = _run_no_copy(dets)
        assert [det.confidence for det in result] == [0.6]
    finally:
        if saved is None:
            del config.CONF_THRESHOLD[pair]
        else:
            config.CONF_THRESHOLD[pair] = saved


def test_min_area_filter() -> None:
    """Площадь меньше MIN_AREA_M2 класса → отбраковка (квадраты 1×1 и 2×2)."""
    dets = [
        _det(_square(0.0, 0.0, 1.0), conf=0.9),
        _det(_square(100.0, 0.0, 2.0), conf=0.9),
    ]
    result = _run_no_copy(dets)
    assert len(result) == 1
    assert abs(result[0].geom.area - 4.0) < 1e-9


def test_simplify_removes_collinear_vertices() -> None:
    """Simplify убирает коллинеарные вершины: кольцо 9 координат → 5."""
    geom = Polygon(
        [
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (10.0, 10.0),
            (5.0, 10.0),
            (0.0, 10.0),
            (0.0, 5.0),
            (0.0, 0.0),
        ]
    )
    result = _run_no_copy([_det(geom, conf=0.9)])
    assert len(result) == 1
    assert len(result[0].geom.exterior.coords) == 5


def test_input_not_mutated() -> None:
    """run() не мутирует входные детекции (копии через dataclasses.replace)."""
    geom_a = Polygon(
        [
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (5.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]
    )
    det_a = _det(geom_a, conf=0.9)
    det_b = _det(_square(0.5, 0.0, 10.0), conf=0.8)
    wkt_a, wkt_b = det_a.geom.wkt, det_b.geom.wkt
    original = [det_a, det_b]
    result = _run_no_copy(original)
    assert len(original) == 2
    assert det_a.geom.wkt == wkt_a and det_b.geom.wkt == wkt_b
    assert len(result) == 1
    assert len(result[0].geom.exterior.coords) == 5


def test_limit_features() -> None:
    """При превышении MAX_FEATURES остаются самые уверенные (лимит 3 из 5)."""
    saved_limit = config.MAX_FEATURES
    config.MAX_FEATURES = 3
    try:
        xs_m = (0.0, 100.0, 200.0, 300.0, 400.0)
        confs = (0.30, 0.35, 0.40, 0.45, 0.50)
        dets = [_det(_square(x_m, 0.0, 10.0), conf=conf) for x_m, conf in zip(xs_m, confs)]
        result = _run_no_copy(dets)
        assert len(result) == 3
        assert sorted(det.confidence for det in result) == [0.40, 0.45, 0.50]
    finally:
        config.MAX_FEATURES = saved_limit


def test_equal_confidence_keeps_first() -> None:
    """Равные confidence: остаётся первый по исходному порядку (детерминизм)."""
    first = _det(_square(0.0, 0.0, 10.0), conf=0.8)
    second = _det(_square(0.5, 0.0, 10.0), conf=0.8)
    result = _run_no_copy([first, second])
    assert len(result) == 1
    assert result[0].geom.equals(first.geom)
    assert not result[0].geom.equals(second.geom)


def test_whole_object_beats_tile_edge_fragment() -> None:
    """Целая копия объекта подавляет обрезок с края тайла, даже если обрезок увереннее."""
    whole = _det(_square(0.0, 0.0, 10.0), conf=0.80)
    fragment = _det(_square(0.0, 0.0, 6.0), conf=0.85)
    fragment.extra["edge_cut"] = True
    result = _run_no_copy([fragment, whole])
    assert len(result) == 1
    assert result[0].geom.equals(whole.geom)


def test_fragment_kept_when_no_whole_copy() -> None:
    """Обрезок без целой копии (объект крупнее перекрытия тайлов) остаётся."""
    fragment = _det(_square(0.0, 0.0, 6.0), conf=0.85)
    fragment.extra["edge_cut"] = True
    other = _det(_square(50.0, 0.0, 6.0), conf=0.9)
    assert len(_run_no_copy([fragment, other])) == 2


def test_config_dicts_cover_all_classes() -> None:
    """MIN_AREA_M2 и SIMPLIFY_M покрывают все CLASSES (страховка KeyError)."""
    for class_name in CLASSES:
        assert class_name in config.MIN_AREA_M2, class_name
        assert class_name in config.SIMPLIFY_M, class_name


def test_speed_20k_under_10s() -> None:
    """Приёмка 6: 20 000 квадратов проходят run() быстрее 10 секунд.

    Сетка 200×100 с шагом 8 м, сторона 6 м, джиттер ±2.5 м: часть соседей
    пересекается и подавляется NMS, так что замер включает реальную работу.
    """
    rng = np.random.default_rng(config.SEED)
    xs_m = 8.0 * np.arange(200) + rng.uniform(-2.5, 2.5, 200)
    ys_m = 8.0 * np.arange(100) + rng.uniform(-2.5, 2.5, 100)
    dets: list[Detection] = []
    for x_m in xs_m:
        for y_m in ys_m:
            geom = _square(float(x_m), float(y_m), 6.0)
            dets.append(_det(geom, conf=float(rng.uniform(0.3, 1.0))))
    assert len(dets) == 20000
    start_s = time.perf_counter()
    result = _run_no_copy(dets)
    elapsed_s = time.perf_counter() - start_s
    assert 0 < len(result) <= 20000
    assert elapsed_s < 10.0
    print(f"20k: {elapsed_s:.2f} с, осталось {len(result)} детекций")


def test_cross_copy_adds_other_kurgan_class() -> None:
    """Курган целый → ещё копия в повреждённые с confidence × 0.5, та же геометрия."""
    geom = _square(0.0, 0.0, 10.0)
    out = _run_with_copy([_det(geom, "kurgany_tselye", conf=0.8)])
    by_class = {det.class_name: det for det in out}
    assert set(by_class) == {"kurgany_tselye", "kurgany_povrezhdennye"}
    copy = by_class["kurgany_povrezhdennye"]
    assert abs(copy.confidence - 0.4) < 1e-9
    assert copy.extra.get("cross_copy") is True
    assert copy.geom.equals(by_class["kurgany_tselye"].geom)
    assert copy.markup_type == "li"


def test_cross_copy_skipped_when_target_has_duplicate() -> None:
    """В целевом классе уже есть дубль (как model.pt в оба класса) → копий нет."""
    dets = [
        _det(_square(0.0, 0.0, 10.0), "kurgany_tselye", conf=0.8),
        _det(_square(0.5, 0.0, 10.0), "kurgany_povrezhdennye", conf=0.6),
    ]
    out = _run_with_copy(dets)
    assert len(out) == 2
    assert not any(det.extra.get("cross_copy") for det in out)


def test_cross_copy_only_for_kurgany_and_bypasses_threshold() -> None:
    """Другие классы не копируются; копия ниже порога не отбрасывается."""
    dets = [
        _det(_square(0.0, 0.0, 30.0), "gorodishcha", conf=0.9),
        _det(_square(100.0, 0.0, 10.0), "kurgany_povrezhdennye", conf=0.3),
    ]
    out = _run_with_copy(dets)
    classes = sorted(det.class_name for det in out)
    assert classes == ["gorodishcha", "kurgany_povrezhdennye", "kurgany_tselye"]
    copy = next(det for det in out if det.class_name == "kurgany_tselye")
    assert copy.confidence < config.DEFAULT_CONF_THRESHOLD


def test_cross_copy_disabled_by_none() -> None:
    """KURGANY_CROSS_COPY = None — выход как раньше, без копий."""
    out = _run_no_copy([_det(_square(0.0, 0.0, 10.0), "kurgany_tselye", conf=0.8)])
    assert [det.class_name for det in out] == ["kurgany_tselye"]


def main() -> int:
    """Запустить все тесты по очереди; вернуть код выхода 0 или 1."""
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures: list[str] = []
    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"{test_fn.__name__}: {message}")
            print(f"FAIL {test_fn.__name__}: {message}")
        else:
            print(f"OK   {test_fn.__name__}")
    print(f"Итог: {len(tests) - len(failures)} из {len(tests)} тестов прошло")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
