"""Шаг 4. Обучение YOLO-seg на датасете из build_dataset.py.

Аугментации выбраны под хиллшейды:
- hsv_h = 0: в цветных отмывках (c, ch) оттенок кодирует высоту, сдвиг оттенка
  ломает смысл;
- отражения и повороты по умолчанию выключены: в серых отмывках (g, i) тень
  задаёт направление освещения (азимут 315°), а на инференсе оно всегда одно.
  Включение (--flips) — отдельный эксперимент со сравнением на val.
Веса по окончании копируются в --export-dir под именем из koz3.config.YOLO.

Скорость:
- тайлы датасета 1024 px (LIDAR["tile_px"]) модель видит сжатыми до
  config.YOLO["imgsz"] (512): в 2.5–4 раза меньше вычислений, инференс — с тем же
  imgsz; датасет при смене imgsz не пересобирается;
- --cache ram (по умолчанию): картинки после первой эпохи берутся из памяти, а не
  распаковываются из PNG; если памяти мало, ultralytics сам предупредит и не
  будет кэшировать — тогда --cache disk;
- детерминизм обучения выключен (быстрее на GPU); seed фиксирован. Регламент
  требует детерминизма инференса, а не обучения;
- --model <путь к model.pt бейзлайна> — старт с весов, уже обученных на
  отмывках: сходится за меньшее число эпох.

Запуск:
  python training/train.py --data /датасет/dataset.yaml --model yolo11s-seg.pt \
      --epochs 100 --batch 16 --name lidar_v1 [--flips] [--cache ram|disk|none]

Перед стартом проверяется GPU (gpu_problem): если torch её не видит или не
поддерживает архитектуру карты (RTX 50xx — нужен torch ≥ 2.7 с CUDA 12.8), обучение не
начинается — иначе оно молча идёт на CPU в десятки раз медленнее.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import settings
from koz3 import config as runtime_config


def sha256(path: Path) -> str:
    """Хеш файла весов — для журнала команды («Модели и веса»)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_problem() -> str | None:
    """Почему обучать на GPU нельзя, или None. RTX 50xx (sm_120) требует torch с CUDA 12.8."""
    import torch

    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            return "установлена CPU-сборка torch"
        return "torch собран с CUDA, но видеокарта не видна (драйвер, docker без --gpus all)"
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if arch not in supported and f"compute_{major}{minor}" not in supported:
        return (
            f"torch {torch.__version__} (CUDA {torch.version.cuda}) не поддерживает {arch}; "
            "для RTX 50xx нужен torch >= 2.7 с CUDA 12.8 (cu128)"
        )
    return None


def main() -> None:
    """Обучить модель и сохранить лучшие веса с паспортом."""
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--model", default="yolo11s-seg.pt", help="стартовые веса (s — быстрее, m — точнее)"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="при imgsz 512 на 16 ГБ — 16–32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="lidar_v1")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--export-dir", type=Path, default=Path("weights_out"))
    parser.add_argument("--flips", action="store_true", help="включить отражения (эксперимент)")
    parser.add_argument(
        "--cache",
        choices=("ram", "disk", "none"),
        default="ram",
        help="кэш картинок: ram — быстрее всего, disk — если не хватает памяти",
    )
    args = parser.parse_args()

    if args.device != "cpu":
        problem = gpu_problem()
        if problem is not None:
            raise SystemExit(
                f"GPU недоступна для обучения: {problem}. "
                "Обучать на CPU — только явно: --device cpu"
            )
        import torch

        print(f"GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=int(runtime_config.YOLO["imgsz"]),
        cache=False if args.cache == "none" else args.cache,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        seed=settings.SEED,
        deterministic=False,  # быстрее; детерминизм нужен инференсу, не обучению
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.1,
        scale=0.25,
        fliplr=0.5 if args.flips else 0.0,
        flipud=0.5 if args.flips else 0.0,
        mosaic=1.0,
        overlap_mask=False,  # маски объектов не сливаются при наложении
        plots=True,
    )
    best = Path(model.trainer.best)
    names = YOLO(str(best)).names
    expected = dict(enumerate(settings.CLASSES))
    if {int(k): v for k, v in names.items()} != expected:
        raise RuntimeError(
            f"классы модели {names} не совпадают с config.SURFACE_CLASSES {expected}"
        )

    args.export_dir.mkdir(parents=True, exist_ok=True)
    target = args.export_dir / str(runtime_config.YOLO["weights_pt"])
    shutil.copy2(best, target)
    passport = {
        "file": target.name,
        "sha256": sha256(target),
        "ultralytics": ultralytics_version,
        "start_weights": args.model,
        "epochs": args.epochs,
        "imgsz": int(runtime_config.YOLO["imgsz"]),
        "tile_px": settings.TILE_PX,
        "run": str(best.parent.parent),
        "classes": settings.CLASSES,
        "flips": args.flips,
    }
    target.with_suffix(".pt.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(passport, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
