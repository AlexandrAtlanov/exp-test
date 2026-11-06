"""Шаг 5. Конвертация весов .pt → TensorRT .engine (FP16) и проверка совпадения.

ВАЖНО: engine привязан к архитектуре GPU и к версии TensorRT. Собирать его
нужно на видеокарте того же поколения, что у платформы (RTX 4090, sm 8.9),
и ВНУТРИ того же Docker-образа, что указан в metadata.json (та же версия
TensorRT). Иначе koz3.yolo_runtime увидит несовпадение в .engine.json и
откатится на .pt — прогон не упадёт, но и ускорения не будет.

Скрипт:
1. экспортирует engine (ultralytics, format="engine", half=True);
2. пишет рядом паспорт <имя>.engine.json: классы, batch, imgsz, версии
   TensorRT/ultralytics, GPU и compute capability;
3. сравнивает предсказания .pt и .engine на N тайлах из val: число объектов
   и IoU масок. Расхождение больше порога — повод не использовать engine.

Запуск (в образе платформы на GPU):
  python training/export_engine.py --weights weights_out/lidar_yolo.pt \
      --samples /датасет/images/val --n 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import settings
from koz3 import config as runtime_config


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU объединённых масок двух предсказаний одного тайла."""
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def union_mask(result, shape_hw) -> np.ndarray:
    """Объединение всех масок результата ultralytics в одну булеву маску."""
    if result.masks is None:
        return np.zeros(shape_hw, dtype=bool)
    return (result.masks.data.cpu().numpy() > 0.5).any(axis=0)[: shape_hw[0], : shape_hw[1]]


def compare(pt_model, engine_model, images: list[np.ndarray], imgsz: int) -> dict:
    """Сравнить .pt (FP16) и engine на одних и тех же тайлах."""
    ious, count_diff, t_pt, t_engine = [], [], 0.0, 0.0
    for image in images:
        bgr = np.ascontiguousarray(image[..., ::-1])
        kwargs = dict(
            imgsz=imgsz, conf=float(runtime_config.YOLO["conf"]), retina_masks=True, verbose=False
        )
        start = time.perf_counter()
        r_pt = pt_model.predict(bgr, half=True, device=0, **kwargs)[0]
        t_pt += time.perf_counter() - start
        start = time.perf_counter()
        r_en = engine_model.predict(bgr, device=0, **kwargs)[0]
        t_engine += time.perf_counter() - start
        shape = image.shape[:2]
        ious.append(mask_iou(union_mask(r_pt, shape), union_mask(r_en, shape)))
        count_diff.append(abs(len(r_pt.boxes) - len(r_en.boxes)))
    return {
        "tiles": len(images),
        "mean_mask_iou": round(float(np.mean(ious)), 4),
        "min_mask_iou": round(float(np.min(ious)), 4),
        "mean_count_diff": round(float(np.mean(count_diff)), 3),
        "ms_per_tile_pt_fp16": round(1000 * t_pt / len(images), 1),
        "ms_per_tile_engine": round(1000 * t_engine / len(images), 1),
    }


def main() -> None:
    """Экспорт engine, паспорт, проверка совпадения с .pt."""
    import tensorrt as trt
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="lidar_yolo.pt")
    parser.add_argument("--samples", type=Path, help="папка тайлов для сравнения (images/val)")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1, help="1 — надёжнее всего")
    parser.add_argument("--workspace", type=float, default=4.0, help="ГБ для сборки")
    args = parser.parse_args()

    imgsz = int(runtime_config.YOLO["imgsz"])
    pt_model = YOLO(str(args.weights), task="segment")
    exported = Path(
        pt_model.export(
            format="engine",
            imgsz=imgsz,
            half=True,
            batch=args.batch,
            device=0,
            workspace=args.workspace,
            simplify=True,
        )
    )
    target = args.weights.with_name(str(runtime_config.YOLO["weights_engine"]))
    if exported.resolve() != target.resolve():
        exported.replace(target)
    meta = {
        "names": {int(k): v for k, v in pt_model.names.items()},
        "batch": args.batch,
        "imgsz": imgsz,
        "half": True,
        "tensorrt": trt.__version__,
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "source_weights": args.weights.name,
    }
    if list(meta["names"].values()) != list(settings.CLASSES):
        raise RuntimeError(f"классы модели {meta['names']} не совпадают с {settings.CLASSES}")

    if args.samples:
        files = sorted(args.samples.glob(f"*.{settings.IMAGE_FORMAT}"))[: args.n]
        images = [np.asarray(Image.open(p).convert("RGB")) for p in files]
        engine_model = YOLO(str(target), task="segment")
        meta["check"] = compare(pt_model, engine_model, images, imgsz)

    meta_path = target.with_suffix(".engine.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    check = meta.get("check")
    if check and check["mean_mask_iou"] < 0.95:
        print("ВНИМАНИЕ: engine заметно расходится с .pt — не отправлять с engine, разобраться")


if __name__ == "__main__":
    main()
