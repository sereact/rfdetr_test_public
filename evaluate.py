#!/usr/bin/env python3
"""Score a trained checkpoint on a COCO-labelled folder (mask quality per class).

    uv run python evaluate.py --checkpoint runs/try1/checkpoint_best_total.pth --data runs/try1/dataset/valid \
        [--threshold 0.5] [--out results.json] [--overlays overlays/]

--data is a folder with `_annotations.coco.json` and its images (same layout as
train.py). Ground-truth categories are matched to the model's classes BY NAME.

Reported per class and overall:
    mask AP@[.5:.95] and AP@.5   (COCO evaluation on the masks, all predictions)
    precision / recall           at --threshold, one-to-one matching at mask IoU >= 0.5
--overlays writes every image with the predicted masks drawn on it.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COLOURS = [(255, 60, 60), (0, 190, 255), (255, 220, 0), (80, 230, 80), (230, 100, 255), (255, 150, 40)]


def load_model(checkpoint: Path):
    from rfdetr import RFDETRSeg2XLarge
    args = torch.load(checkpoint, map_location="cpu", weights_only=False)["args"]
    print(f"classes {getattr(args, 'class_names', None)}")
    os.chdir(Path(__file__).resolve().parent)
    return RFDETRSeg2XLarge(pretrain_weights=str(checkpoint))


def resolve(data: Path, file_name: str) -> Path:
    p = Path(file_name)
    return next(c for c in (p, data / p, data / "images" / p.name, data / p.name) if c.is_file())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return inter / (np.logical_or(a, b).sum() + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="score cut for the precision / recall counts")
    ap.add_argument("--out", type=Path, default=None, help="write the numbers as json")
    ap.add_argument("--overlays", type=Path, default=None, help="folder for images with the predictions drawn")
    args = ap.parse_args()

    data = args.data.resolve()
    gt = COCO(str(data / "_annotations.coco.json"))
    model = load_model(args.checkpoint.resolve())
    # predicted class_id is a 0-based index into the class list stored in the checkpoint
    # (rfdetr's own class_names property numbers them from 1, which is off by one for fine-tunes)
    names = list(getattr(model.model, "class_names", None) or model.class_names.values())
    model_names = {i: n for i, n in enumerate(names)}
    gt_id_by_name = {c["name"].strip().lower(): c["id"] for c in gt.loadCats(gt.getCatIds())}
    to_gt = {i: gt_id_by_name.get(n.strip().lower()) for i, n in model_names.items()}
    print("model class -> ground-truth category:",
          {model_names[i]: (gt.loadCats([g])[0]["name"] if g else "NOT IN GROUND TRUTH") for i, g in to_gt.items()})
    if args.overlays:
        args.overlays.mkdir(parents=True, exist_ok=True)

    results, counts = [], defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for img in gt.loadImgs(gt.getImgIds()):
        image = Image.open(resolve(data, img["file_name"])).convert("RGB")
        det = model.predict(image, threshold=0.05)
        masks = det.mask if det.mask is not None else np.zeros((0, image.height, image.width), bool)
        if masks.shape[0] and masks.shape[1:] != (image.height, image.width):
            masks = np.stack([np.array(Image.fromarray(m).resize(image.size, Image.NEAREST)) for m in masks])
        for m, cid, score in zip(masks, det.class_id, det.confidence):
            g = to_gt.get(int(cid))
            if g is None:
                continue
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            results.append({"image_id": img["id"], "category_id": g, "segmentation": rle, "score": float(score)})
        # precision / recall at the threshold, per class, greedy one-to-one matching at IoU 0.5
        gts = gt.loadAnns(gt.getAnnIds(imgIds=img["id"]))
        gt_masks = [(a["category_id"], gt.annToMask(a).astype(bool)) for a in gts]
        used = set()
        for m, cid, score in sorted(zip(masks, det.class_id, det.confidence), key=lambda t: -t[2]):
            if score < args.threshold or to_gt.get(int(cid)) is None:
                continue
            g = to_gt[int(cid)]
            best, best_iou = None, 0.5
            for k, (gcat, gm) in enumerate(gt_masks):
                if k in used or gcat != g:
                    continue
                iou = mask_iou(m, gm)
                if iou >= best_iou:
                    best, best_iou = k, iou
            if best is None:
                counts[g]["fp"] += 1
            else:
                used.add(best)
                counts[g]["tp"] += 1
        for k, (gcat, _) in enumerate(gt_masks):
            if k not in used:
                counts[gcat]["fn"] += 1
        if args.overlays:
            over = image.copy()
            arr = np.array(over)
            for m, cid, score in zip(masks, det.class_id, det.confidence):
                if score < args.threshold:
                    continue
                col = np.array(COLOURS[int(cid) % len(COLOURS)], dtype=np.uint8)
                arr[m] = (arr[m] * 0.55 + col * 0.45).astype(np.uint8)
            over = Image.fromarray(arr)
            d = ImageDraw.Draw(over)
            for box, cid, score in zip(det.xyxy, det.class_id, det.confidence):
                if score < args.threshold:
                    continue
                d.rectangle(box.tolist(), outline=COLOURS[int(cid) % len(COLOURS)], width=3)
                d.text((box[0] + 4, box[1] + 4), f"{model_names.get(int(cid), cid)} {score:.2f}",
                       fill=COLOURS[int(cid) % len(COLOURS)])
            over.save(args.overlays / Path(img["file_name"]).name, quality=90)

    summary = {"checkpoint": str(args.checkpoint), "data": str(data), "threshold": args.threshold, "classes": {}}
    if results:
        ev = COCOeval(gt, gt.loadRes(results), "segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        summary["mask_AP"] = round(float(ev.stats[0]), 4)
        summary["mask_AP50"] = round(float(ev.stats[1]), 4)
        for cat in gt.loadCats(gt.getCatIds()):
            k = ev.params.catIds.index(cat["id"])
            prec = ev.eval["precision"][:, :, k, 0, -1]          # iou x recall for area=all, maxDets=100
            ap = float(np.mean(prec[prec > -1])) if (prec > -1).any() else 0.0
            p50 = prec[0]
            ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0
            c = counts[cat["id"]]
            summary["classes"][cat["name"]] = {
                "mask_AP": round(ap, 4), "mask_AP50": round(ap50, 4),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                "precision": round(c["tp"] / max(1, c["tp"] + c["fp"]), 4),
                "recall": round(c["tp"] / max(1, c["tp"] + c["fn"]), 4)}
    else:
        print("the model produced no prediction for any ground-truth class")
    print(f"\n{'class':15s} {'mask AP':>8s} {'AP50':>6s} {'prec':>6s} {'recall':>7s}   tp/fp/fn at {args.threshold}")
    for name, s in summary["classes"].items():
        print(f"{name:15s} {s['mask_AP']:8.3f} {s['mask_AP50']:6.3f} {s['precision']:6.3f} {s['recall']:7.3f}   "
              f"{s['tp']}/{s['fp']}/{s['fn']}")
    if "mask_AP" in summary:
        print(f"{'all classes':15s} {summary['mask_AP']:8.3f} {summary['mask_AP50']:6.3f}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
