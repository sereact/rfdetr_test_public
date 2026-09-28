#!/usr/bin/env python3
"""Fine-tune an RF-DETR segmentation model on a COCO-labelled folder.

    uv run python train.py --data datasets/default_dataset --out runs/try1 --epochs 40 --max-minutes 55

--data is a folder with `_annotations.coco.json` and the images it names
(file_name relative to the folder, or inside an `images/` subfolder). The
categories of that file are the classes the model learns, so the annotations
must carry their final category_id before training. Annotations need
`segmentation` polygons; `bbox` and `area` are filled in when missing.

The images are split into train / valid by --val-fraction (or --val-list, a
text file with one image file name per line to hold out). Nothing is copied:
the staged COCO files under <out>/dataset/ point at the original images.

Output (<out>/):
    checkpoint0001.pth, checkpoint0003.pth, ...            every --checkpoint-every epochs (weights only)
    checkpoint_best_regular.pth, checkpoint_best_ema.pth   best epoch so far (written during the run)
    checkpoint_best_total.pth                              the better of the two, written when the
                                                           run ends (epochs done, early stop, or
                                                           --max-minutes reached)
    log.txt, results_mask.json, metrics_plot.png, tensorboard events
Every checkpoint loads in evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

BACKGROUND_NAMES = {"background", "bg", "__background__"}


def load_coco(data: Path) -> tuple[dict, dict[int, Path]]:
    coco = json.loads((data / "_annotations.coco.json").read_text())
    paths = {}
    for im in coco["images"]:
        p = Path(im["file_name"])
        cand = [p, data / p, data / "images" / p.name, data / p.name]
        found = next((c for c in cand if c.is_file()), None)
        if found is None:
            raise SystemExit(f"image not found: {im['file_name']} (looked in {data} and {data / 'images'})")
        paths[im["id"]] = found.resolve()
    return coco, paths


def clean_annotations(coco: dict) -> tuple[list[dict], list[dict]]:
    categories = [c for c in coco["categories"] if c["name"].strip().lower() not in BACKGROUND_NAMES]
    if not categories:
        raise SystemExit("no categories besides background in _annotations.coco.json")
    keep_ids = {c["id"] for c in categories}
    anns, dropped = [], Counter()
    for a in coco["annotations"]:
        if a.get("category_id") not in keep_ids:
            dropped["category not in the category list (or background)"] += 1
            continue
        seg = a.get("segmentation")
        polys = [p for p in seg if len(p) >= 6] if isinstance(seg, list) else []
        if not polys and not isinstance(seg, dict):
            dropped["no polygon"] += 1
            continue
        a = dict(a)
        if polys:
            a["segmentation"] = polys
            xs = [v for p in polys for v in p[0::2]]
            ys = [v for p in polys for v in p[1::2]]
            if not a.get("bbox") or a["bbox"][2] <= 0 or a["bbox"][3] <= 0:
                a["bbox"] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            if not a.get("area"):
                a["area"] = sum(abs(sum(p[i] * p[(i + 3) % len(p)] - p[(i + 2) % len(p)] * p[i + 1]
                                        for i in range(0, len(p), 2))) / 2 for p in polys)
        a.setdefault("iscrowd", 0)
        anns.append(a)
    for reason, n in dropped.items():
        print(f"dropped {n} annotations: {reason}")
    used = {a["category_id"] for a in anns}
    for c in categories:
        if c["id"] not in used:
            print(f"category {c['name']!r} has no annotations and is left out of the classes")
    categories = [c for c in categories if c["id"] in used]
    if not categories:
        raise SystemExit("no annotations left")
    return categories, anns


def stage(coco: dict, paths: dict[int, Path], categories: list[dict], anns: list[dict],
          val_ids: set[int], out: Path) -> None:
    by_img = {}
    for a in anns:
        by_img.setdefault(a["image_id"], []).append(a)
    for split, ids in (("train", [i for i in paths if i not in val_ids]), ("valid", sorted(val_ids))):
        d = out / "dataset" / split
        d.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for new_id, img_id in enumerate(ids, 1):
            im = next(i for i in coco["images"] if i["id"] == img_id)
            images.append({"id": new_id, "file_name": str(paths[img_id]),
                           "width": im["width"], "height": im["height"]})
            for a in by_img.get(img_id, []):
                annotations.append({**a, "id": len(annotations) + 1, "image_id": new_id})
        # A category at id 0 named "background" is required: rfdetr 1.5.2 maps category ids to
        # 0-based labels and never learns label 0 (checked 2026-09-28: the first class stayed at
        # mask AP 0.000 while the others trained normally). Roboflow exports carry it too.
        staged_categories = [{"id": 0, "name": "background", "supercategory": "none"}] + [
            {**c, "id": i + 1} for i, c in enumerate(categories)]
        remap = {c["id"]: i + 1 for i, c in enumerate(categories)}
        annotations = [{**a, "category_id": remap[a["category_id"]]} for a in annotations]
        (d / "_annotations.coco.json").write_text(json.dumps(
            {"images": images, "annotations": annotations, "categories": staged_categories}))
        counts = Counter(staged_categories[a["category_id"]]["name"] for a in annotations)
        print(f"{split}: {len(images)} images, {len(annotations)} annotations {dict(counts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with _annotations.coco.json + images")
    ap.add_argument("--out", type=Path, required=True, help="run folder (checkpoints, logs)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop after this much training time (checked at the end of each epoch)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-list", type=Path, default=None, help="file with one image file name per line to hold out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-ema", action="store_true", help="skip the EMA weights (halves validation time)")
    ap.add_argument("--patience", type=int, default=15, help="early stop after N epochs without improvement")
    ap.add_argument("--checkpoint-every", type=int, default=2, help="save a weights-only checkpoint every N epochs")
    args = ap.parse_args()

    data, out = args.data.resolve(), args.out.resolve()
    coco, paths = load_coco(data)
    categories, anns = clean_annotations(coco)
    if args.val_list:
        names = {line.strip() for line in args.val_list.read_text().splitlines() if line.strip()}
        val_ids = {i for i, p in paths.items() if p.name in names}
    else:
        ids = sorted(paths)
        random.Random(args.seed).shuffle(ids)
        val_ids = set(ids[:max(1, round(len(ids) * args.val_fraction))])
    if not val_ids or len(val_ids) == len(paths):
        raise SystemExit("the validation split must hold at least one image and not all of them")
    print("classes:", [c["name"] for c in categories])
    stage(coco, paths, categories, anns, val_ids, out)

    os.chdir(Path(__file__).resolve().parent)   # rfdetr downloads rf-detr-seg-xxlarge.pt here on the first run
    from rfdetr import RFDETRSeg2XLarge
    model = RFDETRSeg2XLarge()

    t0 = time.time()

    def strip_periodic_checkpoints(_log_stats):
        # the periodic checkpointNNNN.pth files carry the optimizer and EMA state (about 600 MB);
        # keep only the weights so a run with many of them stays small
        from rfdetr.util.misc import strip_checkpoint
        for ck in sorted(out.glob("checkpoint[0-9][0-9][0-9][0-9].pth")):
            if ck.stat().st_size > 250_000_000:
                strip_checkpoint(ck)
    model.callbacks["on_fit_epoch_end"].append(strip_periodic_checkpoints)
    if args.max_minutes:
        def stop_when_out_of_time(_log_stats):
            if time.time() - t0 > args.max_minutes * 60:
                print(f"--max-minutes {args.max_minutes} reached, stopping after this epoch", flush=True)
                model.model.request_early_stop()
        model.callbacks["on_fit_epoch_end"].append(stop_when_out_of_time)

    model.train(
        dataset_dir=str(out / "dataset"), output_dir=str(out),
        epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum,
        lr=args.lr, lr_encoder=args.lr_encoder, num_workers=args.workers,
        multi_scale=True, expanded_scales=False, square_resize_div_64=True,
        use_ema=not args.no_ema, early_stopping=True, early_stopping_patience=args.patience,
        early_stopping_min_delta=0.0005, tensorboard=True, run_test=False,
        checkpoint_interval=args.checkpoint_every,
    )
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    results = out / "results_mask.json"          # results.json holds the box numbers
    if results.exists():
        r = json.loads(results.read_text())
        for row in (r.get("class_map") or {}).get("valid", []):
            print(f"  valid {row['class']:14s} mask mAP@50:95 {row['map@50:95']:.3f}  mAP@50 {row['map@50']:.3f}")


if __name__ == "__main__":
    main()
