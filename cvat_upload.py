#!/usr/bin/env python3
"""Create a CVAT task from images + segmentation polygons, and fetch the labels back.

    uv run python cvat_upload.py upload --data datasets/default_dataset \
        [--labels <name> ...] [--segment-size 100] [--quality 95]
    uv run python cvat_upload.py download --task 1234

upload:   reads <data>/_annotations.coco.json, uploads its images (in file-name
          order) and every polygon as a prelabel under its category name. The
          extra labels (EXTRA_LABELS below, or --labels) are created as empty
          labels so you can switch each polygon to them in CVAT (select the
          shape, change its label). The task is titled with the dataset
          folder name and the upload time. Prints the task id and the job links.
download: exports the task as COCO 1.0 with its images into a new dataset
          folder datasets/task_<id>/ (images/ + _annotations.coco.json), ready
          for train.py --data. An existing folder is never overwritten; a
          second download of the same task becomes task_<id>_2.

Credentials: CVAT_URL, CVAT_USERNAME, CVAT_PASSWORD from the environment, or
from a `.env` file next to this script (KEY=VALUE lines, never commit it).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA_LABELS = ["extra_class_1", "extra_class_2"]   # created in the task next to the dataset's categories
COLOURS = ["#ff3c3c", "#00beff", "#ffdc00", "#50e650", "#e664ff", "#ff9628", "#8c8c8c"]


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    missing = [k for k in ("CVAT_URL", "CVAT_USERNAME", "CVAT_PASSWORD") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} (set them in the environment or in {env})")


def client():
    from cvat_sdk import make_client
    return make_client(host=os.environ["CVAT_URL"],
                       credentials=(os.environ["CVAT_USERNAME"], os.environ["CVAT_PASSWORD"]))


def upload(args) -> None:
    from cvat_sdk.api_client import models
    from cvat_sdk.core.proxies.tasks import ResourceType

    data = args.data.resolve()
    coco = json.loads((data / "_annotations.coco.json").read_text())
    images = sorted(coco["images"], key=lambda im: im["file_name"])
    paths = []
    for im in images:
        p = Path(im["file_name"])
        found = next((c for c in (p, data / p, data / "images" / p.name, data / p.name) if c.is_file()), None)
        if found is None:
            raise SystemExit(f"image not found: {im['file_name']}")
        paths.append(found)
    if len({p.name for p in paths}) != len(paths):
        raise SystemExit("image file names must be unique inside one task")
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    labels = list(dict.fromkeys([c["name"] for c in coco["categories"]] + list(args.labels or [])))

    with client() as cl:
        task = cl.tasks.create_from_data(
            spec=models.TaskWriteRequest(
                name=f"{data.name} {time.strftime('%Y-%m-%d %H:%M')}", segment_size=args.segment_size,
                labels=[models.PatchedLabelRequest(name=n, color=COLOURS[i % len(COLOURS)], type="any")
                        for i, n in enumerate(labels)]),
            resource_type=ResourceType.LOCAL, resources=[str(p) for p in paths],
            data_params={"image_quality": args.quality, "sorting_method": "predefined"})
        frame_of = {Path(f.name).name: i for i, f in enumerate(task.get_frames_info())}
        label_id = {lb.name: lb.id for lb in task.get_labels()}
        shapes = []
        for im in images:
            frame = frame_of[Path(im["file_name"]).name]
            for a in coco["annotations"]:
                if a["image_id"] != im["id"] or not isinstance(a.get("segmentation"), list):
                    continue
                for poly in a["segmentation"]:
                    if len(poly) < 6:
                        continue
                    shapes.append(models.LabeledShapeRequest(
                        type="polygon", frame=frame, label_id=label_id[cat_name[a["category_id"]]],
                        points=[float(v) for v in poly], group=int(a["id"]),
                        occluded=False, z_order=0, source="manual", attributes=[]))
        if shapes:
            cl.api_client.tasks_api.partial_update_annotations(
                action="create", id=task.id,
                patched_labeled_data_request=models.PatchedLabeledDataRequest(shapes=shapes, tags=[], tracks=[]))
        jobs = sorted(task.get_jobs(), key=lambda j: j.start_frame)
        url = os.environ["CVAT_URL"].rstrip("/")
        print(f"task {task.id}: {len(paths)} images, {len(shapes)} prelabel polygons, labels {labels}")
        print(f"{url}/tasks/{task.id}")
        print(f"{len(jobs)} job(s):")
        for job in jobs:
            print(f"  images {job.start_frame + 1}-{job.stop_frame + 1}: {url}/tasks/{task.id}/jobs/{job.id}")


def download(args) -> None:
    out, n = HERE / "datasets" / f"task_{args.task}", 2
    while out.exists():
        out, n = HERE / "datasets" / f"task_{args.task}_{n}", n + 1
    buf = HERE / f".cvat_task_{args.task}.zip"
    with client() as cl:
        cl.tasks.retrieve(args.task).export_dataset("COCO 1.0", str(buf), include_images=True)
    (out / "images").mkdir(parents=True)
    with zipfile.ZipFile(buf) as z:
        coco = json.loads(z.read(next(n for n in z.namelist() if n.endswith(".json"))))
        for name in z.namelist():
            if name.startswith("images/") and not name.endswith("/"):
                (out / "images" / Path(name).name).write_bytes(z.read(name))
    buf.unlink()
    for im in coco["images"]:
        im["file_name"] = Path(im["file_name"]).name
    (out / "_annotations.coco.json").write_text(json.dumps(coco, indent=1))
    counts = {c["name"]: sum(a["category_id"] == c["id"] for a in coco["annotations"]) for c in coco["categories"]}
    empty = sum(not a.get("segmentation") for a in coco["annotations"])
    print(f"task {args.task}: {len(coco['images'])} images, {len(coco['annotations'])} annotations {counts}"
          + (f", {empty} without a mask" if empty else "") + f" -> {out.relative_to(HERE)}")
    print(f"train with: uv run python train.py --data {out.relative_to(HERE)} --out runs/<name>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--data", type=Path, required=True, help="folder with _annotations.coco.json + images")
    up.add_argument("--labels", nargs="*", default=EXTRA_LABELS, help="extra label names to create in the task")
    up.add_argument("--segment-size", type=int, default=100, help="images per job")
    up.add_argument("--quality", type=int, default=95, help="jpeg quality of the images shown in CVAT")
    dl = sub.add_parser("download")
    dl.add_argument("--task", type=int, required=True)
    args = ap.parse_args()
    load_env()
    upload(args) if args.cmd == "upload" else download(args)


if __name__ == "__main__":
    main()
