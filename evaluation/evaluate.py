"""Evaluate a trained model on CASIA validation or an external manifest.

External CSV columns: image,mask,label. Paths are relative to --root.
An authentic row has label 0 and may leave mask blank. A forged row has
label 1 and requires a nonempty mask. Optional id must be unique if supplied.
External masks use --mask-foreground; CASIA uses its audited white polarity.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image

from data.audit import read_binary_mask
from data.dataset import CASIADataset
from data.frequency import ModelInputs, RGB_MEAN, RGB_STD
from .degradations import CONDITIONS, condition_names, evaluation_transform
from .metrics import EvaluationMetrics


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def training_assets(stats_path, audit_path):
    """Read existing training assets and reconstruct the trainer's data ID."""
    stats = json.loads(Path(stats_path).read_text())
    audit = json.loads(Path(audit_path).read_text())

    if (
        audit["source_policy"] != "keep_existing"
        or audit["mask_foreground"] != "white"
    ):
        raise ValueError(
            "Expected the keep_existing, white-mask training audit"
        )

    expected = [
        "zigzag64-signedlog-v1",
        audit["csv_sha256"]["train"],
    ]
    if stats["identity"] != expected:
        raise ValueError("DCT statistics and training audit do not match")

    arrays = [
        np.asarray(stats[key], np.float32)
        for key in ("mean", "std")
    ]
    if any(
        array.shape != (64,) or not np.isfinite(array).all()
        for array in arrays
    ):
        raise ValueError(
            "Expected 64 finite means and standard deviations"
        )
    if (arrays[1] <= 0).any():
        raise ValueError("DCT standard deviations must be positive")

    identity = {
        "csv": audit["csv_sha256"],
        "stats": digest(stats_path),
        "pixels": audit["pixel_audit"]["retained_rgb_and_masks_sha256"],
    }
    data_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()

    return tuple(array[:, None, None] for array in arrays), data_id, audit


class ManifestDataset:
    """Explicit external image/mask pairs with no filename-based guessing."""

    def __init__(self, root, manifest, mask_foreground="white"):
        self.root = Path(root)
        self.mask_foreground = mask_foreground

        if mask_foreground not in ("white", "black"):
            raise ValueError("mask_foreground must be white or black")

        with Path(manifest).open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            reader = csv.DictReader(stream)
            if not {"image", "mask", "label"}.issubset(
                reader.fieldnames or []
            ):
                raise ValueError(
                    "External CSV requires image,mask,label columns"
                )
            rows = list(reader)

        if not rows:
            raise ValueError("External manifest is empty")

        self.rows = []
        ids, paths = set(), set()

        for number, row in enumerate(rows, start=2):
            image = (
                (row.get("image") or "").strip().replace("\\", "/")
            )
            mask = (
                (row.get("mask") or "").strip().replace("\\", "/")
            )
            label = (row.get("label") or "").strip()
            sample_id = (row.get("id") or image).strip()

            if not image or not sample_id or label not in ("0", "1"):
                raise ValueError(
                    f"Invalid image, id, or label on CSV line {number}"
                )
            if label == "1" and not mask:
                raise ValueError(
                    f"Missing forged mask on CSV line {number}"
                )

            image_path = (self.root / image).resolve()
            if sample_id in ids or image_path in paths:
                raise ValueError(
                    f"Duplicate image or id on CSV line {number}"
                )

            for name in (image, mask):
                if name and not (self.root / name).is_file():
                    raise FileNotFoundError(self.root / name)

            ids.add(sample_id)
            paths.add(image_path)
            self.rows.append(
                dict(
                    image=image,
                    mask=mask,
                    label=int(label),
                    id=sample_id,
                )
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(self.root / row["image"]) as image:
            rgb = np.array(image.convert("RGB"))

        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        if row["mask"]:
            image = read_binary_mask(
                self.root / row["mask"], self.mask_foreground
            )
            if image.size != (width, height):
                raise ValueError(
                    f"Image/mask dimensions differ: {row['id']}"
                )
            mask = np.asarray(image) > 0

        if bool(mask.any()) != bool(row["label"]):
            raise ValueError(
                f"Mask and image label disagree: {row['id']}"
            )

        return {
            "rgb": np.ascontiguousarray(rgb.transpose(2, 0, 1)),
            "mask": mask[None].astype(np.float32),
            "valid": np.ones((1, height, width), dtype=bool),
            "path": row["id"],
            "label": np.float32(row["label"]),
        }


def model_sample(sample, mode, condition, stats, seed):
    view = partial(
        evaluation_transform, condition=condition, seed=seed
    )

    if mode == "spatial_only":
        result = view(sample)
        rgb = result["rgb"].astype(np.float32) / 255
        result["rgb"] = (
            (rgb - RGB_MEAN) / RGB_STD
        ) * result["valid"]
        return result

    return ModelInputs(view, stats)(sample)


def predict(model, sample, device):
    import torch
    from models.model import INPUTS

    batch = {
        key: torch.from_numpy(
            np.ascontiguousarray(sample[key])
        ).unsqueeze(0).to(device)
        for key in INPUTS[model.mode]
    }

    with torch.inference_mode():
        output = model(batch)

        if output["mask_logits"].shape != batch["valid"].shape:
            raise ValueError(
                "Predicted mask shape differs from valid-pixel map"
            )
        if output["image_logits"].shape != (1,):
            raise ValueError("Expected one image logit")
        if any(
            not output[key].isfinite().all().item()
            for key in ("mask_logits", "image_logits")
        ):
            raise FloatingPointError(
                f"Nonfinite model output: {sample['path']}"
            )

        mask = (
            output["mask_logits"]
            .float()
            .sigmoid()[0, 0]
            .cpu()
            .numpy()
        )
        score = output["image_logits"].float().sigmoid().item()

    return mask, score


def evaluate_condition(
    dataset,
    predictor,
    mode,
    stats,
    condition,
    *,
    seed=42,
    pixel_threshold=0.5,
    image_threshold=0.5,
    log_every=100,
):
    """Process one native-resolution image at a time, retaining scalar metrics."""
    meter = EvaluationMetrics(pixel_threshold, image_threshold)

    for index in range(len(dataset)):
        sample = model_sample(
            dataset[index], mode, condition, stats, seed
        )
        mask_probability, image_probability = predictor(sample)

        meter.update(
            sample["path"],
            sample["label"],
            image_probability,
            sample["mask"][0],
            mask_probability,
            sample["valid"][0],
        )

        if log_every and (
            (index + 1) % log_every == 0
            or index + 1 == len(dataset)
        ):
            print(
                f"{condition}: {index + 1}/{len(dataset)} images",
                flush=True,
            )

    return meter


def atomic_text(path, content):
    path = Path(path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)

    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty results table")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def save_results(output, condition, meter, report):
    summary = meter.compute()
    save_csv(output / f"{condition}.csv", meter.rows)

    report["results"][condition] = summary
    report["complete"] = (
        len(report["results"]) == len(report["conditions"])
    )

    curves = []
    for name, result in report["results"].items():
        kind, strength = CONDITIONS[name]
        row = dict(condition=name, kind=kind, strength=strength)

        for group in ("detection", "localization"):
            row.update(
                {
                    f"{group}_{key}": value
                    for key, value in result[group].items()
                }
            )
        curves.append(row)

    save_csv(output / "curves.csv", curves)
    atomic_text(
        output / "summary.json",
        json.dumps(report, indent=2, allow_nan=False),
    )


def smoke(device):
    """Synthetic frequency-model forward pass and checkpoint round trip."""
    import torch
    from models.model import ForgeryModel
    from training.checkpoints import restore_checkpoint, save_checkpoint

    torch.manual_seed(42)
    stats = (
        np.zeros((64, 1, 1), np.float32),
        np.ones((64, 1, 1), np.float32),
    )

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        rng = np.random.default_rng(42)

        for label in (0, 1):
            rgb = rng.integers(
                0, 256, (45, 67, 3), dtype=np.uint8
            )
            Image.fromarray(rgb).save(root / f"{label}.png")

        mask = np.zeros((45, 67), np.uint8)
        mask[10:25, 15:35] = 255
        Image.fromarray(mask).save(root / "mask.png")

        manifest = root / "samples.csv"
        manifest.write_text(
            "image,mask,label\n"
            "0.png,,0\n"
            "1.png,mask.png,1\n"
        )
        dataset = ManifestDataset(root, manifest)

        model = ForgeryModel(
            "frequency_only", pretrained_spatial=False
        ).to(device).eval()

        checkpoint = root / "model.pt"
        save_checkpoint(
            checkpoint,
            model,
            stage="frequency",
            epoch=0,
            data_id="smoke",
            config={"smoke": True},
            progress={},
        )
        restore_checkpoint(
            checkpoint, model, data_id="smoke", stage="frequency"
        )

        conditions = condition_names("validation")
        report = dict(
            conditions=list(conditions),
            results={},
            complete=False,
            smoke=True,
        )

        for name in conditions:
            meter = evaluate_condition(
                dataset,
                partial(predict, model, device=device),
                model.mode,
                stats,
                name,
                log_every=0,
            )
            assert meter.compute()["detection"]["count"] == 2
            assert (
                meter.compute()["localization"]["evaluated_images"]
                == 1
            )
            save_results(root, name, meter, report)

        saved = json.loads(
            (root / "summary.json").read_text()
        )
        assert saved["complete"] and len(saved["results"]) == 5

        with (root / "curves.csv").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 5

    print(
        "Evaluation smoke check passed: checkpoint, model inference, "
        "five views, CSV/JSON"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./dataset_clean")
    parser.add_argument(
        "--manifest",
        help="External dataset CSV; omitted means CASIA val",
    )
    parser.add_argument("--dataset-name", default="external")
    parser.add_argument(
        "--mask-foreground",
        choices=("white", "black"),
        default="white",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--mode",
        choices=("spatial_only", "frequency_only", "fusion"),
        default="fusion",
    )
    parser.add_argument(
        "--stats", default="./artifacts/dct_stats.json"
    )
    parser.add_argument(
        "--audit", default="./artifacts/casia_clean_split.json"
    )
    parser.add_argument(
        "--protocol",
        choices=("clean", "validation", "sweep"),
        default="clean",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS)
    )
    parser.add_argument("--pixel-threshold", type=float, default=0.5)
    parser.add_argument("--image-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable; request a GPU allocation or use --device cpu"
        )

    if args.smoke:
        smoke(device)
        return

    if not args.checkpoint or not args.out:
        parser.error("Supply --checkpoint and --out, or run --smoke")
    if not args.manifest and args.mask_foreground != "white":
        parser.error(
            "CASIA validation uses the audited white-mask polarity"
        )

    EvaluationMetrics(args.pixel_threshold, args.image_threshold)

    names = args.conditions
    if names is None:
        names = (
            ["clean"]
            if args.protocol == "clean"
            else list(condition_names(args.protocol))
        )
    if len(set(names)) != len(names):
        parser.error("Conditions must be unique")

    stats, data_id, audit = training_assets(args.stats, args.audit)

    if args.manifest:
        dataset = ManifestDataset(
            args.root, args.manifest, args.mask_foreground
        )
        dataset_info = dict(
            name=args.dataset_name,
            manifest=str(Path(args.manifest).resolve()),
            manifest_sha256=digest(args.manifest),
            mask_foreground=args.mask_foreground,
        )
    else:
        hashes = {
            split: digest(
                Path(args.root) / "csv files" / f"{split}.csv"
            )
            for split in ("train", "val")
        }
        if hashes != audit["csv_sha256"]:
            raise ValueError(
                "Active CASIA CSVs differ from the training audit"
            )

        dataset = CASIADataset(args.root, "val")
        dataset_info = dict(
            name="casia_val",
            csv_sha256=hashes,
            mask_foreground="white",
        )

    from models.model import ForgeryModel
    from training.checkpoints import _read

    checkpoint_hash = digest(args.checkpoint)
    state = _read(args.checkpoint, args.mode, data_id)

    if state["config"].get("smoke", False):
        raise ValueError(
            "Select a full-training checkpoint for reported evaluation"
        )

    model = ForgeryModel(args.mode, pretrained_spatial=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    if digest(args.checkpoint) != checkpoint_hash:
        raise RuntimeError(
            "Checkpoint changed while loading; "
            "use a stable completed checkpoint"
        )

    checkpoint_info = dict(
        path=str(Path(args.checkpoint).resolve()),
        sha256=checkpoint_hash,
        mode=args.mode,
        stage=state["stage"],
        epoch=state["epoch"],
        data_id=data_id,
    )
    del state

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "complete": False,
        "dataset": {
            **dataset_info,
            "root": str(Path(args.root).resolve()),
            "images": len(dataset),
        },
        "checkpoint": checkpoint_info,
        "stats_sha256": digest(args.stats),
        "audit_sha256": digest(args.audit),
        "seed": args.seed,
        "device": str(device),
        "precision": "float32",
        "pixel_threshold": args.pixel_threshold,
        "image_threshold": args.image_threshold,
        "conditions": names,
        "versions": dict(
            torch=str(torch.__version__),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pillow=PIL.__version__,
        ),
        "results": {},
    }

    atomic_text(
        output / "summary.json",
        json.dumps(report, indent=2, allow_nan=False),
    )

    predictor = partial(predict, model, device=device)

    for name in names:
        meter = evaluate_condition(
            dataset,
            predictor,
            args.mode,
            stats,
            name,
            seed=args.seed,
            pixel_threshold=args.pixel_threshold,
            image_threshold=args.image_threshold,
        )
        save_results(output, name, meter, report)
        print(
            json.dumps(
                {"condition": name, **meter.compute()},
                allow_nan=False,
            ),
            flush=True,
        )

    print(f"Evaluation complete: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
