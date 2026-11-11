import csv
import hashlib
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from data.audit import read_binary_mask


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "dataset_clean"
CSV_DIR = ROOT / "csv files"
REPORT = PROJECT / "artifacts" / "duplicate_review.csv"


def data_path(value):
    return ROOT / value.strip().replace("\\", "/")


def image_hash(path):
    # Compare decoded pixels so different filenames can still match.
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB|".encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def annotation(row):
    label = int(row["label"])
    if label == 0:
        return (0, "")
    mask = read_binary_mask(data_path(row["gt"]), foreground="white")
    return (1, hashlib.sha256(mask.tobytes()).hexdigest())


def main():
    rows = {}
    groups = defaultdict(list)

    for split in ("train", "val"):
        print(f"Scanning {split}.csv ...", flush=True)
        with (CSV_DIR / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if split == "train":
                train_columns = reader.fieldnames
            rows[split] = list(reader)

        for index, row in enumerate(rows[split]):
            digest = image_hash(data_path(row["tp"]))
            groups[digest].append((split, index, row))

    remove = set()
    findings = []
    conflicts = 0

    for digest, group in groups.items():
        if len(group) == 1:
            continue

        splits = {split for split, _, _ in group}
        conflict = len({annotation(row) for _, _, row in group}) > 1

        if conflict:
            conflicts += 1
        elif len(splits) == 1:
            # Matching copies within one split are allowed by our audit.
            continue

        for split, index, row in group:
            if conflict:
                action = "review_annotation"
            elif split == "train":
                action = "exclude_from_train"
                remove.add(index)
            else:
                action = "keep_in_validation"

            findings.append((digest, split, row["tp"], row["label"], action))

    if findings:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pixel_hash", "split", "image", "label", "action"])
            writer.writerows(findings)
        print(f"Duplicate report: {REPORT}")

    if conflicts:
        raise SystemExit(
            f"Found {conflicts} duplicate group(s) with conflicting annotations. "
            "Training CSV unchanged. Review the report."
        )

    if not remove:
        print("No cross-split duplicates to remove.")
        return

    train_file = CSV_DIR / "train.csv"
    backup = CSV_DIR / "train.before_dedup.csv"
    if backup.exists() and backup.read_bytes() != train_file.read_bytes():
        raise SystemExit(f"An earlier backup already exists at {backup}. Check it before proceeding.")
    if not backup.exists():
        shutil.copy2(train_file, backup)

    # Finish writing the new CSV before replacing the active training list.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=CSV_DIR, delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
        writer = csv.DictWriter(tmp, fieldnames=train_columns)
        writer.writeheader()
        writer.writerows(row for i, row in enumerate(rows["train"]) if i not in remove)

    os.replace(temp_path, train_file)
    print(f"Excluded {len(remove)} training rows.")
    print(f"Final train: {len(rows['train']) - len(remove)}")
    print(f"Final val:   {len(rows['val'])}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
