"""Audit the CASIA v2 train/validation split before fitting any model.

Expected dataset layout: Au/, Tp/, CASIA 2 Groundtruth/, csv files/{train,val}.csv.
The CSV-only preview uses the Python standard library. A full audit also uses
Pillow and produces one JSON split file for data/dataset.py to read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path, PureWindowsPath


SPLITS = ("train", "val")
COLUMNS = ("tp", "gt", "label", "type", "category", "group", "split")
SOURCE_ID = re.compile(r"(ani|arc|art|cha|ind|nat|pla|sec|txt)_?(\d{5})(?!\d)")
FORMAT = "casia-audited-split-v1"


def _relative_path(value):
    """Accept Windows and POSIX separators and return a portable relative path."""
    p = PureWindowsPath(value)
    if p.anchor or ".." in p.parts or not p.parts:
        raise ValueError(f"Expected a path relative to the dataset root, got {value!r}")
    return p.as_posix()


def _read_csv(root, split):
    path = root / "csv files" / f"{split}.csv"
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    missing = set(COLUMNS) - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
    result = []
    for row in reader:
        location = f"{path}:{reader.line_num}"
        r = {k: (row[k] or "").strip() for k in COLUMNS}
        if r["label"] not in {"0", "1"} or r["split"] != split:
            raise ValueError(f"{location}: invalid label or split")
        r["label"] = int(r["label"])
        if not all(r[k] for k in ("tp", "group", "type", "category")):
            raise ValueError(f"{location}: missing image path, group, type, or category")
        if bool(r["gt"]) != bool(r["label"]):
            raise ValueError(f"{location}: manipulated rows require a mask; authentic rows require blank gt")
        try:
            r["tp"] = _relative_path(r["tp"])
            if r["gt"]:
                r["gt"] = _relative_path(r["gt"])
        except ValueError as exc:
            raise ValueError(f"{location}: {exc}") from exc
        result.append(r)
    if not result:
        raise ValueError(f"Empty manifest: {path}")
    return result, hashlib.sha256(raw).hexdigest()


def _source_ids(row):
    return {category + number for category, number in SOURCE_ID.findall(Path(row["tp"]).stem.lower())}


def _counts(rows):
    return {"images": len(rows), "authentic": sum(r["label"] == 0 for r in rows),
            "manipulated": sum(r["label"] == 1 for r in rows),
            "by_type": dict(sorted(Counter(r["type"] for r in rows).items()))}


def _check_direct_overlap(rows):
    for field in ("tp", "gt", "group"):
        keys = {}
        for split in SPLITS:
            items = [r[field].casefold() for r in rows[split] if r[field]]
            counts = Counter(items)
            if field != "group":
                repeated = next((key for key, count in counts.items() if count > 1), None)
                if repeated:
                    raise ValueError(f"{split}: repeated {field} path {repeated!r}")
            keys[split] = set(counts)
        shared = keys["train"] & keys["val"]
        if shared:
            raise ValueError(f"{len(shared)} {field} values occur in both splits, e.g. {min(shared)!r}")


def plan_split(data_root, *, source_policy="strict"):
    """Return rows and CSV audit results. 'protect_val' excludes conflicting train rows.

    Source IDs are inferred from filenames. Direct path/group overlap is always
    an error. Unrecognized filenames are counted for manual review.
    """
    if source_policy not in {"strict", "keep_existing", "protect_val"}:
        raise ValueError("source_policy must be 'strict' or 'protect_val'")
    rows, hashes = {}, {}
    for split in SPLITS:
        rows[split], hashes[split] = _read_csv(Path(data_root), split)
    _check_direct_overlap(rows)
    before = {s: _counts(rows[s]) for s in SPLITS}
    sources = {s: [_source_ids(r) for r in rows[s]] for s in SPLITS}
    unknown = {s: sum(not ids for ids in sources[s]) for s in SPLITS}
    val_ids = set().union(*sources["val"])
    shared = set().union(*sources["train"]) & val_ids
    if shared and source_policy == "strict":
        raise ValueError(f"{len(shared)} inferred source IDs cross the split. Review the CSVs "
                         "or explicitly select --source-policy protect_val.")
    kept, excluded = [], []
    for row, ids in zip(rows["train"], sources["train"]):
        overlap = ids & val_ids
        if overlap and source_policy == "protect_val":
            excluded.append({"tp": row["tp"], "shared_ids": sorted(overlap)})
        else:
            kept.append(row)
    rows["train"] = kept
    for s in SPLITS:
        if {r["label"] for r in rows[s]} != {0, 1}:
            raise ValueError(f"{s} must contain both authentic and manipulated images")
    summary = {"original": before, "retained": {s: _counts(rows[s]) for s in SPLITS},
               "shared_inferred_source_ids": len(shared), "excluded_train_rows": len(excluded),
               "unrecognized_source_filenames": unknown}
    return {"format": FORMAT, "source_policy": source_policy, "csv_sha256": hashes,
            "summary": summary, "excluded": excluded, "splits": rows}


def read_binary_mask(path, foreground="white"):
    """Return a Pillow L image with values 0 (background) or 255 (forgery).

    Accepts 0/1 and 0/255 masks; grayscale masks use a midpoint threshold.
    Coordinates follow stored pixels, without EXIF rotation or resizing.
    """
    from PIL import Image

    if foreground not in {"white", "black"}:
        raise ValueError("foreground must be 'white' or 'black'")
    with Image.open(path) as im:
        gray = im.convert("L")
    threshold = 0 if gray.getextrema()[1] <= 1 else 127
    lut = [255 if ((v > threshold) == (foreground == "white")) else 0 for v in range(256)]
    return gray.point(lut)


def _audit_pixels(root, planned, mask_foreground):
    from PIL import Image

    seen = {}
    repeated = {s: 0 for s in SPLITS}
    collection = hashlib.sha256()
    for split in SPLITS:
        for row in planned["splits"][split]:
            relative = row["tp"]
            try:
                with Image.open(root / relative) as im:
                    rgb = im.convert("RGB")
                digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB|".encode())
                digest.update(rgb.tobytes())
                key = digest.hexdigest()
                mask_digest = ""
                if row["label"]:
                    mask = read_binary_mask(root / row["gt"], mask_foreground)
                    if mask.size != rgb.size:
                        raise ValueError(f"mask size {mask.size} differs from RGB size {rgb.size}")
                    if mask.getbbox() is None:
                        raise ValueError("manipulated image has an empty mask; check mask polarity")
                    mask_digest = hashlib.sha256(mask.tobytes()).hexdigest()
            except (OSError, ValueError) as exc:
                raise ValueError(f"{split}: {relative} (gt={row['gt']!r}): {exc}") from exc
            if key in seen:
                first_split, first_image, first_mask = seen[key]
                if first_split != split or first_mask != mask_digest:
                    raise ValueError(f"Identical RGB pixels with split or annotation conflict: "
                                     f"{first_image} and {relative}")
                repeated[split] += 1
            else:
                seen[key] = (split, relative, mask_digest)
            record = json.dumps([split, relative, key, mask_digest]).encode()
            collection.update(record + b"\n")
    return {"retained_rgb_and_masks_sha256": collection.hexdigest(),
            "same_split_duplicate_rows": repeated}


def _save_lock(path, result):
    """Publish a complete lock once. Identical reruns can reuse it."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("The split output must be a .json file")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, ensure_ascii=False, indent=1) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        temp.write_bytes(payload)
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"{path} contains a different split. Choose a new output name.")
    finally:
        temp.unlink(missing_ok=True)


def audit_dataset(data_root, output, *, source_policy="strict", mask_foreground="white"):
    """Run CSV and pixel checks. Only a successful full audit publishes a split."""
    if mask_foreground not in {"white", "black"}:
        raise ValueError("mask_foreground must be 'white' or 'black'")
    result = plan_split(data_root, source_policy=source_policy)
    result["mask_foreground"] = mask_foreground
    result["pixel_audit"] = _audit_pixels(Path(data_root), result, mask_foreground)
    _save_lock(output, result)
    return result["summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="CASIA root containing 'csv files'")
    parser.add_argument("--source-policy", choices=("strict", "keep_existing", "protect_val"), default="strict")
    parser.add_argument("--mask-foreground", choices=("white", "black"), default="white")
    parser.add_argument("--manifest-only", action="store_true", help="preview CSV counts, write no file")
    parser.add_argument("--output", type=Path, help="writable local JSON path for the full audit")
    args = parser.parse_args(argv)
    if args.manifest_only and args.output:
        parser.error("--output is only used for a full audit")
    try:
        if args.manifest_only:
            summary = plan_split(args.root, source_policy=args.source_policy)["summary"]
        else:
            output = args.output or Path("artifacts/casia_split.json")
            summary = audit_dataset(args.root, output, source_policy=args.source_policy,
                                    mask_foreground=args.mask_foreground)
            print(f"Audited split: {output.resolve()}")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()