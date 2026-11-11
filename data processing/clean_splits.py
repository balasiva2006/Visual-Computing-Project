import csv
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "dataset"
OUTPUT = PROJECT / "dataset_clean"
REPORT = PROJECT / "artifacts" / "mask_review" / "issues.csv"


def normalize(path):
    return path.strip().replace("\\", "/")


def main():
    # Identify the exact image–mask pairs flagged by check.py.
    flagged = set()
    with REPORT.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["issue"] in {"reversed_dimensions", "other_size_mismatch"}:
                flagged.add((
                    row["split"],
                    normalize(row["image"]),
                    normalize(row["mask"]),
                ))

    if len(flagged) != 17:
        raise ValueError(f"Expected 17 flagged pairs from your scan, found {len(flagged)}.")

    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} already exists. Check it before running again.")

    matched = set()
    counts = {}

    # Prepare both CSVs together before making dataset_clean available.
    with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
        staged = Path(temp) / "dataset_clean"
        csv_out = staged / "csv files"
        csv_out.mkdir(parents=True)

        for split in ("train", "val"):
            csv_in = SOURCE / "csv files" / f"{split}.csv"
            kept = removed = 0

            with csv_in.open(newline="", encoding="utf-8-sig") as src, \
                 (csv_out / f"{split}.csv").open("w", newline="", encoding="utf-8") as dst:
                reader = csv.DictReader(src)
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row in reader:
                    key = (split, normalize(row["tp"]), normalize(row["gt"]))

                    if key in flagged:
                        if row["label"].strip() != "1" or key in matched:
                            raise ValueError(f"Unexpected label or duplicate row: {key}")
                        matched.add(key)
                        removed += 1
                    else:
                        writer.writerow(row)
                        kept += 1

            counts[split] = (kept, removed)

        if matched != flagged:
            raise ValueError(f"Report entries missing from the source CSVs: {flagged - matched}")

        for name in ("Au", "Tp", "CASIA 2 Groundtruth"):
            folder = SOURCE / name
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            (staged / name).symlink_to(folder, target_is_directory=True)

        staged.rename(OUTPUT)

    for split, (kept, removed) in counts.items():
        print(f"{split}: kept {kept}, excluded {removed}")
    print(f"Clean dataset: {OUTPUT}")


if __name__ == "__main__":
    main()
