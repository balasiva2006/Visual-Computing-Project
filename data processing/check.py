import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "dataset"
CSV_DIR = DATASET / "csv files"
OUTPUT = PROJECT / "artifacts" / "mask_review"
T = getattr(Image, "Transpose", Image)


def save_preview(rgb, mask, output_path):
    """Show the original image and four possible mask orientations."""
    orientations = [
        ("A: Clockwise", T.ROTATE_270),
        ("B: Counterclockwise", T.ROTATE_90),
        ("C: Main diagonal", T.TRANSPOSE),
        ("D: Other diagonal", T.TRANSVERSE),
    ]

    red = Image.new("RGB", rgb.size, "red")
    tinted = Image.blend(rgb, red, 0.45)
    panels = [("Original", rgb)]

    for title, operation in orientations:
        overlay = Image.composite(tinted, rgb, mask.transpose(operation))
        panels.append((title, overlay))

    width, height = rgb.size
    panel_height = height + 24
    sheet = Image.new("RGB", (3 * width, 2 * panel_height), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (title, panel) in enumerate(panels):
        x = (i % 3) * width
        y = (i // 3) * panel_height
        draw.text((x + 5, y + 4), title, fill="black")
        sheet.paste(panel, (x, y + 24))

    sheet.save(output_path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "issues.csv"
    counts = Counter()

    with report_path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.writer(report)
        writer.writerow(["split", "csv_line", "image", "mask", "issue", "details", "preview"])

        for split in ("train", "val"):
            csv_path = CSV_DIR / f"{split}.csv"
            print(f"Scanning {csv_path.name} ...", flush=True)

            with csv_path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)

                for row in reader:
                    # Authentic images have no ground-truth forgery mask.
                    if row["label"].strip() == "0":
                        continue

                    counts["manipulated rows checked"] += 1
                    image_rel = row["tp"].strip().replace("\\", "/")
                    mask_rel = row["gt"].strip().replace("\\", "/")
                    prefix = [split, reader.line_num, image_rel, mask_rel]

                    try:
                        with Image.open(DATASET / image_rel) as image:
                            rgb = image.convert("RGB")
                        with Image.open(DATASET / mask_rel) as image:
                            gray = image.convert("L")
                    except OSError as error:
                        writer.writerow(prefix + ["read_error", str(error), ""])
                        counts["read errors"] += 1
                        continue

                    # The project uses white foreground, encoded as 1 or 255.
                    threshold = 0 if gray.getextrema()[1] <= 1 else 127
                    mask = gray.point(lambda v: 255 if v > threshold else 0)
                    empty = mask.getbbox() is None

                    if empty:
                        writer.writerow(prefix + ["empty_mask", "No foreground pixels", ""])
                        counts["empty masks"] += 1

                    if rgb.size == mask.size:
                        if not empty:
                            counts["matching dimensions and nonempty mask"] += 1
                        continue

                    details = f"image={rgb.size}, mask={mask.size}"
                    preview_name = ""

                    if mask.size == rgb.size[::-1]:
                        issue = "reversed_dimensions"
                        if not empty:
                            preview_name = f"{split}_{reader.line_num}_{Path(image_rel).stem}.png"
                            save_preview(rgb, mask, OUTPUT / preview_name)
                    else:
                        issue = "other_size_mismatch"

                    writer.writerow(prefix + [issue, details, preview_name])
                    counts[issue] += 1

    print("\nScan complete.")
    for name, count in counts.items():
        print(f"{name}: {count}")
    print(f"\nIssue report: {report_path}")


if __name__ == "__main__":
    main()
