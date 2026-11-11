"""Load CASIA samples from the existing train and validation CSVs."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import plan_split, read_binary_mask


class CASIADataset:
    """Map-style dataset compatible with torch.utils.data.DataLoader.

    Before transforms: rgb is uint8 [3,H,W], mask is float32 [1,H,W],
    valid is bool [1,H,W], and label is a scalar float32.
    A transform receives and returns a sample dict, keeping mask and valid aligned.
    """

    def __init__(self, root, split="train", transform=None, mask_foreground="white"):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.rows = plan_split(self.root, source_policy="keep_existing")["splits"][split]
        self.transform = transform
        self.mask_foreground = mask_foreground

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["tp"]) as image:
            rgb = np.array(image.convert("RGB"))

        height, width = rgb.shape[:2]
        if row["label"] == 1:
            mask_image = read_binary_mask(self.root / row["gt"], self.mask_foreground)
            if mask_image.size != (width, height):
                raise ValueError(f"Image/mask size mismatch: {row['tp']}")
            mask = np.asarray(mask_image) > 0
            if not mask.any():
                raise ValueError(f"Empty forgery mask: {row['gt']}")
        else:
            mask = np.zeros((height, width), dtype=bool)

        sample = {
            "rgb": np.ascontiguousarray(rgb.transpose(2, 0, 1)),
            "mask": mask[None].astype(np.float32),
            "valid": np.ones((1, height, width), dtype=bool),
            "path": row["tp"],
            "type": row["type"],
        }
        if self.transform is not None:
            sample = self.transform(sample)

        # A crop from a forged image is positive only if it contains forged pixels.
        mask = np.asarray(sample["mask"])
        valid = np.asarray(sample["valid"], dtype=bool)
        if mask.shape != valid.shape or not valid.any():
            raise ValueError(f"Invalid mask or valid-pixel map: {row['tp']}")
        sample["label"] = np.float32(np.any((mask > 0) & valid))
        return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    args = parser.parse_args()

    for split in ("train", "val"):
        dataset = CASIADataset(args.root, split)
        print(f"\n{split}: {len(dataset)} images")

        for label in (0, 1):
            index = next(
                i for i, row in enumerate(dataset.rows) if row["label"] == label
            )
            sample = dataset[index]

            assert int(sample["label"]) == label
            assert sample["rgb"].shape[1:] == sample["mask"].shape[1:]

            print(
                f"  {sample['path']} | label={label} | RGB={sample['rgb'].shape} "
                f"| forged pixels={int(sample['mask'].sum())}"
            )


if __name__ == "__main__":
    main()
