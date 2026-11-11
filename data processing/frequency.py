"""Build model inputs from the same transformed uint8 RGB view."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.fft import dctn
from scipy.ndimage import correlate

from .dataset import CASIADataset
from .transforms import train_transform, validation_transform


# Zigzag order begins with the DC coefficient and ends at high frequencies.
ORDER = sorted(
    np.ndindex(8, 8),
    key=lambda uv: (sum(uv), uv[0] if sum(uv) % 2 else -uv[0]),
)
BANDS = (slice(0, 10), slice(10, 32), slice(32, 64))

RGB_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
RGB_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

FILTERS = [
    np.array([
        [-1,  2, -1],
        [ 2, -4,  2],
        [-1,  2, -1],
    ], np.float32) / 4,

    np.array([
        [-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1],
    ], np.float32) / 12,

    np.array([[1, -2, 1]], np.float32) / 2,
]


def luminance(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Expected transformed uint8 RGB with shape [3,H,W]")

    r, g, b = rgb.astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def block_dct(gray):
    """Return signed-log DCT coefficients with shape [64,H/8,W/8]."""
    h, w = gray.shape
    blocks = (gray - 128).reshape(h // 8, 8, w // 8, 8)

    # Axes 1 and 3 are the two dimensions inside each 8x8 block.
    coeff = dctn(blocks, type=2, axes=(1, 3), norm="ortho")
    ordered = np.stack([coeff[:, u, :, v] for u, v in ORDER])

    scaled = np.sign(ordered) * np.log1p(np.abs(ordered)) / np.log1p(1024)
    return scaled.astype(np.float32)


def load_or_fit_stats(root, path):
    """Fit once on complete 8x8 blocks of unaugmented training images."""
    path = Path(path)
    manifest = Path(root) / "csv files" / "train.csv"
    identity = [
        "zigzag64-signedlog-v1",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    ]

    if path.exists():
        data = json.loads(path.read_text())
        if data["identity"] != identity:
            raise ValueError(
                "Training CSV or feature format changed. Choose a new --stats path."
            )
    else:
        ds = CASIADataset(root, "train")
        total = np.zeros(64, np.float64)
        squares = np.zeros(64, np.float64)
        count = 0

        for i in range(len(ds)):
            gray = luminance(ds[i]["rgb"])
            h, w = gray.shape

            # Only complete blocks of real pixels contribute to the statistics.
            if h >= 8 and w >= 8:
                complete = gray[:h - h % 8, :w - w % 8]
                x = block_dct(complete).reshape(64, -1)

                total += x.sum(axis=1, dtype=np.float64)
                squares += np.einsum("ij,ij->i", x, x, dtype=np.float64)
                count += x.shape[1]

            if (i + 1) % 1000 == 0:
                print(f"Fitting DCT statistics: {i + 1}/{len(ds)}", flush=True)

        if not count:
            raise ValueError("No complete training blocks")

        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean**2, 0))
        std[std < 1e-6] = 1

        data = {
            "identity": identity,
            "blocks": count,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as tmp:
            json.dump(data, tmp)
        os.replace(tmp.name, path)

    mean, std = [
        np.asarray(data[k], np.float32).reshape(64, 1, 1)
        for k in ("mean", "std")
    ]
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError(f"Invalid DCT statistics in {path}")

    return mean, std


class ModelInputs:
    """Pass this callable to CASIADataset(transform=...)."""

    def __init__(self, view, stats):
        self.view = view
        self.mean, self.std = stats

    def __call__(self, sample):
        # Apply the transform once so every branch sees the same image.
        s = self.view(sample)
        rgb = s["rgb"]
        H, W = rgb.shape[1:]
        h, w = map(int, s["view_hw"])
        gray = luminance(rgb[:, :h, :w])

        # Extend partial edge blocks. Fully padded blocks stay zero.
        extended = np.pad(gray, ((0, -h % 8), (0, -w % 8)), mode="edge")
        x = (block_dct(extended) - self.mean) / self.std

        dct = np.zeros((64, H // 8, W // 8), np.float32)
        dct[:, :x.shape[1], :x.shape[2]] = x

        # Filter real image pixels before padding the residual maps.
        srm = np.zeros((3, H, W), np.float32)
        for k, kernel in enumerate(FILTERS):
            residual = correlate(gray, kernel, mode="reflect")
            srm[k, :h, :w] = np.clip(residual, -3, 3) / 3

        spatial = (rgb.astype(np.float32) / 255 - RGB_MEAN) / RGB_STD

        return {
            **s,
            "rgb": spatial * s["valid"],
            "dct": dct,
            "srm": srm,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    parser.add_argument("--stats", default="./artifacts/dct_stats.json")
    args = parser.parse_args()

    stats = load_or_fit_stats(args.root, args.stats)

    for split, view in (("train", train_transform), ("val", validation_transform)):
        ds = CASIADataset(args.root, split, transform=ModelInputs(view, stats))

        for label in (0, 1):
            idx = next(i for i, r in enumerate(ds.rows) if r["label"] == label)
            s = ds[idx]
            h, w = s["rgb"].shape[1:]

            assert s["dct"].shape == (64, h // 8, w // 8)
            assert s["srm"].shape == s["rgb"].shape
            assert all(np.isfinite(s[k]).all() for k in ("rgb", "dct", "srm"))

            print(
                split, "label", int(s["label"]),
                "RGB", s["rgb"].shape,
                "DCT", s["dct"].shape,
                "SRM", s["srm"].shape,
            )


if __name__ == "__main__":
    main()
