"""Aligned CASIA transforms. Keep RGB in uint8 until frequency extraction."""
import argparse
import hashlib
import io
import random

import numpy as np
from PIL import Image, ImageFilter


def _origin(length, size, point):
    """Choose an 8-pixel-aligned crop origin, optionally including a given pixel."""
    last = max(0, (length - size + 7) // 8)
    low, high = 0, last

    if point is not None:
        low = max(0, (int(point) - size + 8) // 8)
        high = min(last, int(point) // 8)

    return 8 * random.randint(low, high)


def degrade(rgb, kind, strength=0, seed=0):
    """Input and output: uint8 RGB [3,H,W]. Noise sigma uses the 0–255 scale."""
    if kind == "clean":
        return rgb

    image = Image.fromarray(np.moveaxis(rgb, 0, -1))

    if kind == "jpeg":
        with io.BytesIO() as buf:
            image.save(buf, format="JPEG", quality=int(strength), subsampling=2)
            buf.seek(0)
            with Image.open(buf) as decoded:
                result = np.array(decoded.convert("RGB"))

    elif kind == "scale":
        filters = getattr(Image, "Resampling", Image)
        small = tuple(max(1, round(n * strength)) for n in image.size)
        result = np.array(
            image.resize(small, filters.BOX).resize(image.size, filters.BICUBIC)
        )

    elif kind == "blur":
        result = np.array(image.filter(ImageFilter.GaussianBlur(strength)))

    elif kind == "noise":
        rng = np.random.default_rng(seed)
        noisy = np.asarray(image) + rng.normal(
            0, strength, size=(*rgb.shape[1:], 3)
        )
        result = np.rint(noisy).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown degradation: {kind}")

    return np.ascontiguousarray(np.moveaxis(result, -1, 0))


def _pad(sample, rgb, mask, valid, height, width):
    h, w = rgb.shape[1:]
    pads = ((0, 0), (0, height - h), (0, width - w))

    return {
        **sample,
        "rgb": np.ascontiguousarray(np.pad(rgb, pads, mode="edge")),
        "mask": np.ascontiguousarray(np.pad(mask, pads, mode="constant")),
        "valid": np.ascontiguousarray(np.pad(valid, pads, mode="constant")),
        "view_hw": np.array([h, w], dtype=np.int64),
    }


def train_transform(sample, crop_size=512):
    if crop_size <= 0 or crop_size % 32:
        raise ValueError("crop_size must be a positive multiple of 32")

    rgb, mask, valid = (sample[k] for k in ("rgb", "mask", "valid"))
    h, w = rgb.shape[1:]

    # When cropping a large forged image, usually retain a forged pixel.
    y = x = None
    if (h > crop_size or w > crop_size) and random.random() < 0.9:
        positives = np.flatnonzero((mask[0] > 0) & valid[0])
        if positives.size:
            y, x = divmod(int(random.choice(positives)), w)

    top = _origin(h, crop_size, y)
    left = _origin(w, crop_size, x)
    region = np.s_[:, top:top + crop_size, left:left + crop_size]

    # Use exactly the same crop, flip, and rotation for all three arrays.
    arrays = [a[region] for a in (rgb, mask, valid)]
    if random.random() < 0.5:
        arrays = [a[:, :, ::-1] for a in arrays]

    turns = random.randrange(4)
    rgb, mask, valid = [np.rot90(a, turns, axes=(1, 2)) for a in arrays]

    # Half of training views remain clean. The other half receive one degradation.
    if random.random() < 0.5:
        kind = random.choice(("jpeg", "scale", "blur", "noise"))
        if kind == "jpeg":
            strength = random.randint(50, 95)
        else:
            limits = {
                "scale": (0.5, 1.0),
                "blur": (0.2, 1.5),
                "noise": (1.0, 10.0),
            }
            strength = random.uniform(*limits[kind])

        rgb = degrade(rgb, kind, strength, random.getrandbits(64))

    return _pad(sample, rgb, mask, valid, crop_size, crop_size)


def validation_transform(sample, view="clean", seed=42):
    views = {
        "clean": ("clean", 0),
        "jpeg70": ("jpeg", 70),
        "scale05": ("scale", 0.5),
        "blur1": ("blur", 1),
        "noise5": ("noise", 5),
    }
    kind, strength = views[view]

    # Give the same validation image the same noise across all model runs.
    key = f"{seed}|{sample['path']}|{kind}".encode()
    image_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")

    rgb = degrade(sample["rgb"], kind, strength, image_seed)
    h, w = rgb.shape[1:]

    # Keep the full validation image and pad only to a multiple of 32.
    return _pad(
        sample, rgb, sample["mask"], sample["valid"],
        h + (-h) % 32, w + (-w) % 32,
    )


def main():
    from .dataset import CASIADataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    args = parser.parse_args()
    random.seed(42)

    for split, transform in (("train", train_transform), ("val", validation_transform)):
        ds = CASIADataset(args.root, split, transform=transform)

        for label in (0, 1):
            index = next(i for i, row in enumerate(ds.rows) if row["label"] == label)
            s = ds[index]
            h, w = s["rgb"].shape[1:]

            assert s["mask"].shape == s["valid"].shape == (1, h, w)
            assert h % 32 == w % 32 == 0
            assert not s["mask"][~s["valid"]].any()
            if split == "train":
                assert (h, w) == (512, 512)

            print(
                split, s["path"], "RGB", s["rgb"].shape,
                "valid pixels", int(s["valid"].sum()), "label", int(s["label"]),
            )


if __name__ == "__main__":
    main()
