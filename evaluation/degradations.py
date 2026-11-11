"""Deterministic evaluation conditions applied before padding and features.

RGB inputs are uint8 [3,H,W]. Masks and validity maps are binary [1,H,W].
Scaling means BOX downsampling followed by BICUBIC restoration to the
original dimensions, matching training and validation. Masks stay aligned
on the original pixel grid. Each condition starts from the original RGB.
"""
import hashlib
from numbers import Integral

import numpy as np

from data.transforms import _pad, degrade


# Ordered from clean to increasing severity within each degradation family.
SWEEP_CONDITIONS = {
    "clean": ("clean", 0),
    "jpeg95": ("jpeg", 95),
    "jpeg85": ("jpeg", 85),
    "jpeg75": ("jpeg", 75),
    "jpeg50": ("jpeg", 50),
    "jpeg30": ("jpeg", 30),
    "jpeg10": ("jpeg", 10),
    "scale075": ("scale", 0.75),
    "scale05": ("scale", 0.5),
    "scale025": ("scale", 0.25),
    "blur05": ("blur", 0.5),
    "blur1": ("blur", 1.0),
    "blur2": ("blur", 2.0),
    "blur3": ("blur", 3.0),
    "noise2": ("noise", 2.0),
    "noise5": ("noise", 5.0),
    "noise10": ("noise", 10.0),
    "noise20": ("noise", 20.0),
}

VALIDATION_CONDITIONS = {
    "clean": ("clean", 0),
    "jpeg70": ("jpeg", 70),
    "scale05": ("scale", 0.5),
    "blur1": ("blur", 1.0),
    "noise5": ("noise", 5.0),
}

CONDITIONS = {**SWEEP_CONDITIONS, **VALIDATION_CONDITIONS}


def condition_names(protocol="sweep"):
    """Return ordered condition names for the sweep or validation protocol."""
    if protocol == "sweep":
        return tuple(SWEEP_CONDITIONS)
    if protocol == "validation":
        return tuple(VALIDATION_CONDITIONS)
    raise ValueError("protocol must be 'sweep' or 'validation'")


def evaluation_transform(sample, condition="clean", seed=42):
    """Transform one raw dataset sample, then pad its dimensions to /32.

    Required sample keys are rgb, mask, valid, and path. Extra keys, including
    label, are preserved. path is the stable image identifier used to seed
    noise. Use the same identifiers and seed across model comparisons.
    Inputs must be unpadded and all pixels valid. This prevents padding
    borders from contaminating JPEG, blur, and downsampling operations.

    Noise strengths share a standard-normal realization per image, scaled
    by sigma in 0..255 intensity units. Seeds match validation_transform.
    Recompute DCT/SRM from this output using the training DCT statistics.
    """
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}; choose from {tuple(CONDITIONS)}"
        )
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(sample["path"], str) or not sample["path"]:
        raise ValueError(
            "sample['path'] must be a nonempty stable image identifier"
        )

    rgb = np.asarray(sample["rgb"])
    mask = np.asarray(sample["mask"])
    valid = np.asarray(sample["valid"])

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Expected raw uint8 RGB with shape [3,H,W]")

    height, width = rgb.shape[1:]
    if height == 0 or width == 0:
        raise ValueError("Image dimensions must be positive")

    if mask.shape != (1, height, width) or valid.shape != mask.shape:
        raise ValueError(
            "mask and valid must have shape [1,H,W] matching RGB"
        )
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be binary 0/1")
    if not np.equal(valid, 1).all():
        raise ValueError("Pass an unpadded sample with all pixels valid")

    if "view_hw" in sample and not np.array_equal(
        sample["view_hw"], [height, width]
    ):
        raise ValueError("Pass the original sample before padding")

    if any(key in sample for key in ("dct", "srm")):
        raise ValueError("Apply degradations before computing DCT and SRM")

    kind, strength = CONDITIONS[condition]
    key = f"{seed}|{sample['path']}|{kind}".encode()
    image_seed = int.from_bytes(
        hashlib.sha256(key).digest()[:8], "little"
    )

    degraded = degrade(rgb, kind, strength, image_seed)

    return _pad(
        sample,
        degraded,
        mask,
        valid.astype(bool),
        height + (-height) % 32,
        width + (-width) % 32,
    )


def main():
    from data.transforms import validation_transform

    rng = np.random.default_rng(7)
    height, width = 45, 67

    rgb = rng.integers(
        0, 256, (3, height, width), dtype=np.uint8
    )
    mask = np.zeros((1, height, width), dtype=np.float32)
    mask[:, 10:20, 15:30] = 1

    sample = {
        "rgb": rgb,
        "mask": mask,
        "valid": np.ones_like(mask, dtype=bool),
        "path": "synthetic/forged.png",
        "label": 1,
    }
    original = {
        key: value.copy()
        for key, value in sample.items()
        if isinstance(value, np.ndarray)
    }

    for name in condition_names():
        result = evaluation_transform(sample, name)

        assert result["rgb"].shape == (3, 64, 96)
        assert result["rgb"].dtype == np.uint8
        assert result["rgb"].flags.c_contiguous
        assert (
            result["mask"].shape
            == result["valid"].shape
            == (1, 64, 96)
        )
        assert int(result["valid"].sum()) == height * width
        assert not result["mask"][~result["valid"]].any()
        assert result["label"] == 1

        np.testing.assert_array_equal(
            result["view_hw"], [height, width]
        )
        np.testing.assert_array_equal(
            result["mask"][:, :height, :width], mask
        )

        if name == "clean":
            np.testing.assert_array_equal(
                result["rgb"][:, :height, :width], rgb
            )

    # Existing validation views must match byte-for-byte, including noise.
    for name in condition_names("validation"):
        expected = validation_transform(
            sample, view=name, seed=42
        )
        actual = evaluation_transform(
            sample, condition=name, seed=42
        )
        for key in ("rgb", "mask", "valid", "view_hw"):
            np.testing.assert_array_equal(
                actual[key], expected[key]
            )

    first = evaluation_transform(sample, "noise10", seed=42)
    evaluation_transform(sample, "noise20", seed=123)
    repeated = evaluation_transform(sample, "noise10", seed=42)
    np.testing.assert_array_equal(
        first["rgb"], repeated["rgb"]
    )

    changed = evaluation_transform(sample, "noise10", seed=43)
    assert not np.array_equal(first["rgb"], changed["rgb"])

    for key, value in original.items():
        np.testing.assert_array_equal(sample[key], value)

    try:
        evaluation_transform(first, "jpeg50")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Already padded samples must be rejected"
        )

    print(
        "Degradation check passed: 18 sweep conditions, "
        "validation parity, deterministic noise, aligned masks"
    )


if __name__ == "__main__":
    main()
