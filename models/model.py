"""End-to-end forgery detection and localization in three modes."""
import argparse

import torch
from torch import nn

from .decoder import FPNDecoder
from .frequency_encoder import FrequencyEncoder
from .fusion import MultiScaleFusion
from .heads import ForgeryHeads


INPUTS = {
    "spatial_only": ("rgb", "valid"),
    "frequency_only": ("dct", "srm", "valid"),
    "fusion": ("rgb", "dct", "srm", "valid"),
}


class ForgeryModel(nn.Module):
    def __init__(self, mode, *, pretrained_spatial=True):
        super().__init__()

        if mode not in INPUTS:
            raise ValueError(f"Unknown mode {mode!r}. Choose from {tuple(INPUTS)}")

        self.mode = mode
        self.spatial_encoder = None
        self.frequency_encoder = None
        self.fusion = None

        if mode in ("spatial_only", "fusion"):
            from .spatial_encoder import SpatialEncoder
            self.spatial_encoder = SpatialEncoder(pretrained=pretrained_spatial)

        if mode in ("frequency_only", "fusion"):
            self.frequency_encoder = FrequencyEncoder()

        if mode == "fusion":
            self.fusion = MultiScaleFusion()

        self.decoder = FPNDecoder()
        self.heads = ForgeryHeads()

    def forward(self, batch, *, return_features=False):
        missing = [key for key in INPUTS[self.mode] if key not in batch]
        if missing:
            raise ValueError(f"{self.mode} requires batch keys {missing}")

        valid = batch["valid"]
        if valid.ndim != 4 or valid.shape[1] != 1:
            raise ValueError("Expected batched valid mask [B,1,H,W]")

        streams = {}
        if self.spatial_encoder is not None:
            streams["spatial"] = self.spatial_encoder(batch["rgb"])

        if self.frequency_encoder is not None:
            streams["frequency"] = self.frequency_encoder(
                batch["dct"], batch["srm"]
            )

        if self.mode == "fusion":
            maps = self.fusion(streams["spatial"], streams["frequency"])
        elif self.mode == "spatial_only":
            maps = streams["spatial"]
        else:
            maps = streams["frequency"]

        decoded = self.decoder(maps, valid.shape[-2:])
        out = self.heads(decoded, valid)

        if return_features:
            # Keep student gradients for the optional distillation loss.
            out["features"] = {
                name: maps_[2:] for name, maps_ in streams.items()
            }

        return out


def main():
    from data.dataset import CASIADataset
    from data.frequency import ModelInputs, load_or_fit_stats
    from data.transforms import train_transform, validation_transform

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_or_fit_stats(args.root, "./artifacts/dct_stats.json")

    # Use the same transformed examples across all three modes.
    samples = {}
    for split, view in (
        ("train", train_transform),
        ("val", validation_transform),
    ):
        ds = CASIADataset(args.root, split, transform=ModelInputs(view, stats))
        samples[split] = ds[0]

    with torch.no_grad():
        for mode, keys in INPUTS.items():
            model = ForgeryModel(mode, pretrained_spatial=False).to(device).eval()

            for split, sample in samples.items():
                # Pass only the inputs required by this mode.
                batch = {
                    k: torch.from_numpy(sample[k]).unsqueeze(0).to(device)
                    for k in keys
                }

                out = model(batch, return_features=True)

                assert out["mask_logits"].shape == batch["valid"].shape
                assert out["image_logits"].shape == (1,)
                assert all(
                    out[k].isfinite().all().item()
                    for k in ("mask_logits", "image_logits")
                )

                h, w = batch["valid"].shape[-2:]
                for coarse in out["features"].values():
                    assert [tuple(t.shape) for t in coarse] == [
                        (1, 96, h // 16, w // 16),
                        (1, 96, h // 32, w // 32),
                    ]

                print(
                    split, mode,
                    "mask", tuple(out["mask_logits"].shape),
                    "image", tuple(out["image_logits"].shape),
                )

            del model

    print("Model check passed: three modes, two sizes, both outputs")


if __name__ == "__main__":
    main()
