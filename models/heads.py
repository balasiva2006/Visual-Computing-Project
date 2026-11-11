"""Prediction heads for forgery localization and image-level detection."""
import argparse

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(features, valid):
    # Fraction of real pixels in each feature cell, including partial edge cells.
    weights = F.adaptive_avg_pool2d(valid.float(), features.shape[-2:])
    total = (features.float() * weights).sum(dim=(-2, -1))
    return total / weights.sum(dim=(-2, -1)).clamp_min(1e-12)


class ForgeryHeads(nn.Module):
    def __init__(self):
        super().__init__()

        # The decoder supplies 32 features at each image pixel.
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1)

        # 96 pooled channels from P2 + 96 pooled channels from P5.
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def image_logit(self, p2, p5, valid):
        # Generalized mean pooling with p=3 emphasizes strong local responses.
        moment = masked_mean(p2.float().relu().pow(3), valid)
        fine = torch.where(
            moment > 0,
            moment.clamp_min(1e-12).pow(1 / 3),
            0.0,
        )

        # Average pooling summarizes the coarser feature map.
        coarse = masked_mean(p5, valid)
        combined = torch.cat((fine, coarse), dim=1)

        # Squeeze only the final dimension so batch size 1 still gives shape [1].
        return self.classifier(combined).squeeze(-1)

    def forward(self, decoded, valid):
        x = decoded["mask_features"]
        if valid.shape != (x.shape[0], 1, x.shape[2], x.shape[3]):
            raise ValueError(
                "Expected valid mask [B,1,H,W] aligned with mask_features"
            )

        return {
            "mask_logits": self.mask_head(x),
            "image_logits": self.image_logit(decoded["p2"], decoded["p5"], valid),
        }


def main():
    from data.dataset import CASIADataset
    from data.frequency import ModelInputs, load_or_fit_stats
    from data.transforms import train_transform, validation_transform
    from .decoder import FPNDecoder
    from .frequency_encoder import FrequencyEncoder
    from .fusion import MultiScaleFusion
    from .spatial_encoder import SpatialEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial = SpatialEncoder(pretrained=False).to(device).eval()
    frequency = FrequencyEncoder().to(device).eval()
    fusion = MultiScaleFusion().to(device).eval()
    decoder = FPNDecoder().to(device).eval()
    heads = ForgeryHeads().to(device).eval()
    stats = load_or_fit_stats(args.root, "./artifacts/dct_stats.json")

    with torch.no_grad():
        # Feature values are 2, 8, 100. Their cells contain 2, 1, 0 real pixels.
        # The expected mean is (2*2 + 8*1 + 100*0) / 3 = 4.
        toy = torch.tensor([2., 8., 100.], device=device).reshape(1, 1, 1, 3)
        valid = torch.tensor(
            [1, 1, 1, 0, 0, 0], device=device
        ).reshape(1, 1, 1, 6)
        torch.testing.assert_close(
            masked_mean(toy, valid), toy.new_tensor([[4.]])
        )

        for split, view in (
            ("train", train_transform),
            ("val", validation_transform),
        ):
            ds = CASIADataset(args.root, split, transform=ModelInputs(view, stats))
            sample = ds[0]

            rgb, dct, srm, valid = [
                torch.from_numpy(sample[k]).unsqueeze(0).to(device)
                for k in ("rgb", "dct", "srm", "valid")
            ]

            maps = fusion(spatial(rgb), frequency(dct, srm))
            decoded = decoder(maps, rgb.shape[-2:])
            out = heads(decoded, valid)

            assert out["mask_logits"].shape == valid.shape
            assert out["image_logits"].shape == (rgb.shape[0],)
            assert all(t.isfinite().all().item() for t in out.values())

            print(split, {k: tuple(t.shape) for k, t in out.items()})

    print("Heads check passed: both logits and padding-aware pooling")


if __name__ == "__main__":
    main()
