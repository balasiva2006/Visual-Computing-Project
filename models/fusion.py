"""Fuse paired spatial and frequency maps at strides 4, 8, 16, and 32."""
import argparse

import torch
from torch import nn


class GatedFusionBlock(nn.Module):
    """Combine the two encoder outputs at one resolution."""

    def __init__(self, channels=96):
        super().__init__()
        self.spatial_norm = nn.GroupNorm(8, channels)
        self.frequency_norm = nn.GroupNorm(8, channels)

        self.gate = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

        # Two zero logits give softmax weights of exactly 0.5 and 0.5.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, spatial, frequency):
        if spatial.shape != frequency.shape:
            raise ValueError(
                f"Unaligned feature maps: {spatial.shape} vs {frequency.shape}"
            )

        s = self.spatial_norm(spatial)
        f = self.frequency_norm(frequency)

        # Give the gate both features, their absolute difference, and their product.
        evidence = torch.cat((s, f, (s - f).abs(), s * f), dim=1)
        weights = self.gate(evidence).softmax(dim=1)

        # Each weight map is [B,1,H,W] and broadcasts over the 96 channels.
        mixed = weights[:, 0:1] * s + weights[:, 1:2] * f
        return self.refine(mixed), weights


class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([GatedFusionBlock() for _ in range(4)])

    def forward(self, spatial_maps, frequency_maps, *, return_gates=False):
        if len(spatial_maps) != 4 or len(frequency_maps) != 4:
            raise ValueError(
                "Expected four maps per encoder in stride order 4, 8, 16, 32"
            )

        features, gates = [], []
        for block, s, f in zip(self.blocks, spatial_maps, frequency_maps):
            z, gate = block(s, f)
            features.append(z)
            gates.append(gate)

        if return_gates:
            return tuple(features), tuple(gates)
        return tuple(features)


def main():
    from data.dataset import CASIADataset
    from data.frequency import ModelInputs, load_or_fit_stats
    from data.transforms import train_transform, validation_transform
    from .spatial_encoder import SpatialEncoder
    from .frequency_encoder import FrequencyEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random weights are sufficient for this integration check.
    spatial = SpatialEncoder(pretrained=False).to(device).eval()
    frequency = FrequencyEncoder().to(device).eval()
    fusion = MultiScaleFusion().to(device).eval()
    stats = load_or_fit_stats(args.root, "./artifacts/dct_stats.json")

    with torch.no_grad():
        for split, view in (
            ("train", train_transform),
            ("val", validation_transform),
        ):
            ds = CASIADataset(args.root, split, transform=ModelInputs(view, stats))

            # Read once: RGB, DCT, and SRM must describe the same transformed view.
            sample = ds[0]
            rgb, dct, srm = [
                torch.from_numpy(sample[k]).unsqueeze(0).to(device)
                for k in ("rgb", "dct", "srm")
            ]

            maps, gates = fusion(
                spatial(rgb), frequency(dct, srm), return_gates=True
            )

            h, w = rgb.shape[-2:]
            for stride, z, gate in zip((4, 8, 16, 32), maps, gates):
                assert z.shape == (1, 96, h // stride, w // stride)
                assert gate.shape == (1, 2, h // stride, w // stride)
                assert torch.isfinite(z).all().item()
                assert torch.all(gate == 0.5).item()

            print(split, [tuple(z.shape) for z in maps])

    print("Fusion check passed: four scales, equal initial gates")


if __name__ == "__main__":
    main()
