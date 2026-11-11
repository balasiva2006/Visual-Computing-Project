"""RGB encoder for spatial-only and fusion modes."""
import argparse

import timm
import torch
from torch import nn


class SpatialEncoder(nn.Module):
    strides = (4, 8, 16, 32)
    channels = (18, 36, 72, 144)
    out_channels = 96

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True,
            feature_location="",
            out_indices=(1, 2, 3, 4),
        )

        info = self.backbone.feature_info
        if (
            tuple(info.channels()) != self.channels
            or tuple(info.reduction()) != self.strides
        ):
            raise RuntimeError(
                f"Unexpected HRNet features: {info.channels()}, {info.reduction()}"
            )

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, self.out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, self.out_channels),
                nn.ReLU(inplace=True),
            )
            for c in self.channels
        ])

        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)

        # With small batches, keep the backbone's running mean/variance fixed.
        # BatchNorm scale and bias can still receive gradients.
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        return self

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Expected normalized floating-point RGB [B,3,H,W]")

        if rgb.shape[-2] % 32 or rgb.shape[-1] % 32:
            raise ValueError("Pad height and width to multiples of 32")

        features = self.backbone(rgb)
        return tuple(
            adapter(feature)
            for adapter, feature in zip(self.adapters, features)
        )


def main():
    from data.dataset import CASIADataset
    from data.frequency import ModelInputs, load_or_fit_stats
    from data.transforms import train_transform, validation_transform

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SpatialEncoder(pretrained=args.pretrained).to(device).eval()
    stats = load_or_fit_stats(args.root, "./artifacts/dct_stats.json")

    with torch.no_grad():
        for split, view in (
            ("train", train_transform),
            ("val", validation_transform),
        ):
            ds = CASIADataset(
                args.root,
                split,
                transform=ModelInputs(view, stats),
            )

            # A dataset sample is [3,H,W]. Add the batch dimension.
            rgb = torch.from_numpy(ds[0]["rgb"]).unsqueeze(0).to(device)
            h, w = rgb.shape[-2:]
            maps = encoder(rgb)

            assert len(maps) == 4
            for stride, feature in zip(encoder.strides, maps):
                assert tuple(feature.shape) == (1, 96, h // stride, w // stride)
                assert torch.isfinite(feature).all().item()

            print(
                split, "RGB", tuple(rgb.shape),
                "->", [tuple(x.shape) for x in maps],
            )


if __name__ == "__main__":
    main()
