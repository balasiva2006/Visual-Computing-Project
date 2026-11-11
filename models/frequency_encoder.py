"""Trainable DCT/SRM encoder for frequency_only and fusion modes."""
import argparse

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(cin, cout, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            cin, cout, kernel,
            stride=stride, padding=kernel // 2, bias=False,
        ),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.body(x))


def stage(cin, cout, stride=1):
    return nn.Sequential(
        conv_block(cin, cout, stride=stride),
        ResidualBlock(cout),
        ResidualBlock(cout),
    )


class FrequencyEncoder(nn.Module):
    strides = (4, 8, 16, 32)
    out_channels = 96
    band_widths = (10, 22, 32)

    def __init__(self):
        super().__init__()

        # DCT channels arrive in zigzag order: low, middle, high frequencies.
        self.band_stems = nn.ModuleList([
            nn.Sequential(
                conv_block(cin, cout, kernel=1),
                conv_block(cout, cout),
            )
            for cin, cout in zip(self.band_widths, (16, 32, 48))
        ])

        # SRM starts at full image resolution.
        self.srm4 = nn.Sequential(
            conv_block(3, 32, stride=2),
            conv_block(32, 64, stride=2),
        )
        self.srm8 = conv_block(64, 96, stride=2)

        # Combine DCT and SRM at strides 4 and 8.
        self.stage2 = stage(64 + 96, 96)
        self.stage3 = stage(96 + 96, 96)

        # Build the two coarser feature maps.
        self.stage4 = stage(96, 192, stride=2)
        self.stage5 = stage(192, 384, stride=2)

        # Give the decoder 96 channels at every scale.
        self.adapter4 = conv_block(192, 96, kernel=1)
        self.adapter5 = conv_block(384, 96, kernel=1)

    def forward(self, dct, srm):
        if srm.ndim != 4 or srm.shape[1] != 3:
            raise ValueError("Expected SRM [B,3,H,W]")

        b, _, h, w = srm.shape
        if h % 32 or w % 32 or dct.shape != (b, 64, h // 8, w // 8):
            raise ValueError(
                "Expected H,W divisible by 32 and DCT [B,64,H/8,W/8]"
            )

        bands = dct.split(self.band_widths, dim=1)
        d8 = torch.cat(
            [stem(band) for stem, band in zip(self.band_stems, bands)],
            dim=1,
        )

        s4 = self.srm4(srm)
        s8 = self.srm8(s4)

        # Each DCT cell describes one 8x8 block. SRM supplies finer detail.
        d4 = F.interpolate(d8, size=s4.shape[-2:], mode="nearest")
        f2 = self.stage2(torch.cat((s4, d4), dim=1))
        f3 = self.stage3(torch.cat((s8, d8), dim=1))
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        return f2, f3, self.adapter4(f4), self.adapter5(f5)


def main():
    from data.dataset import CASIADataset
    from data.frequency import ModelInputs, load_or_fit_stats
    from data.transforms import train_transform, validation_transform

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./dataset_clean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrequencyEncoder().to(device).eval()
    stats = load_or_fit_stats(args.root, "./artifacts/dct_stats.json")

    with torch.no_grad():
        for split, view in (
            ("train", train_transform),
            ("val", validation_transform),
        ):
            dataset = CASIADataset(
                args.root, split, transform=ModelInputs(view, stats)
            )
            sample = dataset[0]

            dct = torch.from_numpy(sample["dct"]).unsqueeze(0).to(device)
            srm = torch.from_numpy(sample["srm"]).unsqueeze(0).to(device)

            maps = encoder(dct, srm)
            h, w = srm.shape[-2:]

            assert len(maps) == 4
            for stride, feature in zip(encoder.strides, maps):
                assert feature.shape == (1, 96, h // stride, w // stride)
                assert torch.isfinite(feature).all().item()

            print(split, [tuple(feature.shape) for feature in maps])

    print("Frequency encoder check passed")


if __name__ == "__main__":
    main()
