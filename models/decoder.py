"""Feature pyramid decoder shared by all three model modes."""
import argparse

import torch
from torch import nn
from torch.nn import functional as F

from .frequency_encoder import conv_block


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.p5 = conv_block(96, 96)
        self.p4 = conv_block(96, 96)
        self.p3 = conv_block(96, 96)
        self.p2 = conv_block(96, 96)

        self.merge = conv_block(384, 128, kernel=1)
        self.half_res = conv_block(128, 64)
        self.full_res = conv_block(64, 32)

    def forward(self, maps, output_size):
        """output_size is the padded RGB (height, width), also used by the mask."""
        if len(maps) != 4:
            raise ValueError("Expected four maps in stride order 4, 8, 16, 32")

        h, w = output_size
        if h % 32 or w % 32:
            raise ValueError("Use the padded input dimensions, divisible by 32")

        batch = maps[0].shape[0]
        for stride, x in zip((4, 8, 16, 32), maps):
            if x.shape != (batch, 96, h // stride, w // stride):
                raise ValueError(
                    f"Unexpected stride-{stride} feature shape: {tuple(x.shape)}"
                )

        # Top-down pyramid: add coarser context to each finer map.
        z2, z3, z4, z5 = maps
        p5 = self.p5(z5)
        p4 = self.p4(z4 + resize(p5, z4.shape[-2:]))
        p3 = self.p3(z3 + resize(p4, z3.shape[-2:]))
        p2 = self.p2(z2 + resize(p3, z2.shape[-2:]))

        # Align all four maps at stride 4, then refine up to image resolution.
        size4 = p2.shape[-2:]
        joined = torch.cat(
            (p2, resize(p3, size4), resize(p4, size4), resize(p5, size4)),
            dim=1,
        )
        x = self.merge(joined)
        x = self.half_res(resize(x, (h // 2, w // 2)))
        mask_features = self.full_res(resize(x, (h, w)))

        return {"mask_features": mask_features, "p2": p2, "p5": p5}


def main():
    from data.dataset import CASIADataset
    from data.frequency import ModelInputs, load_or_fit_stats
    from data.transforms import train_transform, validation_transform
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
    stats = load_or_fit_stats(args.root, "./artifacts/dct_stats.json")

    with torch.no_grad():
        for split, view in (
            ("train", train_transform),
            ("val", validation_transform),
        ):
            ds = CASIADataset(args.root, split, transform=ModelInputs(view, stats))
            sample = ds[0]

            rgb, dct, srm = [
                torch.from_numpy(sample[k]).unsqueeze(0).to(device)
                for k in ("rgb", "dct", "srm")
            ]
            h, w = rgb.shape[-2:]
            s, f = spatial(rgb), frequency(dct, srm)

            for mode, maps in (
                ("spatial_only", s),
                ("frequency_only", f),
                ("fusion", fusion(s, f)),
            ):
                out = decoder(maps, (h, w))
                assert out["mask_features"].shape == (1, 32, h, w)
                assert out["p2"].shape == (1, 96, h // 4, w // 4)
                assert out["p5"].shape == (1, 96, h // 32, w // 32)
                assert all(t.isfinite().all().item() for t in out.values())

                print(split, mode, "mask_features", tuple(out["mask_features"].shape))
                del out

    print("Decoder check passed: three modes, two input sizes")


if __name__ == "__main__":
    main()
