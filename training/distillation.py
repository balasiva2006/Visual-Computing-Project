"""Optional within-stream feature distillation for joint fusion fine-tuning."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


STREAMS = ("spatial", "frequency")
STRIDES = (16, 32)


class FrozenTeachers(nn.Module):
    """Construct after loading the two best standalone checkpoints."""

    def __init__(self, spatial_model, frequency_model):
        super().__init__()

        if (
            spatial_model.mode != "spatial_only"
            or frequency_model.mode != "frequency_only"
        ):
            raise ValueError(
                "Expected the spatial_only and frequency_only models, in that order"
            )

        # Independent copies: updating the student cannot change the teachers.
        self.spatial = deepcopy(spatial_model.spatial_encoder)
        self.frequency = deepcopy(frequency_model.frequency_encoder)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep teacher normalization and dropout in evaluation mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch):
        return {
            "spatial": self.spatial(batch["rgb"])[2:],
            "frequency": self.frequency(batch["dct"], batch["srm"])[2:],
        }


def distillation_loss(student, teacher, valid):
    """Return unweighted mean cosine distance across both streams and scales."""
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("Expected valid mask [B,1,H,W]")

    b, _, h, w = valid.shape
    terms = []

    for name in STREAMS:
        if len(student[name]) != 2 or len(teacher[name]) != 2:
            raise ValueError(f"Expected stride-16 and stride-32 {name} maps")

        for stride, s, t in zip(STRIDES, student[name], teacher[name]):
            if (
                s.ndim != 4
                or t.shape != s.shape
                or s.shape[0] != b
                or s.shape[-2:] != (h // stride, w // stride)
            ):
                raise ValueError(f"Mismatched {name} features at stride {stride}")

            # Compare feature directions across channels at each spatial location.
            t = t.detach().float()
            similarity = F.cosine_similarity(s.float(), t, dim=1, eps=1e-8)
            distance = 1 - similarity.clamp(-1, 1)

            # Partial cells contribute in proportion to their real image area.
            weights = F.adaptive_avg_pool2d(
                valid.float(), s.shape[-2:]
            ).squeeze(1)

            # A zero teacher vector has no direction to match.
            weights = weights * t.ne(0).any(dim=1)

            scale_loss = (distance * weights).sum() / weights.sum().clamp_min(1e-8)
            terms.append(scale_loss)

    return torch.stack(terms).mean()


def main():
    # Left 24 columns are real. The rest are padding.
    valid = torch.zeros(1, 1, 32, 64, dtype=torch.bool)
    valid[..., :24] = True

    student, teacher = {}, {}
    for name in STREAMS:
        student[name], teacher[name] = [], []

        for stride in STRIDES:
            t = torch.zeros(1, 2, 32 // stride, 64 // stride)
            t[:, 0 if name == "spatial" else 1] = 1

            s = t.clone()
            if name == "spatial":
                s = s.flip(1)  # Orthogonal to teacher: cosine distance 1.

            # Deliberately disagree in fully padded cells.
            s[..., 32 // stride:] = -t[..., 32 // stride:]

            student[name].append(s.requires_grad_())
            teacher[name].append(t.requires_grad_())

    loss = distillation_loss(student, teacher, valid)

    # Spatial distance = 1; frequency distance = 0. Their mean is 0.5.
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()

    for name in STREAMS:
        for stride, s, t in zip(STRIDES, student[name], teacher[name]):
            assert t.grad is None
            assert s.grad is not None and torch.isfinite(s.grad).all()
            assert not s.grad[..., 32 // stride:].any()

            if name == "spatial":
                assert s.grad[..., 0].abs().sum() > 0

    print("Distillation check passed: loss=0.5, student gradients, padding excluded")


if __name__ == "__main__":
    main()
