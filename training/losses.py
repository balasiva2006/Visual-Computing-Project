"""Supervised forgery localization and detection losses for all three modes."""
import math

import torch
from torch.nn import functional as F


def classification_weight(epoch, *, standalone):
    """Epochs are numbered starting at 1."""
    if epoch < 1:
        raise ValueError("epoch must start at 1")
    return 0.1 * min(epoch, 5) if standalone else 0.5


def forgery_loss(outputs, batch, *, lambda_cls=0.5):
    """Return scalar tensors. Backpropagate through losses["total"]."""
    logits = outputs["mask_logits"].float()
    target = batch["mask"].float()
    valid = batch["valid"].bool()
    image_logits = outputs["image_logits"].float()
    labels = batch["label"].float()

    if (
        logits.ndim != 4
        or logits.shape[1] != 1
        or target.shape != logits.shape
        or valid.shape != logits.shape
    ):
        raise ValueError("mask_logits, mask and valid must all have shape [B,1,H,W]")

    if image_logits.shape != (logits.shape[0],) or labels.shape != image_logits.shape:
        raise ValueError("image_logits and label must both have shape [B]")

    axes = (1, 2, 3)
    counts = valid.sum(dim=axes)
    if (counts == 0).any():
        raise ValueError("Every sample must have at least one valid pixel")

    # Average over valid pixels in each image, then over images in the batch.
    pixel_terms = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    pixel_bce = ((pixel_terms * valid).sum(dim=axes) / counts).mean()

    # Soft Dice uses probabilities and includes only nonempty target masks.
    prob = logits.sigmoid() * valid
    foreground = target * valid
    area = foreground.sum(dim=axes)
    intersection = (prob * foreground).sum(dim=axes)

    eps = 1e-6
    dice_per_image = 1 - (2 * intersection + eps) / (
        prob.sum(dim=axes) + area + eps
    )

    positive = area > 0
    dice = (dice_per_image * positive).sum() / positive.sum().clamp_min(1)

    image_bce = F.binary_cross_entropy_with_logits(image_logits, labels)
    total = 0.5 * pixel_bce + 0.5 * dice + lambda_cls * image_bce

    return {
        "total": total,
        "pixel_bce": pixel_bce,
        "dice": dice,
        "image_bce": image_bce,
    }


def main():
    # First image: two valid pixels, one forged.
    # Second image: one valid pixel, authentic.
    logits = torch.tensor(
        [[[[0., 0., 40.]]], [[[math.log(3), -40., 40.]]]],
        requires_grad=True,
    )
    image_logits = torch.zeros(2, requires_grad=True)

    batch = {
        "mask": torch.tensor([[[[1., 0., 0.]]], [[[0., 0., 0.]]]]),
        "valid": torch.tensor([
            [[[True, True, False]]],
            [[[True, False, False]]],
        ]),
        "label": torch.tensor([1., 0.]),
    }
    outputs = {"mask_logits": logits, "image_logits": image_logits}
    losses = forgery_loss(outputs, batch)

    # Compare against independently calculated values for this small batch.
    torch.testing.assert_close(
        losses["pixel_bce"], torch.tensor((math.log(2) + math.log(4)) / 2)
    )
    torch.testing.assert_close(losses["dice"], torch.tensor(0.5))
    torch.testing.assert_close(losses["image_bce"], torch.tensor(math.log(2)))

    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(image_logits.grad).all()

    # Padding receives zero mask gradient. The authentic pixel receives a
    # positive gradient, so gradient descent lowers its forgery score.
    assert (logits.grad[~batch["valid"]] == 0).all()
    assert logits.grad[1, 0, 0, 0] > 0

    authentic = forgery_loss(
        {k: v[1:] for k, v in outputs.items()},
        {k: v[1:] for k, v in batch.items()},
    )
    assert authentic["dice"].item() == 0
    assert authentic["pixel_bce"].item() > 0

    print("Loss check passed: per-image BCE, foreground Dice, padding gradients, authentic batch")


if __name__ == "__main__":
    main()
