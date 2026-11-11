"""Detection and localization metrics on CPU NumPy arrays.

Pass probabilities after sigmoid, binary 0/1 masks, and one [H,W] image
at a time. Choose thresholds on validation data and freeze them for testing.
Localization scores are macro averages over nonempty forged masks.
Undefined metrics are None, which serializes as JSON null.
"""
import json

import numpy as np
from scipy.stats import rankdata


def _binary(values, name):
    values = np.asarray(values)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return values.astype(bool)


def _probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(
            "Probabilities must be finite and in [0,1]; apply sigmoid to logits"
        )
    return values


def _threshold(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be finite and in [0,1]")
    return value


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def binary_auc(target, scores):
    """Exact ROC-AUC using average ranks for ties; None for one-class data."""
    target = _binary(target, "target")
    scores = np.asarray(scores, dtype=np.float64)

    if target.ndim != 1 or scores.shape != target.shape or not target.size:
        raise ValueError("Expected matching nonempty 1D target and scores")
    if not np.isfinite(scores).all():
        raise ValueError("AUC scores must be finite")

    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return None

    ranks = rankdata(scores, method="average")
    value = ranks[target].sum() - positives * (positives + 1) / 2
    return float(value / (positives * negatives))


def detection_metrics(labels, probabilities, threshold=0.5):
    """Image-level metrics. F1 is 0 if there are no positives or predictions."""
    labels = _binary(labels, "labels")
    probabilities = _probabilities(probabilities)
    threshold = _threshold(threshold)

    if labels.ndim != 1 or labels.shape != probabilities.shape or not labels.size:
        raise ValueError("Expected matching nonempty 1D labels and probabilities")

    predicted = probabilities >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())

    return {
        "count": int(labels.size),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "accuracy": float((tp + tn) / labels.size),
        "false_positive_rate": _ratio(fp, fp + tn),
        "roc_auc": binary_auc(labels, probabilities),
    }


def localization_metrics(mask, probabilities, valid=None, threshold=0.5):
    """Metrics for one [H,W] image, with padded positions excluded.

    Empty ground-truth masks have None for foreground IoU and Dice/F1.
    Their false-positive area is reported through predicted_fraction.
    Pixel AUC requires both foreground and background among valid pixels.
    """
    mask = np.asarray(mask)
    probabilities = np.asarray(probabilities)

    if mask.ndim != 2 or not mask.size or probabilities.shape != mask.shape:
        raise ValueError("Expected matching nonempty [H,W] mask and probabilities")

    if valid is None:
        valid = np.ones(mask.shape, dtype=bool)
    valid = _binary(valid, "valid")

    if valid.shape != mask.shape or not valid.any():
        raise ValueError(
            "valid must match [H,W] and contain at least one valid pixel"
        )

    target = _binary(mask[valid], "mask")
    scores = _probabilities(probabilities[valid])
    predicted = scores >= _threshold(threshold)

    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    foreground = int(target.sum())

    dice = _ratio(2 * tp, 2 * tp + fp + fn) if foreground else None

    return {
        "valid_pixels": int(target.size),
        "foreground_pixels": foreground,
        "foreground_fraction": float(foreground / target.size),
        "predicted_fraction": float(predicted.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": _ratio(tp, tp + fp + fn) if foreground else None,
        "dice": dice,
        "f1": dice,
        "roc_auc": binary_auc(target, scores),
        "background_fpr": _ratio(fp, fp + tn),
    }


def _mean_present(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


class EvaluationMetrics:
    """Accumulate one condition/dataset/model at a time, storing scalar rows.

    update accepts CPU arrays, one image label, and one image probability.
    A forged image with an empty mask contributes to detection, but is
    excluded from localization means and explicitly counted in compute().
    """

    def __init__(self, pixel_threshold=0.5, image_threshold=0.5):
        self.pixel_threshold = _threshold(pixel_threshold)
        self.image_threshold = _threshold(image_threshold)
        self.rows = []
        self._ids = set()

    def update(
        self,
        sample_id,
        label,
        image_probability,
        mask,
        mask_probability,
        valid=None,
    ):
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a nonempty string")
        if sample_id in self._ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")

        label_array = _binary(label, "label")
        image_array = _probabilities(image_probability)

        if label_array.ndim != 0 or image_array.ndim != 0:
            raise ValueError("label and image_probability must be scalars")

        label = int(label_array.item())
        row = localization_metrics(
            mask, mask_probability, valid, self.pixel_threshold
        )

        if label == 0 and row["foreground_pixels"]:
            raise ValueError(
                "An authentic image must have an empty ground-truth mask"
            )

        row.update(
            sample_id=sample_id,
            label=label,
            image_probability=float(image_array.item()),
        )
        self.rows.append(row)
        self._ids.add(sample_id)
        return dict(row)

    def compute(self):
        if not self.rows:
            raise ValueError("No samples accumulated")

        forged = [row for row in self.rows if row["label"] == 1]
        localized = [row for row in forged if row["foreground_pixels"] > 0]
        authentic = [row for row in self.rows if row["label"] == 0]

        return {
            "detection": detection_metrics(
                [row["label"] for row in self.rows],
                [row["image_probability"] for row in self.rows],
                self.image_threshold,
            ),
            "localization": {
                "threshold": self.pixel_threshold,
                "forged_images": len(forged),
                "evaluated_images": len(localized),
                "empty_forged_masks": len(forged) - len(localized),
                "macro_iou": _mean_present(localized, "iou"),
                "macro_dice": _mean_present(localized, "dice"),
                "macro_f1": _mean_present(localized, "f1"),
                "macro_pixel_roc_auc": _mean_present(localized, "roc_auc"),
                "pixel_auc_images": sum(
                    row["roc_auc"] is not None for row in localized
                ),
                "authentic_images": len(authentic),
                "pristine_false_positive_area": _mean_present(
                    authentic, "predicted_fraction"
                ),
            },
        }


def main():
    target = np.array([[1, 0], [1, 0]])
    scores = np.array([[0.9, 0.8], [0.2, 0.1]])

    result = localization_metrics(target, scores)
    np.testing.assert_allclose(
        [result["iou"], result["dice"], result["roc_auc"]],
        [1 / 3, 0.5, 0.75],
    )
    assert localization_metrics(target, target)["dice"] == 1.0
    assert localization_metrics(target, 1 - target)["dice"] == 0.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5
    assert binary_auc([1, 1], [0.2, 0.8]) is None

    padded_mask = np.pad(target, ((0, 0), (0, 1)), constant_values=0)
    padded_scores = np.pad(scores, ((0, 0), (0, 1)), constant_values=np.nan)
    valid = np.pad(
        np.ones_like(target), ((0, 0), (0, 1)), constant_values=0
    )
    assert localization_metrics(padded_mask, padded_scores, valid) == result

    meter = EvaluationMetrics()
    meter.update("forged", 1, 0.9, target, scores)
    meter.update("authentic", 0, 0.1, np.zeros_like(target), scores)

    summary = meter.compute()
    assert summary["detection"]["f1"] == 1.0
    assert summary["detection"]["roc_auc"] == 1.0
    assert summary["localization"]["macro_dice"] == 0.5
    assert summary["localization"]["pristine_false_positive_area"] == 0.5

    meter.update("empty_forged", 1, 0.8, np.zeros_like(target), scores)
    assert meter.compute()["localization"]["empty_forged_masks"] == 1
    assert meter.compute()["localization"]["macro_dice"] == 0.5

    empty = localization_metrics(
        np.zeros_like(target), np.zeros_like(scores)
    )
    assert empty["dice"] is None and empty["roc_auc"] is None
    assert localization_metrics(
        np.ones_like(target), scores
    )["roc_auc"] is None
    assert detection_metrics([0, 0], [0.1, 0.2])["roc_auc"] is None

    json.dumps(meter.compute(), allow_nan=False)
    print(
        "Metrics check passed: IoU, Dice/F1, AUC ties, padding, "
        "pristine images, undefined AUC"
    )


if __name__ == "__main__":
    main()
