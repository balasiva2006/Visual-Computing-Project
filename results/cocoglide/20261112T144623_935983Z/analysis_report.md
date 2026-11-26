# COCOGlide measured evaluation report

Test: 1024 images, 512 source groups. Manifest: cocoglide_test.csv.
Frozen thresholds: image=0.5, pixel=0.5.
Pixel overlap averages include forged images only. No thresholds were fitted on test data.
Mask policy: predicted image label 0 forces an empty final mask; localization uses this image-gated output.

## Fusion finetuned
Image accuracy 0.545, F1 0.625, ROC-AUC 0.579.
Forged-image macro Dice 0.144, macro IoU 0.099; authentic false-positive area 0.023.
Incorrect image decisions: 466/1024.
Median preprocessing + inference + metric time: 0.069 s/image (hardware-specific; includes first-pass overhead).

## Interpretation limits
Confirm that this external dataset and related source images were not used in training or selection.
Inspect mask polarity/alignment and both image classes. Source-related samples are correlated.
Bootstrap intervals describe source-group sampling uncertainty, not training uncertainty.
Qualitative galleries use declared best/median/worst selection; use aggregate scores for claims.
Preview runs are workflow checks, not full-dataset results. Test threshold curves are descriptive only.