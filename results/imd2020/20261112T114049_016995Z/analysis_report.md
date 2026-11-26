# IMD2020 measured evaluation report

Test: 2424 images, 414 source groups. Manifest: imd2020_test.csv.
Frozen thresholds: image=0.5, pixel=0.5.
Pixel overlap averages include forged images only. No thresholds were fitted on test data.

## Fusion finetuned
Image accuracy 0.778, F1 0.865, ROC-AUC 0.736.
Forged-image macro Dice 0.355, macro IoU 0.275; authentic false-positive area 0.037.
Incorrect image decisions: 538/2424.
Median preprocessing + inference + metric time: 0.271 s/image (hardware-specific; includes first-pass overhead).

## Interpretation limits
Confirm that this external dataset and related source images were not used in training or selection.
Inspect mask polarity/alignment and both image classes. Source-related samples are correlated.
Bootstrap intervals describe source-group sampling uncertainty, not training uncertainty.
Qualitative galleries use declared best/median/worst selection; use aggregate scores for claims.
Preview runs are workflow checks, not full-dataset results. Test threshold curves are descriptive only.