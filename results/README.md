# Results and reproduction guide

[Project run guide](../README.md) | [Notebook instructions](../notebooks/README.md)

This directory contains the supplied checkpoints, training histories/plots, external evaluation reports, and presentation figures. Recorded metrics below describe the final fusion checkpoint on clean IMD2020 and COCOGlide images.

## Recorded results

Image and pixel thresholds were fixed at `0.5`. All values below are fractions, not percentages.

| Dataset and mask policy | Images | Accuracy | Image F1 | Image ROC-AUC | Macro Dice | Macro IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IMD2020, original independent pixel head | 2,424 | 0.7781 | 0.8646 | 0.7355 | 0.3549 | 0.2753 |
| IMD2020, final image-gated masks | 2,424 | 0.7781 | 0.8646 | 0.7355 | 0.3343 | 0.2620 |
| COCOGlide, image-gated masks | 1,024 | 0.5449 | 0.6254 | 0.5786 | 0.1444 | 0.0993 |

| Dataset | Authentic | Forged | Source groups | Recorded run |
| --- | ---: | ---: | ---: | --- |
| IMD2020 | 414 | 2,010 | 414 | [20261112T114049_016995Z](imd2020/20261112T114049_016995Z/) |
| COCOGlide | 512 | 512 | 512 | [20261112T144623_935983Z](cocoglide/20261112T144623_935983Z/) |

The original IMD2020 JSON/CSV measurements remain unchanged. Its gated presentation scores were calculated from stored image decisions and pixel confusion counts, without another model run. The corresponding PNGs are in [imd2020/presentation/IMD2020_20261112T143751Z](imd2020/presentation/IMD2020_20261112T143751Z/). COCOGlide's recorded metadata explicitly identifies the image-gated policy.

## Evaluation protocol

- Images retain native resolution; padded pixels are excluded from metrics.
- Image detection and foreground localization are scored separately.
- Foreground Dice and pixel F1 are the same quantity in this binary setting.
- Macro localization scores average over forged images. Authentic images contribute false-positive area, the fraction of their pixels predicted as forged.
- Under **image gating**, an image probability below the image threshold forces an empty final mask. Forged images classified as authentic then receive zero foreground Dice/IoU.
- Current notebooks apply image gating. The CLI in `evaluation/evaluate.py` scores independent pixel-head outputs, as did the original IMD2020 run.

Mask policies affect pixel metrics while leaving image probabilities and image detection metrics unchanged. Compare runs only after checking the dataset manifest, checkpoint, preprocessing, thresholds, and mask policy. A fresh notebook evaluation should be compared with the gated rows above.

## Reading the findings

On IMD2020, the model detected 1,718 forged images and missed 292; it also flagged 246 of 414 authentic images as forged. On COCOGlide, it detected 389 forged images and missed 123, while flagging 343 of 512 authentic images as forged. High authentic-image false-positive rates limit usefulness despite stronger forged-class recall.

Localization is weaker than image detection, particularly on COCOGlide. These measurements alone do not establish the cause: domain shift, edited-region size, and preprocessing/annotation differences need further investigation.

Confidence intervals resample source groups rather than treating related images as independent. They quantify dataset sampling variability, not uncertainty across model-training seeds. Qualitative examples use declared best/median/worst or failure categories and are not unbiased performance estimates. Clean results do not establish robustness under degradation. Independent dataset/model provenance must be confirmed before making generalization claims.

## Directory map

```text
results/
|-- README.md
|-- checkpoints/
|   |-- 01_spatial_baseline/best.pt
|   |-- 02_frequency_baseline/best.pt
|   |-- 03_fusion_warmup/best.pt
|   `-- 04_fusion_finetuned/best.pt
|-- training/
|   |-- 01_spatial_baseline_results/
|   |-- 02_frequency_baseline_results/
|   |-- 03_fusion_warmup_results/
|   `-- 04_fusion_finetuned_results/
|-- imd2020/
|   |-- 20261112T114049_016995Z/
|   `-- presentation/IMD2020_20261112T143751Z/
`-- cocoglide/
    `-- 20261112T144623_935983Z/
```

New training runs default to `results/training/runs/`. New notebook evaluations create UTC timestamped directories under the selected dataset folder. The submission includes each stage's `best.pt`; resume checkpoints (`last.pt`), original per-image probability archives, and older archive folders are not included here.

## Which files to open

| Goal | File or folder |
| --- | --- |
| Read the measured conclusions | `analysis_report.md` inside a recorded run |
| Inspect aggregate metrics | `summary.csv` or `summary.json` |
| Verify weights, environment, and thresholds | `run_metadata.json` |
| Inspect image decisions and pixel counts | `per_image_results.csv` |
| Inspect source-group uncertainty | `group_bootstrap_ci.csv` |
| Review failure cases | `image_detection_failures.csv`, `localization_ranked_worst_first.csv` |
| Find plots for a report or slides | Each run's `figures/` and the IMD2020 `presentation/` folder |
| Look up figure descriptions | `figure_captions.csv` |
| Review training progress | Stage JSON histories and PNGs under `training/` |

Historical reports may reference prediction files or paths from the original training/evaluation environment. Those paths preserve provenance and may not exist in this submission. Saved aggregate reports and figures remain available; regenerate probability-dependent plots with a fresh evaluation when their arrays are absent.

## Reproduce an evaluation

The current source has invalid `from data processing...` imports that must be corrected to `from data.` before execution. See the [checkout status](../README.md#current-checkout-status). The saved reports remain readable without executing Python.

1. Follow the [setup guide](../README.md#setup) and restore the selected dataset with its original structure.
2. Keep the supplied checkpoint together with `artifacts/dct_stats.json` and `artifacts/casia_clean_split.json`. External evaluation does not require CASIA images, but it does verify the training-asset identity.
3. Open the matching [evaluation notebook](../notebooks/README.md). Begin with a preview manifest to check the workflow.
4. For a full comparable run, select the full test manifest, final fusion checkpoint, `CONDITIONS = ['clean']`, and image/pixel thresholds of `0.5`. Verify mask polarity and use the notebook's image-gated policy.
5. Restart the kernel and run all cells. Review the new timestamped run's metadata, sample counts, summary, and figures before comparing results.

The [command-line guide](../README.md#command-line-evaluation) offers a lighter CSV/JSON workflow with independent pixel-head scoring. Its output format differs from notebook reports: the CLI writes `summary.json`, `curves.csv`, and per-condition CSVs, with a `complete` flag in the summary.

Exact numerical agreement can depend on software versions and hardware. The requirements are version ranges; saved metadata records the historical environment. Do not change test thresholds or refit normalization to match a reported score, and do not treat a preview result as a full test result.
