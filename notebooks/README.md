# Evaluation notebook run guide

[Project setup and training](../README.md) | [Recorded results](../results/README.md)

These notebooks run the trained models on external datasets, audit image/mask pairs, compute image detection and pixel localization metrics, and export reports and PNG figures.

| Notebook | Dataset | Pairing |
| --- | --- | --- |
| [IMD2020_Local_Evaluation.ipynb](IMD2020_Local_Evaluation.ipynb) | IMD2020 authentic images and manipulations | Source-group folders with `*_orig` images and matching `*_mask` files |
| [COCOGlide_Local_Evaluation.ipynb](COCOGlide_Local_Evaluation.ipynb) | COCOGlide real/fake pairs | The dataset's `table.csv` and `real/`, `fake/`, `mask/` directories |

## Current execution blocker

The current Python source contains invalid `from data processing...` imports, which prevent the notebook's evaluation/model modules from loading. Correct that import prefix to `from data.` in the affected Python files before running. See the [checkout status](../README.md#current-checkout-status). The instructions below assume this source issue has been repaired.

## Before running

Complete the [environment setup](../README.md#setup). Restore the selected dataset under `datasets/imd2020/` or `datasets/cocoglide/`, and confirm that these files are present:

- `artifacts/dct_stats.json`
- `artifacts/casia_clean_split.json`
- `results/checkpoints/04_fusion_finetuned/best.pt` (default model)

External evaluation does not require CASIA images. The original training statistics and audit are required, even for the external datasets, because the notebooks verify them against checkpoint identity.

## Start JupyterLab

Run from the project root:

```bash
python -m jupyterlab
```

Select the kernel belonging to the environment where `requirements.txt` was installed. If it is not listed, register it from that environment and select **Forgery detection** in Jupyter:

```bash
python -m ipykernel install --user --name forgery-detection --display-name "Forgery detection"
```

The notebooks search the current directory and its parents for the project. If the kernel starts elsewhere, set `PROJECT_OVERRIDE` in the first configuration cell to the absolute project path, for example `Path('G:/My Drive/vc_project/submission/vc_project')`.

## First run: use a preview

1. Open the notebook for the dataset you restored.
2. In the configuration cell, change `MANIFEST` to the dataset's preview CSV:

   ```python
   # IMD2020 notebook
   MANIFEST = PROJECT / 'evaluation/manifests/imd2020_preview.csv'
   ```

   ```python
   # COCOGlide notebook
   MANIFEST = PROJECT / 'evaluation/manifests/cocoglide_preview.csv'
   ```

3. Keep only the default fusion model and `CONDITIONS = ['clean']`. Choose `DEVICE_REQUEST = 'auto'`, `'cuda'`, or `'cpu'`.
4. Run cells in order. The setup regenerates test/preview manifests from the restored dataset; the selected manifest controls which images are evaluated.
5. Inspect the dataset profile and mask-audit figures. Confirm that masks align with images and that white denotes the forged foreground after normalization.
6. Review the output path printed by the notebook and inspect `summary.csv` and `figures/`.

Preview manifests select five source groups by default. They are subsets for checking the workflow, not independent validation sets or full-dataset performance estimates. Dataset enumeration still checks the restored dataset layout.

## Full evaluation

Restore `MANIFEST` to `imd2020_test.csv` or `cocoglide_test.csv`, review the configuration below, restart the kernel, and run all cells from top to bottom. Restart and rerun after changing models, masks, thresholds, or conditions to keep the outputs consistent.

| Setting | Default | Purpose |
| --- | --- | --- |
| `PROJECT_OVERRIDE` | `None` | Set an explicit project directory if discovery fails |
| `DATA_ROOT` | Dataset folder under `datasets/` | Root for manifest-relative image/mask paths |
| `MANIFEST` | Dataset's `*_test.csv` | Full-test or preview sample selection |
| `STATS_PATH`, `AUDIT_PATH` | Files under `artifacts/` | Original training assets; must match weights |
| `MODEL_SPECS` | Final fusion model | Mapping of display name to `(mode, checkpoint_folder)` |
| `DEVICE_REQUEST` | `'auto'` | Uses CUDA when available; supports explicit `'cpu'`/`'cuda'` |
| `IMAGE_THRESHOLD` | `0.5` | Forged-image decision and mask gating |
| `PIXEL_THRESHOLD` | `0.5` | Final binary pixel mask; notebook requires a value greater than zero |
| `MASK_FOREGROUND` | `'white'` | Change to `'black'` only if required by the annotation convention |
| `CONDITIONS` | `['clean']` | Evaluation conditions; must include `clean` |
| `SEED` | `42` | Randomness and deterministic degradation generation |
| `BOOTSTRAPS` | `1000` | Source-group bootstrap replicates |
| `EXAMPLES_PER_CATEGORY` | `2` | Examples in best/median/worst and authentic false-positive galleries |
| `SAVE_ALL_PREDICTIONS` | `True` | Saves compressed probabilities and binary PNG masks |

### Compare model stages

Uncomment the corresponding entries in `MODEL_SPECS` to compare the supplied spatial baseline, frequency baseline, fusion warmup, and final fusion checkpoint. Folder names are resolved under `results/checkpoints/`, with `best.pt` loaded from each folder. Modes must match their checkpoints:

| Mode | Supplied checkpoint folder |
| --- | --- |
| `spatial_only` | `01_spatial_baseline` |
| `frequency_only` | `02_frequency_baseline` |
| `fusion` | `03_fusion_warmup` or `04_fusion_finetuned` |

For a newly trained model, place its `best.pt` in a distinct folder such as `results/checkpoints/my_fusion_run/`, then add `'My fusion run': ('fusion', 'my_fusion_run')` to `MODEL_SPECS`. Keep its matching training artifacts. Do not replace the supplied checkpoints when comparing with recorded results.

### Add degradation conditions

For the five views used in validation:

```python
CONDITIONS = ['clean', 'jpeg70', 'scale05', 'blur1', 'noise5']
```

Available names are defined in [evaluation/degradations.py](../evaluation/degradations.py). Each condition starts from the original RGB image; masks remain on the native pixel grid. Additional models and conditions increase runtime and storage. Report robustness only for conditions actually evaluated.

## Mask policy and metrics

The notebooks use **image-gated masks**. If `P(forged) < IMAGE_THRESHOLD`, final probabilities are zeroed and the binary mask is empty. Otherwise the pixel probabilities are thresholded normally. A forged image classified as authentic therefore receives zero foreground Dice/IoU.

Image detection and pixel localization are reported separately. Pixel Dice/F1 and IoU are macro averages over forged images; authentic images contribute false-positive mask area. Native-resolution inference excludes padded pixels. Source-group bootstrap intervals describe dataset sampling variability, not variability across training seeds.

The evaluation CLI scores independent pixel-head outputs without image gating. The saved original IMD2020 run also predates the current notebook gate. See the [results guide](../results/README.md#evaluation-protocol) before comparing those values with a new notebook run. Keep thresholds and normalization fixed before testing; sensitivity curves do not justify fitting thresholds on the test set.

## Output files

Each execution of the configuration cell creates a new UTC timestamped run directory:

```text
results/<imd2020-or-cocoglide>/<UTC-timestamp>/
|-- run_metadata.json
|-- test_manifest.csv
|-- dataset_audit.csv
|-- summary.json
|-- summary.csv
|-- per_image_results.csv
|-- group_bootstrap_ci.csv
|-- analysis_report.md
|-- figure_captions.csv
|-- figures/
`-- predictions/
```

| Output | Contents |
| --- | --- |
| `run_metadata.json` | Model/checkpoint hashes, asset hashes, environment, thresholds, and mask policy |
| `test_manifest.csv`, `dataset_audit.csv` | Evaluated samples and audited image/mask properties |
| `summary.json`, `summary.csv` | Metrics by model and condition |
| `per_image_results.csv` | Image probabilities, pixel confusion counts, overlap, timing, and groups |
| `group_bootstrap_ci.csv`, `micro_localization.csv` | Source-group intervals and pooled forged-image overlap |
| `image_detection_failures.csv`, `localization_ranked_worst_first.csv` | Failure analysis tables |
| `analysis_report.md`, `figure_captions.csv` | Measured report and figure descriptions |
| `figures/*.png` | 300-DPI plots and qualitative examples for reports/slides |
| `predictions/model_XX/<condition>/` | Per-image CSV and, when enabled, `.npz` probabilities and `*_mask.png` files |

Additional outputs include calibration, threshold sensitivity, region-size analysis, and pooled pixel curves. COCOGlide also saves `source_table.csv` and prompt-category analysis. Robustness tables/figures require multiple conditions; galleries and plots that consume probability files require `SAVE_ALL_PREDICTIONS = True`.

Native-resolution inference can require substantial memory and time. Set `SAVE_ALL_PREDICTIONS = False` to reduce disk use; core scalar metrics remain available, while saved-probability galleries/plots are skipped. Start with one model and a clean preview run when checking the environment.

Notebook outputs and transient errors were cleared for submission. Existing measurements are retained separately under `results/`; rerunning creates new reports rather than replacing the recorded runs. Some archived prediction arrays are absent from this submission, so plots requiring them must be regenerated from a fresh evaluation.
