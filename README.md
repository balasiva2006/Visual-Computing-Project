# Image Forgery Detection and Localization

A PyTorch research project for detecting manipulated images and locating forged pixels. It combines an HRNet-W18 RGB encoder with DCT/SRM frequency features, multi-scale fusion, a decoder, and image/pixel prediction heads. Spatial-only and frequency-only baselines are included.

Training uses audited CASIA splits. External evaluation covers IMD2020 and COCOGlide. This directory includes four trained checkpoints, training histories, evaluation reports, and PNG figures.

## Navigation

- [Setup](#setup)
- [Datasets and checkpoints](#datasets-and-checkpoints)
- [Quick run: notebook evaluation](#quick-run-notebook-evaluation)
- [Command-line evaluation](#command-line-evaluation)
- [Training guide](#training-guide)
- [Troubleshooting](#troubleshooting)
- [Detailed notebook guide](notebooks/README.md)
- [Recorded results and interpretation](results/README.md)

## Project structure

```text
vc_project/
|-- requirements.txt
|-- configs/          YAML training configurations
|-- data/             CASIA loading, audits, transforms, DCT/SRM features
|-- models/           Encoders, fusion, decoder, and prediction heads
|-- training/         Trainer, losses, checkpoints, optional distillation
|-- evaluation/       Evaluation CLI, metrics, degradations, manifest builders
|   `-- manifests/    Full-test and preview CSVs
|-- notebooks/        IMD2020 and COCOGlide evaluation notebooks
|-- artifacts/        Original DCT statistics and CASIA split audit
|-- datasets/         Local dataset files; excluded from version control
`-- results/          Checkpoints, training histories, reports, and PNG figures
```

## Current checkout status

The source currently contains invalid imports such as `from data processing.dataset import CASIADataset`. Python package names cannot contain spaces; this checkout's package is `data/`. These syntax errors block training, evaluation, and the notebook imports, including the smoke check below.

Before running, correct the affected Python imports from `from data processing.` to `from data.`. The commands in this guide describe the implemented entry points and assume those imports have been repaired. This documentation update does not change source code or recorded model outputs.

## Setup

Run commands from this project directory, where `requirements.txt` is located. Use Python 3.10 or newer with a compatible PyTorch/torchvision build. A CUDA GPU is recommended for training and full native-resolution evaluation; CPU execution is supported but can be slow.

### 1. Create and activate an environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` below.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install the appropriate PyTorch/torchvision build for your system before the remaining requirements. Dependencies include timm, NumPy, SciPy, Pillow, pandas, Matplotlib, scikit-learn, JupyterLab, and ipykernel. `requirements.txt` specifies version ranges rather than an exact reproduction environment.

### 3. Check the environment

```bash
python -c "import torch, torchvision, timm; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m evaluation.evaluate --smoke --device cpu
```

The evaluation smoke check uses synthetic images and temporary frequency-model weights. It checks inference, five evaluation views, checkpoint loading, and CSV/JSON output without requiring datasets or supplied checkpoints. It does not measure trained-model accuracy.

## Datasets and checkpoints

Supply dataset images separately when using a code-only submission. Preserve the extracted filenames and directory structure:

```text
datasets/
|-- casia/
|   |-- Au/
|   |-- Tp/
|   |-- CASIA 2 Groundtruth/
|   `-- csv files/
|       |-- train.csv
|       `-- val.csv
|-- imd2020/
|   `-- <source-group folders containing *_orig images, edits, and *_mask files>
`-- cocoglide/
    |-- table.csv
    |-- real/
    |-- fake/
    `-- mask/
```

CASIA CSVs require `tp,gt,label,type,category,group,split` columns with dataset-relative paths. Label `0` means authentic and has an empty mask path; label `1` means forged and requires a nonempty foreground mask. CASIA data and CSV bytes must match the supplied audit to reproduce training with the included checkpoint identity.

External evaluation needs only the chosen external dataset, a manifest, a checkpoint, and the two training artifacts below. CASIA images are not required for external evaluation.

| Asset | Location |
| --- | --- |
| Training DCT statistics | `artifacts/dct_stats.json` |
| Training split/pixel audit | `artifacts/casia_clean_split.json` |
| Spatial baseline | `results/checkpoints/01_spatial_baseline/best.pt` |
| Frequency baseline | `results/checkpoints/02_frequency_baseline/best.pt` |
| Fusion warmup | `results/checkpoints/03_fusion_warmup/best.pt` |
| Final fusion model (evaluation default) | `results/checkpoints/04_fusion_finetuned/best.pt` |

Keep statistics, audit, and checkpoints together: loading checks their training-data identity. Do not fit normalization statistics on external test datasets. Checkpoint `.pt` files are ignored by Git in this layout, so a fresh clone may require restoring them separately.

Rebuild external manifests after restoring the datasets:

```bash
python -m evaluation.prepare_imd2020 --root datasets/imd2020
python -m evaluation.prepare_cocoglide --root datasets/cocoglide
```

These commands regenerate the corresponding `*_test.csv` and `*_preview.csv` files in `evaluation/manifests/`. Each preview selects five source groups by default. COCOGlide pairing comes from `table.csv`; IMD2020 pairs each edited image with its matching `*_mask` file.

## Quick run: notebook evaluation

1. Complete setup and restore the selected dataset and model assets.
2. Start JupyterLab from the project root:

   ```bash
   python -m jupyterlab
   ```

3. Open [IMD2020_Local_Evaluation.ipynb](notebooks/IMD2020_Local_Evaluation.ipynb) or [COCOGlide_Local_Evaluation.ipynb](notebooks/COCOGlide_Local_Evaluation.ipynb) with the environment where dependencies were installed.
4. Review the first configuration cell. For an initial workflow check, change `MANIFEST` from `*_test.csv` to the matching `*_preview.csv`. Keep `CONDITIONS = ['clean']`; select `DEVICE_REQUEST = 'auto'`, `'cuda'`, or `'cpu'`.
5. Run cells from top to bottom. Inspect mask alignment and polarity before interpreting localization scores.
6. Restore the full `*_test.csv` manifest, restart the kernel, and run all cells for a full evaluation.

Each run creates a UTC timestamped folder under `results/imd2020/` or `results/cocoglide/`, with reports and 300-DPI PNG figures. The default `SAVE_ALL_PREDICTIONS = True` also saves probability arrays and binary masks. See the [notebook guide](notebooks/README.md) for controls and output filenames.

The notebooks use **image-gated localization**: an image probability below the image threshold forces an empty final mask. Image and pixel thresholds default to `0.5`. Images retain native resolution, and padded pixels are excluded from metrics.

## Command-line evaluation

The CLI provides scalar reports without the notebook plotting workflow. Its pixel metrics score the **independent pixel head**; it does not apply the notebook's image gate. Use the notebooks for the current gated presentation protocol.

For a clean IMD2020 preview run:

```bash
python -m evaluation.evaluate --root datasets/imd2020 --manifest evaluation/manifests/imd2020_preview.csv --dataset-name imd2020 --checkpoint results/checkpoints/04_fusion_finetuned/best.pt --mode fusion --protocol clean --device cpu --out results/imd2020/cli_preview
```

For COCOGlide:

```bash
python -m evaluation.evaluate --root datasets/cocoglide --manifest evaluation/manifests/cocoglide_preview.csv --dataset-name cocoglide --checkpoint results/checkpoints/04_fusion_finetuned/best.pt --mode fusion --protocol clean --device cpu --out results/cocoglide/cli_preview
```

Replace `preview` with `test` in the manifest filename for full evaluation. Use a new `--out` directory for every run. Set `--device cuda` when available.

| Option | Behavior |
| --- | --- |
| `--mode` | `fusion`, `spatial_only`, or `frequency_only`; must match the checkpoint |
| `--protocol clean` | Original images only; default |
| `--protocol validation` | `clean`, `jpeg70`, `scale05`, `blur1`, `noise5` |
| `--protocol sweep` | 18 conditions covering clean, JPEG, scaling, blur, and noise |
| `--conditions clean jpeg70` | Explicit conditions, overriding the protocol selection |
| `--image-threshold`, `--pixel-threshold` | Frozen operating points; both default to `0.5` |
| `--stats`, `--audit` | Override the default training-artifact paths |
| `--mask-foreground` | `white` by default; use `black` only when required by external annotations |

Omitting `--manifest` evaluates the audited CASIA validation split under `--root`. CLI outputs are `summary.json`, `curves.csv`, and one `<condition>.csv` per condition. `summary.json` records hashes, settings, versions, and a `complete` flag; partial runs are not completed results.

## Training guide

Training requires prepared CASIA data and matching artifacts. Supplied YAML configurations request CUDA, seed `42`, a `512`-pixel crop, microbatch size `4`, two workers, and mixed precision. CLI options override YAML defaults.

First check the training path with restored CASIA data:

```bash
python -m training.trainer --smoke --device cpu
```

Unlike the evaluation smoke check, this check requires CASIA data and training statistics. It runs a temporary one-epoch frequency model on a tiny subset and reloads its checkpoint.

### Four-stage pipeline

The following commands pass newly trained parent checkpoints to each subsequent stage. Output directories must be empty or absent on fresh runs.

```bash
python -m training.trainer --config configs/spatial.yaml
python -m training.trainer --config configs/frequency.yaml
python -m training.trainer --config configs/fusion_warmup.yaml --spatial-ckpt results/training/runs/spatial_seed42/best.pt --frequency-ckpt results/training/runs/frequency_seed42/best.pt
python -m training.trainer --config configs/fusion_joint.yaml --warmup-ckpt results/training/runs/fusion_warmup_seed42/best.pt --spatial-last-module
```

Spatial and frequency stages train individual streams. Warmup loads and freezes those encoders while training fusion/decoder/heads. Joint tuning starts from warmup. The final command uses `--spatial-last-module` to freeze the HRNet backbone except its final stage-4 module, matching the supplied final run's partial fine-tuning strategy. Omit the flag for broader encoder fine-tuning. Fresh spatial training can download ImageNet-pretrained HRNet weights through timm.

The warmup/joint YAML files point to **supplied** checkpoints by default; the overrides above continue from your new runs. New weights stay under `results/training/runs/`. Update notebook checkpoint selection explicitly when evaluating them.

### Optional teacher distillation

`configs/fusion_teacher.yaml` is an alternative to joint tuning, using frozen spatial/frequency teachers and `beta = 0.05`. Use the same spatial/frequency parents that produced warmup:

```bash
python -m training.trainer --config configs/fusion_teacher.yaml --spatial-ckpt results/training/runs/spatial_seed42/best.pt --frequency-ckpt results/training/runs/frequency_seed42/best.pt --warmup-ckpt results/training/runs/fusion_warmup_seed42/best.pt
```

### Resume and device overrides

```bash
python -m training.trainer --config configs/spatial.yaml --resume
python -m training.trainer --config configs/spatial.yaml --device cpu --workers 0 --batch-size 1 --out results/training/runs/spatial_cpu
```

Resume requires both `last.pt` and `best.pt` plus matching data/configuration. Repeat the original parent paths and flags when resuming later stages. Select a new `--out` for fresh runs.

Microbatch sizes may be `1`, `2`, `4`, or `8`; gradients accumulate toward eight images per update. Training crops must be positive multiples of `32`. Validation uses one native-resolution image at a time across five views, so reducing training batch size may not resolve validation memory limits.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named data/models/evaluation` | Run `python -m ...` from the project root and use the correct notebook kernel. |
| Notebook cannot locate the project | Set `PROJECT_OVERRIDE` to the absolute project directory in its configuration. |
| CUDA requested but unavailable | Use `--device cpu` or `DEVICE_REQUEST = 'cpu'`; check the installed PyTorch build for GPU use. |
| Missing image, mask, or `table.csv` | Restore the expected dataset layout and regenerate manifests. |
| Checkpoint mode or training-asset identity mismatch | Select the correct mode and original checkpoint/statistics/audit combination. |
| Output directory already exists | Choose a new `--out`; use `--resume` only for a matching training run. |
| GPU memory exhausted or evaluation is slow | Begin with one model, a preview manifest, and clean images; try CPU for oversized native-resolution inputs. |
| Prediction exports use too much disk | Set `SAVE_ALL_PREDICTIONS = False`; galleries/plots requiring saved probabilities are then skipped. |
| DataLoader worker issues | Try training with `--workers 0`. |

## Results and reproducibility

See [results/README.md](results/README.md) for recorded metrics, source-group counts, report locations, and historical versus current mask policies. Preview subsets are workflow checks, not independent validation sets. Keep thresholds and normalization fixed before external testing.

Historical metadata may contain paths from the original training environment. Use this guide's current paths to run the project and retain historical paths as provenance. This directory does not include a submission-archive builder or a `scripts/` directory.

## Dataset references

- [IMD2020 dataset](https://staff.utia.cas.cz/novozada/db/)
- [COCOGlide / TruFor](https://github.com/grip-unina/TruFor#cocoglide-dataset)

Dataset images and annotations remain subject to their original terms.
