# Data preparation and original training protocol

This directory provides scripts only. Data, protocol snapshots, weights, and results are not included in Git. No downloads, training, or inference were performed when these scripts were added.

## 1. Obtain the data

Distribution information was checked on September 28, 2026.

- **CSAW-S:** https://zenodo.org/records/4030660 — file access is restricted. Request access through the official page and obtain the authorized original files.
- **INbreast:** use the original DICOM/XML release obtained from the data provider. The scripts do not automatically download unverified public mirrors.
- **CBIS-DDSM:** https://www.cancerimagingarchive.net/collection/cbis-ddsm/ — follow the official page and NBIA download procedure. The study uses full Calc-Test images and the corresponding ROI annotations for external evaluation. Do not substitute converted JPEG datasets.

If you have an authorized direct HTTPS download URL and a trusted SHA-256 checksum:

```bash
# The URL may contain an access token; do not store it in configuration files or Git.
export DATA_URL='https://provider.example/authorized-archive'
export DATA_SHA256='REPLACE_WITH_THE_ACTUAL_64_CHARACTER_SHA256'
bash workflows/download_data.sh csaw  # Also accepts inbreast or cbis.
unset DATA_URL DATA_SHA256
```

`download_data.sh` performs a download when invoked with the required environment variables. It stores the archive under `datasets/downloads/` and marks it complete only after checksum verification. Extract it separately according to its archive format. Existing archives are not overwritten. Without a URL, the script prints acquisition instructions and exits.

## 2. Register data paths and load original runtime records

The following scripts **print commands by default**. Supply `--execute` as the first argument to perform the operations.

```bash
bash workflows/load_data.sh csaw /data/CSAW-S
bash workflows/load_data.sh inbreast /data/INbreast
bash workflows/load_data.sh cbis /data/CBIS-DDSM
# Execute: bash workflows/load_data.sh --execute csaw /data/CSAW-S
```

This registers extracted directories as symbolic links under `datasets/`. The Python loaders still use paths stored in the original manifests. Registration alone does not create training manifests or partitions, or automatically rewrite absolute paths.

To reuse original runtime records from an existing research environment:

```bash
bash workflows/load_data.sh artifacts /path/to/original/project
```

With `--execute`, this makes **independent copies** of the original `runs`, `reports`, `manifests`, and `docs` directories. Mutable run directories are not linked to the source project. This may be a large copy; review the printed paths and available disk space first. The script stops if a destination directory exists rather than merging or overwriting it. Recorded absolute paths and source hashes remain unchanged and must pass the original checks. This operation does not generate a fresh protocol for a new dataset.

## 3. Preprocessing and loading

```bash
bash workflows/preprocess.sh native      # Audit images/masks, create caches and partitions from original manifests.
bash workflows/preprocess.sh baseline    # Prepare dependencies for baseline candidates and masks.
bash workflows/preprocess.sh additional  # Prepare fixed patches and manifests for the additional models.
bash workflows/preprocess.sh test        # Prepare the internal test protocol.
bash workflows/preprocess.sh external    # Bind existing external cohort manifests to the fixed protocol.
```

Each stage calls an existing Python entry point and requires its original inputs. Downloading raw data alone does not satisfy every historical dependency. The wrappers do not reselect the 124-image development set, 32-image test set, or seed 42. Generating external annotation audit JSON files is outside these wrappers; the original input records are required.

## 4. Train the four backbones

```bash
export GPU_INDEX=0
# Models: proposed (PatchUNet), unet2d, utnet, unext.
bash workflows/train.sh unet2d prepare
bash workflows/train.sh unet2d smoke
bash workflows/train.sh unet2d warmup
bash workflows/train.sh unet2d train
bash workflows/train.sh unet2d infer
bash workflows/train.sh unet2d evaluate
```

- Seed 42; folds 0–2; four conditions: C = Base, L = Base + Lesion, H = Base + Background, and LH = Base + Lesion + Background.
- Sixteen epochs, 2,048 patches per epoch, batch size 16, and the original deterministic/BF16 settings.
- Warm-up runs for three epochs across all 12 condition–fold combinations. `verify_warmup.py` checks the shared initial state before training proceeds to epoch 16 through the original implementation.
- The original `preflight.json` and warm-up verification records are required. A smoke run alone is not treated as independent preflight approval. The wrappers neither fabricate approval records nor bypass hash checks.
- Evaluation outputs also contain historical policies. Use `ungated_p5` for the manuscript's OOF results, keeping them separate from internal test and external results.

## 5. Train the additional models

```bash
bash workflows/train.sh deepmica train
bash workflows/train.sh yolo train
```

These commands call the original worker. Its device is fixed to **cuda:1** and is not changed by `GPU_INDEX`. It requires `.venv/bin/python`, the protocol, prepared patches, and model-specific GPU preflight records. DeepMiCa uses 16 epochs, batch size 16, and SGD with learning rate 0.01. YOLO-HLSAM uses 16 epochs for each detector and segmentor stage, with seed 42. SAM weights are expected at `runs/mammo_hypothesis/weights/sam_vit_b.pth`.

The worker performs internal inference and evaluation after training. It fails if its output directory already exists; do not launch duplicate jobs or delete existing results to restart it.

## 6. Evaluation and tmux execution

```bash
bash workflows/evaluate.sh test
bash workflows/evaluate.sh deepmica
bash workflows/evaluate.sh yolo
DEVICE=cuda:1 bash workflows/evaluate.sh external

# These tmux commands execute jobs; they are not previews.
bash workflows/tmux_run.sh unet_training train unet2d train
bash workflows/tmux_run.sh external_validation evaluate external
```

Complete preparation, smoke checks, preflight review, and warm-up before GPU training. All wrappers stop on failure. They preserve the original data and protocol checks and do not introduce new checkpoint or threshold selection.
