# U-Net for Mammographic Microcalcification Segmentation: A Comparative Study

This repository contains the preprocessing, training, internal validation and test evaluation, external evaluation, and computational benchmarking code used in the study. It preserves the source code and relative paths from the original research project.

## Repository structure

| Task | Location |
|---|---|
| Image and mask preprocessing, duplicate checks, and data partitioning | `scripts/prepare_calcification_refinement.py`, `scripts/prepare_calcification_refinement_baselines.py`, `scripts/prepare_calcification_gate_data.py` |
| Shared preprocessing, sampling, losses, and SAM candidate generation | `scripts/calcification_*_common.py`, `scripts/v3_ideas/`, and their shared dependencies |
| PatchUNet, U-Net, and UTNet training and out-of-fold (OOF) evaluation | `scripts/comparative_study/` |
| UNeXt training and OOF evaluation | `scripts/improvement_unext_deterministic/` |
| DeepMiCa and YOLO-HLSAM implementations | `DeepMiCa/`, `YOLO-HLSAM/` |
| Additional-model patch preparation, GPU training, and internal evaluation | `scripts/additional_baselines/training/` |
| Internal test inference, evaluation, and statistics | `scripts/comparative_held32/` |
| External INbreast and CBIS-DDSM inference and evaluation | `scripts/external_current_models/` |
| Shared external evaluation dependencies | `scripts/external_comparative_inbreast/`, `scripts/external_v3/` |
| Inference and training-step benchmarks | `scripts/additional_baselines/training/benchmark_*.py` |
| Configuration files | `configs/` |
| Pinned upstream model sources and licenses | `external/` and model-specific `vendor_*` directories |

The condition identifiers `C`, `L`, `H`, and `LH` correspond to Base, Base + Lesion, Base + Background, and Base + Lesion + Background, respectively. The identifier `held32` is retained in filenames for the internal test code. Historical names, comments, and validation rules in the archived code have been preserved.

## Environment and required inputs

Versions of the main packages in the research environment are recorded in `environment_versions.json`. Model-specific requirements files are also included. Install CUDA builds of PyTorch and torchvision compatible with your GPU and driver. The version record is not a complete environment lockfile.

**This is an archive of the original execution code, not a standalone pipeline that runs from a fresh clone without additional artifacts.** The original code checks data hashes, fixed partitions, initial sampling pools, and prior completion records. Run commands from the repository root after preparing the required paths and records. These checks have not been bypassed.

Prepare the following separately:

- Original CSAW-S images and masks, preprocessing manifests, and native-image caches.
- Original INbreast and CBIS-DDSM images and evaluation annotations.
- Fixed folds, configuration, signatures, sampling pools, and other dependencies from the original `runs/calcification_refinement_v5_seed42/` directory.
- Original protocol and verification records referenced under `runs/`, `manifests/`, and `docs/improvement_team/`. These are excluded from this code archive.
- Trained checkpoints and pretrained SAM weights matching the source and artifact checks in the code.
- Any host-specific data paths retained in the original configuration files.

Example training commands after preparing the inputs, protocol, and required verification records:

```bash
STUDY_BACKBONE=unet2d python scripts/comparative_study/train.py --mode train --variant H --fold 0 --gpu 0
python scripts/improvement_unext_deterministic/train.py --mode train --variant LH --fold 0 --gpu 0
```

Running with new inputs requires consistent regeneration of immutable protocols and source hashes; existing completion records must not simply be edited. No training or inference was performed while creating this archive.

## Provenance and exclusions

`SOURCE_MANIFEST.json` records the original relative path, SHA-256 checksum, and inclusion reason for each copied file. Source files were checked for byte-for-byte agreement when the archive was created. The archive includes shared code needed for static imports and dynamic loading, model tests and verification scripts, and upstream licenses. Modules with historical names are retained where they are dependencies of the study workflows. Subsequent documentation changes are tracked in Git; the manifest remains a record of the original import.

Raw data, patient images, checkpoints, prediction masks, experimental results, logs, manuscript files and PDFs, environment directories, and the parent project's Git history are excluded. Upstream Git histories are also excluded. Third-party code retains its original copyright and license terms. No blanket license is assigned to the combined repository.

## Shell workflows

See [workflows/README.md](workflows/README.md) for data acquisition, path registration, loading original runtime records, preprocessing, training, evaluation, and tmux commands. The guide distinguishes command previews from actual execution. Data and runtime records are not included in the repository.
