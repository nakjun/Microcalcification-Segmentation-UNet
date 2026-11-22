# Reproducing the manuscript comparison

## Results and inputs

Each row of `fold_metrics.csv` is one measured endpoint for one fold. There are 54 internal test evaluations and 36 external evaluations. Repeated metrics from one evaluation share a completion checksum. A YOLO-HLSAM row identifies both the decoder and detector. The internal and external evaluations of each main configuration use the same checkpoints. All internal folds score the same 43 positive and 28 reference-negative images. INbreast point outcomes use 4,339 points and contour overlap uses the 233 images with contours. CBIS-DDSM region outcomes use 324 valid regions. The external scalar exports include mean predicted pixels per image.

Owners of the full evaluation artifacts can regenerate the fold CSV with `python paper_results/export_fold_metrics.py --base runs/CSAW312_Rerun_20261105 --refits runs/CSAW312_FP_Rerun_20261119 --external-refits runs/fp_external_20261122`. It writes CSV to stdout after checking completion records and checkpoint bindings.

Use Python 3.10 and the [recorded package versions](../environment_versions.json) for the original implementation. GPU runs use CUDA-compatible PyTorch and torchvision. Obtain CSAW-S from its controlled-access release, INbreast from its provider, and the CBIS-DDSM calcification test data from TCIA. Retain the original image geometry. The split file contains study image identifiers. Source-folder groups linked by identical images were kept together. [CSAW-S release](https://doi.org/10.5281/zenodo.4030660) · [INbreast publication](https://doi.org/10.1016/j.acra.2011.09.014) · [CBIS-DDSM collection](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

## Predict from a published refit

With a final study checkpoint obtained separately and a native 8-bit grayscale PNG, run from the repository root:

```sh
python scripts/csaw312_rerun/predict_image.py PatchUNet LH 0 /path/to/final.pt /path/to/image.png /tmp/mask.png
```

The command checks the checkpoint SHA-256 against the corresponding fold record, runs native 256 × 256 tiles with stride 192, and writes a binary PNG at probability 0.5. For DeepMiCa and YOLO-HLSAM use the condition `native`. YOLO-HLSAM also requires `--detector /path/to/detector.pt --sam /path/to/sam_vit_b.pth`. Use `--device cpu` for a CPU smoke test. The reported backbone results used BF16 GPU forward passes. For external DICOM data, use the study's native intensity conversion in `scripts/calcification_gate_sam_common.py` via `export_external.py`.

## Data preparation and training

Keep local material under the ignored `runs/` directory. Run commands from the repository root. The source root below contains one subdirectory per CSAW-S source group, with paired `<id>.png` and `<id>_calcifications.png` files.

```sh
python scripts/csaw312_rerun/prepare_data.py --input-manifest paper_results/csaw_inventory.csv \
  --source-root /path/to/CSAW-S --output-root runs/csaw312_rerun_20261105/data
python scripts/csaw312_rerun/configure_local.py
export CSAW312_PROTOCOL=runs/csaw312_rerun_20261105/local_protocol.json
```

Local paths contribute to the preparation signature. The configuration command compares the actual image and mask file digests and all fold assignments with the public inventory before binding a local protocol. The [protocol template](../configs/csaw312_rerun_20261105/protocol.json) records the original data signature. New run signatures and checkpoint hashes may differ.

Full training additionally requires the two upstream sampling-pipeline outputs, their source signatures and completion records, and the SAM ViT-B weights at the protocol path. The builder lists both upstream input trees in `SOURCES` and verifies their masks. With those inputs prepared, run:

```sh
python scripts/csaw312_rerun/build_initial_candidates.py
python scripts/csaw312_rerun/gpu_gate.py
```

Preview the final training sequence with `python scripts/csaw312_rerun/train_paper.py --list`. Run it with `python scripts/csaw312_rerun/train_paper.py --device cuda:0` after the GPU preflight.

The GPU preflight expects the research Python environment at `.venv/bin/python`. Model selection uses the inner validation split. Each refit starts from a fresh initialization and uses the selected epoch count on the outer training set.

| Final configuration | Training entry point |
|---|---|
| PatchUNet, UNeXt and U-Net 2D under C/L/H/LH, UTNet L/H, and DeepMiCa | `worker.py --job-index INDEX --device cuda:0`. Use the matching seed-42 selection and refit jobs from the protocol in dependency order. |
| UTNet C and LH | `fp_utnet.py --device cuda:0` runs the seed-43 configurations on all three folds. |
| YOLO-HLSAM | `fp_yolo.py --device cuda:0`. Use the candidate decoder and its paired detector. The public runner trains fold 0 locally with the zero-input redraw rule. The study run reused the previously trained fold-0 control stages. |

The follow-up runners reuse native image caches from the base training run. The full job grid also contains configurations outside the final manuscript. The fold metrics identify exactly which seed, condition and fold contributed each result.

## Whole-cohort scoring

For a base model with its local refit records, the following commands evaluate one internal fold and one external cohort. Use the same fold for both commands.

```sh
python scripts/csaw312_rerun/export_internal.py --model proposed --variant LH --seed 42 --fold 0 --split test
python scripts/csaw312_rerun/export_external.py --model proposed --variant LH --seed 42 --fold 0 --dataset inbreast
```

External evaluation requires authorized DICOM inputs and the image inventories and annotations under `runs/external_v3/{inputs,annotations}/`. See [external preparation](../scripts/external_v3/prepare.py) and [annotation parsing](../scripts/external_v3/annotations.py). Run base external folds first to populate verified native caches for INbreast and CBIS-DDSM. For UTNet LH and YOLO-HLSAM use `fp_external.py utnet cuda:0` and `fp_external.py yolo cuda:0` after their internal evaluations. Both use the final refits with fixed thresholds. External YOLO masks use detector-predicted boxes.

The public split-preparation and metric tests use synthetic inputs and run without access to mammograms:

```sh
python -m unittest discover -s scripts/csaw312_rerun -p 'test_*.py'
```
