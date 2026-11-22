# Comparative Evaluation of Deep Learning Models for Mammographic Calcification Segmentation

This repository accompanies the manuscript by Nak-Jun Sung and Jun Ma. It contains the training and evaluation code, fixed CSAW-S splits, and fold-level measurements for the final comparison. The internal study uses 312 mammograms, with 241 for development and 71 for testing. Frozen checkpoints were evaluated on 410 INbreast and 284 CBIS-DDSM images.

## Reproduce the result tables

From the repository root, run the following command with Python 3.10 or later. The calculation uses the standard library and the supplied fold measurements.

```sh
python paper_results/summarize.py > /tmp/microcalcification-summary.csv
diff -u paper_results/summary.csv /tmp/microcalcification-summary.csv
```

The [fold measurements](paper_results/fold_metrics.csv) identify the model, loss condition, training seed, fold, evaluation cohort, selected epoch, and checkpoint checksum. The [summary](paper_results/summary.csv) reports the mean and sample standard deviation across the three models evaluated on the same images. External predicted-pixel burden is included for the Discussion.

| Comparison | Models and conditions | Evaluation |
|---|---|---|
| Internal | PatchUNet, UNeXt, U-Net 2D and UTNet under C, L, H and LH, plus DeepMiCa and YOLO-HLSAM | Positive-image macro Dice and IoU, pooled pixel recall, mean FP pixels per negative image, and fraction of negative images marked |
| External | Four LH backbones and the two specialized models | Exact and ±1-pixel point hits and partial-contour overlap on INbreast. Region hits and mean region coverage on CBIS-DDSM |

C combines weighted BCE with Dice on positive patches. L adds the lesion term to C, H adds the hard-background term to C, and LH adds both. UTNet C and LH use seed 43. UTNet L and H and the remaining reported configurations use seed 42. The YOLO-HLSAM adaptation uses predicted boxes as decoder prompts and includes a penalty on the highest 1% of prompt-target-negative logits.

## Use the code

- [Reproduction guide](paper_results/README.md) covers the environment, authorized inputs, data preparation, training, inference, and external evaluation.
- [Final study implementation](scripts/csaw312_rerun/) includes the native-image sampler, model training, specialist adapters and evaluation code.
- [CSAW-S inventory](paper_results/csaw_inventory.csv), [image and mask checksums](paper_results/csaw_file_hashes.csv) and [fixed splits](paper_results/splits.json) identify the cases and the inner and outer partitions for each fold.
- [Training protocol](configs/csaw312_rerun_20261105/protocol.json) records optimization, epoch selection, inference and the original job grid. The release guide maps the reported configurations to their training entry points.
- [Source manifest](SOURCE_MANIFEST.json) records the original import and the checksums of the final release files.
- [Environment versions](environment_versions.json) and the requirements in [DeepMiCa](DeepMiCa/) and [YOLO-HLSAM](YOLO-HLSAM/) describe the software used. Third-party code retains its upstream license.

Image data, private annotations, prediction masks, candidate masks, and model weights must be obtained or generated separately. The earlier shell workflows and `comparative_held32` directory document a prior study split. For the manuscript reported here, start with the reproduction guide above.
