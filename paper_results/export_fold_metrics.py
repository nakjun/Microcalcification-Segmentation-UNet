"""Export public fold measurements from complete internal and external evaluations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

MODELS = {"proposed": "PatchUNet", "unext": "UNeXt", "unet2d": "U-Net 2D",
          "utnet": "UTNet", "deepmica": "DeepMiCa", "yolo": "YOLO-HLSAM"}
ENDPOINTS = {
    "csaw_test": ("positive_macro_dice", "positive_macro_iou", "pixel_recall",
                  "negative_fp_per_image", "negative_fp_image_fraction"),
    "inbreast": ("point_exact_hit_rate", "point_within1_hit_rate", "partial_contour_macro_Dice",
                 "partial_contour_macro_IoU", "mean_predicted_pixels"),
    "cbis_test": ("ROI_any_overlap_rate", "ROI_macro_coverage", "mean_predicted_pixels"),
}


def read(path):
    return json.loads(path.read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(args):
    test_ids = read(Path(__file__).with_name("splits.json"))["test_ids"]
    for model, label in MODELS.items():
        for condition in (("native",) if model in ("deepmica", "yolo") else ("C", "L", "H", "LH")):
            seed = 43 if model == "utnet" and condition in ("C", "LH") else 42
            root = args.base
            if seed == 43:
                root = args.refits / "utnet"
            elif model == "yolo":
                root = args.refits / "yolo/candidate"
            for fold in range(3):
                key = Path(f"{model}_{condition}/seed_{seed}/fold_{fold}")
                internal = root / "evaluation/internal" / key / "test"
                binding = read(internal / "binding.json")
                pipeline = binding["provenance"]
                for cohort in (ENDPOINTS if condition in ("native", "LH") else ("csaw_test",)):
                    if cohort == "csaw_test":
                        marker = internal / "complete.json"
                        done = read(marker)
                        counts = internal / "counts.json"
                        if not (done["complete"] and done["scope"] == "test" and binding["image_ids"] == test_ids
                                and done["binding_sha256"] == checksum(internal / "binding.json")
                                and done["counts_sha256"] == checksum(counts)):
                            raise ValueError(f"Invalid test export at {internal}")
                        metrics = read(counts)["summary"]
                    else:
                        external = args.external_refits if root != args.base else args.base
                        marker = external / "evaluation/external" / cohort / key / "complete.json"
                        done = read(marker)
                        if not (done["complete"] and done["pipeline"] == pipeline and not done["selection_performed"]
                                and done["images"] == {"inbreast": 410, "cbis_test": 284}[cohort]
                                and (done["dataset"], done["model"], done["variant"], done["seed"], done["fold"]) ==
                                (cohort, model, condition, seed, fold)):
                            raise ValueError(f"External evaluation mismatch at {marker}")
                        metrics = done["metrics"]
                    refit = pipeline["refit"]
                    detector = pipeline["detector_refit"] or {}
                    digest = checksum(marker)
                    for metric in ENDPOINTS[cohort]:
                        yield (label, condition, seed, fold, cohort, metric, metrics[metric], refit["selected_epoch"],
                               refit["checkpoint_sha256"], detector.get("selected_epoch", ""),
                               detector.get("checkpoint_sha256", ""), digest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("base", "refits", "external-refits"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    rows = list(export(parser.parse_args()))
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("model", "condition", "seed", "fold", "cohort", "metric", "value", "epoch",
                     "checkpoint_sha256", "detector_epoch", "detector_sha256", "completion_sha256"))
    writer.writerows(rows)
