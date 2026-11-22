"""Run a published fold on one native uint8 grayscale image."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import backbone
from common import ROOT, sha
from export_internal import predict
import specialized


MODELS = {"PatchUNet": "proposed", "UNeXt": "unext", "U-Net 2D": "unet2d",
          "UTNet": "utnet", "DeepMiCa": "deepmica", "YOLO-HLSAM": "yolo"}


def weights(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Checkpoint checksum mismatch for {path}")
    return torch.load(path, weights_only=True, map_location="cpu")["model"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("condition", choices=["C", "L", "H", "LH", "native"])
    parser.add_argument("fold", choices=range(3), type=int)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New binary PNG mask")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--sam", type=Path)
    args = parser.parse_args()
    with (ROOT / "paper_results/fold_metrics.csv").open(newline="") as stream:
        record = next((row for row in csv.DictReader(stream)
                       if (row["model"], row["condition"], row["fold"], row["metric"]) ==
                       (args.model, args.condition, str(args.fold), "positive_macro_dice")), None)
    if record is None:
        parser.error("No published checkpoint for this model and condition.")
    name = MODELS[args.model]
    if name == "yolo" and (args.sam is None or args.detector is None):
        parser.error("YOLO-HLSAM requires --sam and --detector.")
    with Image.open(args.image) as source:
        image = np.asarray(source)
    if image.ndim != 2 or image.dtype != np.uint8:
        parser.error("Supply a native 8-bit grayscale image.")
    device = torch.device(args.device)
    detector = None
    state = weights(args.checkpoint, record["checkpoint_sha256"])
    if name == "yolo":
        if sha(args.sam) != (ROOT / "paper_results/sam_vit_b.sha256").read_text().strip():
            raise ValueError("Unexpected SAM ViT-B weights")
        model, _, _ = specialized.build("segmentor", device, args.sam)
        model.decoder.load_state_dict(state)
        detector, _, _ = specialized.build("detector", device)
        detector.load_state_dict(weights(args.detector, record["detector_sha256"]))
        detector.eval()
    elif name == "deepmica":
        model, _, _ = specialized.build("deepmica", device)
        model.load_state_dict(state)
    else:
        model, _ = backbone.build(name, device)
        model.load_state_dict(state)
    pipeline = dict(model_name=name, model=model.eval(), detector=detector, device=device)
    with torch.inference_mode():
        mask = predict(pipeline, image) >= 0.5
    with args.output.open("xb") as stream:
        Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")


if __name__ == "__main__":
    main()
