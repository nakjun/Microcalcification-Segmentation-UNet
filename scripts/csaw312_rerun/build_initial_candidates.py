"""Verify and index existing image-only SAM candidates for all 312 native inputs.

Reads historical outputs without changing them. Does not read GT masks or run a
model. Empty gate-rejected/no-box masks are recorded as existing empty results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_data import canonical, sha_file

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ("legacy_development", "runs/calcification_refinement_v1/baseline_development",
     "inputs/development_manifest.json", "runs/calcification_refinement_v1/baseline_development/config.json"),
    ("legacy_reserved", "runs/calcification_gate_retinanet_sam_clean_v1",
     "inputs/csaw_manifest.json", "configs/calcification_gate_sam.json"),
]
BITS = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hash(path, expected):
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"SHA mismatch: {path}")
    return actual


def verify_mask(path, case, shape, threshold):
    verify_hash(path, case["mask_sha256"])
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive["shape"].tolist()) != shape:
            raise ValueError(f"Native geometry mismatch: {path}")
        mask, fill, boxes = archive["mask"].copy(), archive["box_fill"], archive["boxes"]
        size = int(np.prod(shape))
        if mask.dtype != np.uint8 or fill.dtype != np.uint8 or mask.shape != ((size + 7) // 8,) or fill.shape != mask.shape:
            raise ValueError(f"Invalid packed native mask: {path}")
        if np.any(mask & ~fill):
            raise ValueError(f"SAM mask outside predicted box fill: {path}")
        if size % 8 and (int(mask[-1]) | int(fill[-1])) & ((1 << (8 - size % 8)) - 1):
            raise ValueError(f"Nonzero packed padding bits: {path}")
        if int(BITS[mask].sum()) != case["predicted_pixels"] or int(BITS[fill].sum()) != case["box_fill_pixels"]:
            raise ValueError(f"Prediction count mismatch: {path}")
        predicted = np.asarray(case["prediction_boxes"], dtype=np.float64).reshape(-1, 5)
        expected = predicted if case["gate_pass"] else predicted[:0]
        if boxes.shape != expected.shape or not np.array_equal(boxes, expected):
            raise ValueError(f"Predicted-box prompt provenance mismatch: {path}")
        if not np.isfinite(predicted).all() or (len(predicted) and (
                np.any(predicted[:, 4] < threshold) or np.any(predicted[:, :4] < 0) or
                np.any(predicted[:, [0, 2]] > shape[1]) or np.any(predicted[:, [1, 3]] > shape[0]) or
                np.any(predicted[:, 2] <= predicted[:, 0]) or np.any(predicted[:, 3] <= predicted[:, 1]))):
            raise ValueError(f"Invalid detector predictions: {path}")
        if case["selected_box_count"] != len(predicted) or case["sam_calls"] != len(expected):
            raise ValueError(f"SAM call/box count mismatch: {path}")
        if not case["gate_pass"] and (mask.any() or fill.any()):
            raise ValueError(f"Rejected gate has nonempty candidates: {path}")
        return mask


def build(data_root, output=None):
    data_root = Path(data_root)
    manifest = read_json(data_root / "manifest.json")
    data_marker = read_json(data_root / "complete.json")
    verify_hash(data_root / "manifest.json", data_marker["file_sha256"]["manifest.json"])
    if not data_marker["technical_complete"] or manifest["signature"] != data_marker["signature"]:
        raise ValueError("Data audit incomplete")
    current = {row["image_id"]: row for row in manifest["images"]}
    images, sources = {}, []
    common_recipe = None
    for name, relative_root, manifest_name, config_name in SOURCES:
        source = ROOT / relative_root
        complete = read_json(source / "complete.json")
        signature = read_json(source / "signature.json")
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if not complete["complete"] or complete.get("smoke", False) or complete["GT_box_prompts_used"] or complete["new_training_performed"] or digest != complete["signature_digest"]:
            raise ValueError(f"Invalid frozen inference completion: {source}")
        recorded = {str((ROOT / path).resolve()): value for path, value in signature["files"].items()}
        outputs = {str((ROOT / path).resolve()): value for path, value in complete["output_files"].items()}
        provenance = {}
        paths = [source / manifest_name, ROOT / config_name, ROOT / "scripts/run_calcification_gate_sam.py",
                 ROOT / "scripts/calcification_gate_sam_common.py", ROOT / "runs/mammo_hypothesis/weights/sam_vit_b.pth"]
        paths += [source / "frozen/gate/classifier.pt", source / "frozen/gate/thresholds.json"]
        paths += [source / "frozen" / arm / item for arm in ("imagenet", "mammo") for item in ("best.pt", "snapshot.json")]
        for path in paths:
            provenance[str(path)] = verify_hash(path, recorded[str(path.resolve())])
        config = read_json(ROOT / config_name)
        gate = read_json(source / "frozen/gate/thresholds.json")["primary"]["threshold_logit"]
        detectors = {arm: read_json(source / "frozen" / arm / "snapshot.json") for arm in ("imagenet", "mammo")}
        recipe = dict(gate_threshold_logit=gate,
                      gate_checkpoint_sha256=recorded[str((source / "frozen/gate/classifier.pt").resolve())],
                      sam_checkpoint_sha256=recorded[str((ROOT / config["sam_weights"]).resolve())],
                      detectors={arm: {key: value for key, value in detectors[arm].items() if key in ("epoch", "threshold", "files")} for arm in detectors},
                      sam_intensity=config["csaw_sam_intensity"], detector_intensity=config["csaw_gate_detector_intensity"],
                      mask_rule=config["mask_rule"], crop_min_side=config["crop_min_side"], crop_context_factor=config["crop_context_factor"])
        if common_recipe is not None and recipe != common_recipe:
            raise ValueError("Legacy cohorts used different frozen recipes")
        common_recipe = recipe
        rows = read_json(source / manifest_name)
        source_nonempty = {"imagenet": 0, "mammo": 0, "union": 0}
        for arm in ("imagenet", "mammo"):
            marker_path = source / "csaw" / arm / "mask_inference_complete.json"
            provenance[str(marker_path)] = verify_hash(marker_path, outputs[str(marker_path.resolve())])
            marker = read_json(marker_path)
            if not marker["complete"] or marker["gt_read_for_inference"] or marker["images"] != len(rows) or marker["signature_digest"] != digest:
                raise ValueError(f"Incomplete image-only inference: {marker_path}")
            expected_ids = {row["image_id"] for row in rows}
            if {path.stem for path in (source / "csaw" / arm / "gated/cases").glob("*.json")} != expected_ids:
                raise ValueError("Incomplete or extra candidate cases")
        for row in rows:
            ident = row["image_id"]
            if ident not in current or ident in images:
                raise ValueError(f"Unexpected or duplicate cohort image: {ident}")
            native = current[ident]
            if row["source_sha256"] != native["image_sha256"] or (row["height"], row["width"]) != (native["height"], native["width"]):
                raise ValueError(f"Raw input changed since image-only inference: {ident}")
            entry = dict(source=name, source_image_sha256=native["image_sha256"],
                         source_image_pixel_sha256=native["image_pixel_sha256"],
                         width=native["width"], height=native["height"], mask_paths=[], masks=[])
            union = None
            for arm in ("imagenet", "mammo"):
                case_path = source / "csaw" / arm / "gated/cases" / f"{ident}.json"
                case_sha = verify_hash(case_path, outputs[str(case_path.resolve())])
                case = read_json(case_path)
                if case["image_id"] != ident or case["arm"] != arm or case["dataset"] != "csaw" or case["route"] != "gated" or case["gt_read_for_inference"] or case["signature_digest"] != digest:
                    raise ValueError(f"Invalid GT-free case provenance: {case_path}")
                if case["gate_pass"] != (case["gate_logit"] >= gate):
                    raise ValueError(f"Gate threshold mismatch: {case_path}")
                mask_path = ROOT / case["mask_path"]
                if str(mask_path.resolve()) != str((source / "csaw" / arm / "gated/masks" / f"{ident}.npz").resolve()):
                    raise ValueError("Case points outside expected candidate route")
                if outputs[str(mask_path.resolve())] != case["mask_sha256"]:
                    raise ValueError("Completion and per-case mask hashes disagree")
                packed = verify_mask(mask_path, case, (native["height"], native["width"]), detectors[arm]["threshold"])
                union = packed if union is None else union | packed
                source_nonempty[arm] += int(case["predicted_pixels"] > 0)
                entry["mask_paths"].append(str(mask_path))
                entry["masks"].append(dict(arm=arm, path=str(mask_path), sha256=case["mask_sha256"],
                                           case_path=str(case_path), case_sha256=case_sha,
                                           predicted_pixels=case["predicted_pixels"], gate_pass=case["gate_pass"],
                                           selected_box_count=case["selected_box_count"], sam_calls=case["sam_calls"]))
            entry["union_pixels"] = int(BITS[union].sum())
            source_nonempty["union"] += int(entry["union_pixels"] > 0)
            images[ident] = entry
        sources.append(dict(name=name, root=str(source), inference_signature=digest, image_count=len(rows),
                            complete_sha256=sha_file(source / "complete.json"),
                            signature_sha256=sha_file(source / "signature.json"),
                            verified_source_files=provenance, nonempty_masks=source_nonempty))
        print(json.dumps({"source": name, "images": len(rows), "nonempty_masks": source_nonempty}), flush=True)
    if set(images) != set(current) or len(images) != 312:
        raise ValueError("Full312 candidate coverage missing; no substitute pools permitted")
    result = dict(schema_version=1, data_signature=manifest["signature"], technical_complete=True,
                  role="initial training candidate proposals only; select current train IDs before sampling; GT rejection training-only",
                  gt_box_prompts=False, gt_read_for_candidate_generation=False, new_inference=False,
                  recipe=common_recipe, sources=sources, images={key: images[key] for key in sorted(images)},
                  image_count=len(images), mask_count=2 * len(images),
                  nonempty_union_images=sum(row["union_pixels"] > 0 for row in images.values()),
                  empty_union_images=sum(row["union_pixels"] == 0 for row in images.values()))
    result["signature"] = hashlib.sha256(canonical(result)).hexdigest()
    output = Path(output) if output is not None else data_root / "initial_candidates.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("signature", "image_count", "mask_count", "nonempty_union_images", "empty_union_images")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="runs/csaw312_rerun_20261105/data")
    parser.add_argument("--output")
    build(**vars(parser.parse_args()))
