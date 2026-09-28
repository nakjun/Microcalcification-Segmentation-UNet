"""Finalize one comparative-study backbone after all technical artifacts exist.

This verifier is intentionally write-once at the end of the workflow.  It
checks the completed training, native inference, independent count, lineage,
and qualitative artifacts before writing ``final_verification.json`` and the
backbone ``complete.json``.  Performance status is recorded separately from
technical completion; no threshold, epoch, model, or held selection is made.
"""
from __future__ import annotations

import argparse
import datetime as datetime_module
import os
from pathlib import Path


BACKBONES = ("proposed", "unet2d", "utnet")
VARIANTS = ("C", "L", "H", "LH")
EXPECTED_LOG_MODES = ("warmup", "train", "infer")
EXPECTED_EPOCHS = 192
EXPECTED_JOBS = 12
EXPECTED_UPDATES = 24576
EXPECTED_MINING_IMAGES = 2976
EXPECTED_OOF_IMAGES = 496
EXPECTED_MASK_COMPARISONS = 2480


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument(
        "--visual-reviewed",
        action="store_true",
        help="Confirm that root independently viewed the fixed qualitative PNG.",
    )
    return parser.parse_args()


ARGS = parse_args()
if not ARGS.visual_reviewed:
    raise SystemExit("refusing to finalize without --visual-reviewed")

# common.py binds all paths and protocol signatures from this environment.
os.environ["STUDY_BACKBONE"] = ARGS.backbone
from common import (  # noqa: E402
    OUT,
    ROOT,
    VARIANTS as COMMON_VARIANTS,
    checksum,
    context,
    immutable,
    job_signature,
    read,
    spec,
    verify,
)


def require(condition, message):
    assert condition, message


def add_file(path, files):
    path = Path(path)
    require(path.is_file(), f"missing artifact: {path}")
    files[str(path)] = checksum(path)


def add_output_files(record, files):
    for filename, expected in record.get("output_files", {}).items():
        require(checksum(filename) == expected, f"output SHA mismatch: {filename}")
        add_file(filename, files)


def check_exit_markers(files):
    log_root = Path("logs/comparative_study") / ARGS.backbone
    for variant in VARIANTS:
        for fold in range(3):
            for mode in EXPECTED_LOG_MODES:
                path = log_root / f"{variant}_fold{fold}_{mode}.exit"
                require(path.read_text().strip() == "0", f"failed exit marker: {path}")
                add_file(path, files)
    for mode in (
        "verify_warmup",
        "evaluate",
        "verify_counts",
        "verify_training_lineage",
        "render_evidence",
    ):
        path = log_root / f"{mode}.exit"
        require(path.read_text().strip() == "0", f"failed exit marker: {path}")
        add_file(path, files)


def check_preflight(c, files):
    path = OUT / "preflight.json"
    preflight = read(path)
    require(preflight["complete"] is True, "preflight incomplete")
    require(preflight["signature"] == c["signature"], "preflight signature mismatch")
    require(preflight["backbone"] == ARGS.backbone, "preflight backbone mismatch")
    require(preflight["warmup_training_authorized"] is True, "warmup not authorized")
    require(preflight["strict_deterministic"] is True, "strict deterministic preflight missing")
    require(preflight["gpu_batch"] == 16, "preflight batch mismatch")
    require(preflight["new_budget"] == {
        "jobs": 12,
        "epochs": 192,
        "patches": 393216,
        "updates": 24576,
        "mining_native_images": 2976,
        "oof_native_images": 496,
    }, "preflight budget mismatch")
    for filename, expected in preflight["source_files"].items():
        require(checksum(filename) == expected, f"preflight source SHA mismatch: {filename}")
        add_file(filename, files)
    add_file(path, files)


def check_runtime_reports(c, files):
    required = (
        "model_source_verification.json",
        "sampler_verification.json",
        "smoke_verification.json",
        "gpu_native_verification.json",
        "determinism_gpu0.json",
        "determinism_gpu1.json",
    )
    for name in required:
        path = OUT / name
        report = read(path)
        require(report["complete"] is True, f"incomplete runtime report: {path}")
        require(report["signature"] == c["signature"], f"runtime signature mismatch: {path}")
        if "performance_evaluation" in report:
            require(report["performance_evaluation"] is False, f"performance mixed into {path}")
        if name == "model_source_verification.json":
            require(report["backbone"] == ARGS.backbone, "source backbone mismatch")
        if name == "gpu_native_verification.json":
            require(report["source_probability_reload_exact"] is True, "native reload check missing")
            require(report["independent_mining_centers_scores_exact"] is True, "native mining check missing")
            for item in report["checks"]:
                require(item["full_native_coverage"] is True, "native coverage check missing")
                require(item["GT_used_for_inference"] is False, "GT-conditioned native check")
        add_file(path, files)

    warmup_root = verify(ROOT / "warmup_verified.json", c["signature"])
    require(warmup_root["jobs"] == EXPECTED_JOBS, "warmup job count mismatch")
    require(warmup_root["all_model_optimizer_pool_equal"] is True, "warmup equality missing")
    add_file(ROOT / "warmup_verified.json", files)
    add_output_files(warmup_root, files)

    warmup_report = verify(OUT / "warmup_verification.json", c["signature"])
    require(warmup_report["jobs"] == EXPECTED_JOBS, "warmup report job count mismatch")
    add_file(OUT / "warmup_verification.json", files)
    add_output_files(warmup_report, files)


def check_training_lineage(c, files):
    path = OUT / "training_lineage.json"
    lineage = read(path)
    require(lineage["complete"] is True, "training lineage incomplete")
    require(lineage["signature"] == c["signature"], "training lineage signature mismatch")
    jobs = lineage["jobs"]
    require(len(jobs) == EXPECTED_JOBS, "training lineage job count mismatch")
    require(sum(job["epochs"] for job in jobs) == EXPECTED_EPOCHS, "training epoch total mismatch")
    require(sum(job["optimizer_steps"] for job in jobs) == EXPECTED_UPDATES, "update total mismatch")
    require(sum(len(job["mining_stages"]) for job in jobs) == 36, "mining stage count mismatch")
    require(sum(stage["training_images"] for job in jobs for stage in job["mining_stages"]) == EXPECTED_MINING_IMAGES,
            "mining image total mismatch")
    seen = set()
    for item in jobs:
        variant, fold = item["variant"], int(item["fold"])
        key = (variant, fold)
        require(variant in VARIANTS and 0 <= fold < 3 and key not in seen, f"duplicate lineage job: {key}")
        seen.add(key)
        require(item["epochs"] == 16 and item["optimizer_steps"] == 2048, f"job budget mismatch: {key}")
        require([stage["epoch"] for stage in item["mining_stages"]] == [3, 7, 11], f"mining stages mismatch: {key}")
        active = dict(c, variant=variant)
        job_root = ROOT / "jobs" / variant / f"fold_{fold}"
        done = verify(job_root / "complete.json", job_signature(active, fold))
        require(done["epochs"] == 16 and done["fold"] == fold and done["variant"] == variant, f"job marker mismatch: {key}")
        require(checksum(job_root / "checkpoint.pt") == item["checkpoint_sha256"], f"checkpoint SHA mismatch: {key}")
        add_file(job_root / "complete.json", files)
        add_output_files(done, files)
    require(seen == {(variant, fold) for variant in VARIANTS for fold in range(3)}, "lineage coverage mismatch")
    require(lineage["verifier_sha256"] == checksum("scripts/comparative_study/verify_training.py"), "lineage verifier SHA mismatch")
    add_file(path, files)


def check_oof_and_counts(c, files):
    oof_path = ROOT / "oof" / "complete.json"
    oof = verify(oof_path, c["signature"])
    require(oof["images"] == 124 and len(oof["pipeline"]) == 4, "OOF image or variant count mismatch")
    require(set(oof["pipeline"]) == set(VARIANTS), "OOF variant coverage mismatch")
    require(oof["policy"] == "epoch16/gate/full_image@0.9", "OOF policy mismatch")
    require(oof["selected"] is None and oof["held_allowed"] is False, "OOF selection/held violation")
    add_file(oof_path, files)
    add_output_files(oof, files)

    count_path = OUT / "count_verification.json"
    counts = read(count_path)
    require(counts["complete"] is True and counts["signature"] == c["signature"], "count report incomplete")
    require(counts["mask_comparisons"] == EXPECTED_MASK_COMPARISONS, "mask comparison count mismatch")
    require(counts["all_pixel_and_component_counts_exact"] is True, "pixel/component exact check missing")
    require(counts["independent_aggregates_exact"] is True, "aggregate exact check missing")
    require(counts["success_conditions_independently_checked"] is True, "success condition check missing")
    require(counts["oof_sha256"] == checksum(oof_path), "count report OOF SHA mismatch")
    require(counts["verifier_sha256"] == checksum("scripts/comparative_study/verify_counts.py"), "count verifier SHA mismatch")
    add_file(count_path, files)


def check_qualitative(c, files):
    path = OUT / "qualitative_evidence.json"
    evidence = read(path)
    require(evidence["complete"] is True and evidence["signature"] == c["signature"], "qualitative report incomplete")
    require(evidence["ids"] == ["105_0", "130_1", "025_1"], "qualitative IDs changed")
    require(evidence["arm"] == "imagenet" and evidence["full_image"] is True, "qualitative scope mismatch")
    require(evidence["GT_crop"] is False, "qualitative crop scope mismatch")
    png = Path(evidence["png"])
    require(png == OUT / "fixed_full_view.png", "qualitative PNG path mismatch")
    require(evidence["png_sha256"] == checksum(png), "qualitative PNG SHA mismatch")
    require(evidence["source_sha256"] == checksum("scripts/comparative_study/render_evidence.py"), "qualitative source SHA mismatch")
    add_file(path, files)
    add_file(png, files)


def performance_summary(oof):
    fields = (
        "historical_local_failures",
        "secondary_v3_local_failures",
        "within_backbone_failures",
        "global_constraint_failures",
        "secondary_v3_global_constraint_failures",
    )
    failures = {field: oof[field] for field in fields}

    def has_failure(value):
        if isinstance(value, dict):
            return any(has_failure(child) for child in value.values())
        if isinstance(value, list):
            return bool(value) or any(has_failure(child) for child in value)
        return False

    return failures, not any(has_failure(value) for value in failures.values())


def main():
    c = context("C")
    require(tuple(COMMON_VARIANTS) == VARIANTS, "variant protocol mismatch")
    files = {}
    check_exit_markers(files)
    check_preflight(c, files)
    check_runtime_reports(c, files)
    check_training_lineage(c, files)
    check_oof_and_counts(c, files)
    check_qualitative(c, files)
    oof = read(ROOT / "oof" / "complete.json")
    failures, performance_success = performance_summary(oof)

    final_path = OUT / "final_verification.json"
    prior = read(final_path) if final_path.exists() else None
    result = dict(
        complete=True,
        signature=c["signature"],
        backbone=ARGS.backbone,
        utc=(prior or {}).get("utc", datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()),
        technical_validation_complete=True,
        source_preflight_complete=True,
        warmup_all12_exact=True,
        jobs=EXPECTED_JOBS,
        epochs=EXPECTED_EPOCHS,
        patches=393216,
        optimizer_updates=EXPECTED_UPDATES,
        mining_native_images=EXPECTED_MINING_IMAGES,
        oof_native_images=EXPECTED_OOF_IMAGES,
        oof_images_per_variant=124,
        independent_mask_comparisons=EXPECTED_MASK_COMPARISONS,
        native_gt_free_verified=True,
        qualitative_reviewed=True,
        qualitative_viewed_by="root",
        performance_success=performance_success,
        performance_failures=failures,
        performance_scope="fixed epoch16 primary and secondary OOF tables; technical completion is separate",
        selected=None,
        held_allowed=False,
        verifier_sha256=checksum(__file__),
        output_files=files,
    )
    if prior is not None:
        require(prior == result, "existing final verification differs")
    immutable(final_path, result)
    root_result = dict(result, output_files={**files, str(final_path): checksum(final_path)})
    immutable(ROOT / "complete.json", root_result)
    print(f"{ARGS.backbone} technical finalization PASS; performance_success={performance_success}")


if __name__ == "__main__":
    main()
