"""Audit native CSAW-S data and freeze nested group splits, without model fitting.

Use a NEW output directory. --audit-only performs the same input and split checks
without creating files. Source images and all legacy experiment files are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixel_hash(array):
    h = hashlib.sha256(canonical({"shape": list(array.shape), "dtype": str(array.dtype)}))
    h.update(memoryview(np.ascontiguousarray(array)))
    return h.hexdigest()


def load_native(path):
    with Image.open(path) as source:
        array = np.asarray(source).copy()
    if array.ndim != 2 or array.dtype != np.uint8 or min(array.shape) < 1:
        raise ValueError(f"Require native nonempty uint8 grayscale: {path}")
    return array


def audit_image(row):
    source_image = Path(row["path"]).resolve()
    image_id = row["image_id"]
    if source_image.stem != image_id:
        raise ValueError(f"Image ID/path mismatch: {image_id}")
    source_mask = source_image.with_name(image_id + "_calcifications.png")
    group_id = source_image.parent.name
    if row.get("grouping_id") and row["grouping_id"] != group_id:
        raise ValueError(f"Folder/group mismatch: {image_id}")
    raw, truth = load_native(source_image), load_native(source_mask)
    if raw.shape != truth.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_id}")
    height, width = raw.shape
    for key, actual in (("height", height), ("width", width)):
        if row.get(key) and int(row[key]) != actual:
            raise ValueError(f"Inventory/native {key} mismatch: {image_id}")
    mask = truth > 0
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:]
    if int(sizes.sum()) != int(mask.sum()) or len(sizes) != component_count:
        raise AssertionError(f"Component count failure: {image_id}")
    return dict(image_id=image_id, group_id=group_id, source_image=str(source_image),
                inventory_source_image=row.get("inventory_source_image", row["path"]),
                source_mask=str(source_mask), width=width, height=height,
                positive=bool(mask.any()), image_sha256=sha_file(source_image),
                mask_sha256=sha_file(source_mask), image_pixel_sha256=pixel_hash(raw),
                mask_pixel_sha256=pixel_hash(truth), gt_pixels=int(mask.sum()),
                component_count=int(component_count),
                mask_values=[int(value) for value in np.unique(truth)],
                component_size_counts={"1_4": int(np.sum(sizes <= 4)),
                                       "5_16": int(np.sum((sizes >= 5) & (sizes <= 16))),
                                       "17_64": int(np.sum((sizes >= 17) & (sizes <= 64))),
                                       "65_plus": int(np.sum(sizes >= 65))})


def union_duplicates(images):
    """Retain every image, linking folders by identical native image pixels."""
    parents = {row["group_id"]: row["group_id"] for row in images}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    duplicate_sets = {}
    for row in images:
        duplicate_sets.setdefault(row["image_pixel_sha256"], []).append(row)
    links = []
    for pixel_sha, rows in sorted(duplicate_sets.items()):
        if len(rows) < 2:
            continue
        if len({row["mask_pixel_sha256"] for row in rows}) != 1:
            raise ValueError("Identical image pixels have inconsistent masks: " +
                             ", ".join(row["image_id"] for row in rows))
        groups = sorted({row["group_id"] for row in rows})
        for group in groups[1:]:
            left, right = sorted((find(groups[0]), find(group)))
            parents[right] = left
        links.append(dict(image_pixel_sha256=pixel_sha,
                          image_ids=sorted(row["image_id"] for row in rows), group_ids=groups))
    for row in images:
        row["split_group_id"] = find(row["group_id"])
    return links


def group_records(images):
    output = {}
    for row in images:
        output.setdefault(row["split_group_id"], []).append(row)
    return [dict(split_group_id=group, positive=any(row["positive"] for row in rows),
                 source_groups=sorted({row["group_id"] for row in rows}),
                 image_ids=sorted(row["image_id"] for row in rows))
            for group, rows in sorted(output.items())]


def make_splits(images):
    groups = group_records(images)
    keys = [row["split_group_id"] for row in groups]
    labels = [int(row["positive"]) for row in groups]
    train, test = train_test_split(keys, test_size=0.2, random_state=42, stratify=labels)
    train, test = sorted(train), sorted(test)
    group_label = {row["split_group_id"]: int(row["positive"]) for row in groups}

    def ids(group_keys):
        allowed = set(group_keys)
        return sorted(row["image_id"] for row in images if row["split_group_id"] in allowed)

    result = dict(schema_version=SCHEMA, split_seed=42, training_seeds=[42, 43, 44],
                  split_unit="exact-pixel-duplicate-linked source folder; patient identity unverified",
                  development_ids=ids(train), test_ids=ids(test), folds=[])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (fit_indices, val_indices) in enumerate(splitter.split(
            np.zeros((len(train), 1)), [group_label[key] for key in train], train)):
        outer_train = sorted(train[index] for index in fit_indices)
        outer_val = sorted(train[index] for index in val_indices)
        inner_train, inner_val = train_test_split(
            outer_train, test_size=0.2, random_state=4242 + fold,
            stratify=[group_label[key] for key in outer_train])
        result["folds"].append(dict(fold=fold, inner_split_seed=4242 + fold,
                                    outer_train_ids=ids(outer_train), outer_val_ids=ids(outer_val),
                                    inner_train_ids=ids(inner_train), inner_val_ids=ids(inner_val)))
    validate_splits(images, result)
    return result


def validate_splits(images, splits):
    lookup = {row["image_id"]: row for row in images}
    if len(lookup) != len(images):
        raise ValueError("Duplicate image IDs")

    def disjoint(left, right):
        for field in ("image_id", "group_id", "split_group_id", "image_pixel_sha256"):
            a = {lookup[key][field] for key in left}
            b = {lookup[key][field] for key in right}
            if a & b:
                raise ValueError(f"Split leakage: {field}: {sorted(a & b)}")

    development, test = splits["development_ids"], splits["test_ids"]
    if len(development) != len(set(development)) or len(test) != len(set(test)):
        raise ValueError("Duplicate partition members")
    if set(development) | set(test) != set(lookup):
        raise ValueError("Partitions do not cover all input images")
    disjoint(development, test)
    oof = Counter()
    if [fold["fold"] for fold in splits["folds"]] != [0, 1, 2]:
        raise ValueError("Require folds 0, 1, 2")
    for fold in splits["folds"]:
        fit, val = fold["outer_train_ids"], fold["outer_val_ids"]
        inner_fit, inner_val = fold["inner_train_ids"], fold["inner_val_ids"]
        if set(fit) | set(val) != set(development) or set(inner_fit) | set(inner_val) != set(fit):
            raise ValueError("Nested partition coverage mismatch")
        disjoint(fit, val)
        disjoint(inner_fit, inner_val)
        for part in (fit, val, inner_fit, inner_val):
            if len(part) != len(set(part)):
                raise ValueError("Duplicate fold partition members")
            if {lookup[key]["positive"] for key in part} != {True, False}:
                raise ValueError("Fold partition lacks positive or negative images")
        oof.update(val)
    if oof != Counter({key: 1 for key in development}):
        raise ValueError("Each development image must have exactly one OOF assignment")
    if splits["training_seeds"] != [42, 43, 44]:
        raise ValueError("All training seeds must share the single frozen split")


def counts(images, ids):
    selected = set(ids)
    rows = [row for row in images if row["image_id"] in selected]
    return dict(images=len(rows), positive_images=sum(row["positive"] for row in rows),
                negative_images=sum(not row["positive"] for row in rows),
                source_groups=len({row["group_id"] for row in rows}),
                split_groups=len({row["split_group_id"] for row in rows}),
                positive_split_groups=sum(group["positive"] for group in group_records(rows)),
                gt_pixels=sum(row["gt_pixels"] for row in rows),
                components=sum(row["component_count"] for row in rows))


def prepare(input_manifest, output_root=None, expected_images=312, expected_source_groups=150,
            workers=2, audit_only=False, source_root=None):
    input_manifest = Path(input_manifest).resolve()
    with input_manifest.open(newline="") as stream:
        source_rows = sorted((row for row in csv.DictReader(stream) if row["dataset"] == "CSAW-S"),
                             key=lambda row: row["image_id"])
    if source_root is not None:
        source_root = Path(source_root).resolve(strict=True)
        for row in source_rows:
            original = Path(row["path"])
            row["inventory_source_image"] = str(original)
            row["path"] = str(source_root / original.parent.name / original.name)
    if len(source_rows) != expected_images:
        raise ValueError(f"Expected {expected_images} CSAW-S images; found {len(source_rows)}")
    if len({row["image_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Repeated input image_id")
    if len({Path(row["path"]).resolve() for row in source_rows}) != len(source_rows):
        raise ValueError("Repeated input path")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = []
        for index, row in enumerate(pool.map(audit_image, source_rows), start=1):
            images.append(row)
            if index % 20 == 0 or index == len(source_rows):
                print(json.dumps({"audited_images": index, "total": len(source_rows)}), flush=True)
    if len({row["group_id"] for row in images}) != expected_source_groups:
        raise ValueError("Unexpected source folder count; no automatic exclusion or regrouping")
    links = union_duplicates(images)
    splits = make_splits(images)
    manifest = dict(schema_version=SCHEMA, images=images)
    audit = dict(schema_version=SCHEMA, source_manifest=str(input_manifest),
                 source_manifest_sha256=sha_file(input_manifest), sklearn_version=sklearn.__version__,
                 explicit_source_root=str(source_root) if source_root is not None else None,
                 split_algorithm="group-record train_test_split .20 seed42; group-record SGKF3 shuffle42; inner .20 seed4242+fold",
                 grouping="source folder, union exact native image-pixel hashes; patient mapping unverified",
                 exclusions=[], groups=group_records(images), duplicate_links=links,
                 full=counts(images, [row["image_id"] for row in images]),
                 development=counts(images, splits["development_ids"]), test=counts(images, splits["test_ids"]),
                 folds=[dict(fold=fold["fold"], **{key: counts(images, value) for key, value in fold.items()
                                                  if key.endswith("_ids")}) for fold in splits["folds"]])
    signature = hashlib.sha256(canonical(dict(manifest=manifest, splits=splits, audit=audit))).hexdigest()
    for item in (manifest, splits, audit):
        item["signature"] = signature
    if not audit_only:
        if output_root is None:
            raise ValueError("--output-root is required unless --audit-only")
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name, value in (("manifest.json", manifest), ("splits.json", splits), ("audit.json", audit)):
            path = output_root / name
            with path.open("x") as stream:
                stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            hashes[name] = sha_file(path)
        marker = dict(schema_version=SCHEMA, signature=signature, technical_complete=True,
                      images=len(images), source_groups=expected_source_groups,
                      file_sha256=hashes, preparation_code_sha256=sha_file(__file__),
                      model_fitting=False, review_pass=False)
        with (output_root / "complete.json").open("x") as stream:
            stream.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"signature": signature, "audit_only": audit_only,
                      "full": audit["full"], "development": audit["development"], "test": audit["test"]}), flush=True)
    return manifest, splits, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", default="runs/audit_v1/image_manifest.csv")
    parser.add_argument("--output-root")
    parser.add_argument("--source-root", help="Explicit relocated directory containing original group folders; preserve inventory paths")
    parser.add_argument("--expected-images", type=int, default=312)
    parser.add_argument("--expected-source-groups", type=int, default=150)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only and not args.output_root:
        parser.error("--output-root is required unless --audit-only")
    prepare(**vars(args))


if __name__ == "__main__":
    main()
