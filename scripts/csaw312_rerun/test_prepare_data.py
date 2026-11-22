"""Leakage and nested-selection tests; no model training and no legacy writes."""
import copy
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).with_name("prepare_data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def inventory():
    rows = []
    for group in range(150):
        for view in range(2 + (group < 12)):
            ident = f"{group:03d}_{view}"
            rows.append(dict(image_id=ident, group_id=f"{group:03d}", positive=group % 2 == 0,
                             image_pixel_sha256=ident, mask_pixel_sha256=f"mask{group % 2}",
                             gt_pixels=int(group % 2 == 0), component_count=int(group % 2 == 0)))
    data.union_duplicates(rows)
    return rows


class PreparationTests(unittest.TestCase):
    def test_nested_coverage_reproducible_and_shared_training_seeds(self):
        rows = inventory()
        first, second = data.make_splits(rows), data.make_splits(copy.deepcopy(rows))
        self.assertEqual(first, second)
        self.assertEqual(first["training_seeds"], [42, 43, 44])
        self.assertEqual(data.counts(rows, first["development_ids"])["source_groups"], 120)
        self.assertEqual(data.counts(rows, first["test_ids"])["source_groups"], 30)
        self.assertEqual(sorted(key for fold in first["folds"] for key in fold["outer_val_ids"]),
                         sorted(first["development_ids"]))
        for fold in first["folds"]:
            self.assertEqual(data.counts(rows, fold["inner_train_ids"])["source_groups"], 64)
            self.assertEqual(data.counts(rows, fold["inner_val_ids"])["source_groups"], 16)

    def test_duplicate_connected_groups_never_cross_any_partition(self):
        rows = inventory()
        lookup = {row["image_id"]: row for row in rows}
        lookup["002_0"]["image_pixel_sha256"] = lookup["000_0"]["image_pixel_sha256"]
        lookup["004_0"]["image_pixel_sha256"] = lookup["002_1"]["image_pixel_sha256"]
        links = data.union_duplicates(rows)
        self.assertEqual({lookup[key]["split_group_id"] for key in ("000_0", "002_0", "004_0")}, {"000"})
        self.assertEqual(len(links), 2)
        splits = data.make_splits(rows)
        data.validate_splits(rows, splits)
        # Simulate a hidden identical-pixel link crossing development/test.
        lookup = {row["image_id"]: row for row in rows}
        lookup[splits["test_ids"][0]]["image_pixel_sha256"] = lookup[splits["development_ids"][0]]["image_pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "image_pixel_sha256"):
            data.validate_splits(rows, splits)

    def test_inconsistent_duplicate_masks_fail_without_exclusion(self):
        rows = inventory()
        rows[3]["image_pixel_sha256"] = rows[0]["image_pixel_sha256"]
        rows[3]["mask_pixel_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "inconsistent masks"):
            data.union_duplicates(rows)

    def test_oof_duplicate_or_missing_assignment_rejected(self):
        rows = inventory()
        splits = data.make_splits(rows)
        splits["folds"][1] = copy.deepcopy(splits["folds"][0])
        splits["folds"][1]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "exactly one OOF"):
            data.validate_splits(rows, splits)

    def test_native_component_geometry_and_missing_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "000"
            folder.mkdir()
            image = np.arange(64, dtype=np.uint8).reshape(8, 8)
            mask = np.zeros_like(image)
            mask[1, 1] = mask[2, 2] = mask[6, 6] = 255
            path = folder / "000_0.png"
            Image.fromarray(image).save(path)
            row = dict(path=str(path), image_id="000_0", grouping_id="000", width="8", height="8")
            with self.assertRaises(FileNotFoundError):
                data.audit_image(row)
            Image.fromarray(mask).save(folder / "000_0_calcifications.png")
            result = data.audit_image(row)
            self.assertEqual(result["component_count"], 2)
            self.assertEqual(result["gt_pixels"], 3)
            self.assertEqual(result["component_size_counts"]["1_4"], 2)
            Image.fromarray(mask[:4]).save(folder / "000_0_calcifications.png")
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                data.audit_image(row)

    def test_relocated_manifest_outputs_and_dry_run_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "native"
            native.mkdir()
            rows = []
            for group in range(45):
                folder = native / f"{group:03d}"
                folder.mkdir()
                for view in range(2):
                    ident = f"{group:03d}_{view}"
                    image = np.full((8, 8), 2 * group + view, dtype=np.uint8)
                    mask = np.zeros_like(image)
                    if group % 2 == 0:
                        mask[1, 1] = 255
                    Image.fromarray(image).save(folder / f"{ident}.png")
                    Image.fromarray(mask).save(folder / f"{ident}_calcifications.png")
                    rows.append(dict(dataset="CSAW-S", image_id=ident,
                                     path=f"/old/{group:03d}/{ident}.png", grouping_id=f"{group:03d}"))
            source = root / "inventory.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            original_sha = data.sha_file(source)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest, splits, audit = data.prepare(source, root / "outputs", 90, 45, source_root=native)
                dry = data.prepare(source, root / "dry", 90, 45, source_root=native, audit_only=True)
            self.assertFalse((root / "dry").exists())
            self.assertEqual(manifest["signature"], dry[0]["signature"])
            self.assertEqual(original_sha, data.sha_file(source))
            self.assertTrue(manifest["images"][0]["inventory_source_image"].startswith("/old/"))
            self.assertTrue(manifest["images"][0]["source_image"].startswith(str(native)))
            marker = json.loads((root / "outputs/complete.json").read_text())
            self.assertTrue(marker["technical_complete"])
            self.assertFalse(marker["review_pass"])
            self.assertEqual(set(marker["file_sha256"]), {"manifest.json", "splits.json", "audit.json"})
            for name, checksum in marker["file_sha256"].items():
                self.assertEqual(checksum, data.sha_file(root / "outputs" / name))


if __name__ == "__main__":
    unittest.main()
