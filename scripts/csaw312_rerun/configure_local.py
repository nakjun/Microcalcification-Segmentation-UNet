"""Bind a local training plan to the published images, masks and splits."""
import csv
from common import ROOT, read, sha, write


def main():
    template = ROOT / "configs/csaw312_rerun_20261105/protocol.json"
    plan = read(template)
    data = ROOT / plan["data_root"]
    local = read(data / "splits.json")
    published = read(ROOT / "paper_results/splits.json")
    if {k: v for k, v in local.items() if k != "signature"} != {
        k: v for k, v in published.items() if k != "signature"
    }:
        raise ValueError("Local split differs from the published split")
    manifest = read(data / "manifest.json")
    with (ROOT / "paper_results/csaw_file_hashes.csv").open() as stream:
        expected = {r["image_id"]: (r["image_sha256"], r["mask_sha256"]) for r in csv.DictReader(stream)}
    observed = {r["image_id"]: (r["image_sha256"], r["mask_sha256"]) for r in manifest["images"]}
    if observed != expected:
        raise ValueError("Local source images or masks differ from the study inventory")
    done = read(data / "complete.json")
    if not done["technical_complete"] or any(sha(data / name) != h for name, h in done["file_sha256"].items()):
        raise ValueError("Incomplete or changed data preparation")
    if manifest['signature'] != local['signature'] or done['signature'] != local['signature']:
        raise ValueError("Inconsistent local data signatures")
    plan["data_signature"] = local["signature"]
    target = ROOT / "runs/csaw312_rerun_20261105/local_protocol.json"
    if target.exists() and read(target) != plan:
        raise ValueError(f"A different run is already configured at {target}")
    write(target, plan)
    print(f"export CSAW312_PROTOCOL={target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
