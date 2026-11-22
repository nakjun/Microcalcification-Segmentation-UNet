"""Reproduce fold means and sample SDs from the public scalar export."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import sys


def summarize(path):
    groups = defaultdict(dict)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = tuple(row[field] for field in ("model", "condition", "seed", "cohort", "metric"))
            fold = int(row["fold"])
            if fold in groups[key]:
                raise ValueError(f"Repeated fold {fold} for {key}")
            groups[key][fold] = float(row["value"])
    for key, values in groups.items():
        if set(values) != {0, 1, 2}:
            raise ValueError(f"Incomplete folds for {key}")
        yield (*key, f"{mean(values.values()):.3f}", f"{stdev(values.values()):.3f}")


if __name__ == "__main__":
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("model", "condition", "seed", "cohort", "metric", "mean", "sample_sd"))
    writer.writerows(summarize(Path(__file__).with_name("fold_metrics.csv")))
