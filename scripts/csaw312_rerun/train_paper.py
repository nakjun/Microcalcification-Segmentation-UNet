"""Run the final study's training stages, sequentially and in dependency order."""
import argparse
from pathlib import Path
import subprocess
import sys

from common import PLAN, ROOT, read


def base_jobs(plan):
    done = set()
    for index, job in enumerate(plan["jobs"]):
        if job["seed"] != 42 or job["model"] == "yolo":
            continue
        if job["model"] == "utnet" and job["variant"] in ("C", "LH"):
            continue
        if not set(job["depends_on"]).issubset(done):
            raise ValueError(f"Missing stage dependency for {job['job_id']}")
        done.add(job["job_id"])
        yield index, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list", action="store_true", help="Print training commands and exit")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    jobs = list(base_jobs(read(PLAN)))
    if len(jobs) != 90:
        raise ValueError("Expected selection and refit for 45 base runs")
    commands = [[str(directory / "worker.py"), "--job-index", str(index)] for index, _ in jobs]
    commands.extend([[str(directory / script)] for script in ("fp_utnet.py", "fp_yolo.py")])
    for command in commands:
        argv = [sys.executable, *command, "--device", args.device]
        if args.list:
            print(*argv)
        else:
            subprocess.run(argv, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
