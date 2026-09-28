"""Persistent resource-aware runner for the last-stage classifier experiment."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from calcification_gate_common import atomic_json, checksum


def gpu_resources(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    for line in output.strip().splitlines():
        index, uuid, total, free, util = [s.strip() for s in line.split(',')]
        if int(index) == gpu:
            result = dict(index=int(index), uuid=uuid, total_mib=int(total),
                          free_mib=int(free), utilization=int(util))
            break
    else:
        raise RuntimeError(f'Configured GPU {gpu} not visible')
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    result['processes'] = []
    for line in apps.strip().splitlines():
        if not line:
            continue
        uuid, pid, used = [s.strip() for s in line.split(',')]
        if uuid == result['uuid']:
            result['processes'].append({'pid': int(pid), 'used_mib': int(used)})
    return result


def record(root, data):
    value = dict(runner_pid=os.getpid(), updated_at_unix=time.time(), **data)
    atomic_json(root / 'pipeline_status.json', value)
    with (root / 'heartbeat.jsonl').open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')
    print(json.dumps(value, allow_nan=False), flush=True)


def main(args):
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'complete.json').exists():
        done = json.loads((root / 'complete.json').read_text())
        assert done['complete'] and checksum(root / 'final_seed42/classifier.pt') == done['classifier_sha256']
        for path, sha in done['signature']['files'].items():
            assert checksum(path) == sha, f'Completed run source changed: {path}'
        print('Fine-tuning, final fit, and development validation already complete and verified.', flush=True)
        return
    started = time.time()
    child = None
    env = dict(os.environ, OMP_NUM_THREADS=str(config['cpu_threads']),
               OPENBLAS_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']))

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Runner received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for stage in ['cache', 'fit']:
            while True:
                resource = gpu_resources(config['gpu'])
                ready = (resource['free_mib'] >= config['gpu_min_free_mib'] and
                         all(p['used_mib'] <= config['gpu_max_other_process_mib'] for p in resource['processes']))
                record(root, {'state': 'resources_ready' if ready else 'waiting_for_resources',
                              'stage': stage, 'resources': resource, 'started_at_unix': started})
                if ready:
                    break
                time.sleep(config['resource_poll_seconds'])
            path = Path('logs') / f"{config['version']}_{stage}.log"
            with path.open('a', buffering=1) as log:
                child = subprocess.Popen([sys.executable, '-u', 'scripts/train_calcification_gate_finetune.py',
                                          '--config', args.config, '--stage', stage],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
                stage_started = time.time()
                while True:
                    code = child.poll()
                    value = {'state': 'running' if code is None else ('stage_complete' if code == 0 else 'failed'),
                             'stage': stage, 'child_pid': child.pid, 'returncode': code,
                             'started_at_unix': started, 'stage_started_at_unix': stage_started,
                             'log': str(path)}
                    latest = stage_started
                    if (root / 'status.json').exists():
                        progress = json.loads((root / 'status.json').read_text())
                        if progress.get('pid') == child.pid:
                            value['progress'] = progress
                            latest = progress['updated_at_unix']
                    record(root, value)
                    if code is not None:
                        if code:
                            raise RuntimeError(f'Stage {stage} failed with exit code {code}; log={path}')
                        break
                    if time.time() - latest > config['max_stage_stall_seconds']:
                        raise RuntimeError(f'Stage {stage} has no progress for {config["max_stage_stall_seconds"]} seconds')
                    try:
                        child.wait(timeout=config['heartbeat_seconds'])
                    except subprocess.TimeoutExpired:
                        pass
        done = json.loads((root / 'final_seed42/evaluation_complete.json').read_text())
        assert done['complete'] and not done['smoke']
        assert checksum(root / 'final_seed42/classifier.pt') == done['classifier_sha256']
        record_done = {**done, 'state': 'complete', 'started_at_unix': started, 'finished_at_unix': time.time()}
        atomic_json(root / 'complete.json', record_done)
        record(root, record_done)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)
        record(root, {'state': 'failed', 'error': repr(exc), 'started_at_unix': started,
                      'child_pid': None if child is None else child.pid})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_gate_finetune.json')
    main(parser.parse_args())
