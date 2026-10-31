#!/usr/bin/env python3
"""Launch only the 40 approved saved-count bootstrap jobs, never circuits.

Parallelism is between jobs; each child retains serial RNG order and BLAS=1.
Completed jobs are revalidated by the numerical runner under --resume.
Progress is separate from the frozen configuration and point-estimate pack.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis_results/c1_primary_fixed_20261018'
RUNNER = ROOT / 'analysis/run_c1_primary_bootstrap_20261018.py'
CONFIG = ROOT / 'analysis/c1_primary_fixed_20261018.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_key(rate):
    return format(float(rate), '.12g').replace('-', 'm').replace('+', '').replace('.', 'p')


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'bootstrap_10000')
    parser.add_argument('--max-jobs', type=int, default=40)
    args = parser.parse_args()
    assert 1 <= args.workers <= 12 and 1 <= args.max_jobs <= 40
    output = args.output_dir.resolve()
    assert output.is_relative_to(BASE) and output != BASE
    assert RUNNER.is_file()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'option2-metric-locked-revision'
    config = json.loads(CONFIG.read_text())
    assert config['ridges'] == dict(zip(('linear_zne', 'quadratic_zne', 'single_exponential_zne', 'dual_exponential_zne'), (0, 0, 0, 1e-6)))
    point_manifest = json.loads((BASE / 'manifest.json').read_text())
    assert point_manifest['sources'][str(CONFIG.relative_to(ROOT))] == sha(CONFIG)
    jobs = []
    for instance in ('n119_a13', 'n119_a50', 'n119_a16'):
        source = ROOT / 'analysis_results/equal_shot_realized' / instance / 'realized/metadata.json'
        for rate in json.loads(source.read_text())['rates']:
            jobs.append({'budget': 'equal_shot', 'instance': instance, 'rate': float(rate)})
        jobs.append({'budget': 'gate_matched', 'instance': instance, 'rate': .0001})
    assert len(jobs) == 40
    jobs.sort(key=lambda j: (0 if j['rate'] == .0001 and j['budget'] == 'equal_shot' else
                            1 if j['budget'] == 'gate_matched' else 2, j['rate'], j['instance']))
    jobs = jobs[:args.max_jobs]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.parallel_launcher.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = output / f'execution_{stamp}'
    logs.mkdir(exist_ok=False)
    signatures = {str(p.relative_to(ROOT)): sha(p) for p in (RUNNER, CONFIG, Path(__file__))}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    started = time.monotonic()
    failures = threading.Event()
    results = []

    def run(job):
        if failures.is_set():
            return {**job, 'status': 'not_started_after_peer_failure', 'returncode': None}
        for name, digest in signatures.items():
            if sha(ROOT / name) != digest:
                failures.set()
                raise RuntimeError('Source changed during production: ' + name)
        log = logs / f"{job['budget']}_{job['instance']}_rate_{rate_key(job['rate'])}.log"
        command = [sys.executable, str(RUNNER), '--budget', job['budget'], '--instance', job['instance'],
                   '--rate', str(job['rate']), '--replicates', '10000', '--resume', '--output-dir', str(output)]
        print('START ' + json.dumps(job), flush=True)
        with log.open('x') as handle:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                  pass_fds=(lock.fileno(),), check=False)
        if proc.returncode:
            failures.set()
        result = {**job, 'returncode': proc.returncode, 'status': 'runner_success' if proc.returncode == 0 else 'failed',
                  'log': str(log.relative_to(ROOT)), 'elapsed_since_launcher_start': time.monotonic() - started}
        print('DONE ' + json.dumps(result), flush=True)
        return result

    def progress():
        state = {'configuration_id': config['configuration_id'], 'started_utc': stamp,
                 'status': 'running', 'scope': '40 approved jobs: 37 equal-shot and 3 gate-matched',
                 'jobs_in_this_invocation': len(jobs), 'workers': args.workers, 'replicates_per_job': 10000,
                 'completed_successfully': sum(r['returncode'] == 0 for r in results),
                 'failed_or_not_started': sum(r['returncode'] != 0 for r in results),
                 'elapsed_seconds': time.monotonic() - started, 'sources': signatures,
                 'new_quantum_shots': 0, 'jobs': jobs, 'results': results}
        atomic(logs / 'progress.json', state)
        atomic(output / 'execution_status.json', state)
        return state

    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run, job) for job in jobs}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            state = progress()
            print(f"PROGRESS {state['completed_successfully']}/{len(jobs)} jobs; failures={state['failed_or_not_started']}; elapsed={state['elapsed_seconds']:.1f}s", flush=True)
    state = progress()
    state['status'] = 'all_requested_jobs_finished' if all(r['returncode'] == 0 for r in results) else 'failure_requires_inspection'
    state['independent_final_verification'] = 'pending'
    atomic(logs / 'progress.json', state)
    atomic(output / 'execution_status.json', state)
    return int(any(r['returncode'] != 0 for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
