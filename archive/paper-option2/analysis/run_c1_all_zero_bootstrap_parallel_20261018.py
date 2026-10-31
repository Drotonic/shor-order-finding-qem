#!/usr/bin/env python3
"""Forty fixed all-zero-ridge jobs, reusing only verified same-draw results."""
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

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis_results/c1_all_zero_fixed_20261018'
RUNNER=ROOT/'analysis/run_c1_all_zero_bootstrap_20261018.py'
CONFIG=ROOT/'analysis/c1_all_zero_fixed_20261018.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def key(rate):
    return format(float(rate),'.12g').replace('-','m').replace('+','').replace('.','p')


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--output-dir',type=Path,default=BASE/'bootstrap_10000')
    p.add_argument('--max-jobs',type=int,default=40)
    args=p.parse_args()
    assert 1<=args.workers<=12 and 1<=args.max_jobs<=40
    out=args.output_dir.resolve()
    assert out.is_relative_to(BASE) and out!=BASE and RUNNER.is_file()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='option2-metric-locked-revision'
    config=json.loads(CONFIG.read_text())
    assert config['configuration_id']=='c1_beta2_all_ridges_zero_20261018'
    assert config['weight_exponent']==2 and set(config['ridges'].values())=={0}
    frozen=json.loads((BASE/'manifest.json').read_text())
    assert frozen['sources'][str(CONFIG.relative_to(ROOT))]==sha(CONFIG)
    jobs=[]
    for instance in ('n119_a13','n119_a50','n119_a16'):
        meta=ROOT/'analysis_results/equal_shot_realized'/instance/'realized/metadata.json'
        jobs += [{'budget':'equal_shot','instance':instance,'rate':float(r)} for r in json.loads(meta.read_text())['rates']]
        jobs.append({'budget':'gate_matched','instance':instance,'rate':.0001})
    assert len(jobs)==40
    jobs.sort(key=lambda j:(0 if j['budget']=='equal_shot' and j['rate']==.0001 else 1 if j['budget']=='gate_matched' else 2,j['rate'],j['instance']))
    jobs=jobs[:args.max_jobs]
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.parallel_launcher.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs=out/f'execution_{stamp}';logs.mkdir(exist_ok=False)
    signatures={str(f.relative_to(ROOT)):sha(f) for f in (RUNNER,CONFIG,Path(__file__))}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    stop=threading.Event();results=[];start=time.monotonic()
    def worker(job):
        if stop.is_set():return {**job,'returncode':None,'status':'not_started_after_peer_failure'}
        for name,digest in signatures.items():
            if sha(ROOT/name)!=digest:
                stop.set();raise RuntimeError('Frozen source changed: '+name)
        logfile=logs/f"{job['budget']}_{job['instance']}_rate_{key(job['rate'])}.log"
        command=[sys.executable,str(RUNNER),'--budget',job['budget'],'--instance',job['instance'],
                 '--rate',str(job['rate']),'--replicates','10000','--resume','--output-dir',str(out)]
        print('START '+json.dumps(job),flush=True)
        with logfile.open('x') as f:
            run=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),check=False)
        if run.returncode:stop.set()
        item={**job,'returncode':run.returncode,'log':str(logfile.relative_to(ROOT)),
              'elapsed_since_launch':time.monotonic()-start}
        print('DONE '+json.dumps(item),flush=True)
        return item
    def progress(final=False):
        ok=sum(r['returncode']==0 for r in results)
        state={'configuration_id':config['configuration_id'],'status':'all_requested_jobs_finished' if final and ok==len(jobs) else 'failure_requires_inspection' if final else 'running',
               'started_utc':stamp,'jobs_in_this_invocation':len(jobs),'completed_successfully':ok,
               'failed_or_not_started':sum(r['returncode']!=0 for r in results),'workers':args.workers,
               'replicates_per_job':10000,'scope':'37 equal-shot plus3 gate-matched, same stored draws',
               'result_origin':'unchanged routes/control caches reused after verification; missing dual0 refitted',
               'new_quantum_shots':0,'sources':signatures,'jobs':jobs,'results':results,
               'elapsed_seconds':time.monotonic()-start,'independent_final_verification':'pending'}
        atomic(out/'execution_status.json',state);atomic(logs/'progress.json',state)
        return state
    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,j) for j in jobs}
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:results.append(f.result())
            s=progress();print(f"PROGRESS {s['completed_successfully']}/{len(jobs)} jobs; failures={s['failed_or_not_started']}; elapsed={s['elapsed_seconds']:.1f}s",flush=True)
    progress(True)
    return int(any(r['returncode']!=0 for r in results))


if __name__=='__main__':
    raise SystemExit(main())
