"""Whole hardware-shaped flow on real CPU Aer only; no IBM account or network."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

pytest.importorskip('qiskit')
runtime=pytest.importorskip('qiskit_ibm_runtime')
aer=pytest.importorskip('qiskit_aer')
from qiskit.providers.fake_provider import GenericBackendV2
from shor_qem import ibm_runtime as ibm
from shor_qem.workflow import _analyze_collected, experiment_config


def test_full_arithmetic_to_local_sampler_to_analysis(tmp_path,monkeypatch):
    target=GenericBackendV2(7,basis_gates=['id','rz','sx','x','cx'],control_flow=True,
                            seed=3,noise_info=False).target
    backend=SimpleNamespace(name='explicit_test_double',num_qubits=7,target=target,simulator=False)
    jobs={}
    service=SimpleNamespace(backend=lambda name:backend,job=lambda identifier:jobs[identifier])
    local_sampler=runtime.SamplerV2(mode=aer.AerSimulator(max_parallel_threads=1,seed_simulator=5))
    class LocalOnly:
        def run(self,pubs):
            job=local_sampler.run(pubs)
            jobs[job.job_id()]=job
            return job
    monkeypatch.setattr(ibm,'_sampler_factory',lambda backend,options:LocalOnly())
    config=experiment_config(3,2,mode='ibm',backend_name=backend.name,
                             shots_per_scale=16,folding_patterns=2)
    prepared=tmp_path/'prepared'
    manifest=ibm.prepare(config,prepared,backend)
    assert manifest['budget']['total_physical_shots']==176
    records=ibm.submit(prepared,service=service,execute=True,max_jobs=8)
    assert all(r['job_id'] for r in records)
    arrays=ibm.collect(prepared,service=service,wait=True)
    assert arrays['raw_counts'].sum()==arrays['reset_extension_counts'].sum()+16==64
    for p in arrays['scale_counts']:
        assert set(np.flatnonzero(p)) <= {0,8}
    summary=_analyze_collected(prepared)
    assert summary['N']==3 and summary['order_evaluation_only']==2
    assert len(summary['points'])==6
    assert (prepared/'analysis/metrics.csv').exists()
    count=len(jobs)
    ibm.submit(prepared,service=service,execute=True,max_jobs=8)
    assert len(jobs)==count
    assert _analyze_collected(prepared)['source_sha256']==summary['source_sha256']
    assert json.loads((prepared/'jobs.json').read_text())['records']
