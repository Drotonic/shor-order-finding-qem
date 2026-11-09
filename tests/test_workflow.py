import json
import subprocess
import sys

import numpy as np
import pytest

from shor_qem.workflow import analyze_counts, experiment_config, main, plan


def make_counts(path, scales=4, shots=100, baseline=400):
    rng=np.random.default_rng(42)
    lambdas=np.linspace(1,3,scales)
    ideal=np.zeros(16);ideal[[0,8]]=.5
    probabilities=[np.exp(-.1*x)*ideal+(1-np.exp(-.1*x))/16 for x in lambdas]
    scale=np.array([rng.multinomial(shots,p) for p in probabilities])
    extension=rng.multinomial(baseline-shots,probabilities[0])
    raw=rng.multinomial(baseline,.7*ideal+.3/16)
    np.savez(path,scale_counts=scale,raw_counts=raw,reset_extension_counts=extension,
             effective_lambdas=lambdas,N=3,a=2)
    return scale,extension,raw,lambdas


def test_plan_has_no_truth_order():
    c=experiment_config(15,4)
    assert c['bits']==8 and 'order' not in c
    assert plan(c)['qubits']==11
    assert plan(c)['total_circuit_shots']==460000
    c=experiment_config(21,8,mode='ibm',backend_name='example_backend')
    assert c['scales']==[1,1.4,1.8,2.2]
    assert c['shots_per_scale']==10000 and c['baseline_shots']==40000
    assert plan(c)['total_circuit_shots']==110000


@pytest.mark.parametrize('N,a',[(True,2),(15,5),(1,2),(15,15),(15,1),(15,4.1)])
def test_invalid_instance(N,a):
    with pytest.raises(ValueError):experiment_config(N,a)


@pytest.mark.parametrize('kw',[{'scales':[1,2,3]},{'scales':[1,1,2,3]},
    {'scales':[1,2,3,float('nan')]},{'rates':[.1,.1]}, {'rates':[float('inf')]},
    {'shots_per_scale':0},{'baseline_shots':1},{'mode':'ibm'},
    {'mode':'ibm','backend_name':'ibm_example','rates':[0]}])
def test_invalid_config(kw):
    with pytest.raises(ValueError):experiment_config(15,4,**kw)


def test_plan_does_not_import_quantum_clients(tmp_path):
    code="from shor_qem.workflow import main; main(['plan','--N','15','--a','4']); import sys; assert 'cupy' not in sys.modules and 'qiskit_ibm_runtime' not in sys.modules and 'shor_qem.simulator' not in sys.modules"
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['qpu_submitted'] is False


def test_dry_run_writes_nothing(tmp_path,capsys):
    dest=tmp_path/'nothing'
    assert main(['simulate','--N','15','--a','4','--output-dir',str(dest)])==0
    assert not dest.exists()
    assert json.loads(capsys.readouterr().out)['configuration']['rates']==[.0001]
    assert main(['ibm','--N','15','--a','4','--backend','ibm_example','--output-dir',str(dest)])==0
    assert not dest.exists()


def test_generic_analysis_four_scales(tmp_path):
    source=tmp_path/'counts.npz';make_counts(source)
    dest=tmp_path/'analysis'
    r=analyze_counts(source,N=3,a=2,output_dir=dest)
    assert r['order_evaluation_only']==2 and len(r['points'])==6
    assert r['bootstrap_replicates']==0 and r['new_quantum_shots']==0
    assert r['shots_per_scale'].tolist()==[100]*4
    assert (dest/'metrics.csv').is_file()
    with np.load(dest/'distributions.npz') as z:
        assert z['selected_decoder_orders'][0]==0
        assert z['selected_decoder_orders'][8]==2
        for m in ['raw','reset','linear_zne','quadratic_zne','single_exponential_zne','dual_exponential_zne']:
            p=z['after__'+m]
            assert np.all(p>=0) and abs(p.sum()-1)<1e-11
    with pytest.raises(FileExistsError):analyze_counts(source,N=3,a=2,output_dir=dest)


def test_old_eight_scale_fit_unchanged(tmp_path):
    from shor_qem.portable_c1 import CountParents,reconstruct,METHODS
    source=tmp_path/'counts.npz'
    scales,ext,raw,lambdas=make_counts(source,8,20000,160000)
    initial=np.zeros(16,dtype=np.int64);initial[0]=20000
    rawextension=np.zeros(16,dtype=np.int64);rawextension[0]=140000
    old=reconstruct(CountParents(scales,ext,initial,rawextension,lambdas),order=2)
    analyze_counts(source,N=3,a=2,output_dir=tmp_path/'new')
    with np.load(tmp_path/'new/distributions.npz') as z:
        for method in METHODS:
            np.testing.assert_array_equal(z['after__'+method],old['arrays']['after__'+method])


def test_analysis_rejects_wrong_instance_metadata(tmp_path):
    source=tmp_path/'counts.npz';make_counts(source)
    with pytest.raises(ValueError,match='metadata'):
        analyze_counts(source,N=5,a=2,output_dir=tmp_path/'analysis')


def test_analysis_rejects_partial_and_float_counts(tmp_path):
    source=tmp_path/'counts.npz';make_counts(source)
    with np.load(source) as z:arrays={k:z[k] for k in z.files}
    arrays['raw_counts']=arrays['raw_counts'].copy();arrays['raw_counts'][0]-=1
    bad=tmp_path/'bad.npz';np.savez(bad,**arrays)
    with pytest.raises(ValueError,match='equal declared'):
        analyze_counts(bad,N=3,a=2,output_dir=tmp_path/'badout')
    arrays['scale_counts']=arrays['scale_counts'].astype(float)
    np.savez(bad,**arrays)
    with pytest.raises(ValueError,match='integer'):
        analyze_counts(bad,N=3,a=2,output_dir=tmp_path/'badout')


def test_resume_verifies_analysis_not_just_directory(tmp_path):
    from shor_qem.workflow import _analyze_or_verify
    source=tmp_path/'counts.npz';make_counts(source)
    dest=tmp_path/'analysis'
    _analyze_or_verify(source,dest,3,2)
    _analyze_or_verify(source,dest,3,2)
    with (dest/'metrics.csv').open('a') as f:f.write('corrupt')
    with pytest.raises(ValueError,match='checksum'):_analyze_or_verify(source,dest,3,2)


def test_simulation_rate_order_and_even_N_rejected():
    with pytest.raises(ValueError,match='odd'):experiment_config(14,3)
    with pytest.raises(ValueError,match='increasing'):experiment_config(15,4,rates=[.01,.001])


def test_ibm_console_suppresses_sdk_secret(monkeypatch,capsys):
    from shor_qem import workflow
    def fail():raise RuntimeError('secret-example-value')
    monkeypatch.setattr(workflow,'main',fail)
    monkeypatch.setattr(sys,'argv',['shor-qem','ibm-run'])
    assert workflow.cli()==1
    err=capsys.readouterr().err
    assert 'secret-example-value' not in err and 'jobs.json' in err


def test_ideal_reference_memory_guard():
    from shor_qem.workflow import _check_ideal_budget
    _check_ideal_budget(14,6)
    with pytest.raises(ValueError,match='working-array'):_check_ideal_budget(22,1024)


def test_hardware_plan_execution_limit():
    with pytest.raises(ValueError,match='10,000,000'):
        experiment_config(15,4,mode='ibm',backend_name='ibm_example',baseline_shots=10000001)
