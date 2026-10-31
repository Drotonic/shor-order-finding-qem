#!/usr/bin/env python3
"""Freeze already calculated all-ridge-zero parent points; no fitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis_results/c1_all_zero_fixed_20261018'
SOURCE = ROOT / 'analysis_results/c1_parameter_search_20261018/search/zero_ridge_beta2'
CONFIG = ROOT / 'analysis/c1_all_zero_fixed_20261018.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    assert not BASE.exists(), 'Never overwrite an existing frozen point pack'
    config = json.loads(CONFIG.read_text())
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    assert manifest['failed_point_count'] == 0 and manifest['point_fit_count'] == 148
    assert manifest['configuration']['alphas'] == [2.0] and manifest['configuration']['ridges'] == [0.0]
    assert sha(SOURCE/'all_points.csv') == manifest['all_points_sha256']
    with (SOURCE/'all_points.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    sources = {str(p.relative_to(ROOT)):sha(p) for p in (CONFIG,Path(__file__),SOURCE/'manifest.json',SOURCE/'all_points.csv')}
    prepared, summaries = [], []
    for method in config['ridges']:
        current = [r for r in rows if r['method'] == method]
        assert len(current) == 37 and len({r['case_id'] for r in current}) == 37
        folder = SOURCE/'families'/f'{method}__alpha_2__ridge_0'
        meta = json.loads((folder/'manifest.json').read_text())
        for name,digest in meta['artifacts'].items():
            assert sha(folder/name) == digest
            sources[str((folder/name).relative_to(ROOT))] = digest
        for r in current:
            assert r['variant']=='C1' and r['status']=='ok' and float(r['ridge'])==0 and float(r['weight_exponent'])==2
            prepared.append({'configuration_id':config['configuration_id'],
                **{k:r[k] for k in ('case_id','instance','a','order','rate','rate_percent','method','weight_exponent','ridge','tvd',
                                   'normalized_grouped_peak_margin','reset_minus_tvd','gap_minus_reset','selected_gamma_index',
                                   'selected_gamma_1','selected_gamma_2','normal_condition_number','intercept_l1_norm','l2_projection_correction')},
                'source_points':str((folder/'points.csv').relative_to(ROOT)),
                'source_selected_fit_cache':str((folder/'selected_fit_cache.npz').relative_to(ROOT)),
                'bootstrap_status_at_point_freeze':'pending'})
        summaries.append({'method':method,'beta':2,'ridge':0,'conditions':37,
            'case_balanced_mean_tvd':float(np.mean([np.mean([float(r['tvd']) for r in current if r['instance']==c])
                                                  for c in ('n119_a13','n119_a50','n119_a16')])),
            'positive_gap_conditions_above_1e-10':sum(float(r['normalized_grouped_peak_margin'])>1e-10 for r in current)})
    BASE.mkdir()
    write_csv(BASE/'point_estimates.csv',prepared)
    write_csv(BASE/'primary_rate_points.csv',[r for r in prepared if float(r['rate'])==.0001])
    write_csv(BASE/'model_summary.csv',summaries)
    w=np.array(config['effective_scales'])**-2;w/=w.mean()
    report={'configuration_id':config['configuration_id'],'configuration':config,'status':'frozen_point_pack',
        'normalized_weights':w.tolist(),'point_rows':148,'case_count':37,'sources':sources,
        'outputs':{p.name:sha(p) for p in BASE.glob('*.csv')},'new_fits':0,'new_quantum_shots':0}
    (BASE/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'frozen','out':str(BASE),'models':summaries},indent=2))


if __name__=='__main__':
    main()
