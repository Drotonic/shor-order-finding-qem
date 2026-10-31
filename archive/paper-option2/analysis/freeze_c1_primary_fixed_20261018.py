#!/usr/bin/env python3
"""Freeze approved mixed-ridge C1 point results from existing family caches.

No fits, bootstrap, circuit simulation, or canonical manuscript edits.
This pack is the input specification for a NEW bootstrap runner, not a claim
that the older scripts have been migrated or that their intervals apply.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'analysis/c1_primary_fixed_20261018.json'
SEARCH = ROOT / 'analysis_results/c1_parameter_search_20261018/search'
OUT = ROOT / 'analysis_results/c1_primary_fixed_20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.loads(CONFIG.read_text())
    assert not OUT.exists(), 'Never overwrite a frozen point pack; use a new version.'
    methods = list(config['ridges'])
    sources = {str(CONFIG.relative_to(ROOT)): sha(CONFIG),
               str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    rows, summary = [], []
    all_ids = None
    for method, ridge in config['ridges'].items():
        suffix = '1em06' if method == 'dual_exponential_zne' else '0'
        stage = 'stage1' if method == 'dual_exponential_zne' else 'zero_ridge_beta2'
        family = SEARCH / stage / 'families' / f'{method}__alpha_2__ridge_{suffix}'
        meta = json.loads((family / 'manifest.json').read_text())
        assert meta['status'] == 'complete' and meta['failed_condition_count'] == 0
        assert meta['alpha'] == 2 and meta['ridge'] == ridge
        for name, digest in meta['artifacts'].items():
            assert sha(family / name) == digest
            sources[str((family / name).relative_to(ROOT))] = digest
        sources[str((family / 'manifest.json').relative_to(ROOT))] = sha(family / 'manifest.json')
        with (family / 'points.csv').open(newline='') as handle:
            current = list(csv.DictReader(handle))
        assert len(current) == 37
        ids = [r['case_id'] for r in current]
        if all_ids is None:
            all_ids = ids
        assert ids == all_ids
        with np.load(family / 'selected_fit_cache.npz', allow_pickle=False) as fit:
            assert np.array_equal(fit['case_ids'], np.array(ids))
            assert np.array_equal(fit['lambdas'], np.array(config['effective_scales']))
            assert float(fit['alpha']) == 2 and float(fit['ridge']) == ridge
        for r in current:
            assert r['status'] == 'ok' and r['variant'] == 'C1'
            assert int(r['bootstrap_replicates']) == 0
            assert int(r['zne_circuit_shots']) == 160000
            rows.append({
                'configuration_id': config['configuration_id'],
                **{k: r[k] for k in ('case_id', 'instance', 'a', 'order', 'rate', 'rate_percent', 'method',
                                    'weight_exponent', 'ridge', 'tvd', 'normalized_grouped_peak_margin',
                                    'reset_minus_tvd', 'gap_minus_reset', 'selected_gamma_index',
                                    'selected_gamma_1', 'selected_gamma_2', 'normal_condition_number',
                                    'intercept_l1_norm', 'l2_projection_correction')},
                'source_points': str((family / 'points.csv').relative_to(ROOT)),
                'source_selected_fit_cache': str((family / 'selected_fit_cache.npz').relative_to(ROOT)),
                'new_bootstrap_available': False})
        per_case_means = [np.mean([float(r['tvd']) for r in current if r['instance'] == inst])
                          for inst in ('n119_a13', 'n119_a50', 'n119_a16')]
        summary.append({'method': method, 'ridge': ridge, 'beta': 2,
                        'case_balanced_mean_tvd': float(np.mean(per_case_means)),
                        'positive_gap_conditions_above_1e-10': sum(float(r['normalized_grouped_peak_margin']) > 1e-10 for r in current),
                        'conditions': 37})
    assert len(rows) == 148
    OUT.mkdir()
    write_csv(OUT / 'point_estimates.csv', rows)
    write_csv(OUT / 'primary_rate_points.csv', [r for r in rows if float(r['rate']) == .0001])
    write_csv(OUT / 'model_summary.csv', summary)
    weights = np.array(config['effective_scales']) ** -2
    weights /= weights.mean()
    report = {'configuration_id': config['configuration_id'],
              'status': 'approved_point_pack_bootstrap_pending',
              'configuration': config, 'normalized_weights': weights.tolist(),
              'point_rows': 148, 'case_count': len(all_ids), 'model_order': methods,
              'new_fits': 0, 'new_bootstrap': 0, 'new_quantum_shots': 0,
              'sources': sources,
              'outputs': {p.name: sha(p) for p in OUT.glob('*.csv')},
              'manuscript_note': 'Existing main/supplement PDFs remain legacy; do not claim migration is complete.'}
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'out': str(OUT), 'summaries': summary}, indent=2))


if __name__ == '__main__':
    main()
