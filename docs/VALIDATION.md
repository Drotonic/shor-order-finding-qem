# Release validation

## v0.1.1: executable workflow and native-count export

The workflow documentation now links native shot generation, count packing, C1 reconstruction, bootstrap, and order decoding. `scripts/pack_simulator_counts.py` supplies the native-NPZ conversion step.

- The full CPU suite passed **87 tests**, including 35 native-export tests.
- Packing the actual `N=119`, `p2=0.01%` records succeeded for all three study instances. All 30 input hashes matched the recorded parent manifests.
- Exported Reset scale and extension frequencies, combined Raw160/Reset160 distributions, and effective scales matched the saved parent arrays exactly. All four direct routes' TVD and normalized-gap metrics matched exactly as well.
- The exporter preserved the recorded initial 20k and extension140k boundaries and verified Reset scale-one shot identity. Actual-data checks wrote only temporary count packs; they used no new trajectories, fitting, or bootstrap.
- The README example reports the recorded results from `primary_rate_results.csv`; its source hash is listed in [Recorded results](REPRODUCIBILITY.md#recorded-results).

The original scientific modules and their 103 source-snapshot copies remain unchanged. The `v0.1.0` validation below records the reconstruction and analysis-interface checks.

## v0.1.0: analysis interface

The `v0.1.0` private-review code snapshot was checked on 2026-10-31. This is software validation, not a new physical experiment or a certification of statistical coverage.

## Checks performed

- All 52 distributed CPU tests passed using Python 3.13.11 and NumPy 2.4.6, with one BLAS/OpenMP thread. The 25 portable-interface tests cover input validation, C1 support, fitting, paired sampling, and strict S6 decoding.
- Editable installation succeeded in a separate virtual environment using already available dependencies; the active GPU simulation environment was not modified. This was not a clean-network dependency-install test across every supported Python/CUDA version.
- The documented synthetic pipeline completed: four-bit count generation, reconstruction, five fresh paired bootstrap replicates, and strict single-candidate decoding. These toy outputs are not paper data and are not distributed.
- The source checksum verifier passed all 103 unchanged original-source copies listed in `SOURCE_MANIFEST.csv`.
- An independent local comparison used the saved `N=119`, `p2=0.01%` parents for `(a,r)=(13,4),(50,2),(16,6)`. All 12 method/case preprojection and postprojection vector pairs were exactly equal to their saved C1/all-zero-ridge vectors in the tested environment. TVD, normalized peak-background gap, direct-route metrics, and selected exponential candidate indices also matched exactly.
- That comparison verified original input hashes and preserved 20k/140k shot boundaries. It did not invent parent blocks by arbitrarily splitting merged histograms. These actual inputs are excluded from the source release, so this check requires separately retained study data.
- Strict S6 decoder masks were tested for all 16,384 outputs: 69, 70, and 48 exact-order-success bins for `(a,r)=(50,2),(13,4),(16,6)`, respectively. Verified nonminimal denominators are not reduced to the true order.
- A read-only release scan found no high-confidence credential tokens, private keys, hardcoded personal home paths, or operational host IPs in the selected text files. This is a bounded packaging check, not an exhaustive security audit.

## Not performed for this release

No new GPU trajectories, quantum-backend submissions, or research-data bootstrap campaign were run. Historical 10,000-replicate caches were not regenerated; the portable fresh bootstrap is not advertised as bitwise historical replay. No statistical-coverage, real-device noise-model, or cross-platform bitwise-reproducibility guarantee follows from these tests.

Recheck the distributed tests and original-source hashes from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests -q
python scripts/verify_source_snapshot.py
git rev-parse 'v0.1.0^{commit}'
```

The release manifest lists source/configuration/documentation payload hashes; it is not a measurement-data manifest. The private repository and absent license must be reviewed before any public software-availability claim.
