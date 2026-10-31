# Release validation

## v0.1.2: public measurement and bootstrap data

The release adds seven scientific-data ZIP assets (1,247,276,792 bytes total), a download index and SHA256 checksums. Extracted data contain 42,758 files, including the two publication manifests and README files. No new trajectories, fits or bootstrap draws were generated for packaging.

- All 11,468 inventoried measurement payloads and 31,286 inventoried bootstrap payloads passed an independent file-SHA256 comparison. All ZIP members were then checked byte-for-byte against those staged public files.
- The measurement bundle preserves 138 native source inputs, 33 completed reset-readout records and 11,259 completed shot chunks. All 37 portable count-parent exports preserve the original initial/extension blocks. Original-parent frequency checks had maximum absolute difference 6.94e-18; 296 direct metric checks had maximum difference 3.47e-18.
- The 40 TVD/gap cases and 37 strict-success cases each retain 10,000 replicates and 400 contiguous batch checkpoints. Final/cache concatenation checks covered 866,840,000 numerical values in 206,800 array-byte comparisons.
- Only private host paths in metadata were made portable. All 192,000 other NPY member payloads checked across the modified C1 checkpoint containers were byte-identical. Final parent and scalar NPZ files retain their original bytes. The publication manifest maps original hashes to distributed hashes.
- Publication scans checked 370 bootstrap text/JSON files and the string metadata of 30,917 NPZs, together with the measurement files; no object-dtype NPZ arrays, private paths or high-confidence credentials were found. Private builder audit reports were excluded.
- `scripts/reproduce_published_results.py` regenerated the full 474-row TVD/gap CSV and 222-row strict-success CSV **byte-identically** from the distributed parent/scalar arrays. This is cache reduction, not a new bootstrap run. The new reader uses distributed-file SHA256 hashes and explicit array labels.
- The CPU suite passed **124 tests**, including 37 new cached-reader tests. The 103 preserved scientific-source copies remain unchanged.

The data guide identifies the separate equal-shot and gate-matched allocations and explains how to reproduce the recorded tables or start fresh reconstruction from the released counts.

## Public availability: 2026-10-31

The repository was made publicly accessible on 2026-10-31. The publication update changes repository visibility and documentation only; the scientific code remains at version `v0.1.1`. Earlier private-review snapshots and their tags are preserved. License selection remains pending.

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
- That comparison verified original input hashes and preserved 20k/140k shot boundaries. It did not invent parent blocks by arbitrarily splitting merged histograms. The actual inputs, originally retained separately from the v0.1.0 source snapshot, are now provided in the v0.1.2 data assets.
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

The release manifest lists source/configuration/documentation payload hashes; it is not a measurement-data manifest. The repository is public, and the separate license decision is recorded in `LICENSE_STATUS.txt`.
