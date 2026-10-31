# Recorded measurement data and bootstrap caches

Download the data from the [v0.1.2 release](https://github.com/Drotonic/shor-order-finding-qem/releases/tag/v0.1.2). The seven ZIP assets share the `shor-paper-data/` directory and contain disjoint file sets. Extract them into the same location. `dataset_assets.json` and `SHA256SUMS.txt` identify the exact download sizes and SHA256 checksums.

| Release asset | Contents |
| --- | --- |
| `shor-measurement-records.zip` | 138 original native inputs; 37 portable count-parent packs; 33 reset-readout records and 11,259 completed shot chunks |
| `shor-bootstrap-equal-shot-n119_a13.zip` | All 11 equal-shot rate conditions for a13, each with 10,000 replicates and 400 batch checkpoints |
| `shor-bootstrap-equal-shot-n119_a50.zip` | All 11 equal-shot rate conditions for a50, with the same complete caches |
| `shor-bootstrap-equal-shot-n119_a16.zip` | All 15 equal-shot rate conditions for a16, with the same complete caches |
| `shor-bootstrap-gate-matched.zip` | Three separate gate-application-matched conditions at p2=0.01%, each with 10,000 replicates and 400 checkpoints |
| `shor-bootstrap-order-success.zip` | 37 strict order-recovery caches, 10,000 paired replicates per condition, 14,800 checkpoints, and the 222-row interval table |
| `shor-data-index.zip` | Aggregate TVD/gap tables, approved configuration, verification record, bootstrap publication manifest and dataset README |

## From the recorded data to the reported results

The data are finite-shot **stochastic simulator** outputs for N119, not quantum-hardware measurements. The three instances are `(a,r)=(50,2),(13,4),(16,6)`. The final reconstruction uses C1 observed support, mean-one weights proportional to lambda^-2, and zero ridge for every fitting model.

```bash
gh release download v0.1.2 --repo Drotonic/shor-order-finding-qem --pattern 'shor-*.zip' --pattern SHA256SUMS.txt --pattern dataset_assets.json --dir downloads
(cd downloads && sha256sum -c SHA256SUMS.txt)
for archive in downloads/shor-*.zip; do unzip -n "$archive" -d data; done
python scripts/reproduce_published_results.py --data-root data/shor-paper-data --output-dir results/recorded-tables
```

The final command reads the saved original-parent estimates and replicate-level arrays to regenerate 474 TVD/gap rows (444 equal-shot and 30 gate-matched) and 222 order-success interval rows, checking them against the published aggregate tables. No new shot trajectories, fitting or bootstrap draws are needed to reduce these caches.

For a fresh point reconstruction from the published count parents:

```bash
python -m shor_qem.portable_c1 reconstruct --input data/shor-paper-data/measurements/counts/n119_a13/rate_0p0001.npz --order 4 --output-dir results/published-a13-point
python -m shor_qem.portable_c1 decode --input results/published-a13-point/parent_estimates.npz --N 119 --a 13 --order 4 --bits 14 --output-dir results/published-a13-order
```

To generate a fresh paired bootstrap from those same empirical parents:

```bash
python -m shor_qem.portable_c1 bootstrap --input data/shor-paper-data/measurements/counts/n119_a13/rate_0p0001.npz --order 4 --replicates 10000 --seed 20260722 --batch-size 25 --confidence .95 --output-dir results/published-a13-new-bootstrap
```

The portable bootstrap performs new refits and resampling; it is distinct from reducing the archived cache. The original cache production used the preserved historical RNG ordering and dependencies. For shot generation from the circuit configurations, follow [RUNNING_EXPERIMENT.md](RUNNING_EXPERIMENT.md).

## Scope and pairing

- Equal-shot: 37 rate/case conditions, eight Reset scales with 20,000 shots each. Raw160k and Reset160k retain separate original 20k and extension 140k empirical parents. Four child RNG streams preserve their pairing.
- Gate matched: three conditions use 11,188 resampled shots per scale and a 148,812-shot Reset extension, with their separate two-stream design. These caches are not interchangeable with the equal-shot caches.
- Order success: the same 37 equal-shot paired draws, evaluated with one `Fraction(y,16384).limit_denominator(119)` candidate and the strict exact-order rule. These are not additional independent physical experiments.
- Reset readout: three cases at p2=0.005% and lambda=1, each with Raw plus errors 0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 and 1 percent; 20,000 shots per record. Their count-based TVD, gap and success table is `measurements/reset_readout/point_metrics.csv`. Incomplete 2% and abandoned partial runs are excluded.

## File integrity and recorded metadata

The native measurement inputs, reset-readout records, original-parent arrays and final scalar-cache NPZs preserve their original bytes. `measurements/manifest.json` lists measurement file hashes and source-relative identities. `bootstrap/publication_manifest.json` lists the source and distributed SHA256 hashes for every bootstrap payload file.

Personal absolute paths were replaced by portable source identifiers in JSON metadata and the `checkpoint_json` metadata field of C1 batch NPZs. Every other NPY member payload and all numerical values, selected candidates, support masks, RNG states and numerical array hashes were preserved. Original embedded checkpoint/source hashes still identify the original source files. Use the manifest's `source_sha256` → `public_sha256` mapping when checking a metadata-sanitized file; do not compare an original container hash directly with its redacted container.

Full pre/postprojection vectors are retained for original-parent estimates. Bootstrap files retain scalar results, support masks and fitting choices; full D-dimensional probability vectors for every replicate were not stored. The 95% summaries are the 2.5th and 97.5th percentiles of the conditional empirical resamples. Their center points remain the original-record estimates. Zero-width central intervals can coexist with nonzero tails; the underlying draws are included for inspection.

All original source copies remain covered by `SOURCE_MANIFEST.csv`. Data-release checksums and the package's `RELEASE_MANIFEST.json` cover the separately distributed data and current code/documentation, respectively.
