# Recorded simulation and hardware data

Download the data from the
[v0.2.1 release](https://github.com/Drotonic/shor-order-finding-qem/releases/tag/v0.2.1).
It contains nine ZIP assets: seven simulation archives carried forward
byte-for-byte from v0.1.2 and two hardware archives carried forward
byte-for-byte from v0.2.0.

`release_assets.json` is the unified v0.2.1 catalogue, and
`SHA256SUMS-all.txt` verifies all nine ZIPs.  The retained
`dataset_assets.json` and `SHA256SUMS.txt` are historical simulation-only
metadata for the original seven v0.1.2 archives.  They intentionally omit the
hardware archives.

| Release asset | Contents |
| --- | --- |
| `shor-measurement-records.zip` | 138 original native inputs; 37 portable count-parent packs; 33 reset-readout records and 11,259 completed shot chunks |
| `shor-bootstrap-equal-shot-n119_a13.zip` | All 11 equal-shot rate conditions for a13, each with 10,000 replicates and 400 batch checkpoints |
| `shor-bootstrap-equal-shot-n119_a50.zip` | All 11 equal-shot rate conditions for a50, with the same complete caches |
| `shor-bootstrap-equal-shot-n119_a16.zip` | All 15 equal-shot rate conditions for a16, with the same complete caches |
| `shor-bootstrap-gate-matched.zip` | Three separate gate-application-matched conditions at p2=0.01%, each with 10,000 replicates and 400 checkpoints |
| `shor-bootstrap-order-success.zip` | 37 strict order-recovery caches, 10,000 paired replicates per condition, 14,800 checkpoints, and the 222-row interval table |
| `shor-data-index.zip` | Aggregate TVD/gap tables, approved configuration, verification record, bootstrap publication manifest and dataset README |
| `shor-hardware-n3-a2-ibm_aachen.zip` | Recorded N=3, a=2 `ibm_aachen` hardware counts, reconstructed distributions, metric/bootstrap summaries, execution metadata and validation |
| `shor-hardware-n3-a2-ibm_boston.zip` | Recorded N=3, a=2 `ibm_boston` hardware data with the same internal filenames and schema as the Aachen archive |

## Download, verify and extract

Download the complete v0.2.1 data release and verify every ZIP against the
unified checksum file:

```bash
gh release download v0.2.1 --repo Drotonic/shor-order-finding-qem \
  --pattern 'shor-*.zip' --pattern release_assets.json \
  --pattern SHA256SUMS-all.txt --pattern dataset_assets.json \
  --pattern SHA256SUMS.txt --dir downloads
(cd downloads && sha256sum -c SHA256SUMS-all.txt)
```

The seven simulation ZIPs contain disjoint paths under one common
`shor-paper-data/` root.  Extract all seven into the same directory:

```bash
mkdir -p data/simulation
for archive in \
  downloads/shor-measurement-records.zip \
  downloads/shor-bootstrap-equal-shot-n119_a13.zip \
  downloads/shor-bootstrap-equal-shot-n119_a16.zip \
  downloads/shor-bootstrap-equal-shot-n119_a50.zip \
  downloads/shor-bootstrap-gate-matched.zip \
  downloads/shor-bootstrap-order-success.zip \
  downloads/shor-data-index.zip; do
  unzip -n "$archive" -d data/simulation
done
```

The hardware ZIPs use the same internal filenames.  Extracting them into one
directory would overwrite or skip one device's files, so use separate device
directories:

```bash
mkdir -p data/hardware/ibm_aachen data/hardware/ibm_boston
unzip -n downloads/shor-hardware-n3-a2-ibm_aachen.zip \
  -d data/hardware/ibm_aachen
unzip -n downloads/shor-hardware-n3-a2-ibm_boston.zip \
  -d data/hardware/ibm_boston
```

The two hardware datasets are separate recorded experiments.

## From the recorded data to the reported results

The seven simulation archives contain finite-shot **stochastic simulator**
outputs for N119, not quantum-hardware measurements. The three instances are
`(a,r)=(50,2),(13,4),(16,6)`. The final reconstruction uses C1 observed
support, mean-one weights proportional to lambda^-2, and zero ridge for every
fitting model.

```bash
python scripts/reproduce_published_results.py \
  --data-root data/simulation/shor-paper-data \
  --output-dir results/recorded-tables
```

The final command reads the saved original-parent estimates and replicate-level arrays to regenerate 474 TVD/gap rows (444 equal-shot and 30 gate-matched) and 222 order-success interval rows, checking them against the published aggregate tables. No new shot trajectories, fitting or bootstrap draws are needed to reduce these caches.

For a fresh point reconstruction from the published count parents:

```bash
python -m shor_qem.portable_c1 reconstruct --input data/simulation/shor-paper-data/measurements/counts/n119_a13/rate_0p0001.npz --order 4 --output-dir results/published-a13-point
python -m shor_qem.portable_c1 decode --input results/published-a13-point/parent_estimates.npz --N 119 --a 13 --order 4 --bits 14 --output-dir results/published-a13-order
```

To generate a fresh paired bootstrap from those same empirical parents:

```bash
python -m shor_qem.portable_c1 bootstrap --input data/simulation/shor-paper-data/measurements/counts/n119_a13/rate_0p0001.npz --order 4 --replicates 10000 --seed 20260722 --batch-size 25 --confidence .95 --output-dir results/published-a13-new-bootstrap
```

The portable bootstrap performs new refits and resampling; it is distinct from reducing the archived cache. The original cache production used the preserved historical RNG ordering and dependencies. For shot generation from the circuit configurations, follow [RUNNING_EXPERIMENT.md](RUNNING_EXPERIMENT.md).

## Hardware-data scope

The two hardware archives preserve acquired records and the fixed analysis
products distributed with v0.2.0.  They support inspection of the counts,
reported reconstructed probability vectors, metric tables, bootstrap summaries
and figure inputs.  They are not simulator outputs and should not be merged
with `shor-paper-data/`.

The current repository includes an opt-in IBM workflow and local lifecycle
tests, but this release does **not** claim that those scripts replay the two
historical QPU jobs end to end or reproduce their reconstructed outputs
bit-for-bit from a new submission.  Re-running a hardware job depends on new
credentials, allocation, queue state and calibration.  Use the archived
hardware products for the recorded results; treat any fresh submission as a new
experiment.

## Scope and pairing

- Equal-shot: 37 rate/case conditions, eight Reset scales with 20,000 shots each. Raw160k and Reset160k retain separate original 20k and extension 140k empirical parents. Four child RNG streams preserve their pairing.
- Gate matched: three conditions use 11,188 resampled shots per scale and a 148,812-shot Reset extension, with their separate two-stream design. These caches are not interchangeable with the equal-shot caches.
- Order success: the same 37 equal-shot paired draws, evaluated with one `Fraction(y,16384).limit_denominator(119)` candidate and the strict exact-order rule. These are not additional independent physical experiments.
- Reset readout: three cases at p2=0.005% and lambda=1, each with Raw plus errors 0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 and 1 percent; 20,000 shots per record. Their count-based TVD, gap and success table is `measurements/reset_readout/point_metrics.csv`. Incomplete 2% and abandoned partial runs are excluded.

## File integrity and recorded metadata

The native measurement inputs, reset-readout records, original-parent arrays and final scalar-cache NPZs preserve their original bytes. `measurements/manifest.json` lists measurement file hashes and source-relative identities. `bootstrap/publication_manifest.json` lists the source and distributed SHA256 hashes for every bootstrap payload file.

Personal absolute paths were replaced by portable source identifiers in JSON metadata and the `checkpoint_json` metadata field of C1 batch NPZs. Every other NPY member payload and all numerical values, selected candidates, support masks, RNG states and numerical array hashes were preserved. Original embedded checkpoint/source hashes still identify the original source files. Use the manifest's `source_sha256` → `public_sha256` mapping when checking a metadata-sanitized file; do not compare an original container hash directly with its redacted container.

Full pre/postprojection vectors are retained for original-parent estimates. Bootstrap files retain scalar results, support masks and fitting choices; full D-dimensional probability vectors for every replicate were not stored. The 95% summaries are the 2.5th and 97.5th percentiles of the conditional empirical resamples. Their center points remain the original-record estimates. Zero-width central intervals can coexist with nonzero tails; the underlying draws are included for inspection.

All original source copies remain covered by `SOURCE_MANIFEST.csv`.
`SHA256SUMS-all.txt` and `release_assets.json` cover all nine v0.2.1 ZIPs;
the legacy `SHA256SUMS.txt` and `dataset_assets.json` remain valid for the seven
simulation ZIPs alone.  The package's `RELEASE_MANIFEST.json` covers the current
code and documentation rather than the separately distributed data.
