# Quantum order-finding error mitigation

Code for **“Mitigating errors of a polynomial-depth quantum circuit: quantum order-finding.”**

The calculation starts with noisy order-finding shots, reconstructs the output distribution using arithmetic-register Reset and zero-noise extrapolation (ZNE), and evaluates TVD, the normalized peak-background gap, and exact-order recovery.

## From shots to results

| Step | Code | Input → output |
| --- | --- | --- |
| 1. Compile the arithmetic circuit | `scripts/build_matched_operation_caches.py` | Instance configurations → compiled operation caches |
| 2. Generate shots | `shor_qem.simulator`, `shor_qem.folding_pattern_simulator`, `shor_qem.followup_cluster` | Circuits, gate-error rates, and noise scales → Raw/Reset shot records and counts |
| 3. Assemble the analysis input | `scripts/pack_simulator_counts.py` | Eight Reset scale records and the two baseline records → `counts.npz`, retaining the original 20k/140k blocks |
| 4. Reconstruct and evaluate | `shor_qem.portable_c1 reconstruct` | `counts.npz` → four ZNE distributions, TVD, and normalized gap |
| 5. Resample and summarize | `shor_qem.portable_c1 bootstrap` | The same count parents → paired bootstrap draws and percentile intervals |
| 6. Evaluate order recovery | `shor_qem.portable_c1 decode` | Reconstructed distributions → strict single-candidate exact-order probabilities |

The [complete execution guide](docs/RUNNING_EXPERIMENT.md) connects these commands and filenames, including generation of the Raw and Reset baseline extensions. The [input-format reference](docs/DATA_FORMAT.md) describes the arrays at each stage.

## Study settings

The three instances are `N=119`, `(a,r)=(50,2),(13,4),(16,6)`. Each uses the retained 17-qubit arithmetic template, 14 measured phase bits, and 49,686 noisy two-qubit gate locations before folding.

- Nominal noise scales: `1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3`. Fits use the effective scales after integer fold rounding.
- Shots: 20,000 at each Reset scale; Raw and unfolded Reset each total 160,000 for the primary comparison.
- Reconstruction: C1 observed-support fitting and simplex projection, inverse-square scale weights normalized to mean one, and **ridge=0 for all four models**.
- Exponential fits: fixed offset `1/16384`, 200 single-rate candidates, and 19,900 dual-rate pairs.
- Configuration `rates` store the nonidentity two-qubit Pauli-fault probability `p2`. For example, `0.0001` means `p2=0.01%`.

The exact grids, effective scales, and resampling conventions are in [Calculation settings and recorded outputs](docs/REPRODUCIBILITY.md).

## Recorded result example

At `p2=0.01%`, the saved shot records gave the following results under the settings above. Each entry is **(TVD, normalized peak-background gap)**; lower TVD and higher gap are better.

| Route | `a=50, r=2` | `a=13, r=4` | `a=16, r=6` |
| --- | --- | --- | --- |
| Raw, 160k | (0.916325, 0.062338) | (0.897319, 0.075225) | (0.915268, 0.034723) |
| Reset, 160k | (0.810413, 0.149188) | (0.784963, 0.167625) | (0.873115, 0.053561) |
| Linear ZNE | (0.729672, 0.212796) | (0.694815, 0.233193) | (0.831447, 0.077574) |
| Quadratic ZNE | (0.539951, 0.374465) | (0.496102, 0.394938) | (0.725293, 0.153715) |
| Single-exponential ZNE | (0.355753, 0.526106) | (0.340198, 0.525216) | (0.582272, 0.248730) |
| Dual-exponential ZNE | (0.045278, 0.909444) | (0.071599, 0.849886) | (0.833333, 0.000000) |

These values were read from the final all-zero-ridge `primary_rate_results.csv` and checked against the saved parent distributions. Reconstructing the three saved count inputs with `portable_c1` reproduced all four models' vectors and TVD/gap values exactly in the tested environment. The [source record](docs/REPRODUCIBILITY.md#recorded-results) identifies the original files and hashes.

The original shot archives and bootstrap caches are stored separately from this code repository. The table describes that recorded run; newly generated shots are another sample of the specified stochastic model.

## Install

From a checkout, use Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

CPU analysis uses NumPy and Matplotlib. For NVIDIA shot simulation, additionally install `.[gpu-cuda12]` or `.[gpu-cuda13]` to match the CUDA environment. Choose one CuPy CUDA distribution.

## Run the analysis

After assembling `counts.npz` with step 3 of the execution guide, the following commands analyze the `a=13, r=4` example:

```bash
python -m shor_qem.portable_c1 reconstruct --input results/n119_a13_p0p0001/counts.npz --order 4 --output-dir results/n119_a13_p0p0001/point
python -m shor_qem.portable_c1 bootstrap --input results/n119_a13_p0p0001/counts.npz --order 4 --replicates 10000 --seed 20260722 --batch-size 25 --confidence .95 --output-dir results/n119_a13_p0p0001/bootstrap
python -m shor_qem.portable_c1 decode --input results/n119_a13_p0p0001/point/parent_estimates.npz --N 119 --a 13 --order 4 --bits 14 --output-dir results/n119_a13_p0p0001/order
```

Read the outputs as follows:

| File | Contents |
| --- | --- |
| `point/points.json` | Original-count TVD and gap for Raw, Reset, and the four ZNE models |
| `point/parent_estimates.npz` | Pre/postprojection vectors, selected exponential rates, and evaluation arrays |
| `bootstrap/intervals.json` | Percentile intervals for individual metrics and paired improvements over Reset160k |
| `bootstrap/scalar_bootstrap_draws.npz` | Replicate-level metrics and rate selections |
| `order/decoder.json` | Exact-order probability for each distribution |

The bootstrap reselects observed support and exponential rates in every replicate. Its four models share the same resampled counts, and Reset160k shares its initial 20,000 shots with the scale-one ZNE input.

## Quick CPU example

To try the file flow before running the GPU simulation, use this small **synthetic four-bit example**:

```bash
mkdir demo
python scripts/make_synthetic_counts.py --output demo/counts.npz --bits 4 --order 4 --seed 1
python -m shor_qem.portable_c1 reconstruct --input demo/counts.npz --order 4 --output-dir demo/point
python -m shor_qem.portable_c1 bootstrap --input demo/counts.npz --order 4 --replicates 5 --seed 20260722 --batch-size 25 --confidence .95 --output-dir demo/bootstrap
python -m shor_qem.portable_c1 decode --input demo/point/parent_estimates.npz --N 119 --a 13 --order 4 --bits 4 --output-dir demo/order
```

The five-replicate example checks the interface; the study analysis uses 14 bits and 10,000 replicates. Choose new output destinations for each run.

## Source organization and checks

`src/shor_qem/` contains the simulator and analysis modules; `configs/` contains the instance and follow-up specifications. `archive/paper-option2/` preserves the original production scripts, including Appendix D reset-readout analysis and the S6 order-recovery export. The [execution guide](docs/RUNNING_EXPERIMENT.md) uses the packaged interfaces; [Calculation settings and recorded outputs](docs/REPRODUCIBILITY.md) maps the production scripts to their recorded results.

```bash
python -m pytest -q
python scripts/verify_source_snapshot.py
python scripts/verify_release.py
git rev-parse HEAD
```

`SOURCE_MANIFEST.csv` records the original source hashes, and `RELEASE_MANIFEST.json` records the current package contents. See [Validation](docs/VALIDATION.md) for the checks performed.

Repository: <https://github.com/Drotonic/shor-order-finding-qem> — private review, version `v0.1.1`. [License selection](docs/LICENSE_STATUS.txt) is pending.
