# Experiment and analysis workflow

The workflow is **circuit compilation → shot simulation → count parents → C1 reconstruction → paired bootstrap → metric tables and order-recovery evaluation**. The repository is currently private. The [README](../README.md) describes the commands and the [input-format guide](DATA_FORMAT.md) specifies the arrays passed between stages.

## Study settings

The study uses `N = 119` and `(a, r) = (50, 2), (13, 4), (16, 6)`. The retained compiled template has 17 simulated qubits, 14 phase-estimation rounds, and 49,686 noisy two-qubit gate locations, including 3,549 per round. The matched configurations use Qiskit optimization level 1 and transpiler seed 42.

Configuration `rates` specify the total probability `p2` of a nonidentity two-qubit Pauli fault. Its relation to the depolarizing-channel mixing parameter is `p2 = 15 ε / 16`. Rates in JSON/CSV files are probabilities: `0.0001` means `p2 = 0.01%`.

There are 37 rate/case conditions. Each case uses the following 11 `p2` points, in percent:

```text
0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06
```

The `a = 16` case additionally uses `0.012, 0.014, 0.016, 0.018%`.

Shot-wise folding selects an exact-cardinality subset for each instance, rate, scale, phase-estimation round, and shot. The nominal scale labels are `1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3`; integer fold counts produce the effective scales used in fitting:

```text
1.0, 1.099746407438715, 1.2000563539025078, 1.4001127078050155,
1.800225415610031, 2.199774584389969, 2.5998872921949845, 3.0
```

## Counts to reconstructed distributions

The primary comparison uses 160,000 circuit shots per route. Raw160 combines its independent 20,000-shot initial block and 140,000-shot extension. Reset160 combines the 20,000-shot Reset scale-one record and an independent 140,000-shot extension. Each ZNE estimate uses eight Reset scale records of 20,000 shots each. Folded ZNE records therefore involve more noisy-gate executions than the direct baselines; the separate gate-matched analysis uses its own allocation.

C1 takes `U` to be the union of outcomes observed in the eight Reset scale records, retaining each outcome's zero counts at scales where it was absent. It fits and projects on `U`; final probabilities outside `U` are zero. Direct extensions and ideal references enter the comparison and evaluation stages, respectively.

| Reconstruction setting | Value |
| --- | --- |
| Outcome dimension | `D = 2^14 = 16384` |
| Weights | `effective_lambda**(-2)`, divided by their mean |
| Models | Linear, quadratic, shared-rate single exponential, dual exponential |
| Ridge penalties | `0` for all four models |
| Exponential offset | `1 / D = 1 / 16384` |
| Single-rate grid | `linspace(0.01, 10, 200)` |
| Dual-rate grid | `linspace(0, 10, 200)`, with all 19,900 pairs satisfying `gamma1 < gamma2` |
| Rate selection | Summed weighted fitting objective across observed outcomes |
| Tie selection | First maximum from native `argmax` |
| Probability constraint | Euclidean simplex projection on `U` |

The portable entry point for this profile is `python -m shor_qem.portable_c1`. Preserved historical estimator interfaces also include the earlier dense-support/nonzero-ridge profiles; their original settings remain in the archive. The final C1 zero-ridge profile was selected during the analysis of the study.

`reconstruct` writes `points.json`, projected distributions in `parent_estimates.npz`, and an input/output hash manifest. Evaluation uses TVD to the finite-resolution ideal QPE distribution and the normalized peak–background gap. For `r = 6`, the normalization uses the finite-resolution ideal gap rather than `1/r`.

## Paired-bootstrap processing

The reported analysis uses 10,000 replicates per condition, seed `20260722`, and batch size 25. Each replicate resamples the eight Reset scale rows, Reset extension, Raw initial block, and Raw extension from their respective parent frequencies. The same resampled Reset scale-one block enters Reset160 and all four ZNE fits. The four models also share the other seven resampled scale rows. Keeping the initial and extension parent records separate preserves this pairing.

For each replicate, the observed support, exponential rate selection, simplex projection, and metrics are recomputed. Weights, ridge penalties, offset, and rate grids stay fixed. Paired improvements are `Reset160 − ZNE` for TVD and `ZNE − Reset160` for the gap. The 2.5th and 97.5th percentiles give the reported central 95% intervals; the center point remains the original-parent estimate. These intervals summarize resampling conditional on the supplied empirical parents and fixed analysis profile; their nominal frequentist coverage, including the earlier profile selection, has not been calibrated.

The portable `bootstrap` command generates fresh paired draws and writes `intervals.json` and `scalar_bootstrap_draws.npz`. The archived runners document the original cache-coupled processing. New shot simulations sample new trajectories; portable resampling samples supplied counts; historical replay follows the archived parent records, random-number ordering, source snapshots, and cache dependencies. The source repository contains the code for these stages, while the original numerical records use the study's archived formats. Record the full Git revision, input hash, seed, batch size, replicate count, and NumPy/BLAS environment with each run.

## Original processing and artifact trace

The following paths identify the original processing stages. Paths beginning `archive/` are preserved source in this repository. Paths beginning `analysis_results/` identify the local study artifacts used to verify the README table; those numerical artifacts are maintained separately from this code repository. The historical runners are processing records for that original artifact layout; the current command-line workflow is documented in the README.

| Stage | Code or configuration | Output / downstream input |
| --- | --- | --- |
| Compile matched arithmetic circuits | `scripts/build_matched_operation_caches.py`; `configs/n119_a13_matched.json`, `configs/n119_a50_matched.json`, `configs/n119_a16_matched.json` | Per-instance operation caches and gate-count checks |
| Generate Reset scale records | `src/shor_qem/folding_pattern_simulator.py` | Shot outcomes/counts and per-rate/per-scale metadata |
| Generate direct baseline extensions | `src/shor_qem/followup_cluster.py`; `configs/n119_a100_followup.json` | Baseline chunks and merged Raw160/Reset160 records, with initial/extension provenance; `A100` names the GPU family |
| Freeze final analysis profile | `archive/paper-option2/analysis/freeze_c1_all_zero_fixed_20261018.py`; `c1_all_zero_fixed_20261018.json` in the same directory | Final C1 zero-ridge profile and point-estimate provenance |
| Process original paired draws | `archive/paper-option2/analysis/run_c1_all_zero_bootstrap_20261018.py` and its parallel runner | `parent_estimates.npz`, scalar draws, selected rates, and manifests per condition |
| Summarize and verify | `archive/paper-option2/analysis/summarize_c1_all_zero_bootstrap_20261018.py`; `verify_c1_all_zero_bootstrap_20261018.py` | Aggregate metrics, paired intervals, and independent verification |
| Reconstruct / bootstrap current count inputs | `src/shor_qem/portable_c1.py` | `points.json`, `parent_estimates.npz`, `intervals.json`, `scalar_bootstrap_draws.npz` |

## Recorded results

The README's `p2 = 0.01%` example uses the final-profile rows from the local study CSV:

```text
analysis_results/c1_all_zero_fixed_20261018/aggregate/primary_rate_results.csv
SHA256: 18e2f65f382daa94457dfb610f503de7587a276e430d45856ed66113688a135d
```

Select `budget = equal_shot`, `rate = 0.0001`, and the relevant `instance`, `route`, and `metric`; the point estimate is the `point` column. The instances are `n119_a50`, `n119_a13`, and `n119_a16`; routes are `raw_160k`, `reset_160k`, and the four `*_zne` models. Metric names are `tvd` and `normalized_grouped_peak_margin`.

These 36 scalar values match the saved original-parent arrays exactly. Their NPZ paths are:

```text
analysis_results/c1_all_zero_fixed_20261018/bootstrap_10000/equal_shot/
  <instance>/rate_0p0001/parent_estimates.npz
```

| Instance | Original-parent NPZ SHA256 |
| --- | --- |
| `n119_a50` | `f6376d7be00bc3345dd8a841aba02fa6c3c72a92b05fe5688c360b814db078d2` |
| `n119_a13` | `cf69d4b9cd05a81dbfc4f2ef497a03bc948b141fe39391a20c3510e32c0d26f6` |
| `n119_a16` | `73eb4b06b722a50efc78350148e4c11d965ab9434904856bd937fad5f86a3ed0` |

In these original archives, read fitted metrics from `scalar_draws[route, quantity, 0]` using `routes` and `quantities` as the index labels. Read direct metrics from `direct_draws[route, metric, 0]` using `direct_routes` and `direct_metrics`. The `after__<model>` fields store projected vectors. The portable archive's metric axes are described separately in [DATA_FORMAT.md](DATA_FORMAT.md).

## Additional evaluations

Appendix D code is retained in `archive/paper-option2/experiments/arithmetic_reset_readout_20261020/` and `arithmetic_reset_readout_refinement_20261020/`. Its completed results comprise 20,000-shot Raw and reset-readout errors `0%, 0.01%, 0.1%, 0.2%, 0.3%, 0.4%, 0.5%`, applied only to arithmetic-register reset measurements. The retained configuration also records the unfinished 1% condition and unexecuted 2% option. Zero reset-readout error retains gate noise.

Supplement S6 uses the export and verification scripts in `archive/paper-option2/supplements/s6_order_recovery_20261031/`. For each outcome, the strict decoder selects `Fraction(y, 2**bits).limit_denominator(N)`, rejects a zero numerator or failure of `pow(a, q, N) == 1`, and retains a verified denominator `q` as-is. The known order enters the evaluation `q == r`. The portable `decode` command applies this rule to reconstructed probability vectors.

## Environment and version record

The CPU package requires Python 3.10+, NumPy, and Matplotlib. `requirements-reproduction.txt` records the CPU analysis package versions. GPU generation uses Qiskit and one of the `gpu-cuda12` or `gpu-cuda13` extras, together with a compatible NVIDIA driver and sufficient GPU memory. `SOURCE_MANIFEST.csv` identifies preserved source files and their hashes; `RELEASE_MANIFEST.json` records the release inventory. Record the full checked-out version with:

```bash
git rev-parse HEAD
```

New run manifests should retain the actual package/runtime versions, configuration and operation-cache hashes, seeds, and source revision. The repository's software-license status is recorded in [LICENSE_STATUS.txt](LICENSE_STATUS.txt).
