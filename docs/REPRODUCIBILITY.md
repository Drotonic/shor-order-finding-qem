# Reproducibility scope

This repository separates a portable CPU analysis interface from preserved historical source. It is a code release, not a complete numerical-results archive. The repository is currently private; no DOI or immutable public data deposit is claimed.

## What can be reproduced from this repository alone?

- Install and exercise the distributed Python package on CPU.
- Generate explicitly synthetic count inputs and run a new C1 analysis with fresh bootstrap draws.
- Inspect the circuit, noise, folding, estimation, metric, order-recovery, and historical analysis implementations.
- On a compatible GPU system, reconstruct operation caches and generate new simulator records.

The manuscript's reported point estimates, intervals, figures, and historical bootstrap draws cannot be regenerated from code alone. Actual measurement/trajectory records, numerical outputs, compiled-operation pickles, checkpoints, and manuscript/figure assets are not included. No promise of access to those excluded artifacts is made here.

## Scientific configuration

The retained study scope is `N = 119`, with `(a, r) = (50, 2), (13, 4), (16, 6)`. Each compiled retained template uses 17 simulated qubits and `L = 14` phase-estimation rounds, with 49,686 noisy two-qubit gate locations, or 3,549 per round. These retained template counts are not a claim that an independently simplified arithmetic circuit is minimal.

The two-qubit error parameter follows `p2 = 15 ε / 16`: a stochastic nonidentity two-qubit Pauli is applied with probability `p2`. Read `ε` as a probability, not a percentage. Single-qubit errors are not silently substituted for this convention.

The matched configurations use a common Qiskit optimization-level-1 compilation rule and transpiler seed 42. Shot-wise folding selects an independent exact-cardinality subset for each instance, physical error rate, scale, phase-estimation round, and shot. The nominal scale labels are:

```text
1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0
```

Integer fold counts give the fitting scales:

```text
1.0, 1.099746407438715, 1.2000563539025078, 1.4001127078050155,
1.800225415610031, 2.199774584389969, 2.5998872921949845, 3.0
```

Use effective scales for fits; nominal labels alone do not specify the same estimator. The `a = 16` augmented configuration adds physical-rate points; it does not change the eight fitting scales.

## Frozen C1 analysis profile

For a given input or bootstrap replicate, let `U` be the union of outcomes observed across the eight Reset scale records. C1 includes each outcome in `U` at every scale, retaining zero frequencies at scales where it was absent. Outcomes outside `U` are excluded from both fitting and simplex projection and receive zero final probability. This is an intentional estimator choice, not merely a lossless compression of a full-outcome fit.

The profile is:

| Setting | Value |
| --- | --- |
| Full outcome dimension | `D = 2^14 = 16384` |
| Regression weights | `lambda_effective**(-2)`, divided by their mean |
| Models | Linear, quadratic, single exponential, dual exponential |
| Ridge/amplitude penalties | Zero for all four models |
| Exponential offset | `1 / D = 1 / 16384` |
| Single-rate grid | 200 equally spaced values from `0.01` to `10` |
| Dual-rate grid | 200 equally spaced values from `0` to `10`; all 19,900 ordered index pairs with `gamma1 < gamma2` |
| Exponential selection | Shared rate or rate pair across observed outcomes, using the summed weighted observed-support fit objective |
| Tie behavior | Native first `argmax`; no tolerance-based tie replacement |
| Final constraint | Euclidean projection onto the probability simplex on `U` |

Ideal distributions, multiplicative orders, and peak groups are evaluation references, not inputs to exponential-rate selection. Evaluation uses the finite-resolution QPE target, which matters particularly when the order does not divide `2^14`.

The all-zero-ridge C1 profile was selected after inspection of earlier analysis results; it was not preregistered or independently validated. Conditional bootstrap intervals do not correct for that analysis selection and are not a post-selection coverage guarantee.

For this profile, use `python -m shor_qem.portable_c1`, not the original `bootstrap_analysis` CLI. Some preserved estimator defaults use a nonzero ridge (`1e-8`) and full-outcome fitting/projection. Those original defaults are retained source history, not the final C1 specification.

## Equal-shot paired bootstrap

The primary budget is 160,000 circuit shots per route:

- Raw160: a 20,000-shot Raw initial block plus its independent 140,000-shot extension.
- Reset160: the 20,000-shot Reset `lambda = 1` row plus its independent 140,000-shot extension.
- Each ZNE model: eight Reset scale rows of 20,000 shots each.

At each replicate, the Reset scale rows, Reset extension, Raw initial block, and Raw extension are resampled from their respective parents. The **same resampled Reset `lambda = 1` block** contributes to both the ZNE fit and Reset160. All four fitted ZNE models use the same resampled scale records. Paired differences are computed within that replicate, not by combining separately generated marginal intervals.

The observed support is reselected from each replicate's Reset scale draws; exponential rates are reselected and projection is repeated. The fixed weighting exponent, ridge values, offset, and grid definitions do not change across replicates. Intervals are conditional on the supplied saved parents; they do not establish coverage over new hardware, new compilation, or a new folding campaign.

Do not split merged 160,000-shot histograms to invent the original parents. Do not independently resample a merged Reset160 histogram when claiming the historical Reset/ZNE pairing. Without all four parent blocks, the paired comparison is unavailable. The portable implementation rejects an incomplete parent specification rather than manufacturing pairing.

Equal shots do **not** mean equal gate executions: the eight folded scale records have greater aggregate noisy-gate cost than the unfolded direct baselines. The archive also contains separate resource-budget investigations; these do not relabel the primary equal-shot analysis as gate-budget matched.

The portable CPU workflow implements new draws from explicitly supplied counts. Its random-number stream, batching, and computational path are not promised to reproduce the exact saved historical bootstrap cache bit for bit, even when its inputs are derived from the same parents.

For repeatable portable runs, retain the count-file hash, full repository commit, NumPy/BLAS environment, seed, batch size, replicate count, and confidence level. The output label is explicitly `portable fresh bootstrap NOT bitwise historical archived replay`. Five-replicate synthetic examples in the README exercise the interface only; they are not the historical 10,000-replicate analysis or an adequate Monte Carlo precision check.

## GPU generation and environment limits

The base package requires Python 3.10+, NumPy, and Matplotlib. The optional GPU extra uses Qiskit and `cupy-cuda12x`; a compatible NVIDIA driver/runtime and adequate GPU memory are also required. For CUDA 13 use a compatible CUDA 13 CuPy distribution in a separate environment, replacing rather than supplementing the CUDA 12 distribution.

From the repository root, the cache builder is:

```bash
python scripts/build_matched_operation_caches.py
```

It generates trusted local operation-cache pickles from the matched configurations and verifies gate counts. Existing destinations are protected unless its explicit overwrite option is requested. Do not load untrusted pickle files. Generated caches and simulator results are intentionally ignored by Git.

The matched shot-wise simulator and the older fixed-pattern simulator are distinct profiles. `shor_qem.folding_pattern_simulator` is the matched shot-wise generator; `shor_qem.simulator` also contains the original circuit construction and no-reset generation interface. The distributed source may preserve historical endpoint conventions; the matched configurations require every scale to come from the shot-wise profile. Do not mix historical endpoints with newly compiled matched records.

Current dependency constraints permit installation; they are not a recovered lockfile for the production machines. Qiskit transpilation, CuPy kernels, CUDA, drivers, floating-point libraries, and scheduling can affect byte-level output. Record actual package/runtime versions, config hashes, cache hashes, seeds, and source revision with any newly generated dataset. Passing template-count checks is not proof of bitwise historical replay.

## Historical archive

`archive/paper-option2/analysis/` preserves the manuscript-era analysis source and frozen specifications. The original package source is retained separately under `archive/paper-option2/shor_error_mitigation/`. These files document historical processing and exploratory/sensitivity profiles, not just the final all-zero-ridge C1 profile.

The archived analysis is not fully portable: it expects omitted saved data, mixed-profile result/bootstrap caches, pinned source snapshots, and parts of the original worktree layout. Manuscript-specific guard/support files that are not software are not included. Archive entry points are provenance references, not supported one-command reproduction instructions; their requirements must be restored and verified before replay. Do not replace these requirements with synthetic data while labeling the output as the paper's results.

Appendix D source is preserved under:

```text
archive/paper-option2/experiments/arithmetic_reset_readout_20261020/
archive/paper-option2/experiments/arithmetic_reset_readout_refinement_20261020/
```

Its reported scope comprises 20,000-shot Raw and `0%, 0.01%, 0.1%, 0.2%, 0.3%, 0.4%, 0.5%` reset/readout-error conditions. The 1% condition was unfinished and the 2% condition was not run. Configuration or loop entries for those conditions are not evidence of reported results.

The Supplement S6 archive at `archive/paper-option2/supplements/s6_order_recovery_20261031/` preserves its original export and verification scripts, but not the saved distributions, original parent manifests, or exported numerical/figure artifacts on which they depended. The `decode` subcommand of `shor_qem.portable_c1` supplies a portable decoder for new inputs. Neither the source archive nor a successful synthetic example substitutes for those excluded historical inputs.

## Pinning and integrity

The root `SOURCE_MANIFEST.csv` and `RELEASE_MANIFEST.json` document the selected source and release inventory. They do not contain raw measurements and do not certify reproduction of numerical results. Compare the release inventory and hashes when auditing the package, and pin the actual release commit before citing the code:

```bash
git rev-parse HEAD
```

Use that full commit identifier in the code-availability statement or an immutable commit URL. Do not substitute a historical source-manifest digest for the repository commit, invent a release tag/DOI, or describe the private repository as publicly available.
