# Quantum order-finding error mitigation

Source code accompanying the study **“Mitigating errors of a polynomial-depth quantum circuit: quantum order-finding.”**

Repository: <https://github.com/Drotonic/shor-order-finding-qem>. This repository is currently **private**. It has no archival DOI, and this source release does not imply publication or acceptance of the manuscript. License selection is pending; see [license status](docs/LICENSE_STATUS.txt).

## Scope

The study compares Raw, workspace Reset, and Reset plus zero-noise extrapolation (ZNE) for three order-finding instances with `N = 119`:

| Base `a` | Multiplicative order `r` | Configuration |
| --- | --- | --- |
| 50 | 2 | `configs/n119_a50_matched.json` |
| 13 | 4 | `configs/n119_a13_matched.json` |
| 16 | 6 | `configs/n119_a16_matched.json` |

The retained 17-qubit compiled template has 14 phase-estimation rounds and 49,686 noisy two-qubit gate locations (3,549 per round). Configuration `rates` are the total nonidentity Pauli-fault probabilities `p2`, not the depolarizing-channel mixing parameter `ε`; their convention is `p2 = 15 ε / 16`. Shot-wise random folding uses eight nominal scales: `1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3`; analysis uses the effective scales after integer fold rounding.

The manuscript-facing C1 estimator fits and projects only on the union of observed Reset outcomes. Its four models use mean-one `effective_lambda**(-2)` weights and zero ridge penalties. The exponential models retain the `1 / 16384` offset and select rates on the shared 200-point single-rate grid or 19,900 dual-rate pairs without ideal-distribution labels. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the exact profile and limitations.

The primary comparison is **equal circuit shots**, not equal noisy-gate executions: Raw and Reset each use 160,000 shots; ZNE uses 20,000 shots at each of eight scales. Folding increases the gate-execution cost of ZNE.

## Repository layout

- `src/shor_qem/`: importable simulator, estimation, metric, and portable CPU-analysis code.
- `configs/`: the three matched `N = 119` configurations, the augmented rate grid for `a = 16`, and the A100 follow-up specification (`A100` names a GPU family, not base `a = 100`).
- `scripts/`: operation-cache construction and synthetic CPU-demo input generation.
- `tests/`: checks for the distributed code.
- `archive/paper-option2/`: preserved analysis, original package source, Appendix D experiment code, and Supplement S6 export/verification source. This is historical provenance, not a second supported installation.
- `docs/`: reproducibility scope, input format, code-availability wording, and license status.
- `SOURCE_MANIFEST.csv` and `RELEASE_MANIFEST.json`: source provenance and release inventory.

## CPU installation

Use Python 3.10 or newer in a fresh virtual environment, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The base installation uses NumPy and Matplotlib and does not require a GPU. The portable C1 workflow accepts count arrays, performs a fresh equal-shot paired bootstrap, and is independent of the original machine layout. Its synthetic demonstration is not a replay of the paper's measurements or historical random draws. The input contract is documented in [DATA_FORMAT.md](docs/DATA_FORMAT.md).

Run the CPU tests:

```bash
python -m pytest tests -q
```

For a small end-to-end demonstration, use a new `demo` directory:

```bash
mkdir demo
python scripts/make_synthetic_counts.py --output demo/counts.npz --bits 4 --order 4 --seed 1
python -m shor_qem.portable_c1 reconstruct --input demo/counts.npz --order 4 --output-dir demo/point
python -m shor_qem.portable_c1 bootstrap --input demo/counts.npz --order 4 --replicates 5 --seed 20260722 --batch-size 25 --confidence .95 --output-dir demo/bootstrap
python -m shor_qem.portable_c1 decode --input demo/point/parent_estimates.npz --N 119 --a 13 --order 4 --bits 4 --output-dir demo/decode
```

This toy example uses `D = 16`, not the study's `D = 16384`, and its five bootstrap replicates are only a smoke test, not an uncertainty estimate suitable for reporting. Outputs include `points.json`, `parent_estimates.npz`, `manifest.json` with input/output hashes, and (for bootstrap) `intervals.json` and `scalar_bootstrap_draws.npz`. Existing output files/directories are protected; choose new destinations for another run. See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for decoder inputs and output conventions.

**Use `shor_qem.portable_c1` for the final all-zero-ridge C1 profile.** The retained legacy `bootstrap_analysis` interface defaults to a different, dense-support/ridge profile; running that legacy CLI is not a reproduction of the final C1 analysis.

## GPU simulation

GPU generation is optional and substantially more expensive than the CPU demonstration. The `gpu-cuda12` extra installs Qiskit and the CUDA 12 CuPy package:

```bash
python -m pip install -e '.[gpu-cuda12]'
python scripts/build_matched_operation_caches.py
```

Compiled-operation pickle files are not distributed; the builder reconstructs them and checks the configured gate counts. Compilation is version-sensitive: matching gate counts alone does not establish byte-for-byte circuit or shot replay. For CUDA 13, install `.[gpu-cuda13]` **instead of** `.[gpu-cuda12]`; do not install both CuPy CUDA distributions in one environment.

The original source includes more than one simulation profile. For the matched shot-wise folding experiment, use `shor_qem.folding_pattern_simulator` with the `*_matched.json` configurations, not a historical fixed-pattern analysis profile. Consult each module's command-line help and the [reproducibility notes](docs/REPRODUCIBILITY.md) before a production run.

After building the caches, inspect a three-instance shot-wise plan without starting trajectories:

```bash
python -m shor_qem.folding_pattern_simulator --configs configs/n119_a13_matched.json configs/n119_a50_matched.json configs/n119_a16_matched.json --shots 20000 --gpu-ids 0 --output-dir simulator_runs/new_matched --dry-run
```

Removing `--dry-run` starts the expensive GPU computation. The configured output is the eight-scale Reset campaign; it does not by itself supply the separate Raw20k, Raw140k, and Reset140k parents required for the portable equal-shot bootstrap. The retained `followup_cluster` implementation and A100 follow-up specification document the additional parent-generation workflow. No new GPU trajectories were launched to validate this release.

## Data and historical replay

This is a **code-only** repository. It does not distribute actual shot records, manuscript or figure assets, result tables, bootstrap checkpoints, or compiled-operation pickles. A synthetic CPU run can demonstrate the analysis but cannot regenerate the paper's numerical claims.

Exact historical replay additionally requires the admissible original parent blocks, their provenance, and the original pinned analysis/cache dependencies. In particular, paired intervals require separate Reset scale counts, the Reset extension, Raw initial counts, and the Raw extension. Merged 160,000-shot counts alone are insufficient to reconstruct that pairing. See [DATA_FORMAT.md](docs/DATA_FORMAT.md).

The reported Appendix D results cover the 20,000-shot conditions Raw and `0%, 0.01%, 0.1%, 0.2%, 0.3%, 0.4%, 0.5%` reset/readout error. The 1% condition was unfinished and the 2% condition was not run; neither is a reported result in this release. Preserved configurations include additional execution options, which are not evidence of completed results. Separate multi-modulus and other ongoing experiments are outside scope.

## Version and reuse

Record the exact checked-out revision with `git rev-parse HEAD` when reporting results. A mutable branch URL is not a code-version pin. The release manifest and source hashes describe the packaged source; they do not replace the final Git commit identifier or provide missing measurement data.

No software license was found in the source provided for this release, and no new license has been inferred. Repository access is not a license grant. See [LICENSE_STATUS.txt](docs/LICENSE_STATUS.txt) before redistribution or reuse.
