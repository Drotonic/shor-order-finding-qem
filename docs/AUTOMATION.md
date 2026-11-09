# N,a-driven experiments

The `shor-qem` command connects circuit construction, execution, counts, ZNE and
order-recovery analysis. Supply `N` and a coprime `a`; the circuit builder retains
all modular-arithmetic rounds and never uses a known order to simplify them.

The output is `counts.npz` followed by `analysis/metrics.csv`,
`analysis/distributions.npz` and `analysis/summary.json`. The metrics are TVD,
normalized peak–background gap, and exact-order recovery for Raw, Reset, linear,
quadratic, single-exponential and dual-exponential ZNE. Analysis uses the existing
C1 estimator: mean-one inverse-square weights, zero ridge, offset `1/D`, and the
same exponential search grids. The archived `portable_c1` commands and published
N119 datasets are unchanged.

## Install

Use a separate Python environment. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hardware,local-quantum-test]'
shor-qem --help
```

For NVIDIA simulation, add **one** CUDA extra appropriate to your installation:

```bash
python -m pip install -e '.[gpu-cuda12]'
```

The hardware client is tested with Qiskit 2.4.1 and IBM Runtime 0.51.x. Optional
CPU circuit tests use Aer 0.18.x. Simulation uses the retained CuPy trajectory
engine; planning and analysis do not import CUDA or create an IBM account.
GPU execution uses Linux or WSL. State-vector/count-array and ideal-reference
memory checks limit the supported instance sizes on a given machine.
The pinned Runtime client still supports `SamplerV2`, although that class is
deprecated upstream; client upgrades should rerun the local integration tests.

## 1. Simulate N,a and analyze automatically

Inspect the full shot allocation first:

```bash
shor-qem plan --N 15 --a 4 --rates-percent 0.001 0.005 0.01
```

Then execute:

```bash
shor-qem simulate --N 15 --a 4 --rates-percent 0.001 0.005 0.01 \
  --shots 20000 --batch-size 48 --output-dir results/n15_a4 --execute
```

Without `--execute`, this command only displays its plan. The default scales are
`1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3`. `--rates-percent 0.01` means the
nonidentity two-qubit Pauli-fault probability is `0.0001`, not `0.01`.
Rates must be increasing. All simulated folds are sampled shot-wise using the
retained engine; fits use the actual effective scales from gate-count rounding.

Default Raw and unfolded Reset totals equal the sum of the scale shots. Reset
shares the scale-one record and adds an independent extension; Raw is independent.
For eight scales and 20,000 shots this is 460,000 generated shots per rate:
160,000 folded-scale shots + 140,000 Reset extension + 160,000 Raw.

The result directory contains:

| Artifact | Produced by |
| --- | --- |
| `manifest.json`, `operations.json` | CPU circuit compilation and explicit run specification |
| `checkpoints/`, `run_state.json` | Completed shot batches, checksums and progress |
| `counts/rate_000.npz`, etc. | Complete count arrays for each requested error rate |
| `analysis/rate_000/metrics.csv`, etc. | Original-count TVD, gap and exact-order probabilities |

Resume an interrupted simulation with the **same arguments** plus `--resume`:

```bash
shor-qem simulate --N 15 --a 4 --rates-percent 0.001 0.005 0.01 \
  --shots 20000 --batch-size 48 --output-dir results/n15_a4 --execute --resume
```

Saved batches are verified before new execution; committed shots are not generated
or counted again. Keep the prepared files, numerical environment and GPU consistent.
The command does not install a scheduler or alter any separately running experiment.

## 2. Build and run real IBM circuits

Choose a backend your account can access. The examples use `ibm_phoenix`; replace
it with your chosen backend. Its target must support dynamic measurement,
conditional operations, reset, and native CX or CZ gates. Preparation checks these
requirements and the physical qubit mapping; ECR-only targets are not supported by
this first native-folding adapter.

The default hardware scale set is `1, 1.4, 1.8, 2.2`. The following plan uses
10,000 shots at each scale, plus 10,000 independent Raw shots. The Reset baseline
is the scale-one 10,000-shot record (`--baseline-shots 10000`):

```bash
shor-qem ibm-plan --N 15 --a 4 --backend ibm_phoenix \
  --shots 10000 --baseline-shots 10000 --folding-patterns 10
```

To use equal-total-shot Raw/Reset baselines instead, omit `--baseline-shots`;
each baseline will then contain 40,000 shots. The displayed plan reports the full
physical shot budget, including any independent Reset extension.

### One-command execution and analysis

```bash
shor-qem ibm --N 15 --a 4 --backend ibm_phoenix \
  --shots 10000 --baseline-shots 10000 --folding-patterns 10 \
  --output-dir results/ibm_n15_a4 --prompt-token \
  --max-jobs 4 --max-execution-time 300 --execute --wait
```

The secure prompt requests an IBM Cloud API key and the instance CRN without
echoing or saving them. Alternatively provide `QISKIT_IBM_TOKEN` and
`QISKIT_IBM_INSTANCE` as process environment variables, and omit `--prompt-token`.
Do not put keys in source code, JSON configurations, command arguments or Git.
No IBM account is saved by this workflow.

This command compiles the circuit, submits the declared jobs, saves their IDs,
waits for results, assembles counts, and runs the analysis. `--execute` authorizes
real QPU use. Without it, `ibm` displays a local plan and does not connect or submit.
`--wait` is optional: omit it to return while jobs are pending and collect later.

`--max-execution-time` is a **per-job** QPU-time limit, not a wall-clock queue limit,
reservation or total-account allocation. `--max-jobs` bounds the number of declared
jobs; it does not guarantee they fit within your remaining account quota. Review
the compiled gate counts and shot budget before submission.
PUBs with equal shot counts are grouped together, and no job exceeds the current
10-million-execution Sampler limit. See IBM's
[job limits](https://quantum.cloud.ibm.com/docs/en/guides/job-limits).

### Prepare, inspect, submit separately

```bash
shor-qem ibm-prepare --N 15 --a 4 --backend ibm_phoenix \
  --shots 10000 --baseline-shots 10000 --folding-patterns 10 \
  --output-dir results/ibm_prepared --prompt-token

shor-qem ibm-run --prepared results/ibm_prepared
# Review compilation_report.json and manifest.json, then authorize submission:
shor-qem ibm-run --prepared results/ibm_prepared --prompt-token \
  --max-jobs 4 --max-execution-time 300 --execute --wait
```

The first command reads the chosen backend's target but submits no jobs. The
second is a local dry run. To compile without an IBM account, add `--offline`
to `ibm-prepare`; this uses a generic local target, not a calibration snapshot.
Offline preparations cannot be submitted as real hardware plans.

The compiled circuit uses the `phase` classical register. Counts are decoded in
Qiskit's displayed MSB-first ordering to the same integer `z` used by the analysis.
Raw and Reset share the routed circuit; only the labeled arithmetic-workspace
resets are removed for Raw. The control-qubit reset and feedforward stay intact.

### Hardware folding and saved jobs

Native folding is applied **after** routing/translation so optimization does not
cancel `GG†G`. Do not transpile the exported folded QPY circuits again. With
`--folding-patterns 10`, each nontrivial scale has ten seeded circuits, each reused
for its allocated shots. This is a hardware pattern ensemble, not an independent
new fold pattern on every shot. To request one pattern per shot, set the pattern
count equal to the shot count and review the much larger payload and job budget.

The hardware run uses the device's intrinsic noise; no simulated `p2` is assigned
to its data. Gate-count effective scales describe folding, not a calibrated
scalar multiplier for every source of hardware noise.

`jobs.json` stores submission intent before each call and the returned job ID
immediately afterwards. After a disconnect, reuse the same prepared directory:

```bash
shor-qem ibm-collect --prepared results/ibm_prepared --prompt-token --wait
```

Collection reads existing IDs and never submits a job. A crash between sending a
request and recording its ID is marked ambiguous and blocks automatic resubmission.
Verify the existing job's backend, PUB ordering and shots in IBM's job details,
then attach its ID explicitly:

```bash
shor-qem ibm-recover --prepared results/ibm_prepared \
  --job-index 0 --job-id EXISTING_JOB_ID --verified-job
```

If a process crash left `.runtime.lock`, first confirm no process owns the
directory and inspect `jobs.json`; only then remove that specific stale lock.
Do not delete the ledger to retry a possibly billable submission.

## 3. Analyze completed counts separately

```bash
shor-qem analyze --input results/ibm_prepared/counts.npz \
  --N 15 --a 4 --output-dir results/ibm_prepared/reanalysis
```

The new generic NPZ schema contains integer `scale_counts` (K,D), `raw_counts`
(D,), `reset_extension_counts` (D,), and `effective_lambdas` (K,). All K>=4 scale
records must have equal shot totals, start with lambda=1, and form a strictly
increasing effective grid. Raw and combined Reset baseline totals must match.
Instance metadata are checked when present. This schema is distinct from the
archived five-array, fixed-20k `portable_c1` parent schema.

Analysis computes the exact order classically **only as a small-instance evaluation
label**, after fitting. The decoder selects one candidate with
`Fraction(z,2**bits).limit_denominator(N)` and verifies it modulo N; no known order,
denominator-multiple search or divisor reduction is used by the decoder.
Exact-order probability is the sum of probabilities decoded to that evaluation
order. For ZNE it is a property of the reconstructed distribution, not a set of
new QPU shots. No bootstrap is silently started by these new commands.

## Tests and IBM references

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python scripts/verify_source_snapshot.py
python scripts/verify_release.py
```

Circuit tests run locally on CPU; submission tests use mock services and local
Sampler results. They make no real QPU submissions. IBM's current API details:
[account initialization](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit-runtime-service),
[dynamic circuits](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits),
and [Sampler options](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options).
