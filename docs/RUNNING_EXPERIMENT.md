# Run the calculation: circuits → shots → distributions → intervals

Run the commands below from the repository root after installation. Steps 1–2 use the NVIDIA simulation environment; steps 3–5 run on CPU. Use a new run directory. The GPU commands generate trajectories and take the full simulation time. `folding_pattern_simulator` and `followup_cluster run-shard` accept `--dry-run` for plan inspection; the Raw `simulator` command starts its computation directly.

## 1. Compile the three circuits

```bash
python scripts/build_matched_operation_caches.py
```

Inputs are the three `configs/n119_a*_matched.json` files. The builder writes operation caches under `data/generated/ops_cache_matched_opt1/` and records compilation metadata. The configurations specify the cache filenames, retained circuit template, transpiler seed, and two-qubit gate counts.

## 2. Generate the shot records

### Initial Raw and eight-scale Reset records

Generate the initial 20,000-shot Raw record for all eleven configured error rates in each instance:

```bash
for instance in n119_a13 n119_a50 n119_a16; do
  python -m shor_qem.simulator --config "configs/${instance}_matched.json" --reset-mode no-reset --lambdas 1 --shots 20000 --gpu-ids 0 --output-dir "simulator_runs/reproduction/no_reset/${instance}"
done
```

Then generate 20,000 Reset shots at each of the eight noise scales:

```bash
python -m shor_qem.folding_pattern_simulator --configs configs/n119_a13_matched.json configs/n119_a50_matched.json configs/n119_a16_matched.json --shots 20000 --gpu-ids 0 --output-dir simulator_runs/reproduction/shotwise
```

For `a=13`, examples of the saved files are:

```text
simulator_runs/reproduction/
  no_reset/n119_a13/safe_random_noareset_N119_a13_p0.000100.pkl
  shotwise/n119_a13/shotwise/lambda_1p0-1p0.npz
  shotwise/n119_a13/shotwise/lambda_1p1-1p1.npz
  ...
  shotwise/n119_a13/shotwise/lambda_3p0-3p0.npz
```

Raw files retain the shot bitstrings; each Reset NPZ contains `rates`, `lambdas`, `outcomes`, `counts`, and folding metadata. The little-endian outcome convention is `y = sum(bit[j] * 2**j)`.

### Extend Raw and Reset to 160,000 shots

The follow-up specification adds 140,000 unfolded shots to each existing Raw/Reset baseline. It also generates the four extra `a=16` rates used in the study. A single-GPU layout is:

```bash
python -m shor_qem.followup_cluster plan --experiment configs/n119_a100_followup.json --workers local=1 --output-dir simulator_runs/reproduction/followup_plan
python -m shor_qem.followup_cluster run-shard --manifest simulator_runs/reproduction/followup_plan/shard_local.json --output-dir simulator_runs/reproduction/followup_local/results --gpu-ids 0
python -m shor_qem.followup_cluster merge --global-manifest simulator_runs/reproduction/followup_plan/global_manifest.json --worker-root local=simulator_runs/reproduction/followup_local --base-run-dir simulator_runs/reproduction --output-dir simulator_runs/reproduction/followup_merged
```

`A100` in the specification filename names the original GPU campaign; it is not the modular base and does not restrict this single-GPU recipe to that model.

Merge validates the shot records and writes:

```text
followup_merged/
  baseline_160k/n119_a13/raw/lambda1_p0.000100.npz
  baseline_160k/n119_a13/reset/lambda1_p0.000100.npz
  ...
  a16_augmented_20k/shotwise/n119_a16/shotwise/lambda_1p0-1p0.npz
  ...
```

Each baseline file retains ordered `outcomes`, the total `counts`, and `source_ranges_json` identifying the initial 20k block followed by seven 20k extension blocks. The resulting grid has 11 rates for `a=13`, 11 for `a=50`, and 15 for `a=16`.

## 3. Assemble one condition's count input

This example selects `N=119, a=13, r=4, p2=0.01%`:

```bash
mkdir -p results/n119_a13_p0p0001
python scripts/pack_simulator_counts.py --reset-scales-dir simulator_runs/reproduction/shotwise/n119_a13/shotwise --raw-baseline simulator_runs/reproduction/followup_merged/baseline_160k/n119_a13/raw/lambda1_p0.000100.npz --reset-baseline simulator_runs/reproduction/followup_merged/baseline_160k/n119_a13/reset/lambda1_p0.000100.npz --rate 0.0001 --output results/n119_a13_p0p0001/counts.npz
```

The exporter verifies the native count/outcome agreement and shot-block boundaries, checks that the initial Reset block matches the scale-one record, and writes the five arrays consumed by the CPU analysis:

| Array | Content |
| --- | --- |
| `reset_scale_counts` | Eight 20k Reset count rows |
| `reset_extension_counts` | The recorded 140k Reset extension |
| `raw_initial_counts` | The recorded initial 20k Raw block |
| `raw_extension_counts` | The recorded 140k Raw extension |
| `effective_lambdas` | Scales calculated from the saved integer folding counts |

For `a=16`, use `followup_merged/a16_augmented_20k/shotwise/n119_a16/shotwise` as the scale directory to access all 15 rates. Change the baseline paths, rate, and output name to match the selected condition. [DATA_FORMAT.md](DATA_FORMAT.md) gives the shapes and conventions.

## 4. Reconstruct and bootstrap

```bash
python -m shor_qem.portable_c1 reconstruct --input results/n119_a13_p0p0001/counts.npz --order 4 --output-dir results/n119_a13_p0p0001/point
python -m shor_qem.portable_c1 bootstrap --input results/n119_a13_p0p0001/counts.npz --order 4 --replicates 10000 --seed 20260722 --batch-size 25 --confidence .95 --output-dir results/n119_a13_p0p0001/bootstrap
```

`reconstruct` fits the four C1 models with mean-one inverse-square weights and zero ridge, projects their extrapolated vectors, and writes `points.json` and `parent_estimates.npz`. The JSON contains TVD, grouped peak margin, and normalized grouped peak margin for each route. The NPZ contains the actual distributions.

`bootstrap` samples the four parent-block streams, refits each model, and writes `scalar_bootstrap_draws.npz` and `intervals.json`. The latter reports both route-level intervals and paired differences against Reset160k. Original-count estimates remain the reported central values. Set `--confidence .90` for 90% percentile intervals in a separate output directory.

## 5. Evaluate strict exact-order recovery

```bash
python -m shor_qem.portable_c1 decode --input results/n119_a13_p0p0001/point/parent_estimates.npz --N 119 --a 13 --order 4 --bits 14 --output-dir results/n119_a13_p0p0001/order
```

For each outcome, the decoder selects `Fraction(y,16384).limit_denominator(119)`, verifies the candidate denominator, and evaluates whether it equals the true order. `order/decoder.json` reports the resulting probability for every input distribution; `order/decoder.npz` also stores the masks. Use `--a 50 --order 2` or `--a 16 --order 6` for the other two instances.

## Reading a saved study result

The README's table follows this same input-to-metric flow on the saved study counts. The original production bootstrap stored each condition's `parent_estimates.npz` and `scalar_bootstrap_draws.npz`; its summarizer collected these into `absolute_locked_metrics.csv` and `primary_rate_results.csv`. [The production-source map](REPRODUCIBILITY.md#recorded-results) links those filenames to the archived scripts.

The commands above generate a new run. Its seeds, environment, input hashes, and resulting manifests identify that run; the original study's shot and bootstrap archives are retained separately.
