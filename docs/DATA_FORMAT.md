# Count inputs and pairing

The shot simulator writes native outcome/count records. `scripts/pack_simulator_counts.py` assembles those records into a NumPy `.npz` input for the CPU reconstruction and bootstrap, representing one instance and one gate-error rate. The [execution guide](RUNNING_EXPERIMENT.md) shows the full filename and command sequence.

## Required parent blocks

For the primary study profile, `D = 16384`, and column `y` denotes phase-estimation outcome `y` in `0 <= y < D`. The portable interface also permits smaller power-of-two dimensions `D >= 2` for synthetic checks. Store nonnegative integer-valued counts, including zero-count outcomes; probabilities or pseudocounts are not a replacement for counts.

| Key | Shape | Required total | Meaning |
| --- | --- | --- | --- |
| `reset_scale_counts` | `(8, D)` | 20,000 in each row | Independent Reset records at the eight effective scales, in scale order |
| `reset_extension_counts` | `(D,)` | 140,000 | Additional unfolded Reset shots, independent of the scale rows |
| `raw_initial_counts` | `(D,)` | 20,000 | Initial no-reset Raw block |
| `raw_extension_counts` | `(D,)` | 140,000 | Additional no-reset Raw shots, independent of the initial block |
| `effective_lambdas` | `(8,)` | Not a count field | Effective, not nominal, scale values corresponding to Reset rows |

The effective scales for the retained template are:

```text
1.0, 1.099746407438715, 1.2000563539025078, 1.4001127078050155,
1.800225415610031, 2.199774584389969, 2.5998872921949845, 3.0
```

The input loads with `allow_pickle=False`. The native-record exporter additionally records instance identifiers, the error rate, source file hashes, and the retained shot-block boundaries in `provenance_json` and metadata fields.

All five listed keys are required; unrelated metadata keys are ignored. `effective_lambdas` must contain finite strictly increasing real values beginning at `1`. All arrays must agree on `D`, and every count parent must have exactly the stated total. The synthetic demonstration uses a four-bit toy distribution and treats its eight scale values as exact.

## Why the parent blocks must remain separate

The unresampled direct distributions are:

```text
Raw160   = (raw_initial_counts + raw_extension_counts) / 160000
Reset160 = (reset_scale_counts[0] + reset_extension_counts) / 160000
```

ZNE is fitted to the normalized eight Reset scale rows. For a paired bootstrap replicate, the very same resampled `reset_scale_counts[0]` contributes both to Reset160 and to every ZNE fit. Its 140,000-shot extension is resampled separately. Raw initial and extension blocks are also drawn from their separate parents.

The native baseline NPZs retain the ordered outcomes and `source_ranges_json`, so the exporter can recover the actual initial and extension blocks. It also checks that the initial Reset outcomes equal the scale-one outcomes. This preserves the Reset/ZNE pairing; a histogram without those shot-block records is not an input to this exporter.

Each scale row is one multinomial parent; no cross-scale shot correspondence is inferred from equal row lengths. The pairing is shared reuse of the unfolded Reset draw and shared scale draws across fitted models, not an invented shot-ID alignment across all scales.

## Fitting support and evaluation references

C1 determines its support from `reset_scale_counts` alone: take the union of outcomes with positive counts in any scale row, retain each such outcome at every scale, and retain its zero counts where appropriate. Neither direct extensions nor ideal probabilities add outcomes to this fitting support. The union is recomputed after each bootstrap draw.

The full outcome dimension remains `D`, even though fitting and projection use only the observed union. In particular, the exponential offset remains `1 / D`, not `1 / |U|`. Values outside the observed union are zero after C1 projection.

Evaluation references must correspond to the same `N`, base, phase-bit count, and outcome encoding as the counts. The ideal finite-resolution QPE distribution and peak groups are used for evaluation only, not for fitting or rate selection. Do not pass an ideal distribution as if it were measured counts.

## Native simulator records

The eight Reset files contain `rates`, `lambdas`, `outcomes`, `counts`, and `folded_gate_counts_by_round`. Each merged baseline NPZ contains `outcomes`, `counts`, `mode`, `rate`, and `source_ranges_json`. The exporter checks instance and circuit metadata, count/outcome agreement, and the preserved 20k/140k boundaries before assembling the analysis input.

Effective scales are calculated as `1 + 2*sum(folded_gate_counts_by_round)/two_qubit_gate_count`. The exporter accepts the native N=119 matched-template layout, including the augmented r=6 scale directory. It reads NPZ files only. The Raw generator's original bitstring pickle is consumed by the baseline merge stage, which produces the ordered baseline NPZ used here.

## Portable outputs and decoder input

`reconstruct` writes `points.json`, `parent_estimates.npz`, and `manifest.json`. `bootstrap` additionally writes `intervals.json` and `scalar_bootstrap_draws.npz`. Output destinations must be new; existing files are not silently replaced.

The four fitted routes are `linear_zne`, `quadratic_zne`, `single_exponential_zne`, and `dual_exponential_zne`. Direct diagnostic routes are `raw_20k`, `raw_160k`, `reset_20k`, and `reset_160k`. The 20k direct routes are precision diagnostics, not equal-budget replacements for the primary 160k comparison.

The metric order is `tvd`, `grouped_peak_margin`, `normalized_grouped_peak_margin`. Bootstrap `scalar_draws` are indexed by `(4 fitted methods, 3 metrics, replicate)`; `direct_draws` use `(4 direct routes, 3 metrics, replicate)`. Positive `paired_reset_improvements` consistently mean an improvement over Reset160: `Reset160 − ZNE` for TVD and `ZNE − Reset160` for the two margins. Percentile intervals are pointwise, not simultaneous confidence bands.

The `decode` subcommand accepts a non-pickle NPZ containing `probabilities` with shape `(D,)` or `(K,D)` and optional `probability_routes` with shape `(K,)`. Alternatively, it can read projected `after__*` vectors from `parent_estimates.npz`, or explicit keys selected using repeated `--key` arguments. Signed preprojection `before__*` vectors are not probability distributions and must not be decoded. Specify the matching `--N`, `--a`, `--order`, and `--bits`; the study uses `--bits 14`. The README's four-bit example is explicitly synthetic.

The strict S6 rule forms exactly one rational candidate for outcome `y`: `Fraction(y, 2**bits).limit_denominator(N)`. A zero numerator or failure of `pow(a, q, N) == 1` is rejected. A verified denominator `q` is retained as-is: the decoder does not scan all convergents, test denominator multiples, factor-reduce `q`, or combine multiple shots. The supplied true order enters only the final evaluation `q == order`, not candidate selection.

## File provenance

The count export records the basenames and SHA256 hashes of its ten input files. Each analysis command then writes an input/output `manifest.json`. Together these records identify the shot inputs, reconstructed distributions, and bootstrap outputs for the run. The [public data release](DATA_AVAILABILITY.md) supplies the original study records and final caches as downloadable archives alongside the source repository.

The published historical scalar caches retain their original axes and labels; these are distinct from the portable output layout described above. Use `scripts/reproduce_published_results.py` to read the archived labels and regenerate the published point estimates and percentile tables. The final scalar and parent NPZ files retain their original bytes. Metadata-sanitized batch checkpoints are paired with original/public SHA256 mappings in `bootstrap/publication_manifest.json`.
