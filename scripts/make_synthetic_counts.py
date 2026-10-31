#!/usr/bin/env python3
"""Create a small, explicitly synthetic CPU demonstration of the count schema.

This is a multinomial toy distribution, not a circuit/noise simulation or new
scientific dataset. It never submits jobs or accesses a quantum/GPU backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from shor_qem.metrics import ideal_distribution
from shor_qem.portable_c1 import CountParents


def synthetic_counts(*, bits: int = 4, order: int = 4, seed: int = 1) -> dict[str, np.ndarray]:
    if isinstance(bits, bool) or int(bits) != bits or not 1 <= bits <= 16:
        raise ValueError("synthetic demo bits must be an integer in [1,16]")
    bits = int(bits)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    lambdas = np.asarray([1, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0])
    ideal = ideal_distribution(bits, order)
    # Suppress one toy bin at every scale, exercising the C1 zero-lift rule.
    # Extension-only mass is allowed and must NOT enter C1 observed support.
    uniform = np.ones(ideal.size)
    omitted = int(np.argmin(ideal))
    uniform[omitted] = 0
    uniform /= uniform.sum()
    toy_ideal = ideal.copy()
    toy_ideal[omitted] = 0
    if toy_ideal.sum() == 0:
        raise ValueError("synthetic order must leave positive toy ideal mass")
    toy_ideal /= toy_ideal.sum()
    rng = np.random.default_rng(int(seed))
    reset = [np.exp(-0.2 * lam) * toy_ideal + (1 - np.exp(-0.2 * lam)) * uniform for lam in lambdas]
    raw = 0.65 * toy_ideal + 0.35 / ideal.size
    extension = reset[0] * 0.995
    extension[omitted] += 0.005
    arrays = {
        "reset_scale_counts": np.stack([rng.multinomial(20_000, parent) for parent in reset]),
        "reset_extension_counts": rng.multinomial(140_000, extension),
        "raw_initial_counts": rng.multinomial(20_000, raw),
        "raw_extension_counts": rng.multinomial(140_000, raw),
        "effective_lambdas": lambdas,
    }
    CountParents(**arrays)
    arrays["synthetic_demo_only"] = np.asarray(True)
    arrays["synthetic_seed"] = np.asarray(seed)
    return arrays


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new NPZ file; parent directory must already exist")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        arrays = synthetic_counts(bits=args.bits, order=args.order, seed=args.seed)
        with args.output.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote SYNTHETIC CPU demo counts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
