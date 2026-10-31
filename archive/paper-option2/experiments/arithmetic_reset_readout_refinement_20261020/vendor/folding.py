"""Deterministic random-folding utilities with no GPU dependencies."""

from __future__ import annotations

import numpy as np


MASK64 = (1 << 64) - 1
FOLDING_ARCHIVE_SCHEMA_VERSION = 3
FOLDING_SEED_VERSION = 2
FOLDING_PATTERN_SCOPE = "per_rate_lambda_qpe_round_shot"
FOLDING_SEED_FIELDS = (
    "folding_seed",
    "N",
    "a",
    "physical_error_rate",
    "lambda",
    "qpe_round",
    "shot_id",
)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    mixed = value
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & MASK64
    mixed ^= mixed >> 27
    mixed = (mixed * 0x94D049BB133111EB) & MASK64
    mixed ^= mixed >> 31
    return mixed & MASK64


def folding_seed_for_shot(
    *,
    global_seed: int,
    N: int,
    a: int,
    rate: float,
    lam: float,
    qpe_round: int,
    shot_id: int,
) -> int:
    """Return the deterministic folding seed for one simulated shot.

    The seed is independent of GPU identity and task scheduling, so changing
    the number of visible GPUs does not change the simulated experiment.
    """

    if N <= 0 or a <= 0 or rate < 0.0 or lam < 1.0:
        raise ValueError("invalid folding-seed parameters")
    if qpe_round < 0 or shot_id < 0:
        raise ValueError("qpe_round and shot_id must be nonnegative")

    rate_code = int(round(float(rate) * 1_000_000_000.0))
    lambda_code = int(round(float(lam) * 10.0))
    components = (
        FOLDING_SEED_VERSION,
        int(N),
        int(a),
        rate_code,
        lambda_code,
        int(qpe_round),
        int(shot_id),
    )
    state = _splitmix64((int(global_seed) & MASK64) ^ 0xA24BAED4963EE407)
    for component in components:
        state = _splitmix64(state ^ (int(component) & MASK64))
    return state


def build_shotwise_folding_masks(
    current_configs: list[tuple[float, float]],
    shots: int,
    shot_offset: int,
    step_i: int,
    n_targets: int,
    *,
    global_seed: int,
    N: int,
    a: int,
) -> np.ndarray:
    """Return exact-cardinality folding masks with one pattern per shot.

    A subset depends on ``(global_seed, N, a, rate, lambda, QPE round,
    shot_id)``. Each output column contains exactly
    ``round((lambda - 1) * n_targets / 2)`` selected locations. GPU identity
    and task ordering are deliberately absent from the seed hierarchy.
    """

    if shots <= 0 or n_targets < 0 or shot_offset < 0 or step_i < 0:
        raise ValueError("invalid shot-wise folding dimensions")
    total = len(current_configs) * int(shots)
    masks = np.zeros((int(n_targets), total), dtype=np.int32)
    for config_i, (rate, lam) in enumerate(current_configs):
        start = config_i * shots
        stop = (config_i + 1) * shots
        fold_count = int(round((float(lam) - 1.0) * n_targets / 2.0))
        fold_count = min(max(fold_count, 0), int(n_targets))
        if fold_count == n_targets:
            masks[:, start:stop] = 1
        elif fold_count:
            choose_complement = fold_count > n_targets // 2
            draw_count = n_targets - fold_count if choose_complement else fold_count
            if choose_complement:
                masks[:, start:stop] = 1
            for local_shot in range(shots):
                shot_id = int(shot_offset) + local_shot
                folding_seed = folding_seed_for_shot(
                    global_seed=global_seed,
                    N=N,
                    a=a,
                    rate=rate,
                    lam=lam,
                    qpe_round=step_i,
                    shot_id=shot_id,
                )
                selected = np.random.default_rng(folding_seed).choice(
                    n_targets, draw_count, replace=False, shuffle=False
                )
                masks[selected, start + local_shot] = 0 if choose_complement else 1
    return masks
