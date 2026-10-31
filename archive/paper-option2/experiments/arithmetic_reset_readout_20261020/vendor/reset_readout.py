"""Pure NumPy readout-error masks; no quantum simulation and no global RNG."""
from __future__ import annotations

import hashlib
import math
import struct
import numpy as np

DOMAIN = b"QST:arithmetic-reset-assignment:v1"


def validate_probability(p):
    value = float(p)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reset readout error must be a finite probability in [0, 1]")
    return value


def context_key(*, seed, N, a, gate_error, noise_scale, round_index, qubit):
    if not 0 <= int(seed) < 2**64:
        raise ValueError("reset seed must be uint64")
    n = int(N).bit_length()
    if not 0 <= int(round_index) < 2*n:
        raise ValueError("invalid phase-estimation round")
    if not n+1 <= int(qubit) <= 2*n+2:
        raise ValueError("readout-error stream is restricted to arithmetic-register qubits")
    if not math.isfinite(float(gate_error)) or not 0 <= float(gate_error) <= 1:
        raise ValueError("invalid gate error")
    if not math.isfinite(float(noise_scale)) or float(noise_scale) < 1:
        raise ValueError("invalid noise scale")
    packed = struct.pack("<QQQddQQ", int(seed), int(N), int(a), float(gate_error),
                         float(noise_scale), int(round_index), int(qubit))
    return int.from_bytes(hashlib.blake2b(DOMAIN+packed, digest_size=8).digest(), "little")


def error_uniforms(*, shots, shot_offset, **context):
    """Stable across chunk boundaries; does not touch np.random or the gate RNG."""
    shots, shot_offset = int(shots), int(shot_offset)
    if shots < 0 or shot_offset < 0 or shots+shot_offset >= 2**63:
        raise ValueError("invalid shot interval")
    key = context_key(**context)
    counter = np.arange(shot_offset, shot_offset+shots, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (counter ^ np.uint64(key)) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def error_mask(p, *, shots, shot_offset, **context):
    p = validate_probability(p)
    if p == 0:
        # Validate even a zero-noise request, but never generate random numbers.
        context_key(**context)
        if int(shots) < 0 or int(shot_offset) < 0:
            raise ValueError("invalid shot interval")
        return np.zeros(int(shots), dtype=np.int32)
    return (error_uniforms(shots=shots, shot_offset=shot_offset, **context) < p).astype(np.int32)


def batch_error_mask(p, *, configs, shots, shot_offset, seed, N, a, round_index, qubit):
    """Match the frozen engine's configuration-major / shot-minor ordering."""
    return np.concatenate([error_mask(p, shots=shots, shot_offset=shot_offset,
        seed=seed, N=N, a=a, gate_error=rate, noise_scale=lam,
        round_index=round_index, qubit=qubit) for rate, lam in configs])


def fault_counts(p, *, shots, shot_offset, seed, N, a, gate_error, noise_scale):
    n = int(N).bit_length()
    return np.array([[int(error_mask(p, shots=shots, shot_offset=shot_offset,
        seed=seed, N=N, a=a, gate_error=gate_error, noise_scale=noise_scale,
        round_index=i, qubit=q).sum()) for q in range(n+1, 2*n+3)]
        for i in range(2*n)], dtype=np.int64)
