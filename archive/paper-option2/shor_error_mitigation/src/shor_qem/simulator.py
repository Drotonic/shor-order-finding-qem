"""Multi-GPU simulator extracted from the historical experiment notebook.

The numerical kernels and seed formulas are preserved.  The command-line
driver makes the instance, reset mode, output path, and overwrite policy
explicit.  Folding subsets remain fixed for each (lambda, QPE-round) pair and
are reused across shots and physical error rates, exactly as in the saved run.
"""


import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cupy as cp
import multiprocessing as mp

from .folding import build_shotwise_folding_masks as _build_shotwise_folding_masks

from qiskit import QuantumCircuit, transpile
# Linux file lock (for ops_cache)
import fcntl

# ============================================================
# 1. CONFIG & PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = os.environ.get(
    "SHOR_QEM_RESULTS_DIR", str(PROJECT_ROOT / "simulator_runs")
)
OPS_CACHE_DIR = os.environ.get(
    "SHOR_QEM_OPS_CACHE_DIR", str(PROJECT_ROOT / "data" / "raw" / "ops_cache")
)

# ============================================================
# 2. GLOBAL CONFIG
# ============================================================
GLOBAL_SEED = 42

SHOTS_STANDARD   = 20000
SHOTS_BLOCK_SCAN = 80000  # total shots for segment(block_scan)  (이번 실행에서는 사용 안함)

USE_FP32 = True

# VRAM safety controls
VRAM_SAFETY_FACTOR = 2.5          # divide free VRAM by this (extra safety)
VRAM_FREE_FRACTION = 0.70         # use only this fraction of FREE mem (multi-proc friendly)

TARGET_BASIS = ["cz", "id", "rx", "rz", "rzz", "sx", "x", "measure", "reset"]
NOISY_2Q_GATES = ['cz', 'rzz']
SINGLE_QUBIT_GATES = {'x', 'y', 'z', 'h', 'sx', 'rz', 'rx', 'id'}
USE_FUSED_2Q_KERNELS = True

if USE_FP32:
    c_type = cp.complex64
    r_type = cp.float32
    np_c_type = np.complex64
    cpp_complex = 'complex<float>'
    cpp_real = 'float'
    epsilon = 1e-6

    raw_real_type = 'float'
    raw_vec_type = 'float2'
    kernel_sig = 'complex64 u00, complex64 u01, complex64 u10, complex64 u11'
    bytes_per_complex = 8
else:
    c_type = cp.complex128
    r_type = cp.float64
    np_c_type = np.complex128
    cpp_complex = 'complex<double>'
    cpp_real = 'double'
    epsilon = 1e-15

    raw_real_type = 'double'
    raw_vec_type = 'double2'
    kernel_sig = 'complex128 u00, complex128 u01, complex128 u10, complex128 u11'
    bytes_per_complex = 16

# NOTE: keep these as NumPy matrices (passed as scalars into ElementwiseKernel)
X_MAT  = np.array([[0, 1], [1, 0]], dtype=np_c_type)
H_MAT  = np.array([[1, 1], [1, -1]], dtype=np_c_type) / np.sqrt(2)
SX_MAT = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np_c_type) / 2

# ============================================================
# 3. KERNELS: 1Q, CZ, RZZ
# ============================================================
gate_kernel_masked = cp.ElementwiseKernel(
    f'{kernel_sig}, int32 target_mask, int32 current_batch, raw int32 cond_mask, bool use_mask',
    f'raw {c_type.__name__} state',
    f'''
    int batch_idx = i % current_batch;
    if (use_mask && cond_mask[batch_idx] == 0) return;
    int state_idx = i / current_batch;
    if ((state_idx & target_mask) == 0) {{
        long long partner_idx = i + (long long)target_mask * current_batch;
        {cpp_complex} v0 = state[i]; {cpp_complex} v1 = state[partner_idx];
        state[i] = u00 * v0 + u01 * v1; state[partner_idx] = u10 * v0 + u11 * v1;
    }}
    ''',
    'gate_kernel_masked'
)

cz_mask_kernel = cp.ElementwiseKernel(
    'int32 q1, int32 q2, int32 current_batch, raw int32 cond_mask',
    f'raw {c_type.__name__} state',
    f'''
    int batch_idx = i % current_batch;
    if (cond_mask[batch_idx] == 0) return;
    int state_idx = i / current_batch;
    int b1 = (state_idx >> q1) & 1; int b2 = (state_idx >> q2) & 1;
    if (b1 == 1 && b2 == 1) state[i] = -state[i];
    ''',
    'cz_mask_kernel'
)

rzz_mask_kernel = cp.ElementwiseKernel(
    f'int32 q1, int32 q2, {r_type.__name__} theta, int32 current_batch, raw int32 cond_mask',
    f'raw {c_type.__name__} state',
    f'''
    int batch_idx = i % current_batch;
    if (cond_mask[batch_idx] == 0) return;
    int state_idx = i / current_batch;
    int b1 = (state_idx >> q1) & 1; int b2 = (state_idx >> q2) & 1;
    {cpp_real} ht = theta * 0.5; {cpp_real} c = cos(ht); {cpp_real} s = sin(ht);
    {cpp_complex} val = state[i];
    if (b1 == b2) state[i] = {cpp_complex}(val.real()*c + val.imag()*s, val.imag()*c - val.real()*s);
    else          state[i] = {cpp_complex}(val.real()*c - val.imag()*s, val.imag()*c + val.real()*s);
    ''',
    'rzz_mask_kernel'
)

# ============================================================
# 4. DETERMINISTIC RNG / HASH (host)
# ============================================================
MASK64 = (1 << 64) - 1

def _splitmix64_py(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z ^= (z >> 30)
    z = (z * 0xBF58476D1CE4E5B9) & MASK64
    z ^= (z >> 27)
    z = (z * 0x94D049BB133111EB) & MASK64
    z ^= (z >> 31)
    return z & MASK64

def make_gate_key(step_i: int, op_idx: int, inv_flag: int, noise_tag: int) -> np.uint64:
    x = 0
    x ^= (GLOBAL_SEED & 0xFFFFFFFF)
    x ^= ((step_i + 1)    & 0xFFFF) << 16
    x ^= ((op_idx + 1)    & 0xFFFF) << 32
    x ^= ((inv_flag + 1)  & 0xFFFF) << 48
    x ^= ((noise_tag + 1) & 0xFF)   << 8
    return np.uint64(_splitmix64_py(int(x)))

def build_seed_shot_array(current_configs, shots: int, shot_offset: int = 0):
    n_cfg = len(current_configs)
    total = n_cfg * shots

    rate_code = np.zeros(total, dtype=np.int64)
    lam_code  = np.zeros(total, dtype=np.int16)
    shot_id   = np.zeros(total, dtype=np.int32)

    for i_cfg, (r, lam) in enumerate(current_configs):
        s = i_cfg * shots
        e = (i_cfg + 1) * shots
        rc = int(round(float(r) * 1_000_000_000.0))
        lc = int(round(float(lam) * 10.0))
        rate_code[s:e] = rc
        lam_code[s:e]  = lc
        shot_id[s:e]   = shot_offset + np.arange(shots, dtype=np.int32)

    seed_shot = np.zeros(total, dtype=np.uint64)
    base = int(GLOBAL_SEED) & MASK64
    for i in range(total):
        x = base
        x ^= (int(rate_code[i]) & MASK64)
        x ^= (int(lam_code[i]) & 0xFFFF) << 32
        x ^= (int(shot_id[i])  & 0xFFFFFFFF) << 1
        seed_shot[i] = np.uint64(_splitmix64_py(x))

    return (
        rate_code.astype(np.int64),
        lam_code.astype(np.int16),
        shot_id.astype(np.int32),
        seed_shot.astype(np.uint64),
    )

def make_draw_key(step_i: int, draw_id: int) -> np.uint64:
    x = 0
    x ^= (GLOBAL_SEED & 0xFFFFFFFF)
    x ^= ((step_i + 1) & 0xFFFF) << 16
    x ^= ((draw_id + 1) & 0xFFFF) << 40
    return np.uint64(_splitmix64_py(int(x)))

# ============================================================
# 5. DEVICE RNG KERNEL: gen_uniform(out[batch], seed_shot[batch], draw_key)
# ============================================================
rand_kernel_code = f'''
extern "C" __device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {{
    x += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = x;
    z ^= (z >> 30);
    z *= 0xBF58476D1CE4E5B9ULL;
    z ^= (z >> 27);
    z *= 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return z;
}}

extern "C" __global__ void gen_uniform_{raw_real_type}(
    {raw_real_type}* out,
    const unsigned long long* seed_shot,
    unsigned long long draw_key,
    long long n
) {{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned long long x = seed_shot[tid] ^ draw_key;
    x = splitmix64(x);
    unsigned int r = (unsigned int)(x >> 32);
    out[tid] = ({raw_real_type})r / ({raw_real_type})4294967296.0;
}}
'''
gen_uniform_kernel = cp.RawKernel(rand_kernel_code, f'gen_uniform_{raw_real_type}')

def gen_uniform(seed_shot_gpu: cp.ndarray, draw_key_u64: np.uint64) -> cp.ndarray:
    n = int(seed_shot_gpu.size)
    out = cp.empty(n, dtype=r_type)
    block = 256
    grid = (n + block - 1) // block
    gen_uniform_kernel((grid,), (block,), (out, seed_shot_gpu, cp.uint64(draw_key_u64), cp.int64(n)))
    return out

# ============================================================
# 6. 2Q DEPOL KERNEL (seed_shot + gate_key)
# ============================================================
raw_2q_depol_code = f'''
extern "C" __device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {{
    x += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = x;
    z ^= (z >> 30);
    z *= 0xBF58476D1CE4E5B9ULL;
    z ^= (z >> 27);
    z *= 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return z;
}}

extern "C" __global__ void fused_2q_depol_raw(
    {raw_vec_type}* state,
    int q1, int q2,
    int batch_size,
    {raw_real_type}* p_array,
    const unsigned long long* seed_shot,
    unsigned long long gate_key,
    long long total_tasks,
    const int* mask_ptr
) {{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_tasks) return;

    int batch_idx = (int)(tid % batch_size);
    if (mask_ptr != 0 && mask_ptr[batch_idx] == 0) return;

    {raw_real_type} p = p_array[batch_idx];
    if (p <= 0.0) return;

    unsigned long long key = seed_shot[batch_idx] ^ gate_key;
    key = key * 2862933555777941757ULL + 3037000493ULL;
    {raw_real_type} r = ({raw_real_type})(key >> 32) / ({raw_real_type})4294967296.0;

    if (r < p) {{
        key = key * 2862933555777941757ULL + 3037000493ULL;
        int type = (int)((({raw_real_type})(key >> 32) / ({raw_real_type})4294967296.0) * 15.0) + 1;
        int op1 = (type >> 2) & 3;
        int op2 = type & 3;

        long long group_idx = tid / batch_size;
        long long k = group_idx;

        long long mask_low = ((1LL << q2) - 1);
        long long part_low = k & mask_low;
        long long part_high = k >> q2;
        long long with_q2 = (part_high << (q2 + 1)) | part_low;

        long long mask_low_q1 = ((1LL << q1) - 1);
        part_low = with_q2 & mask_low_q1;
        part_high = with_q2 >> q1;
        long long base_idx = (part_high << (q1 + 1)) | part_low;

        long long idx_00 = base_idx * batch_size + batch_idx;
        long long idx_01 = idx_00 + (1LL << q2) * batch_size;
        long long idx_10 = idx_00 + (1LL << q1) * batch_size;
        long long idx_11 = idx_01 + (1LL << q1) * batch_size;

        {raw_vec_type} v00 = state[idx_00]; {raw_vec_type} v01 = state[idx_01];
        {raw_vec_type} v10 = state[idx_10]; {raw_vec_type} v11 = state[idx_11];

        if (op2 == 1) {{ {raw_vec_type} t; t=v00; v00=v01; v01=t; t=v10; v10=v11; v11=t; }}
        else if (op2 == 2) {{
            {raw_vec_type} t00=v00, t01=v01, t10=v10, t11=v11;
            v00.x=t01.y; v00.y=-t01.x;
            v01.x=-t00.y; v01.y=t00.x;
            v10.x=t11.y; v10.y=-t11.x;
            v11.x=-t10.y; v11.y=t10.x;
        }}
        else if (op2 == 3) {{ v01.x=-v01.x; v01.y=-v01.y; v11.x=-v11.x; v11.y=-v11.y; }}

        if (op1 == 1) {{ {raw_vec_type} t; t=v00; v00=v10; v10=t; t=v01; v01=v11; v11=t; }}
        else if (op1 == 2) {{
            {raw_vec_type} t00=v00, t01=v01, t10=v10, t11=v11;
            v00.x=t10.y; v00.y=-t10.x;
            v10.x=-t00.y; v10.y=t00.x;
            v01.x=t11.y; v01.y=-t11.x;
            v11.x=-t01.y; v11.y=t01.x;
        }}
        else if (op1 == 3) {{ v10.x=-v10.x; v10.y=-v10.y; v11.x=-v11.x; v11.y=-v11.y; }}

        state[idx_00] = v00; state[idx_01] = v01; state[idx_10] = v10; state[idx_11] = v11;
    }}
}}
'''
fused_2q_depol_raw = cp.RawKernel(raw_2q_depol_code, 'fused_2q_depol_raw')

# The legacy depolarizing kernel above derives the same shot-level Pauli code
# independently in every four-amplitude group.  For an n-qubit state that
# repeats identical integer RNG work 2**(n-2) times per shot and gate.  The
# optimized path derives each code once, then applies the diagonal gate and
# selected Pauli in a single state-vector pass.
raw_2q_error_code_code = f'''
extern "C" __global__ void build_2q_error_codes(
    unsigned char* out,
    const {raw_real_type}* p_array,
    const unsigned long long* seed_shot,
    const unsigned long long* gate_keys,
    int batch_size,
    long long total_codes,
    const int* fold_masks,
    int variants,
    int use_fold_masks
) {{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_codes) return;

    int batch_idx = (int)(tid % batch_size);
    long long gate_idx = tid / batch_size;
    int variant = (int)(gate_idx % variants);
    long long target_idx = gate_idx / variants;
    if (
        use_fold_masks &&
        variant != 0 &&
        fold_masks[target_idx * batch_size + batch_idx] == 0
    ) {{
        out[tid] = 0;
        return;
    }}
    {raw_real_type} p = p_array[batch_idx];
    unsigned char code = 0;
    if (p > 0.0) {{
        unsigned long long key = seed_shot[batch_idx] ^ gate_keys[gate_idx];
        key = key * 2862933555777941757ULL + 3037000493ULL;
        {raw_real_type} r =
            ({raw_real_type})(key >> 32) /
            ({raw_real_type})4294967296.0;
        if (r < p) {{
            key = key * 2862933555777941757ULL + 3037000493ULL;
            code = (unsigned char)(
                (int)((({raw_real_type})(key >> 32) /
                       ({raw_real_type})4294967296.0) * 15.0) + 1
            );
        }}
    }}
    out[tid] = code;
}}
'''
build_2q_error_codes_raw = cp.RawKernel(
    raw_2q_error_code_code, 'build_2q_error_codes'
)

raw_fused_folded_diagonal_2q_code = f'''
extern "C" __device__ __forceinline__ {raw_vec_type} cmul_folded_diag(
    {raw_vec_type} value, {raw_real_type} dr, {raw_real_type} di
) {{
    {raw_vec_type} out;
    out.x = value.x * dr - value.y * di;
    out.y = value.x * di + value.y * dr;
    return out;
}}

extern "C" __device__ __forceinline__ void apply_pauli_code(
    {raw_vec_type}* v00,
    {raw_vec_type}* v01,
    {raw_vec_type}* v10,
    {raw_vec_type}* v11,
    int type
) {{
    int op1 = (type >> 2) & 3;
    int op2 = type & 3;

    if (op2 == 1) {{
        {raw_vec_type} t;
        t=*v00; *v00=*v01; *v01=t;
        t=*v10; *v10=*v11; *v11=t;
    }} else if (op2 == 2) {{
        {raw_vec_type} t00=*v00, t01=*v01, t10=*v10, t11=*v11;
        v00->x=t01.y; v00->y=-t01.x;
        v01->x=-t00.y; v01->y=t00.x;
        v10->x=t11.y; v10->y=-t11.x;
        v11->x=-t10.y; v11->y=t10.x;
    }} else if (op2 == 3) {{
        v01->x=-v01->x; v01->y=-v01->y;
        v11->x=-v11->x; v11->y=-v11->y;
    }}

    if (op1 == 1) {{
        {raw_vec_type} t;
        t=*v00; *v00=*v10; *v10=t;
        t=*v01; *v01=*v11; *v11=t;
    }} else if (op1 == 2) {{
        {raw_vec_type} t00=*v00, t01=*v01, t10=*v10, t11=*v11;
        v00->x=t10.y; v00->y=-t10.x;
        v10->x=-t00.y; v10->y=t00.x;
        v01->x=t11.y; v01->y=-t11.x;
        v11->x=-t01.y; v11->y=t01.x;
    }} else if (op1 == 3) {{
        v10->x=-v10->x; v10->y=-v10->y;
        v11->x=-v11->x; v11->y=-v11->y;
    }}
}}

extern "C" __device__ __forceinline__ void apply_local_1q(
    {raw_vec_type}* v0,
    {raw_vec_type}* v1,
    {raw_real_type} u00r,
    {raw_real_type} u00i,
    {raw_real_type} u01r,
    {raw_real_type} u01i,
    {raw_real_type} u10r,
    {raw_real_type} u10i,
    {raw_real_type} u11r,
    {raw_real_type} u11i
) {{
    {raw_vec_type} a00 = cmul_folded_diag(*v0, u00r, u00i);
    {raw_vec_type} a01 = cmul_folded_diag(*v1, u01r, u01i);
    {raw_vec_type} a10 = cmul_folded_diag(*v0, u10r, u10i);
    {raw_vec_type} a11 = cmul_folded_diag(*v1, u11r, u11i);
    {raw_vec_type} out0;
    {raw_vec_type} out1;
    out0.x = a00.x + a01.x;
    out0.y = a00.y + a01.y;
    out1.x = a10.x + a11.x;
    out1.y = a10.y + a11.y;
    *v0 = out0;
    *v1 = out1;
}}

extern "C" __global__ void fused_folded_diagonal_2q_with_pauli(
    {raw_vec_type}* state,
    int q1,
    int q2,
    int batch_size,
    int use_pre_q1,
    {raw_real_type} q1u00r,
    {raw_real_type} q1u00i,
    {raw_real_type} q1u01r,
    {raw_real_type} q1u01i,
    {raw_real_type} q1u10r,
    {raw_real_type} q1u10i,
    {raw_real_type} q1u11r,
    {raw_real_type} q1u11i,
    int use_pre_q2,
    {raw_real_type} q2u00r,
    {raw_real_type} q2u00i,
    {raw_real_type} q2u01r,
    {raw_real_type} q2u01i,
    {raw_real_type} q2u10r,
    {raw_real_type} q2u10i,
    {raw_real_type} q2u11r,
    {raw_real_type} q2u11i,
    {raw_real_type} d00r,
    {raw_real_type} d00i,
    {raw_real_type} d01r,
    {raw_real_type} d01i,
    {raw_real_type} d10r,
    {raw_real_type} d10i,
    {raw_real_type} d11r,
    {raw_real_type} d11i,
    const unsigned char* base_error_codes,
    const unsigned char* inverse_error_codes,
    const unsigned char* forward_error_codes,
    long long total_tasks,
    const int* fold_mask,
    int folding_mode
) {{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_tasks) return;

    int batch_idx = (int)(tid % batch_size);
    long long k = tid / batch_size;
    long long mask_low = ((1LL << q2) - 1);
    long long part_low = k & mask_low;
    long long part_high = k >> q2;
    long long with_q2 = (part_high << (q2 + 1)) | part_low;

    long long mask_low_q1 = ((1LL << q1) - 1);
    part_low = with_q2 & mask_low_q1;
    part_high = with_q2 >> q1;
    long long base_idx = (part_high << (q1 + 1)) | part_low;

    long long idx_00 = base_idx * batch_size + batch_idx;
    long long idx_01 = idx_00 + (1LL << q2) * batch_size;
    long long idx_10 = idx_00 + (1LL << q1) * batch_size;
    long long idx_11 = idx_01 + (1LL << q1) * batch_size;

    {raw_vec_type} v00 = state[idx_00];
    {raw_vec_type} v01 = state[idx_01];
    {raw_vec_type} v10 = state[idx_10];
    {raw_vec_type} v11 = state[idx_11];

    if (use_pre_q2) {{
        apply_local_1q(
            &v00, &v01,
            q2u00r, q2u00i, q2u01r, q2u01i,
            q2u10r, q2u10i, q2u11r, q2u11i
        );
        apply_local_1q(
            &v10, &v11,
            q2u00r, q2u00i, q2u01r, q2u01i,
            q2u10r, q2u10i, q2u11r, q2u11i
        );
    }}
    if (use_pre_q1) {{
        apply_local_1q(
            &v00, &v10,
            q1u00r, q1u00i, q1u01r, q1u01i,
            q1u10r, q1u10i, q1u11r, q1u11i
        );
        apply_local_1q(
            &v01, &v11,
            q1u00r, q1u00i, q1u01r, q1u01i,
            q1u10r, q1u10i, q1u11r, q1u11i
        );
    }}

    v00 = cmul_folded_diag(v00, d00r, d00i);
    v01 = cmul_folded_diag(v01, d01r, d01i);
    v10 = cmul_folded_diag(v10, d10r, d10i);
    v11 = cmul_folded_diag(v11, d11r, d11i);
    apply_pauli_code(
        &v00, &v01, &v10, &v11, (int)base_error_codes[batch_idx]
    );

    bool folded =
        folding_mode == 2 ||
        (folding_mode == 1 && fold_mask[batch_idx] != 0);
    if (folded) {{
        int inverse_code = (int)inverse_error_codes[batch_idx];
        int forward_code = (int)forward_error_codes[batch_idx];
        if (inverse_code != 0 || forward_code != 0) {{
            v00 = cmul_folded_diag(v00, d00r, -d00i);
            v01 = cmul_folded_diag(v01, d01r, -d01i);
            v10 = cmul_folded_diag(v10, d10r, -d10i);
            v11 = cmul_folded_diag(v11, d11r, -d11i);
            apply_pauli_code(
                &v00, &v01, &v10, &v11, inverse_code
            );

            v00 = cmul_folded_diag(v00, d00r, d00i);
            v01 = cmul_folded_diag(v01, d01r, d01i);
            v10 = cmul_folded_diag(v10, d10r, d10i);
            v11 = cmul_folded_diag(v11, d11r, d11i);
            apply_pauli_code(
                &v00, &v01, &v10, &v11, forward_code
            );
        }}
    }}

    state[idx_00] = v00;
    state[idx_01] = v01;
    state[idx_10] = v10;
    state[idx_11] = v11;
}}
'''
fused_folded_diagonal_2q_with_pauli_raw = cp.RawKernel(
    raw_fused_folded_diagonal_2q_code,
    'fused_folded_diagonal_2q_with_pauli',
)

# ============================================================
# 7. HELPER FUNCTIONS (gates, measurement, noise)
# ============================================================
def get_free_vram_gb(fraction=VRAM_FREE_FRACTION) -> float:
    free_b, total_b = cp.cuda.Device().mem_info
    return (free_b * float(fraction)) / (1024**3)

def apply_gate_1q(states, gate_2x2, target, cond_mask=None, dummy_mask=None):
    target_mask = 1 << target
    u00, u01 = gate_2x2[0, 0], gate_2x2[0, 1]
    u10, u11 = gate_2x2[1, 0], gate_2x2[1, 1]
    if dummy_mask is None:
        dummy_mask = cp.ones(1, dtype=cp.int32)
    cond_mask_ptr = cond_mask if cond_mask is not None else dummy_mask
    gate_kernel_masked(
        u00, u01, u10, u11,
        target_mask, states.shape[1],
        cond_mask_ptr, (cond_mask is not None),
        states, size=states.size
    )
    return states

def apply_cz_masked(state, qubits, mask):
    cz_mask_kernel(qubits[0], qubits[1], state.shape[1], mask, state, size=state.size)
    return state

def apply_rzz_masked(state, qubits, theta, mask):
    rzz_mask_kernel(qubits[0], qubits[1], theta, state.shape[1], mask, state, size=state.size)
    return state

# diag mask cache (per process)
_DIAG_MASK_CACHE = {}
_DIAG_VEC_CACHE  = {}

def _get_diag_mask_idx(dim, q_high, q_low):
    key = (dim, q_high, q_low)
    out = _DIAG_MASK_CACHE.get(key, None)
    if out is None:
        idx = cp.arange(dim, dtype=cp.int32)
        out = (((idx >> q_high) & 1) * 2 + ((idx >> q_low) & 1)).astype(cp.int32)
        _DIAG_MASK_CACHE[key] = out
    return out

def _get_diag_vec_gpu(diag_vec_cpu):
    key = diag_vec_cpu.tobytes()
    out = _DIAG_VEC_CACHE.get(key, None)
    if out is None:
        out = cp.array(diag_vec_cpu, dtype=c_type)
        _DIAG_VEC_CACHE[key] = out
    return out

def apply_diagonal_gate_base(states, diag_vec_cpu, qubits, dim):
    q1, q2 = qubits
    q_high, q_low = (q1, q2) if q1 > q2 else (q2, q1)
    mask_idx = _get_diag_mask_idx(dim, q_high, q_low)
    diag = _get_diag_vec_gpu(np.array(diag_vec_cpu, dtype=np_c_type))
    states *= diag[mask_idx].reshape((dim, 1))
    return states

def apply_true_2q_noise(states, q1, q2, p_array_gpu, seed_shot_gpu, gate_key_u64, gate_mask, all_ones_mask):
    if q1 < q2:
        q1, q2 = q2, q1
    dim = states.shape[0]
    batch = states.shape[1]
    total_tasks = (dim // 4) * batch
    block = 256
    grid = (total_tasks + block - 1) // block
    mask_ptr = gate_mask if gate_mask is not None else all_ones_mask

    fused_2q_depol_raw(
        (grid,), (block,),
        (states, cp.int32(q1), cp.int32(q2), cp.int32(batch),
         p_array_gpu,
         seed_shot_gpu,
         cp.uint64(gate_key_u64),
         cp.int64(total_tasks),
         mask_ptr)
    )
    return states

def build_round_2q_error_codes(
    p_array_gpu: cp.ndarray,
    seed_shot_gpu: cp.ndarray,
    gate_keys: np.ndarray,
    *,
    variants: int = 1,
    fold_masks: cp.ndarray | None = None,
    dummy_mask: cp.ndarray | None = None,
) -> cp.ndarray:
    """Derive one deterministic depolarizing-Pauli code per gate and shot."""

    keys_gpu = cp.asarray(np.asarray(gate_keys, dtype=np.uint64))
    batch = int(seed_shot_gpu.size)
    if variants <= 0 or int(keys_gpu.size) % int(variants):
        raise ValueError("gate-key count must be divisible by variants")
    if fold_masks is not None:
        expected = (int(keys_gpu.size) // int(variants), batch)
        if tuple(fold_masks.shape) != expected:
            raise ValueError(
                f"fold-mask shape {fold_masks.shape} does not match {expected}"
            )
    if dummy_mask is None:
        dummy_mask = cp.ones(1, dtype=cp.int32)
    mask_ptr = fold_masks if fold_masks is not None else dummy_mask
    codes = cp.empty((int(keys_gpu.size), batch), dtype=cp.uint8)
    total = int(codes.size)
    block = 256
    grid = (total + block - 1) // block
    build_2q_error_codes_raw(
        (grid,),
        (block,),
        (
            codes,
            p_array_gpu,
            seed_shot_gpu,
            keys_gpu,
            cp.int32(batch),
            cp.int64(total),
            mask_ptr,
            cp.int32(variants),
            cp.int32(fold_masks is not None),
        ),
    )
    return codes

def apply_fused_folded_diagonal_2q_with_noise(
    states: cp.ndarray,
    qubits: list[int],
    diagonal: np.ndarray,
    base_error_codes: cp.ndarray,
    inverse_error_codes: cp.ndarray | None,
    forward_error_codes: cp.ndarray | None,
    fold_mask: cp.ndarray | None,
    *,
    fold_all: bool,
    dummy_mask: cp.ndarray,
    pre_unitaries: dict[int, np.ndarray] | None = None,
) -> cp.ndarray:
    """Apply a noisy base gate and its optional folded pair in one pass."""

    q1, q2 = map(int, qubits)
    if q1 < q2:
        q1, q2 = q2, q1
    diagonal = np.asarray(diagonal, dtype=np_c_type)
    if diagonal.shape != (4,):
        raise ValueError(f"expected a four-entry diagonal, got {diagonal.shape}")
    if (inverse_error_codes is None) != (forward_error_codes is None):
        raise ValueError("folded error-code arrays must be supplied together")

    real_scalar = np.float32 if USE_FP32 else np.float64
    pre_unitaries = pre_unitaries or {}
    identity = np.eye(2, dtype=np_c_type)
    pre_q1 = np.asarray(pre_unitaries.get(q1, identity), dtype=np_c_type)
    pre_q2 = np.asarray(pre_unitaries.get(q2, identity), dtype=np_c_type)
    if pre_q1.shape != (2, 2) or pre_q2.shape != (2, 2):
        raise ValueError("pre-unitary matrices must be 2x2")

    def matrix_parts(matrix: np.ndarray) -> tuple[np.floating, ...]:
        return tuple(
            real_scalar(value)
            for amplitude in matrix.ravel()
            for value in (amplitude.real, amplitude.imag)
        )

    pre_q1_parts = matrix_parts(pre_q1)
    pre_q2_parts = matrix_parts(pre_q2)
    scalar_parts = tuple(
        real_scalar(value)
        for amplitude in diagonal
        for value in (amplitude.real, amplitude.imag)
    )
    batch = int(states.shape[1])
    total_tasks = (int(states.shape[0]) // 4) * batch
    block = 256
    grid = (total_tasks + block - 1) // block
    if inverse_error_codes is None:
        inverse_error_codes = base_error_codes
        forward_error_codes = base_error_codes
        folding_mode = 0
    elif fold_all:
        folding_mode = 2
    else:
        if fold_mask is None:
            raise ValueError("a folding mask is required unless fold_all is set")
        folding_mode = 1
    mask_ptr = fold_mask if fold_mask is not None else dummy_mask
    fused_folded_diagonal_2q_with_pauli_raw(
        (grid,),
        (block,),
        (
            states,
            cp.int32(q1),
            cp.int32(q2),
            cp.int32(batch),
            cp.int32(q1 in pre_unitaries),
            *pre_q1_parts,
            cp.int32(q2 in pre_unitaries),
            *pre_q2_parts,
            *scalar_parts,
            base_error_codes,
            inverse_error_codes,
            forward_error_codes,
            cp.int64(total_tasks),
            mask_ptr,
            cp.int32(folding_mode),
        ),
    )
    return states

# ✅ PATCH: norm-free measure (faster + fewer nondet reductions)
def measure_and_collapse(states, qubit, n_qubits, dim, pre_rand=None):
    B = states.shape[1]
    s = states.reshape((2**(n_qubits - 1 - qubit), 2, 2**qubit, B))

    p0 = cp.sum(cp.abs(s[:, 0, :, :])**2, axis=(0, 1))
    p0 = cp.clip(p0, 0.0, 1.0)

    if pre_rand is None:
        rand = cp.random.random(B).astype(r_type, copy=False)
    else:
        rand = pre_rand

    meas = (rand > p0).astype(cp.int32)

    mask0 = (meas == 0)
    mask1 = ~mask0

    s[:, 1, :, mask0] = 0.0
    s[:, 0, :, mask1] = 0.0

    denom0 = cp.sqrt(cp.maximum(p0, epsilon))
    denom1 = cp.sqrt(cp.maximum(1.0 - p0, epsilon))
    denom  = cp.where(mask0, denom0, denom1)
    s /= denom.reshape((1, 1, 1, B))

    return meas, s.reshape((dim, B))

def reset_qubit_batch(states, qubit, n_qubits, dim, pre_rand=None, dummy_mask=None):
    m, states = measure_and_collapse(states, qubit, n_qubits, dim, pre_rand)
    states = apply_gate_1q(states, X_MAT, qubit, cond_mask=m, dummy_mask=dummy_mask)
    return states

# ============================================================
# 8. COMPILATION (with ops_cache lock + atomic write)
# ============================================================
def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    g, x1, y1 = extended_gcd(b % a, a)
    return (g, y1 - (b // a) * x1, x1)

def mod_inverse(a, N):
    g, x, _ = extended_gcd(a, N)
    return x % N if g == 1 else None

def c_QFT(size):
    qc = QuantumCircuit(size)
    k = 0
    qc.h(0)
    for _ in range(1, size):
        for m in range(1, size - k):
            qc.cp(np.pi / float(2 ** m), m + k, k)
        k += 1
        qc.h(k)
    return qc

def c_QFT_i(s): return c_QFT(s).inverse()

def q_add(s, a):
    qc = QuantumCircuit(s)
    for m in range(s):
        qc.p(a * 2*np.pi / float(2**(s-m)), m)
    return qc

def q_sub(s, a):
    qc = QuantumCircuit(s)
    for m in range(s):
        qc.p(-a * 2*np.pi / float(2**(s-m)), m)
    return qc

def q_mod_add(size, a, n):
    qc = QuantumCircuit(size + 3)
    qc.append(q_add(size, a).control(2), range(0, size + 2))
    qc.append(q_sub(size, n), range(2, size + 2))
    qc.append(c_QFT_i(size), range(2, size + 2))
    qc.cx(2, size + 2)
    qc.append(c_QFT(size), range(2, size + 2))
    qc.append(q_add(size, n).control(1), ([size + 2] + list(range(2, size + 2))))
    qc.append(q_sub(size, a).control(2), range(0, size + 2))
    qc.append(c_QFT_i(size), range(2, size + 2))
    qc.x(2)
    qc.cx(2, size + 2)
    qc.x(2)
    qc.append(c_QFT(size), range(2, size + 2))
    qc.append(q_add(size, a).control(2), range(0, size + 2))
    return qc

def q_cmult(size, a, n):
    qc = QuantumCircuit(2 * size + 1)
    qc.append(c_QFT(size), range(size, 2 * size))
    for m in range(size - 1):
        qc.append(q_mod_add(size, (a * (2 ** m)) % n, n), ([0] + [m + 1] + list(range(size, 2 * size + 1))))
    qc.append(c_QFT_i(size), range(size, 2 * size))
    return qc

def c_Ua(size, a, n):
    qc = QuantumCircuit(2 * size + 1)
    qc.append(q_cmult(size, a, n), range(2 * size + 1))
    for m in range(size - 1):
        qc.cswap(0, m + 1, -m + 2 * size - 1)
    qc.append(q_cmult(size, mod_inverse(a, n), n).inverse(), range(2 * size + 1))
    return qc

def get_gate_matrix(name, params):
    if name == 'x':  return X_MAT
    if name == 'h':  return H_MAT
    if name == 'sx': return SX_MAT
    if name == 'id': return np.eye(2, dtype=np_c_type)
    if name == 'rz':
        th = float(params[0])
        return np.array([[np.exp(-0.5j*th), 0], [0, np.exp(0.5j*th)]], dtype=np_c_type)
    if name == 'rx':
        th = float(params[0])
        return np.array([[np.cos(th/2), -1j*np.sin(th/2)], [-1j*np.sin(th/2), np.cos(th/2)]], dtype=np_c_type)
    return None

def fuse_shor_ops(ops_map):
    fused_map = {}
    for k_val, ops in ops_map.items():
        new_ops = []
        skip_indices = set()
        for i in range(len(ops)):
            if i in skip_indices:
                continue
            name, qbs, params = ops[i]
            if name in SINGLE_QUBIT_GATES:
                current_mat = get_gate_matrix(name, params)
                target_qubit = qbs[0]
                j = i + 1
                while j < len(ops):
                    next_name, next_qbs, next_params = ops[j]
                    if next_name in SINGLE_QUBIT_GATES and next_qbs[0] == target_qubit:
                        current_mat = get_gate_matrix(next_name, next_params) @ current_mat
                        skip_indices.add(j)
                        j += 1
                    else:
                        break
                new_ops.append(('unitary', [target_qubit], [current_mat]))
            else:
                new_ops.append(ops[i])
        fused_map[k_val] = new_ops
    return fused_map

def build_fused_2q_execution_plan(ops):
    """Fuse eligible ideal 1Q runs into the following noisy diagonal gate.

    A one-qubit operation can move across other one-qubit operations on
    different qubits.  Within each contiguous 1Q run, operations acting on
    either qubit of the following CZ/RZZ are therefore attached to that gate.
    Original operation indices are retained so stochastic gate keys and the
    random-folding target order do not change.
    """

    plan = []
    pending = []

    def flush_pending():
        for original_idx, pending_op in pending:
            plan.append((original_idx, pending_op, None))
        pending.clear()

    for original_idx, op in enumerate(ops):
        name, qubits, params = op
        if name == 'unitary':
            pending.append((original_idx, op))
            continue
        if name not in NOISY_2Q_GATES:
            flush_pending()
            plan.append((original_idx, op, None))
            continue

        touched = set(map(int, qubits))
        retained = []
        pre_unitaries = {}
        for pending_idx, pending_op in pending:
            target = int(pending_op[1][0])
            if target not in touched:
                retained.append((pending_idx, pending_op))
                continue
            matrix = np.asarray(pending_op[2][0], dtype=np_c_type)
            if target in pre_unitaries:
                pre_unitaries[target] = matrix @ pre_unitaries[target]
            else:
                pre_unitaries[target] = matrix.copy()
        pending.clear()
        for retained_idx, retained_op in retained:
            plan.append((retained_idx, retained_op, None))
        plan.append((original_idx, op, pre_unitaries or None))

    flush_pending()
    return plan

def compile_shor_ops_cached(N, a):
    bg = ",".join(TARGET_BASIS)
    key = f"N{N}_a{a}_opt3_basis[{bg}]"
    path = os.path.join(OPS_CACHE_DIR, key + ".pkl")
    lock_path = path + ".lock"

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)

        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        n_bits = N.bit_length()
        num_measurements = 2 * n_bits
        q = n_bits + 1

        ops_map = {}
        k_list = sorted(list(set([2**(num_measurements - 1 - i) for i in range(num_measurements)])))
        for k_val in k_list:
            t_qc = transpile(
                c_Ua(q, pow(a, k_val, N), N),
                basis_gates=TARGET_BASIS,
                optimization_level=3
            )
            ops_map[k_val] = [
                (inst.operation.name,
                 [t_qc.find_bit(b).index for b in inst.qubits],
                 inst.operation.params)
                for inst in t_qc.data
            ]

        final_map = fuse_shor_ops(ops_map)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix=key + "_", suffix=".tmp", dir=OPS_CACHE_DIR)
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(final_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

        return final_map


def load_compiled_ops_cache(path):
    """Load an explicitly selected trusted cache without inferring its variant."""

    cache_path = Path(path)
    with cache_path.open("rb") as handle:
        ops_map = pickle.load(handle)
    if not isinstance(ops_map, dict) or not ops_map:
        raise ValueError(f"{cache_path}: operation cache is empty or malformed")
    for k_pow, operations in ops_map.items():
        for operation_i, (name, _, parameters) in enumerate(operations):
            if name == "unitary":
                matrix = (
                    np.asarray(parameters[0])
                    if isinstance(parameters, (list, tuple)) and parameters
                    else np.asarray(None)
                )
                if matrix.shape != (2, 2):
                    raise ValueError(
                        f"{cache_path}: fused one-qubit operation {operation_i} "
                        f"in round {k_pow} has no executable 2x2 matrix"
                    )
    return ops_map

# ============================================================
# [PATCH] HBM-bottleneck friendly chunking + smart memory GC
# ============================================================
MAX_BATCH_COLS_PER_CHUNK = 10000
MIN_CONFIGS_PER_CHUNK    = 1
SMART_GC_LOW_FREE_GB     = 8.0

def smart_mem_gc(*, force=False, low_free_gb=SMART_GC_LOW_FREE_GB, tag=""):
    free_b, total_b = cp.cuda.Device().mem_info
    free_gb = free_b / (1024**3)
    if force or (free_gb < float(low_free_gb)):
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        if tag:
            print(f"[smart_mem_gc] freed pools (free_gb={free_gb:.2f}) tag={tag}")

# ============================================================
# 9. ENGINE (chunked)  [PATCHED BLOCK]
#   ✅ reset_ancilla 옵션 추가
# ============================================================
def _run_gpu_engine_chunked(method, rates, lambdas, N, a, shots, ops_map,
                            restrict_folding_to_step=None,
                            shot_offset=0,
                            reset_ancilla=True):   # ✅ 핵심 옵션
    n_bits = N.bit_length()
    q = n_bits + 1
    n_qubits = 2 * q + 1
    dim = 2 ** n_qubits
    num_measurements = 2 * n_bits

    configs = [(r, lam) for r in rates for lam in lambdas]
    num_configs = len(configs)

    free_vram_gb = get_free_vram_gb(VRAM_FREE_FRACTION)
    effective_vram_gb = (free_vram_gb / VRAM_SAFETY_FACTOR)

    state_size_gb = (dim * bytes_per_complex) / (1024**3)
    max_shots_safe = max(1, int(effective_vram_gb / max(state_size_gb, 1e-12)))
    configs_per_chunk_vram = max(1, max_shots_safe // shots)

    configs_per_chunk_hbm = max(MIN_CONFIGS_PER_CHUNK, int(MAX_BATCH_COLS_PER_CHUNK) // int(shots))
    configs_per_chunk = max(1, min(configs_per_chunk_vram, configs_per_chunk_hbm))

    final_results = []
    num_chunks = (num_configs + configs_per_chunk - 1) // configs_per_chunk

    lam_codes = sorted({int(round(float(lam) * 10.0)) for lam in lambdas})

    for chunk_i in range(num_chunks):
        smart_mem_gc(low_free_gb=SMART_GC_LOW_FREE_GB, tag=f"chunk_start {chunk_i}/{num_chunks}")

        start_cfg = chunk_i * configs_per_chunk
        end_cfg = min((chunk_i + 1) * configs_per_chunk, num_configs)
        current_configs = configs[start_cfg:end_cfg]

        chunk_total_batch = len(current_configs) * shots

        _, _, _, seed_shot_cpu = build_seed_shot_array(current_configs, shots, shot_offset=shot_offset)

        batch_noise = np.zeros(chunk_total_batch, dtype=np.float32)
        batch_lam_code_cpu = np.zeros(chunk_total_batch, dtype=np.int16)
        for i_cfg, (r, lam) in enumerate(current_configs):
            s = i_cfg * shots
            e = (i_cfg + 1) * shots
            batch_noise[s:e] = float(r)
            batch_lam_code_cpu[s:e] = int(round(float(lam) * 10.0))

        batch_noise_gpu = cp.array(batch_noise, dtype=r_type)
        batch_lam_code_gpu = cp.array(batch_lam_code_cpu, dtype=cp.int16)
        seed_shot_gpu = cp.array(seed_shot_cpu, dtype=cp.uint64)

        dummy_mask_1 = cp.ones(1, dtype=cp.int32)
        all_ones_mask = cp.ones(chunk_total_batch, dtype=cp.int32)

        lam_masks = {}
        for code in lam_codes:
            lam_masks[code] = (batch_lam_code_gpu == cp.int16(code)).astype(cp.int32)

        psi = cp.zeros((dim, chunk_total_batch), dtype=c_type)
        psi[0, :] = 1.0
        psi = apply_gate_1q(psi, X_MAT, 1, dummy_mask=dummy_mask_1)

        meas_store = cp.zeros((chunk_total_batch, num_measurements), dtype=cp.int32)

        def apply_op(s, op, *, inv=False, mask=None, step_i=None, op_idx=None, noise_tag=0):
            name, qbs, p = op

            if name == 'unitary':
                s = apply_gate_1q(s, p[0], qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
                return s

            par = [-float(v) for v in p] if inv else [float(v) for v in p]

            if name == 'x':
                s = apply_gate_1q(s, X_MAT, qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
            elif name == 'sx':
                s = apply_gate_1q(s, SX_MAT, qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
            elif name == 'rz':
                th = par[0]
                s = apply_gate_1q(
                    s,
                    np.array([[np.exp(-0.5j * th), 0],
                              [0, np.exp(0.5j * th)]], dtype=np_c_type),
                    qbs[0], cond_mask=None, dummy_mask=dummy_mask_1
                )
            elif name == 'rx':
                th = par[0]
                s = apply_gate_1q(
                    s,
                    np.array([[np.cos(th / 2), -1j * np.sin(th / 2)],
                              [-1j * np.sin(th / 2), np.cos(th / 2)]], dtype=np_c_type),
                    qbs[0], cond_mask=None, dummy_mask=dummy_mask_1
                )
            elif name == 'h':
                s = apply_gate_1q(s, H_MAT, qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
            elif name == 'cz':
                if mask is not None:
                    s = apply_cz_masked(s, qbs, mask)
                else:
                    s = apply_diagonal_gate_base(s, np.array([1, 1, 1, -1], dtype=np_c_type), qbs, dim)
            elif name == 'rzz':
                if mask is not None:
                    s = apply_rzz_masked(s, qbs, par[0], mask)
                else:
                    th = par[0]
                    s = apply_diagonal_gate_base(
                        s,
                        np.array([np.exp(-0.5j * th), np.exp(0.5j * th),
                                  np.exp(0.5j * th), np.exp(-0.5j * th)], dtype=np_c_type),
                        qbs, dim
                    )

            if name in NOISY_2Q_GATES:
                gate_key = make_gate_key(
                    step_i if step_i is not None else -1,
                    op_idx  if op_idx  is not None else -1,
                    1 if inv else 0,
                    noise_tag
                )
                s = apply_true_2q_noise(
                    s, qbs[0], qbs[1],
                    batch_noise_gpu,
                    seed_shot_gpu,
                    gate_key,
                    gate_mask=mask,
                    all_ones_mask=all_ones_mask
                )
            return s

        # ✅ 핵심: ancilla reset을 끄면 draws_per_step도 줄여서 gen_uniform 비용 줄임
        draws_per_step = ((q + 1) if reset_ancilla else 0) + 2

        for i in range(num_measurements):
            psi = apply_gate_1q(psi, H_MAT, 0, dummy_mask=dummy_mask_1)

            if i > 0:
                for k2 in range(i):
                    pval = np.exp(1j * (-np.pi / float(2 ** (i - k2))))
                    psi = apply_gate_1q(
                        psi,
                        np.array([[1, 0], [0, pval]], dtype=np_c_type),
                        0,
                        cond_mask=meas_store[:, k2].astype(cp.int32),
                        dummy_mask=dummy_mask_1
                    )

            k_pow = 2 ** (num_measurements - 1 - i)

            if k_pow in ops_map:
                ops = ops_map[k_pow]
                targets = [idx for idx, (n, _, _) in enumerate(ops) if n in NOISY_2Q_GATES]

                do_folding_this_step = True
                if restrict_folding_to_step is not None and i != restrict_folding_to_step:
                    do_folding_this_step = False

                op_to_lam_codes = {}
                if do_folding_this_step:
                    N_targets = len(targets)
                    for lam in lambdas:
                        folds = int(round((lam - 1.0) / 2.0 * N_targets))
                        if folds <= 0:
                            continue

                        lam_code = int(round(float(lam) * 10.0))
                        if method == 'front':
                            folded_indices = set(range(min(folds, N_targets)))
                        elif method == 'back':
                            folded_indices = set(range(max(0, N_targets - folds), N_targets))
                        else:
                            # random / random_noareset / block_scan 등
                            sel_rng = np.random.RandomState(GLOBAL_SEED + i * 999 + lam_code * 10007)
                            folded_indices = set(sel_rng.choice(range(N_targets), min(folds, N_targets), replace=False))

                        for idx_in_targets in folded_indices:
                            op_real_idx = targets[idx_in_targets]
                            op_to_lam_codes.setdefault(op_real_idx, []).append(lam_code)

                for op_idx, op in enumerate(ops):
                    nm, qbs, _ = op

                    psi = apply_op(psi, op, inv=False, mask=None, step_i=i, op_idx=op_idx, noise_tag=0)

                    if nm in NOISY_2Q_GATES and op_idx in op_to_lam_codes:
                        codes = op_to_lam_codes[op_idx]
                        fm = None
                        for code in codes:
                            m = lam_masks[code]
                            fm = m if fm is None else (fm | m)

                        psi = apply_op(psi, op, inv=True,  mask=fm, step_i=i, op_idx=op_idx, noise_tag=1)
                        psi = apply_op(psi, op, inv=False, mask=fm, step_i=i, op_idx=op_idx, noise_tag=2)

            rand_list = []
            for draw_id in range(draws_per_step):
                draw_key = make_draw_key(i, draw_id)
                rand_list.append(gen_uniform(seed_shot_gpu, draw_key))

            draw_ptr = 0

            # ✅ ancilla reset ON일 때만 수행
            if reset_ancilla:
                for r in range(q, 2 * q + 1):
                    psi = reset_qubit_batch(
                        psi, r, n_qubits, dim,
                        pre_rand=rand_list[draw_ptr],
                        dummy_mask=dummy_mask_1
                    )
                    draw_ptr += 1

            psi = apply_gate_1q(psi, H_MAT, 0, dummy_mask=dummy_mask_1)

            m, psi = measure_and_collapse(
                psi, 0, n_qubits, dim,
                pre_rand=rand_list[draw_ptr]
            )
            draw_ptr += 1
            meas_store[:, i] = m

            psi = reset_qubit_batch(
                psi, 0, n_qubits, dim,
                pre_rand=rand_list[draw_ptr],
                dummy_mask=dummy_mask_1
            )
            draw_ptr += 1

        meas_cpu = cp.asnumpy(meas_store)
        for local_idx in range(len(current_configs)):
            res_slice = meas_cpu[local_idx * shots: (local_idx + 1) * shots]
            final_results.append((start_cfg + local_idx, res_slice))

        smart_mem_gc(low_free_gb=SMART_GC_LOW_FREE_GB, tag=f"chunk_end {chunk_i}/{num_chunks}")

    return final_results

def _run_gpu_engine_chunked_shotwise(method, rates, lambdas, N, a, shots, ops_map,
                            restrict_folding_to_step=None,
                            shot_offset=0,
                            reset_ancilla=True,
                            folding_pattern_per_shot=True,
                            fused_2q_kernels=None):   # ✅ 핵심 옵션
    if fused_2q_kernels is None:
        fused_2q_kernels = USE_FUSED_2Q_KERNELS
    fused_2q_kernels = bool(fused_2q_kernels)
    n_bits = N.bit_length()
    q = n_bits + 1
    n_qubits = 2 * q + 1
    dim = 2 ** n_qubits
    num_measurements = 2 * n_bits

    configs = [(r, lam) for r in rates for lam in lambdas]
    num_configs = len(configs)

    free_vram_gb = get_free_vram_gb(VRAM_FREE_FRACTION)
    effective_vram_gb = (free_vram_gb / VRAM_SAFETY_FACTOR)

    state_size_gb = (dim * bytes_per_complex) / (1024**3)
    max_shots_safe = max(1, int(effective_vram_gb / max(state_size_gb, 1e-12)))
    configs_per_chunk_vram = max(1, max_shots_safe // shots)

    configs_per_chunk_hbm = max(MIN_CONFIGS_PER_CHUNK, int(MAX_BATCH_COLS_PER_CHUNK) // int(shots))
    configs_per_chunk = max(1, min(configs_per_chunk_vram, configs_per_chunk_hbm))

    final_results = []
    num_chunks = (num_configs + configs_per_chunk - 1) // configs_per_chunk

    lam_codes = sorted({int(round(float(lam) * 10.0)) for lam in lambdas})

    for chunk_i in range(num_chunks):
        smart_mem_gc(low_free_gb=SMART_GC_LOW_FREE_GB, tag=f"chunk_start {chunk_i}/{num_chunks}")

        start_cfg = chunk_i * configs_per_chunk
        end_cfg = min((chunk_i + 1) * configs_per_chunk, num_configs)
        current_configs = configs[start_cfg:end_cfg]

        chunk_total_batch = len(current_configs) * shots

        _, _, _, seed_shot_cpu = build_seed_shot_array(current_configs, shots, shot_offset=shot_offset)

        batch_noise = np.zeros(chunk_total_batch, dtype=np.float32)
        batch_lam_code_cpu = np.zeros(chunk_total_batch, dtype=np.int16)
        for i_cfg, (r, lam) in enumerate(current_configs):
            s = i_cfg * shots
            e = (i_cfg + 1) * shots
            batch_noise[s:e] = float(r)
            batch_lam_code_cpu[s:e] = int(round(float(lam) * 10.0))

        batch_noise_gpu = cp.array(batch_noise, dtype=r_type)
        batch_lam_code_gpu = cp.array(batch_lam_code_cpu, dtype=cp.int16)
        seed_shot_gpu = cp.array(seed_shot_cpu, dtype=cp.uint64)

        dummy_mask_1 = cp.ones(1, dtype=cp.int32)
        all_ones_mask = cp.ones(chunk_total_batch, dtype=cp.int32)

        lam_masks = {}
        for code in lam_codes:
            lam_masks[code] = (batch_lam_code_gpu == cp.int16(code)).astype(cp.int32)

        psi = cp.zeros((dim, chunk_total_batch), dtype=c_type)
        psi[0, :] = 1.0
        psi = apply_gate_1q(psi, X_MAT, 1, dummy_mask=dummy_mask_1)

        meas_store = cp.zeros((chunk_total_batch, num_measurements), dtype=cp.int32)

        def apply_op(
            s,
            op,
            *,
            inv=False,
            mask=None,
            step_i=None,
            op_idx=None,
            noise_tag=0,
            error_codes=None,
            fold_inverse_error_codes=None,
            fold_forward_error_codes=None,
            fold_mask=None,
            fold_all=False,
            pre_unitaries=None,
        ):
            name, qbs, p = op

            if name == 'unitary':
                s = apply_gate_1q(s, p[0], qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
                return s

            par = [-float(v) for v in p] if inv else [float(v) for v in p]

            if error_codes is not None and name in NOISY_2Q_GATES:
                if name == 'cz':
                    diagonal = np.array([1, 1, 1, -1], dtype=np_c_type)
                elif name == 'rzz':
                    th = par[0]
                    diagonal = np.array(
                        [
                            np.exp(-0.5j * th),
                            np.exp(0.5j * th),
                            np.exp(0.5j * th),
                            np.exp(-0.5j * th),
                        ],
                        dtype=np_c_type,
                    )
                else:
                    raise ValueError(f"unsupported fused two-qubit gate: {name}")
                return apply_fused_folded_diagonal_2q_with_noise(
                    s,
                    qbs,
                    diagonal,
                    error_codes,
                    fold_inverse_error_codes,
                    fold_forward_error_codes,
                    fold_mask,
                    fold_all=bool(fold_all),
                    dummy_mask=dummy_mask_1,
                    pre_unitaries=pre_unitaries,
                )

            if name == 'x':
                s = apply_gate_1q(s, X_MAT, qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
            elif name == 'sx':
                s = apply_gate_1q(s, SX_MAT, qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
            elif name == 'rz':
                th = par[0]
                s = apply_gate_1q(
                    s,
                    np.array([[np.exp(-0.5j * th), 0],
                              [0, np.exp(0.5j * th)]], dtype=np_c_type),
                    qbs[0], cond_mask=None, dummy_mask=dummy_mask_1
                )
            elif name == 'rx':
                th = par[0]
                s = apply_gate_1q(
                    s,
                    np.array([[np.cos(th / 2), -1j * np.sin(th / 2)],
                              [-1j * np.sin(th / 2), np.cos(th / 2)]], dtype=np_c_type),
                    qbs[0], cond_mask=None, dummy_mask=dummy_mask_1
                )
            elif name == 'h':
                s = apply_gate_1q(s, H_MAT, qbs[0], cond_mask=None, dummy_mask=dummy_mask_1)
            elif name == 'cz':
                if mask is not None:
                    s = apply_cz_masked(s, qbs, mask)
                else:
                    s = apply_diagonal_gate_base(s, np.array([1, 1, 1, -1], dtype=np_c_type), qbs, dim)
            elif name == 'rzz':
                if mask is not None:
                    s = apply_rzz_masked(s, qbs, par[0], mask)
                else:
                    th = par[0]
                    s = apply_diagonal_gate_base(
                        s,
                        np.array([np.exp(-0.5j * th), np.exp(0.5j * th),
                                  np.exp(0.5j * th), np.exp(-0.5j * th)], dtype=np_c_type),
                        qbs, dim
                    )

            if name in NOISY_2Q_GATES:
                gate_key = make_gate_key(
                    step_i if step_i is not None else -1,
                    op_idx  if op_idx  is not None else -1,
                    1 if inv else 0,
                    noise_tag
                )
                s = apply_true_2q_noise(
                    s, qbs[0], qbs[1],
                    batch_noise_gpu,
                    seed_shot_gpu,
                    gate_key,
                    gate_mask=mask,
                    all_ones_mask=all_ones_mask
                )
            return s

        # ✅ 핵심: ancilla reset을 끄면 draws_per_step도 줄여서 gen_uniform 비용 줄임
        draws_per_step = ((q + 1) if reset_ancilla else 0) + 2

        for i in range(num_measurements):
            psi = apply_gate_1q(psi, H_MAT, 0, dummy_mask=dummy_mask_1)

            if i > 0:
                for k2 in range(i):
                    pval = np.exp(1j * (-np.pi / float(2 ** (i - k2))))
                    psi = apply_gate_1q(
                        psi,
                        np.array([[1, 0], [0, pval]], dtype=np_c_type),
                        0,
                        cond_mask=meas_store[:, k2].astype(cp.int32),
                        dummy_mask=dummy_mask_1
                    )

            k_pow = 2 ** (num_measurements - 1 - i)

            if k_pow in ops_map:
                ops = ops_map[k_pow]
                targets = [idx for idx, (n, _, _) in enumerate(ops) if n in NOISY_2Q_GATES]

                do_folding_this_step = True
                if restrict_folding_to_step is not None and i != restrict_folding_to_step:
                    do_folding_this_step = False

                op_to_lam_codes = {}
                shotwise_fold_masks_gpu = None
                shotwise_full_folding = False
                target_positions = {
                    op_real_idx: local_idx
                    for local_idx, op_real_idx in enumerate(targets)
                }
                execution_plan = (
                    build_fused_2q_execution_plan(ops)
                    if fused_2q_kernels
                    else [
                        (original_idx, op, None)
                        for original_idx, op in enumerate(ops)
                    ]
                )
                has_any_folding = False
                if do_folding_this_step:
                    N_targets = len(targets)
                    if folding_pattern_per_shot and method == 'random':
                        fold_counts = [
                            min(
                                max(
                                    int(
                                        round(
                                            (float(lam) - 1.0)
                                            * N_targets
                                            / 2.0
                                        )
                                    ),
                                    0,
                                ),
                                N_targets,
                            )
                            for _, lam in current_configs
                        ]
                        has_any_folding = any(count > 0 for count in fold_counts)
                        shotwise_full_folding = (
                            fused_2q_kernels
                            and has_any_folding
                            and all(count == N_targets for count in fold_counts)
                        )
                        if (
                            not fused_2q_kernels
                            or (has_any_folding and not shotwise_full_folding)
                        ):
                            shotwise_fold_masks_cpu = _build_shotwise_folding_masks(
                                current_configs,
                                shots,
                                shot_offset,
                                i,
                                N_targets,
                                global_seed=GLOBAL_SEED,
                                N=N,
                                a=a,
                            )
                            shotwise_fold_masks_gpu = cp.asarray(
                                shotwise_fold_masks_cpu, dtype=cp.int32
                            )
                            del shotwise_fold_masks_cpu
                    else:
                        for lam in lambdas:
                            folds = int(round((lam - 1.0) / 2.0 * N_targets))
                            if folds <= 0:
                                continue
                            has_any_folding = True

                            lam_code = int(round(float(lam) * 10.0))
                            if method == 'front':
                                folded_indices = set(range(min(folds, N_targets)))
                            elif method == 'back':
                                folded_indices = set(range(max(0, N_targets - folds), N_targets))
                            else:
                                # random / random_noareset / block_scan 등
                                sel_rng = np.random.RandomState(GLOBAL_SEED + i * 999 + lam_code * 10007)
                                folded_indices = set(sel_rng.choice(range(N_targets), min(folds, N_targets), replace=False))

                            for idx_in_targets in folded_indices:
                                op_real_idx = targets[idx_in_targets]
                                op_to_lam_codes.setdefault(op_real_idx, []).append(lam_code)

                error_code_variants = 1 + (2 if has_any_folding else 0)
                round_error_codes_gpu = None
                if fused_2q_kernels and targets:
                    gate_keys = []
                    for op_real_idx in targets:
                        gate_keys.append(make_gate_key(i, op_real_idx, 0, 0))
                        if has_any_folding:
                            gate_keys.append(make_gate_key(i, op_real_idx, 1, 1))
                            gate_keys.append(make_gate_key(i, op_real_idx, 0, 2))
                    round_error_codes_gpu = build_round_2q_error_codes(
                        batch_noise_gpu,
                        seed_shot_gpu,
                        np.asarray(gate_keys, dtype=np.uint64),
                        variants=error_code_variants,
                        fold_masks=shotwise_fold_masks_gpu,
                        dummy_mask=dummy_mask_1,
                    )

                for op_idx, op, pre_unitaries in execution_plan:
                    nm, qbs, _ = op
                    target_position = target_positions.get(op_idx)
                    base_error_codes = (
                        round_error_codes_gpu[
                            target_position * error_code_variants
                        ]
                        if (
                            round_error_codes_gpu is not None
                            and target_position is not None
                        )
                        else None
                    )

                    if (
                        fused_2q_kernels
                        and nm in NOISY_2Q_GATES
                        and target_position is not None
                    ):
                        fm = None
                        fold_all = False
                        fold_this_op = False
                        if (
                            shotwise_fold_masks_gpu is not None
                            or shotwise_full_folding
                        ):
                            fold_this_op = True
                            fold_all = shotwise_full_folding
                            if not fold_all:
                                fm = shotwise_fold_masks_gpu[target_position]
                        elif op_idx in op_to_lam_codes:
                            fold_this_op = True
                            for code in op_to_lam_codes[op_idx]:
                                m = lam_masks[code]
                                fm = m if fm is None else (fm | m)

                        inverse_error_codes = (
                            round_error_codes_gpu[
                                target_position * error_code_variants + 1
                            ]
                            if fold_this_op
                            else None
                        )
                        forward_error_codes = (
                            round_error_codes_gpu[
                                target_position * error_code_variants + 2
                            ]
                            if fold_this_op
                            else None
                        )
                        psi = apply_op(
                            psi,
                            op,
                            inv=False,
                            mask=None,
                            step_i=i,
                            op_idx=op_idx,
                            noise_tag=0,
                            error_codes=base_error_codes,
                            fold_inverse_error_codes=inverse_error_codes,
                            fold_forward_error_codes=forward_error_codes,
                            fold_mask=fm,
                            fold_all=fold_all,
                            pre_unitaries=pre_unitaries,
                        )
                        continue

                    psi = apply_op(
                        psi,
                        op,
                        inv=False,
                        mask=None,
                        step_i=i,
                        op_idx=op_idx,
                        noise_tag=0,
                    )

                    if (
                        nm in NOISY_2Q_GATES
                        and (
                            shotwise_fold_masks_gpu is not None
                            or shotwise_full_folding
                        )
                        and op_idx in target_positions
                    ):
                        fm = (
                            None
                            if shotwise_full_folding
                            else shotwise_fold_masks_gpu[target_position]
                        )
                        psi = apply_op(
                            psi,
                            op,
                            inv=True,
                            mask=fm,
                            step_i=i,
                            op_idx=op_idx,
                            noise_tag=1,
                        )
                        psi = apply_op(
                            psi,
                            op,
                            inv=False,
                            mask=fm,
                            step_i=i,
                            op_idx=op_idx,
                            noise_tag=2,
                        )
                    elif nm in NOISY_2Q_GATES and op_idx in op_to_lam_codes:
                        codes = op_to_lam_codes[op_idx]
                        fm = None
                        for code in codes:
                            m = lam_masks[code]
                            fm = m if fm is None else (fm | m)
                        psi = apply_op(
                            psi,
                            op,
                            inv=True,
                            mask=fm,
                            step_i=i,
                            op_idx=op_idx,
                            noise_tag=1,
                        )
                        psi = apply_op(
                            psi,
                            op,
                            inv=False,
                            mask=fm,
                            step_i=i,
                            op_idx=op_idx,
                            noise_tag=2,
                        )

                if shotwise_fold_masks_gpu is not None:
                    del shotwise_fold_masks_gpu
                if round_error_codes_gpu is not None:
                    del round_error_codes_gpu

            rand_list = []
            for draw_id in range(draws_per_step):
                draw_key = make_draw_key(i, draw_id)
                rand_list.append(gen_uniform(seed_shot_gpu, draw_key))

            draw_ptr = 0

            # ✅ ancilla reset ON일 때만 수행
            if reset_ancilla:
                for r in range(q, 2 * q + 1):
                    psi = reset_qubit_batch(
                        psi, r, n_qubits, dim,
                        pre_rand=rand_list[draw_ptr],
                        dummy_mask=dummy_mask_1
                    )
                    draw_ptr += 1

            psi = apply_gate_1q(psi, H_MAT, 0, dummy_mask=dummy_mask_1)

            m, psi = measure_and_collapse(
                psi, 0, n_qubits, dim,
                pre_rand=rand_list[draw_ptr]
            )
            draw_ptr += 1
            meas_store[:, i] = m

            psi = reset_qubit_batch(
                psi, 0, n_qubits, dim,
                pre_rand=rand_list[draw_ptr],
                dummy_mask=dummy_mask_1
            )
            draw_ptr += 1

        meas_cpu = cp.asnumpy(meas_store)
        for local_idx in range(len(current_configs)):
            res_slice = meas_cpu[local_idx * shots: (local_idx + 1) * shots]
            final_results.append((start_cfg + local_idx, res_slice))

        smart_mem_gc(low_free_gb=SMART_GC_LOW_FREE_GB, tag=f"chunk_end {chunk_i}/{num_chunks}")

    return final_results

# ============================================================
# 10. SAFE RUNNER (FINAL + PARTIAL TASK SUPPORT)
# ============================================================
def get_safe_filename(method, N, a, rate):
    return os.path.join(RESULTS_DIR, f"safe_{method}_N{N}_a{a}_p{rate:.6f}.pkl")

def get_part_filename(method, N, a, rate, tag):
    return os.path.join(RESULTS_DIR, f"part_safe_{method}_N{N}_a{a}_p{rate:.6f}_{tag}.pkl")

def _write_pkl_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

# ---------------------------
# (A) λ-part runner
# ---------------------------
def run_lambda_part(N, a, rate, method, lambdas_subset, shots, ops_map, *,
                    reset_ancilla=True,
                    shot_offset=0):   # ✅ 추가
    """
    1 rate + subset lambdas만 계산해서 part 파일로 저장.
    shot_offset으로 동일 (rate,lam)을 여러 파트로 쪼갤 수 있음.
    """
    results = _run_gpu_engine_chunked(
        method, [rate], lambdas_subset, N, a, shots,
        ops_map,
        restrict_folding_to_step=None,
        shot_offset=int(shot_offset),       # ✅ 전달
        reset_ancilla=reset_ancilla,
    )

    data_map = {}
    for local_cfg_idx, data in results:
        l_idx = local_cfg_idx % len(lambdas_subset)
        lam = lambdas_subset[l_idx]
        data_map[(rate, lam)] = data

    # ✅ shot range tag (merge에서 정렬용)
    tag = f"L{len(lambdas_subset)}_{min(lambdas_subset):.1f}-{max(lambdas_subset):.1f}_R{int(reset_ancilla)}_S{int(shot_offset):06d}-{int(shot_offset+shots-1):06d}"
    part_path = get_part_filename(method, N, a, rate, tag)

    _write_pkl_atomic(part_path, {
        "kind": "lambda_part",
        "rate": float(rate),
        "method": method,
        "lambdas": list(map(float, lambdas_subset)),
        "reset_ancilla": bool(reset_ancilla),
        "shot_offset": int(shot_offset),     # ✅ 저장
        "shots": int(shots),                 # ✅ 저장
        "payload": data_map,
    })
    return part_path


# ---------------------------
# merge helpers (MAIN process only)
# ---------------------------
def merge_parts_for_rate_method(N, a, rate, method, *, full_lambdas=None, expected_total_shots=None):
    final_path = get_safe_filename(method, N, a, rate)

    part_glob = os.path.join(RESULTS_DIR, f"part_safe_{method}_N{N}_a{a}_p{rate:.6f}_*.pkl")
    part_files = sorted(glob.glob(part_glob))
    if not part_files:
        raise FileNotFoundError(f"No part files found for merge: {part_glob}")

    # collect parts per (rate, lam): list of (shot_offset, shots, arr)
    parts = {}  # (rate, lam) -> list
    for pf in part_files:
        with open(pf, "rb") as f:
            obj = pickle.load(f)
        assert obj.get("kind") == "lambda_part"
        shot_offset = int(obj.get("shot_offset", 0))
        shots = int(obj.get("shots", 0))
        payload = obj["payload"]

        for k, arr in payload.items():
            parts.setdefault(k, []).append((shot_offset, shots, arr))

    # stitch
    merged = {}
    for k, plist in parts.items():
        plist.sort(key=lambda x: x[0])  # shot_offset ascending
        # sanity: contiguous offsets (optional but recommended)
        offs = [p[0] for p in plist]
        lens = [p[1] for p in plist]
        for i in range(1, len(plist)):
            if offs[i] != offs[i-1] + lens[i-1]:
                raise RuntimeError(f"Non-contiguous shot_offsets for {k}: {offs} with lens {lens}")

        merged[k] = np.vstack([p[2] for p in plist])

    # sanity: expected lambdas present
    if full_lambdas is not None:
        for lam in full_lambdas:
            if (rate, lam) not in merged:
                raise RuntimeError(f"Missing (rate,lam)=({rate},{lam}) in merged parts for method={method}")

    # sanity: total shots check
    if expected_total_shots is not None:
        for lam in (full_lambdas if full_lambdas is not None else sorted({k[1] for k in merged.keys()})):
            arr = merged[(rate, lam)]
            if arr.shape[0] != int(expected_total_shots):
                raise RuntimeError(f"Shot count mismatch for (rate,lam)=({rate},{lam}): {arr.shape[0]} != {expected_total_shots}")

    _write_pkl_atomic(final_path, merged)
    return final_path, part_files

def cleanup_part_files(part_files):
    for pf in part_files:
        try:
            os.remove(pf)
        except:
            pass


# ============================================================
# 11. MULTI-GPU DRIVER (Dynamic Queue, PART TASKS)
# ============================================================
def worker_process_dynamic(
    gpu_id, task_queue, N, a, results_dir, ops_cache_path, global_seed
):
    global RESULTS_DIR, OPS_CACHE_DIR, GLOBAL_SEED
    RESULTS_DIR = str(results_dir)
    OPS_CACHE_DIR = str(Path(ops_cache_path).parent)
    GLOBAL_SEED = int(global_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OPS_CACHE_DIR, exist_ok=True)

    with cp.cuda.Device(gpu_id):
        print(f"🚀 [GPU {gpu_id}] Ready to work!")

        ops_map = load_compiled_ops_cache(ops_cache_path)

        while True:
            task = task_queue.get()   # ✅ blocking get
            if task is None:          # ✅ sentinel
                print(f"👋 [GPU {gpu_id}] Got sentinel. Exiting.")
                break

            try:
                kind = task["kind"]
                rate = task["rate"]

                if kind == "lambda_part":
                    method = task["method"]
                    lambdas_subset = task["lambdas"]
                    shots = task["shots"]
                    reset_ancilla = task.get("reset_ancilla", True)
                    shot_offset = task.get("shot_offset", 0)

                    print(f"▶️ [GPU {gpu_id}] lambda_part: rate={rate:.6f}, method={method}, lambdas={lambdas_subset}, shots={shots}, shot_offset={shot_offset}, reset_ancilla={reset_ancilla}")

                    run_lambda_part(
                        N, a, rate, method, lambdas_subset, shots, ops_map,
                        reset_ancilla=reset_ancilla,
                        shot_offset=shot_offset,
                    )
                else:
                    raise ValueError(f"Unknown task kind: {kind}")

            except Exception as e:
                print(f"❌ [GPU {gpu_id}] Error task={task}: {e}")
                import traceback
                traceback.print_exc()
                raise


def _project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_run_config(path):
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "instance", "N", "a", "rates", "lambdas", "shots_per_lambda",
        "global_seed", "ops_cache",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")
    return config, config_path


def _shot_partitions(total_shots, shots_per_part):
    if total_shots <= 0 or shots_per_part <= 0:
        raise ValueError("shot counts must be positive")
    parts = []
    offset = 0
    while offset < total_shots:
        size = min(shots_per_part, total_shots - offset)
        parts.append((size, offset))
        offset += size
    return parts


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _existing_outputs(N, a, rates, method):
    existing = []
    for rate in rates:
        final_path = Path(get_safe_filename(method, N, a, rate))
        if final_path.exists():
            existing.append(final_path)
        part_pattern = os.path.join(
            RESULTS_DIR,
            f"part_safe_{method}_N{N}_a{a}_p{float(rate):.6f}_*.pkl",
        )
        existing.extend(Path(path) for path in glob.glob(part_pattern))
    metadata = Path(RESULTS_DIR) / "run_metadata.json"
    if metadata.exists():
        existing.append(metadata)
    return sorted(set(existing))


def _remove_explicit_outputs(paths):
    for path in paths:
        Path(path).unlink()


def _parse_gpu_ids(value, available):
    if value is None:
        return list(range(available))
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError("--gpu-ids must contain distinct comma-separated IDs")
    if min(result) < 0 or max(result) >= available:
        raise ValueError(f"GPU IDs {result} are outside the available range 0..{available - 1}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the historical N=119 CuPy simulator with an explicit, "
            "non-destructive configuration."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--reset-mode",
        required=True,
        choices=("reset", "no-reset"),
        help="Reset or retain the modular-work ancillas after each QPE round.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ops-cache", type=Path)
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--lambdas", nargs="+", type=float)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--shots-per-part", type=int, default=10000)
    parser.add_argument("--lambda-group-size", type=int, default=1)
    parser.add_argument(
        "--gpu-ids", help="Comma-separated CUDA device IDs; default: every visible GPU."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace only matching outputs in --output-dir.",
    )
    parser.add_argument(
        "--keep-parts", action="store_true", help="Retain temporary shot-part files."
    )
    return parser.parse_args(argv)


def main(argv=None):
    global RESULTS_DIR, OPS_CACHE_DIR, GLOBAL_SEED

    args = parse_args(argv)
    config, config_path = _load_run_config(args.config)
    N = int(config["N"])
    a = int(config["a"])
    rates = [float(value) for value in (args.rates or config["rates"])]
    configured_lambdas = [1.0] if args.reset_mode == "no-reset" else config["lambdas"]
    lambdas = [float(value) for value in (args.lambdas or configured_lambdas)]
    shots = int(args.shots or config["shots_per_lambda"])
    global_seed = int(config["global_seed"])
    reset_ancilla = args.reset_mode == "reset"
    method = "random" if reset_ancilla else "random_noareset"

    if not rates or len(set(rates)) != len(rates) or min(rates) < 0.0:
        raise ValueError("rates must be distinct and nonnegative")
    if not lambdas or len(set(lambdas)) != len(lambdas) or min(lambdas) < 1.0:
        raise ValueError("lambdas must be distinct and at least 1")
    if not reset_ancilla and lambdas != [1.0]:
        raise ValueError("the historical no-reset baseline is defined only at lambda=1")
    if args.lambda_group_size <= 0:
        raise ValueError("--lambda-group-size must be positive")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "simulator_runs" / config["instance"] / args.reset_mode
    )
    cache_path = _project_path(args.ops_cache or config["ops_cache"])
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Saved operation cache not found: {cache_path}. Refusing an implicit recompile."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = str(output_dir)
    OPS_CACHE_DIR = str(cache_path.parent)
    GLOBAL_SEED = global_seed
    os.environ["SHOR_QEM_RESULTS_DIR"] = RESULTS_DIR
    os.environ["SHOR_QEM_OPS_CACHE_DIR"] = OPS_CACHE_DIR

    existing = _existing_outputs(N, a, rates, method)
    if existing and not args.overwrite:
        preview = "\n".join(f"  - {path}" for path in existing[:10])
        raise FileExistsError(
            "Matching outputs already exist; choose a new --output-dir or pass "
            f"--overwrite explicitly:\n{preview}"
        )
    if existing:
        _remove_explicit_outputs(existing)
        print(f"Removed {len(existing)} explicitly matched prior output(s).")

    available_gpus = int(cp.cuda.runtime.getDeviceCount())
    if available_gpus == 0:
        raise RuntimeError("No CUDA GPUs found")
    gpu_ids = _parse_gpu_ids(args.gpu_ids, available_gpus)
    partitions = _shot_partitions(shots, int(args.shots_per_part))

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    tasks = []
    for rate in rates:
        for start in range(0, len(lambdas), args.lambda_group_size):
            subset = lambdas[start : start + args.lambda_group_size]
            for part_shots, shot_offset in partitions:
                tasks.append(
                    {
                        "kind": "lambda_part",
                        "rate": rate,
                        "method": method,
                        "lambdas": subset,
                        "shots": part_shots,
                        "shot_offset": shot_offset,
                        "reset_ancilla": reset_ancilla,
                    }
                )
    np.random.default_rng(global_seed).shuffle(tasks)
    task_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in gpu_ids:
        task_queue.put(None)

    print(
        f"Starting {len(tasks)} tasks on GPUs {gpu_ids}: N={N}, a={a}, "
        f"mode={args.reset_mode}, shots={shots}, lambdas={lambdas}"
    )
    processes = []
    for gpu_id in gpu_ids:
        process = mp.Process(
            target=worker_process_dynamic,
            args=(
                gpu_id,
                task_queue,
                N,
                a,
                RESULTS_DIR,
                str(cache_path),
                global_seed,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"one or more GPU workers failed with exit codes {failures}")

    final_paths = []
    for rate in rates:
        final_path, part_files = merge_parts_for_rate_method(
            N,
            a,
            rate,
            method,
            full_lambdas=lambdas,
            expected_total_shots=shots,
        )
        final_paths.append(Path(final_path))
        if not args.keep_parts:
            cleanup_part_files(part_files)
        print(f"Merged {final_path} from {len(part_files)} part file(s).")

    metadata = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "shor_qem.simulator", *sys.argv[1:]],
        "source_config": str(config_path),
        "instance": config["instance"],
        "N": N,
        "a": a,
        "order": config.get("order"),
        "rates": rates,
        "lambdas": lambdas,
        "shots_per_lambda": shots,
        "shot_partitions": [
            {"shots": size, "shot_offset": offset} for size, offset in partitions
        ],
        "global_seed": global_seed,
        "method": method,
        "reset_ancilla": reset_ancilla,
        "gpu_ids": gpu_ids,
        "folding_design": (
            "one seeded subset per lambda and QPE round, reused across all shots "
            "and physical error rates"
        ),
        "folding_seed_formula": (
            "global_seed + qpe_round * 999 + round(lambda * 10) * 10007"
        ),
        "shot_seed_fields": ["physical_error_rate", "lambda", "shot_id"],
        "two_qubit_gate_count": config.get("two_qubit_gate_count"),
        "ops_cache": str(cache_path),
        "ops_cache_sha256": _sha256(cache_path),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "qiskit": _package_version("qiskit"),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        },
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in final_paths
        ],
    }
    metadata_path = output_dir / "run_metadata.json"
    _atomic_json(metadata_path, metadata)
    print(f"Completed run; metadata written to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
