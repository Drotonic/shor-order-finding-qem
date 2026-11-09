"""CPU-only generalized dynamic Shor construction and native gate folding.

No order is supplied to synthesis, no GPU module is imported, and no backend
job is submitted.  Compile the Reset arm once, derive the matched Raw arm by
removing only marked workspace resets, and fold *after* that final compile.
Never transpile or optimize a returned folded circuit: that can cancel folds.

The arithmetic helpers are copied verbatim from the frozen public simulator;
their source and AST hashes make this independent CPU implementation auditable.
"""

from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import math
from numbers import Integral, Real
import random

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Reset
from qiskit.transpiler import Target


SOURCE_PATH = "src/shor_qem/simulator.py"
SOURCE_SHA256 = "3e00444360affc759d419e71145f3728fcdb77eb98283107b31c9c174a15f170"
ARITHMETIC_AST_SHA256 = "e31c1d65205c4f3fb7aaa178a15f959cff0335131b7c01b56040f5b385d9936d"
ARITHMETIC_AST_FORMAT = "CPython 3.12 ast.dump(include_attributes=False), including empty fields"
ARITHMETIC_HELPERS = (
    "extended_gcd", "mod_inverse", "c_QFT", "c_QFT_i", "q_add", "q_sub",
    "q_mod_add", "q_cmult", "c_Ua",
)
CIRCUIT_FAMILY = "template_retained_reference"
RESET_PREFIX = "workspace_reset:"


# BEGIN VERBATIM FROZEN ARITHMETIC (source lines 842--910).
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
# END VERBATIM FROZEN ARITHMETIC.


def build_dynamic_shor_circuit(
    N: int, a: int, workspace_reset: bool = True,
) -> QuantumCircuit:
    """Build all ``L=2*N.bit_length()`` semiclassical QPE rounds.

    There are ``2*N.bit_length()+3`` qubits: recycled control 0, persistent
    value 1..n, and workspace n+1..2*n+2.  Phase round i measures phase[i],
    with integer outcome ``y=sum(phase[i]*2**i)``.  Count strings display
    phase[L-1] first.  Multiplier-one rounds retain the complete arithmetic
    template.  The only difference between arms is the marked workspace
    resets after each modular multiplication, including the final round.
    """
    if isinstance(N, bool) or not isinstance(N, Integral) or N < 3:
        raise ValueError("N must be an integer at least 3")
    if isinstance(a, bool) or not isinstance(a, Integral) or not 1 < a < N:
        raise ValueError("a must be an integer with 1 < a < N")
    N, a = int(N), int(a)
    if math.gcd(a, N) != 1:
        raise ValueError("a and N must be coprime for reversible modular multiplication")
    if not isinstance(workspace_reset, bool):
        raise TypeError("workspace_reset must be bool")
    n = N.bit_length()
    L, q = 2 * n, n + 1
    qubits = QuantumRegister(2 * q + 1, "q")
    phase = ClassicalRegister(L, "phase")
    arm = "reset" if workspace_reset else "raw"
    qc = QuantumCircuit(qubits, phase, name=f"shor_N{N}_a{a}_{arm}")
    powers = [2 ** (L - 1 - i) for i in range(L)]
    multipliers = [pow(a, exponent, N) for exponent in powers]
    qc.metadata = {
        "circuit_family": CIRCUIT_FAMILY,
        "source_path": SOURCE_PATH,
        "source_sha256": SOURCE_SHA256,
        "arithmetic_ast_sha256": ARITHMETIC_AST_SHA256,
        "arithmetic_ast_format": ARITHMETIC_AST_FORMAT,
        "N": N, "a": a, "n_bits": n, "measurement_bits": L,
        "control_qubit": 0,
        "persistent_value_qubits": list(range(1, n + 1)),
        "workspace_qubits": list(range(q, 2 * q + 1)),
        "workspace_reset": workspace_reset,
        "round_exponents": powers,
        "round_multipliers": multipliers,
        "phase_bit_order": "y=sum_i phase[i]*2**i",
        "known_order_used_for_synthesis": False,
        "all_QPE_rounds_retained": True,
    }
    qc.x(qubits[1])
    for i, multiplier in enumerate(multipliers):
        qc.h(qubits[0])
        for j in range(i):
            with qc.if_test((phase[j], 1)):
                qc.p(-np.pi / float(2 ** (i - j)), qubits[0])
        arithmetic = c_Ua(q, multiplier, N)
        arithmetic.name = f"c_Ua_N{N}_a{multiplier}"
        qc.append(arithmetic, qubits)
        if workspace_reset:
            for workspace_qubit in range(q, 2 * q + 1):
                qc.append(
                    Reset(label=f"{RESET_PREFIX}{i}:{workspace_qubit}"),
                    [qubits[workspace_qubit]],
                )
        qc.h(qubits[0])
        qc.measure(qubits[0], phase[i])
        qc.reset(qubits[0])
    return qc


def is_workspace_reset(instruction) -> bool:
    op = instruction.operation
    return op.name == "reset" and (op.label or "").startswith(RESET_PREFIX)


def strip_workspace_resets(circuit: QuantumCircuit) -> QuantumCircuit:
    """Preserve the routed circuit, dropping only marked top-level resets."""
    result = circuit.copy_empty_like(name=f"{circuit.name}_raw")
    for instruction in circuit.data:
        if not is_workspace_reset(instruction):
            result.append(instruction)
    result.metadata = dict(circuit.metadata or {})
    result.metadata["workspace_reset"] = False
    if result.metadata.get("native_qubit_index_space") == "physical_target":
        result.metadata.update(native_instruction_maps(result))
    return result


def walk_instructions(circuit, physical_qubits=None):
    """Yield instructions with root/physical qargs, recursing into branches."""
    if physical_qubits is None:
        physical_qubits = tuple(range(circuit.num_qubits))
    for instruction in circuit.data:
        qubits = tuple(physical_qubits[circuit.find_bit(q).index] for q in instruction.qubits)
        yield instruction, qubits
        for block in getattr(instruction.operation, "blocks", ()):
            yield from walk_instructions(block, qubits)


def native_instruction_maps(circuit) -> dict:
    """Read actual routed qargs; source roles/final layout are not role maps."""
    measurements, workspace_resets, other_resets = [], [], []
    for index, instruction in enumerate(circuit.data):
        op = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        item = {"top_level_instruction_index": index, "physical_qubit_indices": qubits}
        if op.name.startswith("measure"):
            item["phase_clbit_indices"] = [circuit.find_bit(c).index for c in instruction.clbits]
            measurements.append(item)
        elif op.name == "reset":
            if is_workspace_reset(instruction):
                _, round_index, logical_qubit = op.label.split(":")
                item.update({
                    "source_workspace_reset_label": op.label,
                    "source_round_index": int(round_index),
                    "source_logical_qubit_index": int(logical_qubit),
                })
                workspace_resets.append(item)
            else:
                other_resets.append(item)
    return {
        "native_control_measurements": measurements,
        "native_workspace_reset_sites": workspace_resets,
        "native_other_reset_sites": other_resets,
    }


def _restricted_target(target: Target) -> Target:
    """Retain target connectivity/properties but require CX/CZ native gates.

    Broad simulator targets advertise non-native composite instructions.  A
    restricted view forces their decomposition; target support is validated
    again against the original target after routing.  CZ is preferred when
    both entanglers are advertised.  No synthetic coupling edges are added.
    """
    entangler = next((name for name in ("cz", "cx") if name in target.operation_names), None)
    if entangler is None:
        raise ValueError("native folding requires a target with CZ or CX; ECR-only targets are unsupported")
    for required in ("measure", "reset", "if_else"):
        if required not in target.operation_names:
            raise ValueError(f"dynamic Shor requires target instruction {required!r}")
    restricted = Target(
        description="CX/CZ folding-compatible view of supplied target",
        num_qubits=target.num_qubits, dt=target.dt,
        granularity=target.granularity, min_length=target.min_length,
        pulse_alignment=target.pulse_alignment, acquire_alignment=target.acquire_alignment,
        qubit_properties=target.qubit_properties,
        concurrent_measurements=target.concurrent_measurements,
    )
    for name in target.operation_names:
        op = target.operation_from_name(name)
        if name == "if_else":
            restricted.add_instruction(op, name=name)
        elif (
            name == entangler
            or name in {"reset", "measure", "measure_2", "delay"}
            or isinstance(op, Gate) and op.num_qubits == 1
        ):
            restricted.add_instruction(op, properties=target[name], name=name)
    return restricted


def _compact_layout(target, count):
    if target.num_qubits is None or count > target.num_qubits:
        raise ValueError(f"logical circuit needs {count} qubits, target has {target.num_qubits}")
    coupling = target.build_coupling_map()
    if coupling is None:
        return list(range(count))
    graph = {q: set() for q in range(target.num_qubits)}
    for left, right in coupling.get_edges():
        graph[left].add(right)
        graph[right].add(left)
    for start in sorted(graph, key=lambda q: (-len(graph[q]), q)):
        queue, seen, layout = deque([start]), {start}, []
        while queue and len(layout) < count:
            node = queue.popleft()
            layout.append(node)
            for neighbor in sorted(graph[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(layout) == count:
            return layout
    raise ValueError("target has no connected component large enough for this circuit")


def validate_native(circuit: QuantumCircuit, target: Target | None = None) -> dict:
    """Validate final native operations and, when supplied, actual target sites.

    Feedforward blocks must contain only one-qubit gates.  Two-qubit gates in
    branches would make a static gate-count noise multiplier path-dependent,
    so this implementation refuses them rather than reporting a false scale.
    """
    if target is not None and (target.num_qubits is None or circuit.num_qubits > target.num_qubits):
        raise ValueError("compiled circuit width exceeds target qubit capacity")
    checked, two_qubit = 0, Counter()

    def visit(block, physical, inside_block=False):
        nonlocal checked
        for instruction in block.data:
            op = instruction.operation
            qubits = tuple(physical[block.find_bit(q).index] for q in instruction.qubits)
            checked += 1
            children = getattr(op, "blocks", ())
            if children:
                if op.name != "if_else":
                    raise ValueError(f"unsupported native control flow {op.name!r}; expected if_else")
            elif op.name == "barrier":
                continue
            elif isinstance(op, Gate):
                if op.num_qubits == 2:
                    if inside_block:
                        raise ValueError("two-qubit gates inside control-flow blocks cannot have a fixed executed scale")
                    if op.name not in {"cz", "cx"}:
                        raise ValueError(f"unsupported native two-qubit gate {op.name!r}; compile to CZ or CX first")
                    if op.params or op.inverse() != op:
                        raise ValueError("folding requires exact self-inverse native CX/CZ gates")
                    two_qubit[op.name] += 1
                elif op.num_qubits != 1:
                    raise ValueError(f"uncompiled {op.num_qubits}-qubit gate {op.name!r}")
            elif op.name not in {"reset", "measure", "measure_2", "delay"}:
                raise ValueError(f"uncompiled or unsupported native operation {op.name!r}")
            if inside_block and not children and not isinstance(op, Gate):
                raise ValueError("feedforward blocks may contain only one-qubit unitary gates")
            if target is not None and not target.instruction_supported(
                operation_name=op.name, qargs=qubits, parameters=op.params,
            ):
                raise ValueError(f"target does not support {op.name!r} at physical qubits {qubits}")
            for child in children:
                visit(child, qubits, True)

    visit(circuit, tuple(range(circuit.num_qubits)))
    return {
        "passed": True, "recursively_checked_instructions": checked,
        "native_two_qubit_counts": dict(sorted(two_qubit.items())),
        "target_qargs_checked": target is not None,
    }


def compile_pair(N, a, backend, seed=42, optimization_level=1):
    """Return ``(raw, reset, metadata)`` without executing any backend job.

    A backend object with a Qiskit Target (or the Target itself) is required.
    Only Reset is routed.  Raw is its exact instruction sequence minus marked
    workspace resets, so the two arms have matching physical unitary gates.
    The caller is responsible for any live backend discovery/authorization.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(optimization_level, bool)
        or not isinstance(optimization_level, Integral)
        or optimization_level not in (0, 1, 2, 3)
    ):
        raise ValueError("optimization_level must be one of 0, 1, 2, 3")
    source = build_dynamic_shor_circuit(N, a, workspace_reset=True)
    target = backend if isinstance(backend, Target) else getattr(backend, "target", None)
    if not isinstance(target, Target):
        raise TypeError("backend must expose a Qiskit Target; no backend is contacted automatically")
    compile_target = _restricted_target(target)
    layout = _compact_layout(compile_target, source.num_qubits)
    reset = transpile(
        source, target=compile_target, initial_layout=layout,
        routing_method="sabre", seed_transpiler=int(seed),
        optimization_level=int(optimization_level), approximation_degree=1.0,
        qubits_initially_zero=False, num_processes=1,
    )
    source_markers = [i.operation.label for i in source.data if is_workspace_reset(i)]
    native_markers = [i.operation.label for i in reset.data if is_workspace_reset(i)]
    if sorted(source_markers) != sorted(native_markers):
        raise ValueError("transpilation changed or lost marked workspace resets")
    source_counts = Counter(i.operation.name for i, _ in walk_instructions(source))
    native_counts = Counter(i.operation.name for i, _ in walk_instructions(reset))
    for name in ("measure", "reset", "if_else"):
        if source_counts[name] != native_counts[name]:
            raise ValueError(f"transpilation changed required dynamic instruction count for {name!r}")
    if [(r.name, r.size) for r in reset.cregs] != [("phase", source.num_clbits)]:
        raise ValueError("transpilation changed the measured phase register")
    measured_phase = [reset.find_bit(i.clbits[0]).index for i in reset.data if i.operation.name == "measure"]
    if measured_phase != list(range(source.num_clbits)):
        raise ValueError("transpilation changed round-to-phase-bit measurement order")
    reset.name = f"shor_N{int(N)}_a{int(a)}_reset_native"
    reset.metadata = dict(source.metadata)
    for role in ("control_qubit", "persistent_value_qubits", "workspace_qubits"):
        reset.metadata[f"source_{role}"] = reset.metadata.pop(role)
    reset.metadata.update({
        "native_qubit_index_space": "physical_target",
        "source_role_index_space": "logical_source",
        "source_role_indexing_notice": "source_* indices and reset labels are logical; actual routed sites are in native_* maps",
        "native_pairing_policy": "route Reset once; Raw removes only labeled workspace resets",
    })
    reset.metadata.update(native_instruction_maps(reset))
    raw = strip_workspace_resets(reset)
    raw.name = f"shor_N{int(N)}_a{int(a)}_raw_native"
    validation = validate_native(reset, target)
    validate_native(raw, target)
    metadata = {
        "N": int(N), "a": int(a), "seed_transpiler": int(seed),
        "optimization_level": int(optimization_level),
        "logical_qubits": source.num_qubits, "phase_bits": source.num_clbits,
        "initial_physical_layout_logical_order": layout,
        "final_logical_to_physical_layout": reset.layout.final_index_layout(),
        "compile_basis": sorted(compile_target.operation_names),
        "pairing_policy": reset.metadata["native_pairing_policy"],
        "same_physical_unitary_routing_both_arms": True,
        "workspace_reset_labels_preserved_exactly": True,
        "native_validation": validation,
        "native_raw_instruction_counts": dict(sorted(Counter(i.operation.name for i, _ in walk_instructions(raw)).items())),
        "native_reset_instruction_counts": dict(sorted(native_counts.items())),
        "known_order_used_for_synthesis": False,
        "backend_jobs_submitted": 0,
    }
    return raw, reset, metadata


def fold_native(circuit: QuantumCircuit, scale: float, seed: int):
    """Return ``(folded, effective_scale, metadata)`` for final CX/CZ circuits.

    Each original gate receives an integer number of exact local G G† G
    folds.  The total number of inserted inverse/forward pairs is the nearest
    integer to ``(scale-1)*M/2`` (ties to even).  Excess folds use a seeded,
    uniformly chosen subset of original gates.  This is one fixed pattern for
    every shot of the circuit, NOT per-shot random folding.  Arbitrary finite
    scales >=1 are supported.  No transpilation/optimization is performed.
    """
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale < 1:
        raise ValueError("scale must be finite and at least 1")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scale, seed = float(scale), int(seed)
    validation = validate_native(circuit)
    original_count = sum(validation["native_two_qubit_counts"].values())
    if original_count == 0:
        raise ValueError("cannot define a two-qubit amplification scale without native CX/CZ gates")
    total_folds = round((scale - 1) * original_count / 2)
    uniform_folds, extra_count = divmod(total_folds, original_count)
    selected = set(random.Random(seed).sample(range(original_count), extra_count))
    folded = circuit.copy_empty_like(name=f"{circuit.name}_lambda{scale:.12g}")
    ordinal = 0
    for instruction in circuit.data:
        folded.append(instruction)
        op = instruction.operation
        # Control-flow instructions may span two qubits but are not gates.
        if isinstance(op, Gate) and op.num_qubits == 2:
            for _ in range(uniform_folds + (ordinal in selected)):
                folded.append(op.inverse(), instruction.qubits)
                folded.append(instruction)
            ordinal += 1
    final_validation = validate_native(folded)
    actual_count = sum(final_validation["native_two_qubit_counts"].values())
    if actual_count != original_count + 2 * total_folds:
        raise ValueError("actual folded gate count does not match the exact local folding rule")
    effective = actual_count / original_count
    info = {
        "nominal_scale": scale, "effective_scale": effective,
        "original_two_qubit_count": original_count,
        "added_two_qubit_count": 2 * total_folds,
        "final_two_qubit_count": actual_count,
        "original_two_qubit_counts": validation["native_two_qubit_counts"],
        "final_two_qubit_counts": final_validation["native_two_qubit_counts"],
        "uniform_fold_pairs_per_original_gate": uniform_folds,
        "extra_fold_original_gate_count": extra_count,
        "extra_fold_original_gate_ordinals_sha256": hashlib.sha256(
            json.dumps(sorted(selected), separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "pattern_seed": seed,
        "pattern_scope": "one fixed gate subset per materialized circuit, shared by all its shots",
        "per_shot_randomization": False,
        "fold_identity": "native G -> G (G inverse G)^k for self-inverse CX/CZ",
        "post_fold_optimization": False,
        "do_not_transpile_after_folding": True,
    }
    folded.metadata = dict(circuit.metadata or {})
    folded.metadata["native_fold"] = info
    if folded.metadata.get("native_qubit_index_space") == "physical_target":
        folded.metadata.update(native_instruction_maps(folded))
    return folded, effective, info
