"""Offline CPU provenance, dynamic-circuit pairing and native-fold tests.

Qiskit and Aer are optional; no credentials, network or backend submission is
used.  Small-instance tests do not prove large-instance arithmetic correctness.
"""

import ast
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate, HGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import Target

from shor_qem import hardware_circuits as hardware


def _arithmetic_ast(source):
    helpers = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in hardware.ARITHMETIC_HELPERS
    ]
    tree = ast.Module(body=helpers, type_ignores=[])
    # The frozen digest uses CPython 3.12's dump format.  Python 3.13 began
    # omitting empty fields by default, while 3.10/3.11 lack type_params.
    # Normalize only that schema/display difference, not arithmetic semantics.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "type_params" not in node._fields:
            node._fields = (*node._fields, "type_params")
            node.type_params = []
    options = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
    return ast.dump(tree, include_attributes=False, **options)


def _non_entanglers(circuit):
    return [i for i in circuit.data if i.operation.name not in {"cx", "cz"}]


def test_frozen_arithmetic_matches_public_source_without_importing_it():
    source = Path(hardware.__file__).with_name("simulator.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == hardware.SOURCE_SHA256
    copied = _arithmetic_ast(Path(hardware.__file__).read_text())
    assert copied == _arithmetic_ast(source.decode())
    assert hashlib.sha256(copied.encode()).hexdigest() == hardware.ARITHMETIC_AST_SHA256
    assert "r" not in inspect.signature(hardware.build_dynamic_shor_circuit).parameters
    tree = ast.parse(Path(hardware.__file__).read_text())
    imports = [
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (
            [alias.name for alias in node.names] if isinstance(node, ast.Import)
            else [node.module or ""]
        )
    ]
    assert not any("cupy" in name or "simulator" in name for name in imports)


@pytest.mark.parametrize("N,a", [(3, 2), (15, 2), (21, 8), (25, 7)])
def test_full_template_width_rounds_feedforward_and_workspace_resets(N, a):
    circuit = hardware.build_dynamic_shor_circuit(N, a)
    n, L = N.bit_length(), 2 * N.bit_length()
    assert (circuit.num_qubits, circuit.num_clbits) == (2 * n + 3, L)
    assert [(r.name, r.size) for r in circuit.cregs] == [("phase", L)]
    assert circuit.metadata["known_order_used_for_synthesis"] is False
    assert circuit.count_ops()["if_else"] == L * (L - 1) // 2
    assert circuit.count_ops()["measure"] == L
    assert circuit.count_ops()["reset"] == L * (n + 3)
    arithmetic = [i.operation for i in circuit.data if i.operation.name.startswith("c_Ua_")]
    assert len(arithmetic) == L
    assert [op.name for op in arithmetic] == [
        f"c_Ua_N{N}_a{pow(a, 2 ** (L - 1 - i), N)}" for i in range(L)
    ]
    # Includes multiplication, n controlled swaps and unmultiplication, even
    # when a repeated modular-power multiplier happens to equal one.
    assert all(len(op.definition.data) == n + 2 for op in arithmetic)
    measured = [i for i in circuit.data if i.operation.name == "measure"]
    assert [circuit.find_bit(i.clbits[0]).index for i in measured] == list(range(L))
    assert all([circuit.find_bit(q).index for q in i.qubits] == [0] for i in measured)
    markers = [i.operation.label for i in circuit.data if hardware.is_workspace_reset(i)]
    assert markers == [f"workspace_reset:{i}:{q}" for i in range(L) for q in range(n + 1, 2 * n + 3)]


def test_stripping_only_workspace_resets_matches_independently_built_raw():
    raw = hardware.build_dynamic_shor_circuit(3, 2, workspace_reset=False)
    reset = hardware.build_dynamic_shor_circuit(3, 2)
    stripped = hardware.strip_workspace_resets(reset)
    assert raw.decompose(reps=20) == stripped.decompose(reps=20)
    assert stripped.count_ops()["reset"] == raw.num_clbits
    assert stripped.metadata["workspace_reset"] is False
    assert reset.metadata["workspace_reset"] is True


@pytest.mark.parametrize("N,a", [(True, 2), (2, 1), (15.0, 4), (15, True), (15, 1), (15, 15), (15, 3)])
def test_builder_rejects_invalid_modular_instances(N, a):
    with pytest.raises(ValueError):
        hardware.build_dynamic_shor_circuit(N, a)


def test_builder_rejects_non_boolean_reset_policy():
    with pytest.raises(TypeError):
        hardware.build_dynamic_shor_circuit(15, 2, workspace_reset=1)


def test_small_clean_workspace_arithmetic_preserves_relative_phase():
    unitary = hardware.c_Ua(3, 2, 3)
    dimension = 2 ** unitary.num_qubits
    for value in range(3):
        initial = np.zeros(dimension, complex)
        initial[value << 1] = 1 / np.sqrt(2)
        initial[(value << 1) | 1] = 1j / np.sqrt(2)
        expected = np.zeros(dimension, complex)
        expected[value << 1] = 1 / np.sqrt(2)
        expected[(((2 * value) % 3) << 1) | 1] = 1j / np.sqrt(2)
        np.testing.assert_allclose(Statevector(initial).evolve(unitary).data, expected, atol=2e-12)


@pytest.fixture(scope="module")
def compiled_pair():
    edges = [(i, i + 1) for i in range(6)] + [(i + 1, i) for i in range(6)]
    backend = GenericBackendV2(
        7, basis_gates=["rz", "sx", "x", "cx"], coupling_map=edges,
        control_flow=True, seed=15,
    )
    with patch.object(hardware, "transpile", wraps=hardware.transpile) as compile_mock:
        raw, reset, info = hardware.compile_pair(3, 2, backend, seed=42)
        assert compile_mock.call_count == 1
    return raw, reset, info, backend


def test_native_pair_is_exactly_matched_and_target_sites_are_valid(compiled_pair):
    raw, reset, info, backend = compiled_pair
    assert raw == hardware.strip_workspace_resets(reset)
    assert raw.layout == reset.layout
    assert raw.count_ops()["cx"] == reset.count_ops()["cx"]
    assert raw.count_ops()["reset"] == 4
    assert reset.count_ops()["reset"] == 20
    assert info["native_validation"]["target_qargs_checked"] is True
    assert hardware.validate_native(reset, backend.target)["passed"]
    assert raw.metadata["native_workspace_reset_sites"] == []
    assert len(reset.metadata["native_workspace_reset_sites"]) == 16
    assert "workspace_qubits" not in reset.metadata
    assert reset.metadata["source_workspace_qubits"] == [3, 4, 5, 6]
    assert info["backend_jobs_submitted"] == 0
    json.dumps(info)


def test_compile_rejects_insufficient_width_and_non_dynamic_target():
    small = GenericBackendV2(5, basis_gates=["rz", "sx", "x", "cx"], control_flow=True)
    with pytest.raises(ValueError, match="needs 7 qubits"):
        hardware.compile_pair(3, 2, small)
    non_dynamic = GenericBackendV2(7, basis_gates=["rz", "sx", "x", "cx"])
    with pytest.raises(ValueError, match="if_else"):
        hardware.compile_pair(3, 2, non_dynamic)


def test_target_validation_checks_actual_physical_qubits_and_basis():
    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(1, 2): None})
    target.add_instruction(HGate(), {(0,): None, (1,): None, (2,): None})
    valid = QuantumCircuit(3)
    valid.h(0)
    valid.cx(1, 2)
    assert hardware.validate_native(valid, target)["passed"]
    wrong_edge = QuantumCircuit(3)
    wrong_edge.cx(0, 1)
    with pytest.raises(ValueError, match="physical qubits"):
        hardware.validate_native(wrong_edge, target)
    with pytest.raises(ValueError, match="capacity"):
        hardware.validate_native(QuantumCircuit(4), target)
    unsupported = QuantumCircuit(2)
    unsupported.cz(0, 1)
    with pytest.raises(ValueError, match="does not support"):
        hardware.validate_native(unsupported, target)


@pytest.mark.parametrize("scale", [1, 1.4, 2.2, 3, 4.4, 5])
def test_exact_unitary_folds_scale_quantization_and_seeded_pattern(scale):
    circuit = QuantumCircuit(3, global_phase=0.37)
    for i in range(10):
        circuit.h(i % 3)
        circuit.cx(i % 3, (i + 1) % 3) if i % 2 else circuit.cz(i % 3, (i + 1) % 3)
    folded, effective, info = hardware.fold_native(circuit, scale, 42)
    repeated, _, repeated_info = hardware.fold_native(circuit, scale, 42)
    assert folded == repeated
    assert info == repeated_info
    assert folded.global_phase == circuit.global_phase
    assert _non_entanglers(folded) == _non_entanglers(circuit)
    assert info["final_two_qubit_count"] == 10 + 2 * round((scale - 1) * 10 / 2)
    assert effective == info["final_two_qubit_count"] / 10
    assert info["per_shot_randomization"] is False
    np.testing.assert_allclose(Operator(folded).data, Operator(circuit).data, atol=2e-12)
    json.dumps(info)


def test_folds_preserve_dynamic_order_phase_register_and_refresh_maps(compiled_pair):
    _, reset, _, backend = compiled_pair
    folded, effective, info = hardware.fold_native(reset, 1.4, 42)
    assert _non_entanglers(folded) == _non_entanglers(reset)
    assert folded.cregs == reset.cregs
    assert folded.clbits == reset.clbits
    assert folded.layout == reset.layout
    assert effective == info["final_two_qubit_count"] / info["original_two_qubit_count"]
    assert hardware.validate_native(folded, backend.target)["passed"]
    assert folded.metadata["native_control_measurements"] == hardware.native_instruction_maps(folded)["native_control_measurements"]
    assert reset.metadata["native_control_measurements"] != folded.metadata["native_control_measurements"]
    _, _, other = hardware.fold_native(reset, 1.4, 43)
    assert info["extra_fold_original_gate_ordinals_sha256"] != other["extra_fold_original_gate_ordinals_sha256"]


@pytest.mark.parametrize("scale", [0.9, float("nan"), float("inf"), True])
def test_fold_rejects_invalid_scales(scale):
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    with pytest.raises(ValueError, match="scale"):
        hardware.fold_native(circuit, scale, 42)


def test_fold_rejects_uncompiled_gates_and_branch_dependent_two_qubit_counts():
    with pytest.raises(ValueError, match="without native"):
        hardware.fold_native(QuantumCircuit(2), 1.4, 42)
    unsupported = QuantumCircuit(2)
    unsupported.cp(0.3, 0, 1)
    with pytest.raises(ValueError, match="CZ or CX"):
        hardware.fold_native(unsupported, 1.4, 42)
    uncompiled = QuantumCircuit(3)
    uncompiled.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="uncompiled"):
        hardware.fold_native(uncompiled, 1.4, 42)
    branch = QuantumCircuit(2, 1)
    branch.cx(0, 1)
    with branch.if_test((branch.clbits[0], 1)):
        branch.cx(0, 1)
    with pytest.raises(ValueError, match="inside control-flow"):
        hardware.fold_native(branch, 1.4, 42)


def test_ideal_small_dynamic_raw_reset_and_folded_on_cpu():
    aer = pytest.importorskip("qiskit_aer")
    simulator = aer.AerSimulator(method="statevector", device="CPU", max_parallel_threads=1)
    raw, reset, _ = hardware.compile_pair(3, 2, simulator)
    assert "cz" in reset.count_ops()  # Prefer CZ when the target exposes both.
    folded, _, _ = hardware.fold_native(reset, 1.4, 101)
    result = simulator.run([raw, reset, folded], shots=256, seed_simulator=31031).result()
    for counts in result.get_counts():
        assert set(counts) == {"0000", "1000"}
        assert abs(counts["0000"] / 256 - 0.5) < 0.13


def test_ideal_N15_coherent_arithmetic_and_full_dynamic_distribution_on_cpu():
    aer = pytest.importorskip("qiskit_aer")
    # A coherent clean-workspace input catches a relative phase error between
    # the multiply/identity control branches, not just wrong output values.
    unitary = hardware.c_Ua(5, 2, 15)
    initial = np.zeros(2 ** 11, complex)
    initial[7 << 1] = 1 / np.sqrt(2)
    initial[(7 << 1) | 1] = 1j / np.sqrt(2)
    expected = np.zeros_like(initial)
    expected[7 << 1] = 1 / np.sqrt(2)
    expected[(14 << 1) | 1] = 1j / np.sqrt(2)
    np.testing.assert_allclose(Statevector(initial).evolve(unitary).data, expected, atol=2e-11)

    simulator = aer.AerSimulator(method="statevector", device="CPU", max_parallel_threads=1)
    raw, reset, info = hardware.compile_pair(15, 2, simulator)
    assert (info["logical_qubits"], info["phase_bits"]) == (11, 8)
    folded, _, _ = hardware.fold_native(reset, 1.4, 115)
    result = simulator.run([raw, reset, folded], shots=8, seed_simulator=1502).result()
    # Order-four phase support is an independent expected result, never a
    # synthesis input. Eight shots are a support smoke test, not a precision
    # test of the four probabilities or a full-space unitary proof.
    expected_support = {"00000000", "01000000", "10000000", "11000000"}
    for counts in result.get_counts():
        assert sum(counts.values()) == 8
        assert set(counts) <= expected_support


def test_non_binary_eigenphase_feedforward_and_phase_endianness():
    aer = pytest.importorskip("qiskit_aer")
    simulator = aer.AerSimulator(method="statevector", device="CPU", max_parallel_threads=1)

    def eigenphase_oracle(size, multiplier, N):
        assert N == 7
        oracle = QuantumCircuit(2 * size + 1)
        oracle.p(2 * np.pi * {1: 0, 2: 1 / 3, 4: 2 / 3}[multiplier], 0)
        return oracle

    with patch.object(hardware, "c_Ua", side_effect=eigenphase_oracle):
        _, circuit, _ = hardware.compile_pair(7, 2, simulator)
    shots = 2048
    counts = simulator.run(circuit, shots=shots, seed_simulator=19031).result().get_counts()
    M = 2 ** circuit.num_clbits
    observed = np.zeros(M)
    for bits, count in counts.items():
        observed[int(bits, 2)] = count / shots
    expected = abs(np.exp(2j * np.pi * np.outer(np.arange(M), 1 / 3 - np.arange(M) / M)).mean(axis=0)) ** 2
    assert np.abs(observed - expected).sum() / 2 < 0.06
    assert int(observed.argmax()) == 21
