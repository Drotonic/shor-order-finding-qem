import collections
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shor_qem.folding_pattern_simulator import (
    _bits_to_outcomes,
    _load_operation_map,
    _select_worker_gpu_ids,
    _shared_config_grid,
    build_tasks,
)
from shor_qem.folding import (
    build_shotwise_folding_masks,
    folding_seed_for_shot,
)


class FoldingPatternSimulatorTests(unittest.TestCase):
    def test_shotwise_patterns_are_exact_reproducible_and_rate_independent(self):
        configs = [(0.0004, 1.4), (0.0005, 1.4)]
        first = build_shotwise_folding_masks(
            configs, 20, 0, 3, 50, global_seed=42, N=119, a=13
        )
        second = build_shotwise_folding_masks(
            configs, 20, 0, 3, 50, global_seed=42, N=119, a=13
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[:, :20], first[:, 20:]))
        np.testing.assert_array_equal(first.sum(axis=0), np.full(40, 10))
        self.assertGreater(np.unique(first[:, :20], axis=1).shape[1], 1)

    def test_every_nontrivial_shot_configuration_has_a_distinct_seed(self):
        seeds = [
            folding_seed_for_shot(
                global_seed=42,
                N=119,
                a=a,
                rate=rate,
                lam=lam,
                qpe_round=qpe_round,
                shot_id=shot_id,
            )
            for a in (13, 50, 16)
            for rate in (0.0004, 0.0005)
            for lam in (1.1, 1.4)
            for qpe_round in (0, 3)
            for shot_id in range(20)
        ]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_gpu_selection_tracks_visible_devices_and_pending_work(self):
        self.assertEqual(
            _select_worker_gpu_ids(None, available=1, pending_tasks=12), [0]
        )
        self.assertEqual(
            _select_worker_gpu_ids(None, available=4, pending_tasks=12),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            _select_worker_gpu_ids(None, available=8, pending_tasks=3),
            [0, 1, 2],
        )
        self.assertEqual(
            _select_worker_gpu_ids("1,3", available=4, pending_tasks=12),
            [1, 3],
        )

    def test_bits_to_outcomes_uses_historical_little_endian_order(self):
        bits = np.asarray([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int32)
        np.testing.assert_array_equal(_bits_to_outcomes(bits), [5, 6, 0])

    def test_task_plan_replicates_only_nontrivial_folding_scales(self):
        historical_rates = [
            0.00001,
            0.00002,
            0.00003,
            0.00004,
            0.00005,
            0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "instance": "test",
                "N": 119,
                "a": 13,
                "order": 4,
                "measurement_bits": 14,
                "global_seed": 42,
                "ops_cache_path": "/tmp/not-read.pkl",
            }
            tasks = build_tasks(
                [config],
                rates=historical_rates,
                lambdas=[1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0],
                shots=20000,
                lambda_group_size=2,
                output_dir=Path(directory),
                simulate_shared_lambda_one=False,
                simulate_shared_lambda_three=False,
            )
        self.assertEqual(len(tasks), 3)
        self.assertEqual(
            sorted({lam for task in tasks for lam in task["lambdas"]}),
            [1.1, 1.2, 1.4, 1.8, 2.2, 2.6],
        )
        self.assertEqual({task["shots"] for task in tasks}, {20000})
        self.assertEqual({task["folding_seed"] for task in tasks}, {42})
        self.assertTrue(all(task["rates"] == historical_rates for task in tasks))

    def test_gate_matched_config_forces_fresh_endpoint_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "instance": "matched",
                "N": 119,
                "a": 13,
                "order": 4,
                "measurement_bits": 14,
                "global_seed": 42,
                "ops_cache_path": "/tmp/not-read.pkl",
                "require_all_lambdas_from_shotwise": True,
            }
            tasks = build_tasks(
                [config],
                rates=[0.00001],
                lambdas=[1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0],
                shots=10,
                lambda_group_size=2,
                output_dir=Path(directory),
                simulate_shared_lambda_one=False,
                simulate_shared_lambda_three=False,
            )
        self.assertEqual(len(tasks), 5)
        self.assertEqual(
            sorted(lam for task in tasks for lam in task["lambdas"]),
            [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0],
        )

    def test_operation_cache_loader_rejects_count_only_unitaries(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory) / "bad.pkl"
            good_path = Path(directory) / "good.pkl"
            with bad_path.open("wb") as handle:
                pickle.dump({1: [("unitary", [0], [None])]}, handle)
            with good_path.open("wb") as handle:
                pickle.dump(
                    {1: [("unitary", [0], [np.eye(2, dtype=np.complex64)])]},
                    handle,
                )
            with self.assertRaisesRegex(ValueError, "no executable 2x2 matrix"):
                _load_operation_map(bad_path)
            self.assertEqual(list(_load_operation_map(good_path)), [1])

    def test_matched_caches_have_identical_two_qubit_counts(self):
        project_root = Path(__file__).resolve().parents[1]
        totals = []
        for config_name in (
            "n119_a13_matched.json",
            "n119_a50_matched.json",
            "n119_a16_matched.json",
        ):
            config = json.loads(
                (project_root / "configs" / config_name).read_text(encoding="utf-8")
            )
            operation_map = _load_operation_map(
                project_root / config["ops_cache"]
            )
            per_round = []
            total_counts = collections.Counter()
            for operations in operation_map.values():
                counts = collections.Counter(operation[0] for operation in operations)
                per_round.append(counts["cz"] + counts["rzz"])
                total_counts.update(counts)
            self.assertEqual(set(per_round), {3549})
            self.assertEqual(total_counts["cz"], 19600)
            self.assertEqual(total_counts["rzz"], 30086)
            totals.append(total_counts["cz"] + total_counts["rzz"])
        self.assertEqual(totals, [49686, 49686, 49686])

    def test_n119_a16_has_order_six(self):
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (project_root / "configs" / "n119_a16_matched.json").read_text(
                encoding="utf-8"
            )
        )
        order = int(config["order"])
        self.assertEqual(order, 6)
        self.assertEqual(pow(int(config["a"]), order, int(config["N"])), 1)
        self.assertTrue(
            all(
                pow(int(config["a"]), exponent, int(config["N"])) != 1
                for exponent in range(1, order)
            )
        )

    def test_n21_a2_r6_cache_has_expected_order_and_gate_counts(self):
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (project_root / "configs" / "n21_a2_r6.json").read_text(
                encoding="utf-8"
            )
        )
        order = int(config["order"])
        self.assertEqual(pow(int(config["a"]), order, int(config["N"])), 1)
        self.assertTrue(
            all(
                pow(int(config["a"]), exponent, int(config["N"])) != 1
                for exponent in range(1, order)
            )
        )
        operation_map = _load_operation_map(project_root / config["ops_cache"])
        total_counts = collections.Counter()
        per_round = []
        for operations in operation_map.values():
            counts = collections.Counter(operation[0] for operation in operations)
            per_round.append(counts["cz"] + counts["rzz"])
            total_counts.update(counts)
        self.assertEqual(len(per_round), int(config["measurement_bits"]))
        self.assertEqual(set(per_round), {1675})
        self.assertEqual(total_counts["cz"], 7600)
        self.assertEqual(total_counts["rzz"], 9150)
        self.assertEqual(
            total_counts["cz"] + total_counts["rzz"],
            int(config["two_qubit_gate_count"]),
        )

    def test_n15_a2_cache_has_expected_order_and_gate_counts(self):
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (project_root / "configs" / "n15_a2_local.json").read_text(
                encoding="utf-8"
            )
        )
        order = int(config["order"])
        self.assertEqual(order, 4)
        self.assertEqual(pow(int(config["a"]), order, int(config["N"])), 1)
        self.assertTrue(
            all(
                pow(int(config["a"]), exponent, int(config["N"])) != 1
                for exponent in range(1, order)
            )
        )
        operation_map = _load_operation_map(project_root / config["ops_cache"])
        total_counts = collections.Counter()
        per_round = []
        for operations in operation_map.values():
            counts = collections.Counter(operation[0] for operation in operations)
            per_round.append(counts["cz"] + counts["rzz"])
            total_counts.update(counts)
        self.assertEqual(len(per_round), int(config["measurement_bits"]))
        self.assertEqual(set(per_round), {1044})
        self.assertEqual(total_counts["cz"], 4096)
        self.assertEqual(total_counts["rzz"], 4256)
        self.assertEqual(
            total_counts["cz"] + total_counts["rzz"],
            int(config["two_qubit_gate_count"]),
        )

    def test_common_config_grid_uses_all_eleven_historical_rates(self):
        rates = [
            0.00001,
            0.00002,
            0.00003,
            0.00004,
            0.00005,
            0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
        ]
        configs = [{"rates": rates}, {"rates": list(rates)}]
        self.assertEqual(_shared_config_grid(configs, "rates", "--rates"), rates)
        with self.assertRaises(ValueError):
            _shared_config_grid(
                [{"rates": rates}, {"rates": rates[:-1]}],
                "rates",
                "--rates",
            )


if __name__ == "__main__":
    unittest.main()
