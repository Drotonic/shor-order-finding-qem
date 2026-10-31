import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from shor_qem.followup_cluster import (
    MODE_RAW,
    MODE_RESET,
    WorkerCapacity,
    _run_atomic_task,
    build_followup_tasks,
    create_plan,
    distribute_tasks,
    load_experiment,
    validate_task_archive,
)


class FollowupClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.experiment_path = (
            cls.project_root / "configs" / "n119_a100_followup.json"
        )
        cls.experiment = load_experiment(cls.experiment_path)
        cls.tasks = build_followup_tasks(cls.experiment)

    def test_production_plan_has_the_expected_554_atomic_tasks(self):
        self.assertEqual(len(self.tasks), 554)
        self.assertEqual(len({task["task_id"] for task in self.tasks}), 554)
        self.assertEqual(
            Counter(task["purpose"] for task in self.tasks),
            {
                "existing_rate_lambda_one_extension": 462,
                "additional_rate_lambda_one": 64,
                "additional_rate_zne_scale": 28,
            },
        )
        self.assertEqual(
            Counter(task["instance"] for task in self.tasks),
            {"n119_a13": 154, "n119_a50": 154, "n119_a16": 246},
        )

    def test_existing_baselines_start_after_the_saved_twenty_thousand(self):
        selected = [
            task
            for task in self.tasks
            if task["instance"] == "n119_a13"
            and task["mode"] == MODE_RAW
            and task["rate"] == 0.00001
        ]
        self.assertEqual(
            [task["shot_offset"] for task in selected],
            [20000, 40000, 60000, 80000, 100000, 120000, 140000],
        )
        self.assertEqual({task["shots"] for task in selected}, {20000})

    def test_new_a16_rates_include_full_baselines_and_all_zne_scales(self):
        rate = 0.00012
        reset = [
            task
            for task in self.tasks
            if task["instance"] == "n119_a16"
            and task["mode"] == MODE_RESET
            and task["rate"] == rate
        ]
        raw = [
            task
            for task in self.tasks
            if task["instance"] == "n119_a16"
            and task["mode"] == MODE_RAW
            and task["rate"] == rate
        ]
        self.assertEqual(
            sorted({task["lambda"] for task in reset}),
            [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0],
        )
        lambda_one = [task for task in reset if task["lambda"] == 1.0]
        self.assertEqual(
            [task["shot_offset"] for task in lambda_one],
            [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000],
        )
        self.assertEqual(
            [task["shot_offset"] for task in raw],
            [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000],
        )
        nontrivial = [task for task in reset if task["lambda"] != 1.0]
        self.assertEqual(len(nontrivial), 7)
        self.assertEqual({task["shot_offset"] for task in nontrivial}, {0})
        self.assertEqual({task["shots"] for task in nontrivial}, {20000})

    def test_four_two_one_gpu_distribution_is_proportional_and_balanced(self):
        workers = [
            WorkerCapacity("a100x4", 4),
            WorkerCapacity("a100x2", 2),
            WorkerCapacity("a100x1", 1),
        ]
        shards, slots = distribute_tasks(self.tasks, workers)
        self.assertEqual(
            {name: len(tasks) for name, tasks in shards.items()},
            {"a100x4": 317, "a100x2": 158, "a100x1": 79},
        )
        loads = [float(slot["estimated_work"]) for slot in slots]
        self.assertLessEqual(max(loads) - min(loads), 40000.0)
        assigned = [task["task_id"] for tasks in shards.values() for task in tasks]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {task["task_id"] for task in self.tasks})

    def test_plan_files_preserve_the_global_task_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            result = create_plan(
                experiment_path=self.experiment_path,
                output_dir=Path(directory),
                workers=[
                    WorkerCapacity("a100x4", 4),
                    WorkerCapacity("a100x2", 2),
                    WorkerCapacity("a100x1", 1),
                ],
            )
            self.assertEqual(result["task_count"], 554)
            self.assertTrue(Path(result["global_manifest"]).is_file())
            self.assertEqual(set(result["shards"]), {"a100x4", "a100x2", "a100x1"})

    def test_atomic_runner_forwards_global_shot_offset_to_both_engines(self):
        class FakeCore:
            OPS_CACHE_DIR = ""
            GLOBAL_SEED = -1

            def __init__(self):
                self.calls = []

            def load_compiled_ops_cache(self, path):
                return {1: []}

            def smart_mem_gc(self, **kwargs):
                return None

            def _result(self, name, args, kwargs):
                self.calls.append((name, args, kwargs))
                shots = int(args[5])
                return [(0, np.zeros((shots, 14), dtype=np.int32))]

            def _run_gpu_engine_chunked_shotwise(self, *args, **kwargs):
                return self._result("reset", args, kwargs)

            def _run_gpu_engine_chunked(self, *args, **kwargs):
                return self._result("raw", args, kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            core = FakeCore()
            for mode in (MODE_RESET, MODE_RAW):
                template = next(
                    task for task in self.tasks if task["mode"] == mode
                )
                task = dict(template)
                task["task_id"] = f"unit-{mode}"
                task["shots"] = 3
                task["shot_offset"] = 23456
                task["output_relpath"] = f"{mode}.npz"
                path = _run_atomic_task(task, output, core, {})
                validate_task_archive(path, task)
                call_name, _, kwargs = core.calls[-1]
                self.assertEqual(call_name, mode)
                self.assertEqual(kwargs["shot_offset"], 23456)
                self.assertEqual(kwargs["reset_ancilla"], mode == MODE_RESET)

    def test_atomic_runner_subchunks_shots_without_reusing_global_ids(self):
        class FakeCore:
            OPS_CACHE_DIR = ""
            GLOBAL_SEED = -1

            def __init__(self):
                self.calls = []

            def load_compiled_ops_cache(self, path):
                return {1: []}

            def smart_mem_gc(self, **kwargs):
                return None

            def _result(self, name, args, kwargs):
                shots = int(args[5])
                self.calls.append(
                    (name, shots, int(kwargs["shot_offset"]))
                )
                return [(0, np.zeros((shots, 14), dtype=np.int32))]

            def _run_gpu_engine_chunked_shotwise(self, *args, **kwargs):
                return self._result("reset", args, kwargs)

            def _run_gpu_engine_chunked(self, *args, **kwargs):
                return self._result("raw", args, kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for mode in (MODE_RESET, MODE_RAW):
                core = FakeCore()
                template = next(task for task in self.tasks if task["mode"] == mode)
                task = dict(template)
                task["task_id"] = f"unit-subchunk-{mode}"
                task["shots"] = 13
                task["shot_offset"] = 23456
                task["output_relpath"] = f"subchunk-{mode}.npz"
                path = _run_atomic_task(
                    task,
                    output,
                    core,
                    {},
                    shot_subchunk_size=5,
                )
                validate_task_archive(path, task)
                self.assertEqual(
                    core.calls,
                    [
                        (mode, 5, 23456),
                        (mode, 5, 23461),
                        (mode, 3, 23466),
                    ],
                )
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(int(archive["engine_shot_subchunk_size"]), 5)
                    self.assertEqual(int(archive["engine_shot_subchunk_count"]), 3)


if __name__ == "__main__":
    unittest.main()
