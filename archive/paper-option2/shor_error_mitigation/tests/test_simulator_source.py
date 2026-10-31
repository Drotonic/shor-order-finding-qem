import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _function_ast(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"function {name!r} not found")


class SimulatorExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = (
            PROJECT_ROOT
            / "data/raw/source_snapshot/original_experiment_notebook.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.original = "".join(notebook["cells"][17]["source"])
        cls.extracted = (
            PROJECT_ROOT / "src/shor_qem/simulator.py"
        ).read_text(encoding="utf-8")

    def test_numerical_and_seed_functions_match_notebook(self):
        # These functions include the complete shot-seed formula and the GPU
        # evolution/folding loop.  The CLI and worker setup intentionally differ.
        preserved = (
            "_splitmix64_py",
            "make_gate_key",
            "build_seed_shot_array",
            "make_draw_key",
            "_run_gpu_engine_chunked",
            "run_lambda_part",
            "merge_parts_for_rate_method",
        )
        for name in preserved:
            with self.subTest(function=name):
                self.assertEqual(
                    _function_ast(self.extracted, name),
                    _function_ast(self.original, name),
                )

    def test_clean_driver_has_no_automatic_purge(self):
        self.assertNotIn("purge_shor_results_for_key", self.extracted)
        self.assertIn('parser.add_argument(\n        "--overwrite"', self.extracted)


if __name__ == "__main__":
    unittest.main()
