"""Release boundaries, package wiring, and user-facing command behavior."""

import ast
import builtins
import contextlib
import io
import json
import subprocess
import symtable
import sys
import tempfile
import tomllib
import types
from pathlib import Path
from unittest import TestCase, mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
PACKAGE = SOURCE / "llm_pruning"
sys.path.insert(0, str(SOURCE))

from llm_pruning.advanced_cli import parse_args as parse_research_args
from llm_pruning.cli import build_research_args, main, parse_args
from llm_pruning.config import DEFAULT_BO_EVAL_PROMPT_FILES


class CommandTests(TestCase):
    def test_help_without_ml_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)
        self.assertNotIn("--third_stage", result.stdout)

    def test_profiles_translate_to_valid_research_arguments(self):
        for profile, population, trials in [("quick", 4, 4), ("paper", 40, 40)]:
            args = parse_args(
                ["--model", "test/model", "--profile", profile, "--gpus", "0, 1"]
            )
            settings = parse_research_args(build_research_args(args))
            self.assertEqual(settings.proposed_pop_size, population)
            self.assertEqual(settings.second_stage_bayes_trials, trials)
            self.assertTrue(settings.second_stage_bayes)
            self.assertTrue(settings.second_stage_heads and settings.second_stage_mlp)
            self.assertFalse(settings.save_selected_stage2_model)
            self.assertEqual(settings.second_stage_bayes_candidate_parallel_gpus, "0,1")

    def test_optional_settings_are_forwarded(self):
        args = parse_args(
            [
                "--model",
                "test/model",
                "--calibration",
                "mixed",
                "--importance",
                "taylor",
                "--objective",
                "params",
                "--evaluate",
                "--save-model",
                "--stage1-only",
                "--fresh",
            ]
        )
        settings = parse_research_args(build_research_args(args))
        self.assertEqual(settings.proposed_first_objective, "kl_mix")
        self.assertEqual(settings.second_stage_prompt_set, "calibration_mix")
        self.assertEqual(settings.second_stage_method, "taylor")
        self.assertEqual(settings.proposed_second_objective, "params")
        self.assertTrue(
            settings.evaluate_downstream and settings.evaluate_perplexity_all
        )
        self.assertTrue(settings.save_selected_stage2_model)
        self.assertTrue(settings.stage1_only and settings.force_fresh_run)

    def test_invalid_sparsity_and_gpu_inputs(self):
        for extra in [
            ["--sparsity", "0"],
            ["--sparsity", "1"],
            ["--sparsity", "nan"],
            ["--gpus", "0,0"],
            ["--gpus", "0,"],
            ["--gpus", "-1"],
        ]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--model", "test/model", *extra])

    def test_dry_run_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new_run"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                main(
                    ["--model", "test/model", "--dry-run", "--output-dir", str(output)]
                )
            self.assertIn("llm_pruning.advanced", printed.getvalue())
            self.assertFalse(output.exists())

    def test_execution_isolates_outputs_and_restores_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            fake_pipeline = types.ModuleType("llm_pruning.pipeline")
            received = []

            def execute(settings):
                received.append((Path.cwd(), settings))

            fake_pipeline.main = execute
            with mock.patch.dict(sys.modules, {"llm_pruning.pipeline": fake_pipeline}):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["--model", "test/model", "--output-dir", str(output)])
            self.assertEqual(Path.cwd(), original)
            self.assertEqual(received[0][0], output.resolve())
            self.assertEqual(
                Path(received[0][1].pipeline_checkpoint_path).parent, output.resolve()
            )

    def test_execution_restores_working_directory_on_failure(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            fake_pipeline = types.ModuleType("llm_pruning.pipeline")
            fake_pipeline.main = mock.Mock(
                side_effect=RuntimeError("simulated GPU failure")
            )
            with mock.patch.dict(sys.modules, {"llm_pruning.pipeline": fake_pipeline}):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(RuntimeError),
                ):
                    main(["--model", "test/model", "--output-dir", directory])
            self.assertEqual(Path.cwd(), original)

    def test_local_model_paths_remain_absolute(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--model", directory])
            self.assertEqual(args.model, str(Path(directory).resolve()))


class PackageTests(TestCase):
    def test_calibration_is_bundled_and_valid(self):
        self.assertEqual(len(DEFAULT_BO_EVAL_PROMPT_FILES), 16)
        for name in DEFAULT_BO_EVAL_PROMPT_FILES:
            path = Path(name)
            self.assertTrue(path.is_relative_to(PACKAGE / "data"))
            self.assertTrue(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(
            len(list((PACKAGE / "data" / "calibration200").glob("*.json"))), 16
        )

    def test_package_metadata_and_entrypoints(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["project"]["scripts"]["llm-prune"], "llm_pruning.cli:main"
        )
        self.assertIn("research", metadata["project"]["optional-dependencies"])
        self.assertIn(
            "data/calibration/*.json",
            metadata["tool"]["setuptools"]["package-data"]["llm_pruning"],
        )

    def test_all_local_imports_resolve(self):
        for path in PACKAGE.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    target = base.joinpath(*(node.module or "").split("."))
                elif node.module and node.module.startswith("llm_pruning"):
                    target = SOURCE.joinpath(*node.module.split("."))
                else:
                    continue
                source = target.with_suffix(".py")
                if not source.exists():
                    source = target / "__init__.py"
                self.assertTrue(source.exists(), f"{path}: missing {node.module}")
                table = symtable.symtable(
                    source.read_text(encoding="utf-8"), str(source), "exec"
                )
                for alias in node.names:
                    self.assertTrue(
                        alias.name in table.get_identifiers()
                        or (target / (alias.name + ".py")).exists(),
                        f"{path}: missing {node.module}.{alias.name}",
                    )

    def test_no_unresolved_global_names(self):
        for path in PACKAGE.rglob("*.py"):
            table = symtable.symtable(
                path.read_text(encoding="utf-8"), str(path), "exec"
            )
            bound = {
                symbol.get_name()
                for symbol in table.get_symbols()
                if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
            }
            allowed = bound | set(vars(builtins)) | {"__file__", "__name__"}
            pending = [table]
            while pending:
                scope = pending.pop()
                for symbol in scope.get_symbols():
                    if symbol.is_global() and symbol.is_referenced():
                        self.assertIn(
                            symbol.get_name(),
                            allowed,
                            f"{path}: unresolved {symbol.get_name()}",
                        )
                pending.extend(scope.get_children())

    def test_training_schema_owns_downstream_columns(self):
        # Execute only the schema declarations, without importing the GPU runtime.
        tree = ast.parse((PACKAGE / "metrics.py").read_text(encoding="utf-8"))
        nodes = [
            n
            for n in tree.body
            if (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id.startswith("TRAINING_LOG_")
                    for t in n.targets
                )
            )
            or (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and isinstance(n.value.func.value, ast.Name)
                and n.value.func.value.id == "TRAINING_LOG_FIELDNAMES"
            )
        ]
        namespace = {}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "schema", "exec"),
            namespace,
        )
        fields = namespace["TRAINING_LOG_FIELDNAMES"]
        self.assertEqual(len(fields), len(set(fields)))
        for task in namespace["TRAINING_LOG_DOWNSTREAM_TASK_COLUMNS"]:
            self.assertIn(f"downstream_{task}_acc", fields)

    def test_no_backup_sources_or_weights(self):
        for path in PACKAGE.rglob("*"):
            if path.is_file():
                self.assertNotIn("backup", path.name.lower())
                self.assertNotIn(path.suffix, {".pt", ".pth", ".safetensors", ".bak"})
