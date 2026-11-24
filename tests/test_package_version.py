"""Offline checks for package metadata and the unchanged original initializer."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import shor_qem


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_INITIALIZER = "archive/original_package/src/shor_qem/__init__.py"
ORIGINAL_SHA256 = "54380146a0b0a17dc1ca31de525300dad9f41dd6244e059f7eb99dcdc53d5e57"


def test_package_version_matches_project_metadata():
    # A simple metadata read also works on Python 3.10, without a TOML dependency.
    version = re.search(r'^version\s*=\s*"([^"]+)"\s*$',
                        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    assert version is not None
    assert shor_qem.__version__ == version.group(1) == "0.2.1"


def test_original_initializer_identity_and_bytes_are_preserved():
    with (ROOT / "SOURCE_MANIFEST.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    original = [row for row in rows if row["source_id"] == "original_package/src/shor_qem/__init__.py"]
    assert len(original) == 1
    assert original[0]["path"] == ORIGINAL_INITIALIZER
    assert original[0]["role"] == "unchanged_scientific_module"
    assert original[0]["sha256"] == ORIGINAL_SHA256
    content = (ROOT / ORIGINAL_INITIALIZER).read_bytes()
    assert len(content) == int(original[0]["bytes"]) == 79
    assert hashlib.sha256(content).hexdigest() == ORIGINAL_SHA256
    assert not any(row["path"] == "src/shor_qem/__init__.py" for row in rows)


def test_release_builder_version_and_scope_without_changing_real_manifest(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("release_manifest_version_test",
                                                 ROOT / "scripts/build_release_manifest.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    template = {"schema_version": 1, "source_snapshot_copies": 103, "files": []}
    destination = tmp_path / "RELEASE_MANIFEST.json"
    destination.write_text(json.dumps(template), encoding="utf-8")
    (tmp_path / "README.md").write_text("Offline fixture\n", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: b"README.md\0RELEASE_MANIFEST.json\0")
    builder.main()
    manifest = json.loads(destination.read_text(encoding="utf-8"))
    assert manifest["version"] == shor_qem.__version__
    assert manifest["source_snapshot_copies"] == 103
    assert manifest["payload_file_count"] == 1
    assert manifest["files"][0]["path"] == "README.md"
    assert "seven unchanged simulation archives from v0.1.2" in manifest["scope"]
    assert "two unchanged hardware archives from v0.2.0" in manifest["scope"]
