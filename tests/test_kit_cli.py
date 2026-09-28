"""Store-facing commands operate on inert bytes until a campaign is approved."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from bindocracy.cli import app


@pytest.fixture
def kit(tmp_path):
    root = tmp_path / "authored"
    root.mkdir()
    (root / "optimizer.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('executed').touch()\n"
        "raise RuntimeError('metadata commands must never execute this code')\n"
    )
    (root / "data.txt").write_text("a bundled supporting file\n")
    (root / "kit.yaml").write_text(yaml.safe_dump({
        "kit_schema": 1,
        "name": "contract-example",
        "version": "1.0.0",
        "kind": "optimizer",
        "summary": "Synthetic transport fixture, not a scientific optimizer.",
        "entrypoint": "optimizer.py",
        "python": ">=3.12",
        "requires": [],
        "inputs": ["sequence"],
        "loss_models": [],
        "capabilities": {"changes_length": False, "max_children": 2},
        "metrics": {"fixture_value": {"direction": "none"}},
        "cost": {"gpu": "none", "compile_seconds": 0, "per_parent_seconds": 0},
        "args": {"label": "default", "enabled": False},
        "dependency_args": {},
    }))
    return root


def invoke(*args):
    return CliRunner().invoke(app, ["--json", *map(str, args)])


def result(response):
    assert response.exit_code == 0, response.output
    assert not response.stderr
    envelope = json.loads(response.stdout)
    assert envelope["ok"] is True
    return envelope["result"]


def request(tmp_path):
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "tool": "optimize",
        "name": "fixture",
        "design_set": str(tmp_path / "set.json"),
        "sharding": {"jobs": 1},
        "resources": {"gpus": 1, "cpus": 1, "memory_gb": 1, "walltime": "00:01:00"},
    }))
    return path


def test_inspection_and_dependency_check_never_execute_or_modify_kit(kit):
    before = {p: p.read_bytes() for p in kit.rglob("*") if p.is_file()}
    inspected = result(invoke("kit", "inspect", kit))
    check_response = invoke("kit", "check", kit)
    assert check_response.exit_code == 1
    checked = json.loads(check_response.stdout)["result"]
    assert checked["status"] == "unknown"
    assert checked["execution_authorized"] is False
    schema = result(invoke("config", "schema", "--kind", "kit"))
    Draft202012Validator(schema).validate(inspected["manifest"])
    assert inspected["execution_authorized"] is False
    assert checked["runtime_qualified"] is False
    assert checked["bundle_digest"] == inspected["digest"]
    assert {p: p.read_bytes() for p in kit.rglob("*") if p.is_file()} == before


def test_published_bundle_resolves_offline_without_repeating_contract(kit, tmp_path):
    digest = result(invoke("kit", "inspect", kit))["digest"]
    store = tmp_path / "store"
    cache = tmp_path / "cache"
    published = result(invoke("kit", "publish", kit, "--store", store, "--publisher", "lab"))
    reference = "lab/contract-example@1.0.0"
    assert published["release"]["reference"] == reference
    releases = result(invoke("kit", "list", "--store", store))["releases"]
    assert [row["reference"] for row in releases] == [reference]
    fetched = result(invoke(
        "kit", "fetch", reference, "--store", store, "--cache", cache, "--digest", digest,
    ))
    assert fetched["digest"] == digest
    assert (Path(fetched["root"]) / "data.txt").read_bytes() == (kit / "data.txt").read_bytes()
    shutil.rmtree(store)
    shutil.rmtree(kit)
    overrides = tmp_path / "parameters.yaml"
    overrides.write_text("label: reviewed\nenabled: true\n")
    resolved = result(invoke(
        "kit", "resolve", fetched["root"], "--cache", cache,
        "--config", request(tmp_path), "--parameters", overrides,
        "--output-dir", tmp_path / "resolved", "--approve", digest,
    ))
    config = yaml.safe_load(Path(resolved["config"]).read_text())
    lock = json.loads(Path(resolved["lock"]).read_text())
    schema = result(invoke("config", "schema", "--kind", "kit-lock"))
    Draft202012Validator(schema).validate(lock)
    assert lock["bundle_digest"] == digest
    assert lock["approved_digest"] == digest
    assert config["kit_lock"] == resolved["lock"]
    assert config["inputs"] == ["sequence"] and config["loss_models"] == []
    assert "fixture_value" in config["metrics"]
    assert config["args"] == ["--enabled", "--label", "reviewed"]
    assert resolved["campaign_approval_required"] is True
    assert not (Path(fetched["root"]) / "executed").exists()


def test_resolve_refuses_unreviewed_bytes_and_contract_overrides(kit, tmp_path):
    digest = result(invoke("kit", "inspect", kit))["digest"]
    campaign = request(tmp_path)
    directory = tmp_path / "resolved"
    args = (
        "kit", "resolve", kit, "--cache", tmp_path / "cache",
        "--config", campaign, "--output-dir", directory,
    )
    wrong = invoke(*args, "--approve", "sha256:" + "0" * 64)
    assert wrong.exit_code != 0 and not wrong.stdout
    assert json.loads(wrong.stderr)["error"]["code"] == "kit_error"
    assert not directory.exists()
    raw = yaml.safe_load(campaign.read_text())
    raw["loss_models"] = []
    campaign.write_text(yaml.safe_dump(raw))
    conflict = invoke(*args, "--approve", digest)
    assert conflict.exit_code != 0 and not conflict.stdout
    assert json.loads(conflict.stderr)["error"]["code"] == "kit_error"
    assert not directory.exists()
    assert not (kit / "executed").exists()


def test_resolve_keeps_site_configuration_outside_immutable_bundle(kit, tmp_path):
    digest = result(invoke("kit", "inspect", kit))["digest"]
    destination = kit / "resolved"
    refused = invoke(
        "kit", "resolve", kit, "--cache", tmp_path / "cache",
        "--config", request(tmp_path), "--output-dir", destination, "--approve", digest,
    )
    assert refused.exit_code != 0
    assert not destination.exists()
    assert result(invoke("kit", "inspect", kit))["digest"] == digest
