"""Synthetic transport-only kits exercise the ordinary frozen execution path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bindocracy.campaign import check as readiness
from bindocracy.campaign.plan import PlanError, build_plan, load_plan
from bindocracy.kits.bundle import cache_kit
from bindocracy.kits.models import KitError
from bindocracy.kits.optimizer import optimizer_config, validate_optimizer_lock
from bindocracy.kits.resolve import resolve_kit, write_lock
from bindocracy.runs.manifest import ManifestError, RunManifest
from bindocracy.runs.steps import generate_step
from bindocracy.tools import collect_run, launch_spec, load_configs, plan
from bindocracy.tools.optimize.preflight import OptimizePreflightError, preflight_optimize


@pytest.fixture
def make_kit(tmp_path, write_design_set):
    def build(*, container=False):
        source = tmp_path / "authored-kit"
        source.mkdir()
        (source / "helper.py").write_text(
            "from pathlib import Path\n"
            "VALUE = float(Path(__file__).with_name('payload.txt').read_text())\n"
        )
        (source / "payload.txt").write_text("2.5\n")
        (source / "transport.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "from helper import VALUE\n"
            "from bindocracy_io import run_optimization\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--payload-root', type=Path, required=True)\n"
            "def transport(parent, context, args):\n"
            "    value = float((args.payload_root / 'value.txt').read_text())\n"
            "    yield {'sequence': parent['sequence'], "
            "'metrics': {'transport_value': VALUE + value}}\n"
            "raise SystemExit(run_optimization(transport, parser=parser))\n"
        )
        asset = tmp_path / "transport-data"
        asset.mkdir()
        (asset / "value.txt").write_text("1.0\n")
        requirements = [{"name": "transport-data", "kind": "weights"}]
        bindings = {"transport-data": {"kind": "weights", "path": str(asset)}}
        if container:
            image = tmp_path / "fixture.sif"
            image.write_bytes(b"not an executable image; command inspection only")
            overlay = tmp_path / "overlay"
            overlay.mkdir()
            (overlay / "placeholder.txt").write_text("synthetic dependency inventory\n")
            requirements.extend(
                [
                    {"name": "runtime", "kind": "container", "provides": "interpreter"},
                    {
                        "name": "overlay",
                        "kind": "source_overlay",
                        "target": "/opt/mosaic/src/mosaic",
                        "mount": "read_only",
                    },
                ]
            )
            bindings.update(
                {
                    "runtime": {"kind": "container", "path": str(image), "interpreter": "python"},
                    "overlay": {"kind": "source_overlay", "path": str(overlay)},
                }
            )
        (source / "kit.yaml").write_text(
            yaml.safe_dump(
                {
                    "kit_schema": 1,
                    "name": "synthetic-transport",
                    "version": "1.0.0",
                    "kind": "optimizer",
                    "summary": "Transport unchanged fixture rows using sibling code and data.",
                    "entrypoint": "transport.py",
                    "python": ">=3.10",
                    "requires": requirements,
                    "inputs": ["sequence"],
                    "loss_models": [],
                    "capabilities": {"changes_length": False, "max_children": 1},
                    "metrics": {"transport_value": {"direction": "none"}},
                    "cost": {
                        "gpu": "none",
                        "compile_seconds": 0,
                        "per_parent_seconds": 0,
                        "measured_on": {"scope": "synthetic transport fixture"},
                    },
                    "dependency_args": {"transport-data": "--payload-root"},
                }
            )
        )
        bindings_path = tmp_path / "bindings.yaml"
        bindings_path.write_text(yaml.safe_dump({"bindings_schema": 1, **bindings}))
        release = cache_kit(source, tmp_path / "cache")
        lock = resolve_kit(release.root, bindings_path, approved_digest=release.digest)
        lock_path = write_lock(lock, tmp_path / "kit-lock.json")
        target = tmp_path / "target.fasta"
        target.write_text(">synthetic\nACDEFGHIK\n")
        general_path = tmp_path / "general.yaml"
        general_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "campaign": {"name": "transport-only"},
                    "target": {"name": "synthetic", "sequence_fasta": str(target), "chain_id": "A"},
                    "cluster": {"executor": "slurm", "account": "test", "default_partition": "test"},
                }
            )
        )
        request = {
            "schema_version": 1,
            "name": "transport",
            "design_set": str(write_design_set(["ACDE", "GGGG"])),
            "sharding": {"jobs": 1},
            "resources": {"gpus": 1, "cpus": 1, "memory_gb": 1, "walltime": "00:01:00"},
        }
        model = optimizer_config(lock, request, lock_path)
        model_path = tmp_path / "optimize.yaml"
        model_path.write_text(yaml.safe_dump(model.model_dump(mode="json")))
        index = tmp_path / "index.yaml"
        index.write_text(
            yaml.safe_dump(
                {
                    "database": str(tmp_path / "campaign.duckdb"),
                    "run_root": str(tmp_path / "runs"),
                    "general_config": str(general_path),
                    "runs": [{"name": "transport", "config": str(model_path)}],
                }
            )
        )
        site = tmp_path / "site.yaml"
        site.write_text(
            yaml.safe_dump(
                {
                    "controller": {
                        "account": "test", "partition": "test", "cpus": 1,
                        "memory_gb": 1, "walltime": "00:01:00", "gpus": 0,
                    },
                    "max_workers": 1,
                    "max_total_gpus": 1,
                }
            )
        )
        return SimpleNamespace(
            source=source, root=release.root, lock=lock, lock_path=lock_path,
            bindings=bindings_path, asset=asset, request=request, model=model,
            general=general_path, config=model_path, index=index, site=site,
            plan_dir=tmp_path / "frozen",
        )

    return build


@pytest.mark.parametrize(
    "field", ["script", "args", "inputs", "loss_models", "metrics", "driver_script", "kit_lock"]
)
def test_request_cannot_redeclare_kit_owned_configuration(make_kit, field):
    case = make_kit()
    # Even an identical declaration is refused rather than silently replaced.
    value = case.model.model_dump(mode="json")[field]
    with pytest.raises(KitError, match="kit-owned"):
        optimizer_config(case.lock, {**case.request, field: value}, case.lock_path)


def test_runtime_and_capabilities_are_not_campaign_overrides(make_kit):
    case = make_kit()
    with pytest.raises(KitError, match="scratch"):
        optimizer_config(case.lock, {**case.request, "runtime": {"container": None}}, case.lock_path)
    with pytest.raises(KitError, match="max_children"):
        optimizer_config(case.lock, {**case.request, "max_children": 2}, case.lock_path)
    with pytest.raises(KitError, match="length_delta"):
        optimizer_config(case.lock, {**case.request, "length_delta": 1}, case.lock_path)
    model = optimizer_config(
        case.lock, {**case.request, "runtime": {"scratch": str(case.root.parent / "scratch")}},
        case.lock_path,
    )
    assert validate_optimizer_lock(model).digest == case.lock.digest


@pytest.mark.parametrize(
    "override",
    [
        {"script": "/different.py"},
        {"args": ["--different"]},
        {"inputs": ["provenance"]},
        {"loss_models": ["boltz2"]},
        {"metrics": {}},
        {"driver_script": "/different-driver.py"},
        {"runtime": {"container_python": "different-python"}},
    ],
)
def test_effective_yaml_cannot_bypass_lock_configuration(make_kit, override):
    case = make_kit()
    case.config.write_text(yaml.safe_dump({**case.model.model_dump(mode="json"), **override}))
    with pytest.raises(OptimizePreflightError, match="kit-owned"):
        load_configs(case.general, case.config)


def test_unapproved_lock_is_inspectable_but_cannot_configure_or_launch(make_kit):
    case = make_kit()
    unapproved = resolve_kit(case.root, case.bindings)
    path = write_lock(unapproved, case.lock_path.with_name("unapproved.json"))
    with pytest.raises(KitError, match="approv"):
        optimizer_config(unapproved, case.request, path)
    loaded = load_configs(case.general, case.config)
    changed = loaded.model.model_copy(update={"kit_lock": path})
    with pytest.raises(OptimizePreflightError, match="approv"):
        preflight_optimize(loaded.general, changed)


def test_cached_kit_executes_sibling_imports_data_and_helper_through_worker(make_kit):
    case = make_kit()
    shutil.rmtree(case.source)
    frozen = build_plan(case.index, case.site, case.plan_dir)
    manifest = RunManifest.read(frozen["runs"][0]["manifest"])
    task = manifest.tasks[0]
    assert generate_step(
        manifest.directory / "run.json", task.task_id,
        manifest.path(task.log), manifest.path(task.status),
    ) == 0
    collected = collect_run(manifest.directory / "run.json")
    assert sorted(design.sequence for design in collected.designs) == ["ACDE", "GGGG"]
    assert [metric.value for metric in collected.metrics if metric.name == "transport_transport_value"] == [3.5, 3.5]
    assert collected.run.workflow_metadata["kit"]["digest"] == case.lock.digest
    assert not tuple(case.root.rglob("__pycache__"))
    assert load_plan(case.plan_dir / "plan.json")["digest"] == frozen["digest"]
    assert build_plan(case.index, case.site, case.plan_dir)["digest"] == frozen["digest"]


@pytest.mark.parametrize("mutation", ["bundle_change", "bundle_addition", "dependency", "lock"])
def test_frozen_plan_and_worker_refuse_changed_execution_closure(make_kit, mutation):
    case = make_kit()
    frozen = build_plan(case.index, case.site, case.plan_dir)
    manifest = RunManifest.read(frozen["runs"][0]["manifest"])
    if mutation == "bundle_change":
        (case.root / "payload.txt").write_text("9.0\n")
    elif mutation == "bundle_addition":
        (case.root / "new-module.py").write_text("# changes complete inventory\n")
    elif mutation == "dependency":
        (case.asset / "new-data.txt").write_text("new dependency data\n")
    else:
        payload = json.loads(case.lock_path.read_text())
        payload["approved_digest"] = None
        case.lock_path.write_text(json.dumps(payload))
    with pytest.raises((PlanError, ManifestError, KitError)):
        load_plan(case.plan_dir / "plan.json")
    with pytest.raises((ManifestError, KitError)):
        launch_spec(manifest, 0)
    with pytest.raises((PlanError, ManifestError, KitError)):
        build_plan(case.index, case.site, case.plan_dir)
    assert not manifest.path(manifest.tasks[0].status).exists()


@pytest.mark.parametrize("gpus", [0, 1])
def test_container_gpu_visibility_matches_requested_allocation(make_kit, tmp_path, gpus):
    case = make_kit(container=True)
    model = case.model.model_copy(update={
        "resources": case.model.resources.model_copy(update={"gpus": gpus}),
    })
    case.config.write_text(yaml.safe_dump(model.model_dump(mode="json")))
    manifest = plan(load_configs(case.general, case.config), tmp_path / "allocation-test")
    launch = launch_spec(manifest, 0)
    assert ("--nv" in launch.argv) is bool(gpus)
    assert launch.resources["gres"] == (f"gpu:{gpus}" if gpus else "")


def test_readiness_is_read_only_and_reports_actual_lock_evidence(make_kit, tmp_path, monkeypatch):
    case = make_kit()

    def forbidden(*args, **kwargs):
        raise AssertionError("readiness must not probe or execute")

    monkeypatch.setattr(readiness, "_probe", forbidden)
    before = {
        str(path): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for path in tmp_path.rglob("*")
    }
    report = readiness.check_campaign(case.index, case.site)
    after = {
        str(path): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for path in tmp_path.rglob("*")
    }
    assert after == before
    run = report["runs"][0]
    assert run["status"] == "unknown"
    assert any(
        check["code"] == "kit.content.transport-data" and check["status"] == "unknown"
        for check in run["checks"]
    )
    assert run["kit"]["digest"] == case.lock.digest
    assert not any(check["code"] == "kit_dependencies" for check in run["checks"])
    assert report["scheduler_status"] == "unknown"


def test_container_command_binds_declared_assets_read_only_without_running_image(make_kit, tmp_path):
    case = make_kit(container=True)
    loaded = load_configs(case.general, case.config)
    manifest = plan(loaded, tmp_path / "container-run")
    spec = launch_spec(manifest, 0)
    bindings = {spec.argv[index + 1] for index, value in enumerate(spec.argv) if value == "--bind"}
    for dependency in case.lock.dependencies.values():
        path = dependency["path"]
        if dependency["kind"] == "harness_module":
            assert f"{path}:{path}:ro" not in bindings
            for relative in dependency["identity"]["files"]:
                file = str(Path(path) / relative)
                assert f"{file}:{file}:ro" in bindings
        else:
            assert f"{path}:{path}:ro" in bindings
        if dependency.get("target"):
            assert f"{path}:{dependency['target']}:ro" in bindings
    assert f"{case.root}:{case.root}:ro" in bindings
    report = readiness.check_campaign(case.index, case.site)
    assert report["runs"][0]["status"] == "unknown"
    assert not (tmp_path / "container-run" / "tasks" / "0000" / "status.json").exists()


