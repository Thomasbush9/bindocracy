"""Explicit bounded kit execution through the production planning/runtime seams.

Local CPU mode is trusted-code execution, not a sandbox. GPU work is only
prepared here, then submitted through ordinary approved campaign control.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import yaml

from bindocracy.kits.models import KitError
from bindocracy.kits.optimizer import validate_optimizer_lock
from bindocracy.kits.scoring import validate_function_lock


def _run_bounded(spec, *, seconds: int) -> int:
    for path in spec.mkdirs:
        path.mkdir(parents=True, exist_ok=True)
    spec.log.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **spec.env, "CUDA_VISIBLE_DEVICES": "",
                   "SINGULARITYENV_CUDA_VISIBLE_DEVICES": "",
                   "ROCR_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    with spec.log.open("xb") as log:
        process = subprocess.Popen(
            spec.argv, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired as error:
            raise KitError(f"smoke exceeded {seconds}s; logs preserved at {spec.log}") from error
        finally:
            # Kill descendants as well as a timed-out wrapper. External scheduler
            # jobs are deliberately not launched by this execution path.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _result(manifest, model) -> dict:
    task_dir = manifest.directory / manifest.tasks[0].directory
    if model.tool == "optimize":
        from bindocracy.tools.optimize.contract import read_outputs

        _, counts = read_outputs(
            task_dir / "optimizer_output.jsonl",
            n_parents=manifest.workflow["n_designs"],
            max_children=model.max_children, declared=model.metrics,
        )
        if counts["rejected"] or counts["n_torn_lines"]:
            raise KitError(f"optimizer smoke violated the production contract: {counts}")
        return {"n_produced": counts["n_children"], "n_refused": counts["n_failed"],
                "counts": counts}
    payload = json.loads((task_dir / "result.json").read_text())
    if payload.get("error"):
        raise KitError(f"scoring smoke violated the production contract: {payload['error']}")
    summary = payload["summary"]
    if summary.get("process_error"):
        raise KitError(f"scoring smoke runtime failed: {summary['process_error']}")
    if not summary.get("output_present"):
        raise KitError("scoring smoke produced no output file; an explicit empty file is required")
    if summary.get("protocol_error"):
        raise KitError(f"scoring smoke violated the production contract: {summary}")
    return {"n_produced": summary["n_scored"], "counts": summary}


def smoke_kit(
    config: Path,
    general: Path,
    output_dir: Path,
    *,
    local_cpu: bool = False,
    plan_only: bool = False,
    site: Path | None = None,
    database: Path | None = None,
    max_inputs: int = 2,
    timeout_seconds: int = 60,
) -> dict:
    """Execute a bounded CPU test, or prepare (never submit) a campaign canary.

    Inputs are the supplied config's actual frozen design set and target; no
    substitutions or fabricated sequences. Oversized sets are refused rather
    than silently sampled. Every runtime choice still comes from its kit lock.
    """
    from bindocracy.runs.designset import DesignSet
    from bindocracy.tools import load_configs, plan, plugin_for

    if local_cpu == plan_only:
        raise KitError("choose exactly one of --local-cpu or --plan-only; static checks use kit check")
    if not 1 <= max_inputs <= 16 or not 1 <= timeout_seconds <= 600:
        raise KitError("smoke requires max_inputs in 1..16 and timeout_seconds in 1..600")
    loaded = load_configs(general, config)
    model = loaded.model
    bounded = model
    if model.tool == "optimize":
        lock = validate_optimizer_lock(model)
        if model.sharding.jobs != 1:
            raise KitError("smoke requires sharding.jobs=1")
    elif model.tool == "function" and model.function is not None:
        lock = validate_function_lock(model.function)
    else:
        raise KitError("smoke requires a resolved optimizer or custom scoring-function kit config")
    if lock is None:
        raise KitError("smoke requires a resolved kit lock; raw scripts remain supported outside kit smoke")
    design_set = DesignSet.read(model.design_set)
    if design_set.n_designs != len(design_set.entries):
        raise KitError("design set count disagrees with its real input inventory")
    if not 1 <= design_set.n_designs <= max_inputs:
        raise KitError(f"smoke requires 1..{max_inputs} real inputs, got {design_set.n_designs}; build a smaller design set")
    if local_cpu and (model.resources.gpus or lock.manifest.cost.gpu != "none"):
        raise KitError("local smoke refuses GPU work; use --plan-only with --site and approved campaign qualification")
    if local_cpu and os.environ.get("BINDOCRACY_ATTEMPT_ID"):
        if os.environ.get("BINDOCRACY_LOCAL_CPU_ALLOCATION") != "agent":
            raise KitError("benchmark local smoke requires the runner's reserved agent CPU allocation; use the approved campaign route")
        if model.resources.cpus > 1:
            raise KitError("benchmark agent-local smoke is bounded to its one reserved CPU")
    if plan_only and (site is None or database is None):
        raise KitError("--plan-only requires --site and --database for the ordinary campaign route")
    if model.resources.gpus > 1 or model.resources.cpus > (2 if local_cpu else 8):
        raise KitError("smoke is bounded to one GPU and two local/eight scheduled CPUs")
    if model.resources.memory_gb > (4 if local_cpu else 64):
        raise KitError("smoke is bounded to 4 GiB locally or 64 GiB through campaign control")
    seconds = min(model.resources.walltime_seconds, timeout_seconds, 300 if local_cpu else 600)
    bounded = bounded.model_copy(update={"resources": model.resources.model_copy(update={
        "walltime": f"00:{seconds // 60:02d}:{seconds % 60:02d}",
    })})
    if model.tool == "optimize":
        bounded = bounded.model_copy(update={"timeout_seconds": min(bounded.timeout_seconds, seconds)})
    else:
        bounded = bounded.model_copy(update={"function": bounded.function.model_copy(update={
            "timeout_seconds": min(bounded.function.timeout_seconds, seconds),
        })})
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise KitError(f"smoke output directory already exists: {output_dir}")
    if output_dir.is_relative_to(lock.kit_root):
        raise KitError("smoke outputs must be outside the immutable kit bundle")
    output_dir.mkdir(parents=True, exist_ok=False)
    # Persist the exact bounded config rather than attach an unrecorded timeout
    # override to a frozen plan. Resources remain explicitly campaign-owned.
    bounded_path = output_dir / "smoke.yaml"
    bounded_path.write_text(yaml.safe_dump(bounded.model_dump(mode="json"), sort_keys=False))
    report = {"bundle_digest": lock.bundle_digest, "lock_digest": lock.digest,
              "n_inputs": design_set.n_designs, "timeout_seconds": seconds,
              "output_dir": str(output_dir), "runtime_qualified": False}
    try:
        loaded = load_configs(general, bounded_path)
        if plan_only:
            from bindocracy.campaign.plan import build_plan

            index_path = output_dir / "index.yaml"
            index_path.write_text(yaml.safe_dump({
                "database": str(database.resolve()), "run_root": str(output_dir / "runs"),
                "general_config": str(Path(general).resolve()),
                "runs": [{"name": "kit-smoke", "config": str(bounded_path)}],
            }))
            campaign = build_plan(index_path, site, output_dir / "plan")
            report.update(status="planned", executed=False,
                          plan=str(output_dir / "plan" / "plan.json"),
                          plan_digest=campaign["digest"],
                          next_step="Review plan and use campaign qualify with its exact --approve digest; GPU also requires --allow-gpu")
        else:
            manifest = plan(loaded, output_dir / "run", name="kit-smoke")
            if len(manifest.tasks) != 1:
                raise KitError("smoke requires exactly one production task")
            spec = plugin_for(model.tool).launch_spec(manifest, 0)
            returncode = _run_bounded(spec, seconds=seconds)
            if returncode:
                raise KitError(f"smoke runtime exited {returncode}; logs preserved at {spec.log}")
            counts = _result(manifest, bounded)
            report.update(status="passed", executed=True, contract_valid=True,
                          scientific_output="produced" if counts["n_produced"] else "zero_output",
                          log=str(spec.log), **counts)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        report.update(status="failed", error=str(error))
        (output_dir / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise KitError(f"{error}; report: {output_dir / 'smoke-report.json'}") from error
    (output_dir / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
