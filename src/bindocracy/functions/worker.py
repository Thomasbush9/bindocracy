"""Execute frozen function inputs; this module never opens a campaign database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bindocracy.config.load import sha256_file
from bindocracy.functions.contract import FunctionInput
from bindocracy.functions.models import BuiltinFunction, CustomFunction
from bindocracy.functions.runner import FunctionError, kit_bindings, run_custom
from bindocracy.runs.launch import task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.runs.status import TaskStatus, write_task_status
from bindocracy.store.records import utc_now


def execute(manifest: RunManifest, task_id: int) -> int:
    task = task_of(manifest, task_id)
    started = utc_now()
    output = manifest.path(task.designs)
    task_dir = manifest.path(task.directory)
    payload = {"records": [], "summary": {}, "error": None}
    status = "failed"
    attempted = 0
    produced = 0
    try:
        manifest.verify_inputs()
        workflow = manifest.workflow
        cls = CustomFunction if workflow["function_type"] == "custom" else BuiltinFunction
        raw = dict(workflow["function"])
        if cls is CustomFunction:
            raw["kit_lock"] = None
        raw["script"] = str(manifest.path(manifest.provenance[workflow["script_label"]].path))
        function = cls.model_validate(raw)
        if sha256_file(function.script) != workflow["script_sha256"]:
            raise FunctionError("function script changed while archiving")
        bindings = []
        if workflow.get("kit"):
            from bindocracy.kits.resolve import KitLock, verify_dependencies

            lock = KitLock.model_validate_json(json.dumps(workflow["kit"]))
            verify_dependencies(lock)
            root = manifest.directory / "provenance" / "kit"
            expected = {str(root / name) for name in lock.files}
            helper = manifest.path(manifest.provenance[workflow["helper_label"]].path)
            expected.add(str(helper))
            actual = {str(path) for path in root.rglob("*") if path.is_file()}
            if actual != expected or any(path.is_symlink() for path in root.rglob("*")):
                raise FunctionError("archived kit inventory changed")
            for relative, digest in lock.files.items():
                if sha256_file(root / relative) != digest:
                    raise FunctionError(f"kit file changed while archiving: {relative}")
            bindings.extend(kit_bindings(lock, root))
        inputs = []
        skipped = workflow["skipped_metrics"]
        for frozen in workflow["frozen_rows"]:
            if frozen["design_id"] in skipped:
                continue
            row = dict(frozen)
            source = workflow["source_artifacts"].get(row["design_id"])
            if source:
                row["structure"] = manifest.path(manifest.provenance[source["archive_label"]].path)
                if sha256_file(row["structure"]) != source["sha256"]:
                    raise FunctionError(f"structure changed while archiving: {source['artifact_id']}")
            inputs.append(FunctionInput(**row))
        attempted = len(inputs)
        result = run_custom(function, inputs, run_id=manifest.run_id,
                            work_dir=task_dir / "function", measured_at=started,
                            prepared=True, retain_partial=True,
                            gpus=manifest.resources["gpus"], dependency_bindings=tuple(bindings))
        produced = result.n_scored
        records = []
        for metric in result.records:
            details = {**(metric.details or {}),
                       "source_metrics": workflow["metric_sources"].get(metric.design_id, {})}
            if metric.design_id in workflow["source_artifacts"]:
                source = workflow["source_artifacts"][metric.design_id]
                details["source_artifact"] = {**source, "consumed_path": str(manifest.path(manifest.provenance[source["archive_label"]].path))}
            records.append(metric.model_copy(update={"details": details}).model_dump(mode="json"))
        payload["records"] = records
        payload["summary"] = {**result.summary, "skipped_metrics": skipped,
                              "incomplete": {str(k): list(v) for k, v in result.incomplete.items()}}
        status = "succeeded" if produced == len(workflow["frozen_rows"]) and not result.failures and not result.incomplete else "partial" if produced else "failed"
    except (FunctionError, ValueError, OSError, RuntimeError) as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["summary"]["protocol_error"] = isinstance(error, FunctionError)
    task_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    write_task_status(manifest.path(task.status), TaskStatus(
        task_id=task_id, status=status, started_at=started, finished_at=utc_now(),
        n_attempted=attempted, n_produced=produced, error=payload["error"],
        output_file=task.designs,
    ))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    args = parser.parse_args()
    return execute(RunManifest.read(args.manifest), args.task_id)
