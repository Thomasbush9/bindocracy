"""Scoring functions use the same planning and allocation seam as GPU tools."""

from __future__ import annotations

import sys
from pathlib import Path

from bindocracy.config.load import sha256_file
from bindocracy.functions.freeze import freeze_function
from bindocracy.functions.models import CustomFunction
from bindocracy.functions.run import FunctionRunConfig, FunctionRunError
from bindocracy.functions.runner import IO_HELPER
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import ToolPlan
from bindocracy.store.records import RunKind
from bindocracy.tools.base import ToolPlugin
from bindocracy.tools.function.adapter import FunctionOutputAdapter

DRIVER = IO_HELPER.parent / "functions" / "run_function.py"


class FunctionPlugin(ToolPlugin):
    tool = "function"
    config_type = FunctionRunConfig
    adapter_type = FunctionOutputAdapter

    def preflight(self, general, model):
        if model.resources.walltime_seconds <= 0:
            raise FunctionRunError("function walltime must be positive")
        if model.function is not None and model.function.kit_lock is not None:
            from bindocracy.kits.scoring import validate_function_lock

            lock = validate_function_lock(model.function)
            if lock.manifest.cost.gpu == "required" and not model.resources.gpus:
                raise FunctionRunError("this scoring kit requires a positive GPU allocation")
        return freeze_function(general, model)

    def tool_plan(self, loaded):
        pre = loaded.preflight
        function = pre.function
        archives = {"driver": DRIVER, "bindocracy_io": IO_HELPER,
                    "design_set": loaded.model.design_set,
                    "target_fasta": loaded.general.target.sequence_fasta}
        destinations = {"driver": "driver/run_function.py", "design_set": "inputs/design_set.json",
                        "target_fasta": "inputs/target.fasta"}
        lock = None
        if isinstance(function, CustomFunction) and function.kit_lock is not None:
            from bindocracy.kits.scoring import validate_function_lock

            lock = validate_function_lock(function)
            for relative in lock.files:
                label = f"kit:{relative}"
                archives[label] = lock.kit_root / relative
                destinations[label] = f"kit/{relative}"
            script_label = f"kit:{lock.manifest.entrypoint}"
            helper_relative = str(Path(lock.manifest.entrypoint).parent / IO_HELPER.name)
            if helper_relative in lock.files:
                if sha256_file(lock.kit_root / helper_relative) != sha256_file(IO_HELPER):
                    raise FunctionRunError("kit supplies a conflicting bindocracy_io.py")
                archives.pop("bindocracy_io")
                helper_label = f"kit:{helper_relative}"
            else:
                destinations["bindocracy_io"] = f"kit/{helper_relative}"
                helper_label = "bindocracy_io"
        else:
            archives["function_script"] = function.script
            destinations["function_script"] = f"function/{function.script.name}"
            destinations["bindocracy_io"] = "function/bindocracy_io.py"
            script_label = "function_script"
            helper_label = "bindocracy_io"
        structures = {}
        for design_id, (path, model, source) in pre.structures.items():
            label = f"structure:{design_id}"
            archives[label] = path
            destinations[label] = f"structures/{design_id}{path.suffix}"
            structures[design_id] = {**source, "archive_label": label, "source_model": model}
        return ToolPlan(
            kind=RunKind.EVALUATE, jobs=1, designs_per_task=len(pre.rows), designs_file="result.json",
            archives=archives, archive_paths=destinations,
            container=function.container or function.script,
            inputs={"container": function.container} if function.container and lock is None else {},
            workflow={
                "function": function.model_dump(mode="json"),
                "function_type": "custom" if isinstance(function, CustomFunction) else "builtin",
                "script_label": script_label, "helper_label": helper_label,
                "script_sha256": sha256_file(function.script),
                "frozen_rows": pre.rows, "source_artifacts": structures,
                "metric_sources": pre.metric_sources, "skipped_metrics": pre.skipped,
                "design_set_manifest": str(loaded.model.design_set),
                "design_set_digest": pre.design_set.digest, "scope_id": pre.design_set.scope_id,
                "n_designs": len(pre.rows), "kit": lock.model_dump(mode="json") if lock else None,
                "database": str(loaded.model.database.resolve()),
            },
        )

    def launch_spec(self, manifest, task_id):
        general, model = self.configs_of(manifest)
        manifest.verify_inputs()
        task = task_of(manifest, task_id)
        driver = manifest.path(manifest.provenance["driver"].path)
        return LaunchSpec(
            argv=(sys.executable, str(driver), "--manifest", str(manifest.directory / "run.json"),
                  "--task-id", str(task_id)),
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            resources=slurm_resources(general.cluster, model.resources),
            log=manifest.path(task.log), outputs=(manifest.path(task.designs), manifest.path(task.status)),
            mkdirs=(manifest.path(task.directory),),
        )

    def resources(self, loaded):
        return slurm_resources(loaded.general.cluster, loaded.model.resources)
