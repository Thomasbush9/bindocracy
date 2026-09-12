"""Optimization output -> normalized records.

The first adapter here that emits **both** designs and metrics. A generator
emits designs; a scorer emits metrics about designs that already exist. An
optimizer produces new candidates *and* reports what its loss thought of them,
and the two have to stay distinguishable: `<name>_loss` is the optimizer's own
opinion of a child, not a measurement of it, and nothing downstream should be
able to mistake one for the other. Hence the metric prefix is the optimizer's
name and `loss_models` is stored on every child.

Three things are recorded that no other adapter records, each because a
question about optimized designs cannot be answered without it:

* `parent_design_id` -- the lineage, so "which designs are original" is a query
  rather than an archaeology exercise
* `loss_models` -- which models already had a say, so a later ranking can
  exclude them instead of measuring its own optimizer
* `n_substitutions` / `length_delta` -- how far the child moved from its parent,
  so a run that changed nothing is visible as such rather than as a hundred new
  candidates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bindocracy.adapters.base import CollectionError, OutputAdapter
from bindocracy.adapters.common import (
    artifact,
    provenance_artifacts,
    read_manifest,
    run_status,
    run_window,
    unique_by_uri,
)
from bindocracy.adapters.scoring import metric_records
from bindocracy.functions.models import MetricDeclaration
from bindocracy.runs.designset import DesignSet, DesignSetEntry
from bindocracy.runs.status import read_task_status
from bindocracy.store.records import (
    ArtifactRecord,
    CandidateType,
    CollectedRun,
    DesignRecord,
    MetricRecord,
    RunRecord,
    stable_id,
)
from bindocracy.tools.optimize.contract import read_outputs

CHILDREN_FILE = "children.jsonl"


class OptimizeOutputAdapter(OutputAdapter):
    tool = "optimize"

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        manifest = read_manifest(run_dir)
        if manifest.run_id != run.run_id:
            raise CollectionError(
                f"manifest run_id {manifest.run_id} does not match {run.run_id}"
            )

        workflow = manifest.workflow
        prefix = workflow["metric_prefix"]
        loss_models = list(workflow.get("loss_models") or ())
        max_children = int(workflow.get("max_children", 1))
        declared = {
            key: MetricDeclaration.model_validate(value)
            for key, value in (workflow.get("declared_metrics") or {}).items()
        }
        specs = {key: value.as_spec(key) for key, value in declared.items()}

        design_set = DesignSet.read(workflow["design_set_manifest"])
        by_index = {entry.index: entry for entry in design_set.entries}

        designs: list[DesignRecord] = []
        metrics: list[MetricRecord] = []
        artifacts: list[ArtifactRecord] = []
        statuses = []
        per_task: dict[str, Any] = {}
        totals = {
            "n_children": 0,
            "n_failed": 0,
            "n_torn_lines": 0,
            "n_unknown_parent": 0,
            "n_unchanged": 0,
        }
        rejected: dict[str, int] = {}
        parents_touched: set[int] = set()

        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            statuses.append(status)
            path = run_dir / task.designs
            if not path.is_file():
                per_task[f"task-{task.task_id:04d}"] = {
                    "status": status.status if status else None,
                    "n_children": 0,
                    "missing_output": True,
                }
                continue

            rows, counts = read_outputs(
                path,
                # The driver has already translated shard-local indices into
                # design-set indices, so the whole set is in range here.
                n_parents=max(by_index) + 1 if by_index else 0,
                max_children=max_children,
                declared=declared,
            )
            for reason, count in (counts.get("rejected") or {}).items():
                rejected[reason] = rejected.get(reason, 0) + count
            totals["n_torn_lines"] += counts["n_torn_lines"]
            totals["n_failed"] += counts["n_failed"]

            for row in rows:
                parent = by_index.get(row.parent_index)
                if parent is None:
                    totals["n_unknown_parent"] += 1
                    continue
                parents_touched.add(row.parent_index)
                if row.is_failure:
                    continue

                child = _design_record(
                    run_id=run.run_id,
                    parent=parent,
                    row=row,
                    loss_models=loss_models,
                    optimizer=prefix,
                    seed=workflow.get("seed"),
                    created_at=manifest.created_at,
                )
                designs.append(child)
                totals["n_children"] += 1
                if child.sequence == parent.sequence:
                    totals["n_unchanged"] += 1

                if row.metrics:
                    metrics.extend(
                        metric_records(
                            run_id=run.run_id,
                            design_id=child.design_id,
                            model=prefix,
                            values=row.metrics,
                            replicate=0,
                            measured_at=manifest.created_at,
                            details={"seconds": row.seconds, "child": row.child},
                            specs=specs,
                        )
                    )

                for relative, kind in (
                    (row.structure, "optimized_structure"),
                    (row.trajectory, "optimization_trajectory"),
                ):
                    if not relative:
                        continue
                    record = artifact(
                        run_dir,
                        run.run_id,
                        f"{task.directory}/structures/{relative}",
                        kind,
                        design_id=child.design_id,
                    )
                    if record is not None:
                        artifacts.append(record)

            per_task[f"task-{task.task_id:04d}"] = {
                "status": status.status if status else None,
                "n_children": counts["n_children"],
                "n_failed": counts["n_failed"],
                "rejected": counts.get("rejected") or {},
            }
            for relative, kind in (
                (task.designs, "optimized_designs"),
                (task.status, "task_status"),
                (task.log, "log"),
            ):
                record = artifact(run_dir, run.run_id, relative, kind)
                if record is not None:
                    artifacts.append(record)

        artifacts.extend(
            provenance_artifacts(
                run_dir,
                run.run_id,
                manifest,
                {"driver": "driver_script", "general": "general_config", "model": "model_config"},
            )
        )

        started, finished = run_window(statuses)
        complete = all(status is not None for status in statuses)
        n_requested = int(workflow["n_designs"])

        updated = run.model_copy(
            update={
                "status": run_status(len(designs), statuses, complete),
                # The three numbers of harness-design 2, for a run that is both
                # shapes at once: how many PARENTS it was given, how many it
                # reached a verdict on, and how many CHILDREN exist as a
                # result. n_produced counts children, because children are what
                # this run created.
                "n_requested": n_requested,
                "n_attempted": len(parents_touched),
                "n_produced": len(designs),
                "count_details": {
                    "tasks": per_task,
                    **totals,
                    "rejected": rejected,
                    "n_parents_untouched": n_requested - len(parents_touched),
                    "optimizer": prefix,
                    # Stored here as well as on every child: a query asking
                    # "which runs let boltz2 see the design" should not have to
                    # unpack per-design metadata.
                    "loss_models": loss_models,
                    "script_sha256": workflow.get("script_sha256"),
                    "protocol_sha256": workflow.get("protocol_sha256"),
                    "design_set_digest": workflow.get("design_set_digest"),
                    "scope_id": workflow.get("scope_id"),
                },
                "started_at": started,
                "finished_at": finished,
                "output_uri": str(run_dir),
            }
        )
        return CollectedRun(
            run=updated,
            designs=tuple(designs),
            artifacts=tuple(unique_by_uri(artifacts)),
            metrics=tuple(metrics),
        )

    def succeeded(self, run_dir: Path) -> bool:
        """Artifacts exist and parse, per harness-design 6.

        Exit status is not consulted, for the same reason the scorer does not:
        a shard that optimized everything it was given and then tripped over a
        trailing shell command is a success.
        """
        try:
            manifest = read_manifest(run_dir)
        except CollectionError:
            return False
        for task in manifest.tasks:
            status = read_task_status(run_dir / task.status)
            if status is None or status.status != "succeeded":
                return False
            if not (run_dir / task.designs).is_file():
                return False
        return True


def _design_record(
    *,
    run_id: str,
    parent: DesignSetEntry,
    row: Any,
    loss_models: list[str],
    optimizer: str,
    seed: int | None,
    created_at: Any,
) -> DesignRecord:
    """One child, as a design with a parent.

    `native_id` extends the parent's rather than being fresh, so a person
    reading a FASTA can see the lineage without a join -- and so re-collecting
    the same run produces the same `design_id`, which is what makes ingestion
    idempotent. A random id here would make every re-collect a new set of
    designs.
    """
    native_id = f"{parent.native_id}.{optimizer}{row.child}"
    return DesignRecord(
        design_id=stable_id("design", run_id, native_id),
        run_id=run_id,
        parent_design_id=parent.design_id,
        native_id=native_id,
        candidate_type=CandidateType.SEQUENCE,
        sequence=row.sequence,
        seed=seed if isinstance(seed, int) else None,
        metadata={
            "optimizer": optimizer,
            "child": row.child,
            "parent_index": parent.index,
            "parent_native_id": parent.native_id,
            "parent_tool": parent.tool,
            "parent_run_name": parent.run_name,
            # Which models the loss consulted. On every child, not only on the
            # run, because a design outlives the query that found it and this
            # is the fact that stops a later ranking from measuring the
            # optimizer instead of the binder.
            "loss_models": loss_models,
            # How far it moved. A child identical to its parent is a real
            # result -- the optimizer found nothing -- and is worth being able
            # to count rather than discovering as a duplicate sequence later.
            "n_substitutions": _substitutions(parent.sequence, row.sequence),
            "length_delta": len(row.sequence) - parent.length,
        },
        created_at=created_at,
    )


def _substitutions(parent: str, child: str) -> int | None:
    """Hamming distance, or None when the lengths differ.

    None rather than an alignment: this is a cheap diagnostic, and a number
    that silently meant "edit distance" for some rows and "substitutions" for
    others would be worse than an absence. A length change is already recorded
    beside it.
    """
    if len(parent) != len(child):
        return None
    return sum(1 for left, right in zip(parent, child) if left != right)
