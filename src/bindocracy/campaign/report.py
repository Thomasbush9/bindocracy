"""Offline scientific reporting and integrity inspection; never schedule or repair."""

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import duckdb

from bindocracy.campaign.plan import load_plan
from bindocracy.runs.inputs import digest_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.runs.selection import read_only
from bindocracy.runs.staging import read_collected, staged_digest
from bindocracy.runs.status import read_task_status
from bindocracy.store import reporting


def campaign_report(database: str | Path, *, runs=()) -> dict:
    with read_only(database) as connection:
        selected = reporting.selected_runs(connection, runs)
        reports = [reporting.summarize_run(connection, run) for run in selected]
    return {
        "database": str(Path(database).resolve()),
        "run_ids": [run["run_id"] for run in selected],
        "run_status_counts": dict(Counter(run["status"] for run in selected)),
        "runs": reports,
        "interpretation": "Counts and metrics are per owning run; evaluator output counts are not new designs. Replica coverage lists observed indices, not an inferred expected replica count. Scheduler completion and ingestion-marker presence are not scientific success.",
    }


def report_text(report: dict) -> str:
    lines = [f"Campaign: {report['database']}"]
    if not report["runs"]:
        lines.append("No stored runs selected.")
    for item in report["runs"]:
        run, outcome = item["run"], item["scientific_outcome"]
        lines.extend(
            [
                f"{run['name']} [{run['run_id']}] {run['kind']}: {outcome['status']}",
                f"  requested={outcome['requested']} attempted={outcome['attempted']} produced={outcome['produced']} ({outcome['count_unit']}); new designs={outcome['new_design_rows']}",
            ]
        )
        for metric in item["metrics"]:
            lines.append(
                f"  {metric['name']}: finite ok={metric['finite_ok_rows']}/{metric['rows']}, nonfinite={metric['nonfinite_rows']}, mean={metric['mean']}, range=[{metric['min']}, {metric['max']}]"
            )
        for cohort in item["cohorts"]:
            passed = sum(member["passed"] is True for member in cohort["members"])
            lines.append(
                f"  {cohort['kind']} {cohort['name']} scope={cohort['scope_id']}: {len(cohort['members'])} decisions, {passed} passed"
            )
    lines.append(report["interpretation"])
    return "\n".join(lines)


def _destination(output: str | Path, database: str | Path) -> Path:
    path = Path(output)
    if path.resolve() == Path(database).resolve():
        raise ValueError("output must not overwrite the source database")
    # Refusing every existing destination also protects hardlinks, symlinks and
    # input/artifact files without guessing which extensions are scientific data.
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists; refusing to overwrite: {path}")
    return path


def write_report(report: dict, output: str | Path) -> Path:
    path = _destination(output, report["database"])
    content = (
        json.dumps(report, indent=2, allow_nan=False) + "\n"
        if path.suffix.lower() == ".json"
        else report_text(report) + "\n"
    )
    with path.open("x") as handle:
        handle.write(content)
    return path


def export_campaign(database: str | Path, output: str | Path, *, runs=(), table="metrics") -> dict:
    """Export one native long-form table with many-to-one provenance joins.

    CSV uses \\N for NULL, JSON strings for nested columns, and ISO timestamps.
    Parquet retains native numeric/null/timestamp types; nested DB JSON is text.
    """
    path = _destination(output, database)
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("export output must end in .csv or .parquet")
    with read_only(database) as connection:
        selected = reporting.selected_runs(connection, runs)
        run_ids = tuple(run["run_id"] for run in selected)
        sql, params = reporting.export_query(table, run_ids)
        cursor = connection.execute(sql, params)
        columns = [column[0] for column in cursor.description]
        if path.suffix.lower() == ".parquet":
            from pyarrow import parquet

            count = 0
            reader = cursor.to_arrow_reader(batch_size=4096)
            with path.open("xb") as handle, parquet.ParquetWriter(handle, reader.schema) as writer:
                for batch in reader:
                    writer.write_batch(batch)
                    count += batch.num_rows
        else:
            count = 0
            with path.open("x", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                while batch := cursor.fetchmany(4096):
                    for row in batch:
                        writer.writerow(_csv_value(value) for value in row)
                    count += len(batch)
    return {
        "output": str(path.resolve()),
        "table": table,
        "format": path.suffix[1:].lower(),
        "rows": count,
        "columns": columns,
        "run_ids": list(run_ids),
        "database": str(Path(database).resolve()),
    }


def _csv_value(value):
    if value is None:
        return "\\N"
    if isinstance(value, str) and value.startswith("\\"):
        return "\\" + value
    return value.isoformat() if hasattr(value, "isoformat") else value


def audit_campaign(plan_path: str | Path) -> dict:
    """Compare the frozen plan, execution artifacts, bundle and actual stored rows.

    Missing work is incomplete, not corrupt. A completion marker raises the
    required evidence threshold; the marker never substitutes for that evidence.
    """
    findings = []
    result = {
        "plan": str(Path(plan_path).resolve()),
        "integrity": "incomplete",
        "findings": findings,
        "runs": [],
        "scheduler": "not queried",
    }

    def finding(code, message, *, run_id=None, path=None, severity="corrupt", **details):
        findings.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "run_id": run_id,
                "path": str(path) if path is not None else None,
                **details,
            }
        )

    try:
        plan = load_plan(plan_path, verify=False)
    except (OSError, ValueError, RuntimeError) as error:
        finding("plan_invalid", str(error), path=plan_path)
        result["integrity"] = "corrupt"
        return result
    result.update(digest=plan["digest"], database=plan["database"])
    for group in ("artifacts", "referenced_inputs"):
        for path, expected in plan.get(group, {}).items():
            _check_file(Path(path), expected, None, finding, code="frozen_artifact")
    with ExitStack() as stack:
        connection = None
        database = Path(plan["database"])
        if database.is_file():
            try:
                connection = stack.enter_context(read_only(database))
            except (duckdb.Error, OSError, RuntimeError) as error:
                finding("database_unreadable", str(error), path=database)
        else:
            finding(
                "database_not_created",
                "No database yet; ingestion cannot be verified.",
                path=database,
                severity="incomplete",
            )
        for frozen in plan["runs"]:
            entry = {
                "name": frozen["name"],
                "manifest": frozen["manifest"],
                "tasks": [],
                "ingestion": "not_ingested",
            }
            result["runs"].append(entry)
            try:
                manifest = RunManifest.read(frozen["manifest"])
                entry["run_id"] = manifest.run_id
                identity = {
                    "name": manifest.name,
                    "tool": manifest.tool,
                    "kind": manifest.kind.value,
                    "tasks": len(manifest.tasks),
                }
                if any(frozen.get(key) != value for key, value in identity.items()):
                    finding(
                        "manifest_identity_mismatch",
                        "Manifest disagrees with frozen run identity.",
                        run_id=manifest.run_id,
                        path=frozen["manifest"],
                    )
                if (
                    Path(frozen["manifest"]).resolve()
                    != (manifest.directory / "run.json").resolve()
                ):
                    finding(
                        "manifest_location_mismatch",
                        "Manifest run directory differs from frozen location.",
                        run_id=manifest.run_id,
                    )
                manifest.verify_inputs()
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                finding("manifest_invalid", str(error), path=frozen["manifest"])
                continue
            run_id = manifest.run_id
            directory = manifest.directory
            marker_path = directory / "ingested.json"
            marker_exists = marker_path.exists()
            generation_done = (directory / "generation.done").exists()
            entry.update(ingested_marker=marker_exists, generation_done_marker=generation_done)
            if marker_exists:
                try:
                    marker = json.loads(marker_path.read_text())
                    if (
                        not isinstance(marker, dict)
                        or Path(marker.get("database", "")).resolve() != database.resolve()
                        or not isinstance(marker.get("inserted"), bool)
                    ):
                        raise ValueError(
                            "ingested marker does not identify the planned database and insertion result"
                        )
                except (OSError, ValueError, TypeError) as error:
                    finding("ingested_marker_invalid", str(error), run_id=run_id, path=marker_path)
            for task in manifest.tasks:
                task_result = {
                    "task_id": task.task_id,
                    "status": "unstarted",
                    "output_exists": manifest.path(task.designs).is_file(),
                }
                entry["tasks"].append(task_result)
                try:
                    status = read_task_status(manifest.path(task.status))
                    if status is None:
                        task_result["status"] = (
                            "output_without_status" if task_result["output_exists"] else "unstarted"
                        )
                        finding(
                            "task_status_missing",
                            "Task has not recorded an outcome.",
                            run_id=run_id,
                            path=manifest.path(task.status),
                            severity="corrupt"
                            if generation_done or marker_exists
                            else "incomplete",
                        )
                    else:
                        task_result["status"] = status.status
                        if status.task_id != task.task_id:
                            finding(
                                "task_identity_mismatch",
                                "Task status belongs to another task.",
                                run_id=run_id,
                                path=manifest.path(task.status),
                            )
                        if not task_result["output_exists"] and (
                            status.status == "succeeded" or (status.n_produced or 0) > 0
                        ):
                            finding(
                                "task_output_missing",
                                "Task reports produced work but its output is absent.",
                                run_id=run_id,
                                path=manifest.path(task.designs),
                            )
                    output = manifest.path(task.designs)
                    if output.is_file():
                        if output.suffix.lower() in {".jsonl", ".csv", ".tsv", ".json"}:
                            count = _task_output_rows(output)
                            if status is not None and (status.n_produced or 0) > 0 and count == 0:
                                raise ValueError(
                                    "task claims produced work but output contains no records"
                                )
                        else:
                            finding(
                                "task_output_format_unverified",
                                "No syntax reader for this output format; artifact hashes are checked separately.",
                                run_id=run_id,
                                path=output,
                                severity="incomplete",
                            )
                except (OSError, ValueError, TypeError, RuntimeError, csv.Error) as error:
                    task_result["status"] = "invalid"
                    finding(
                        "task_artifact_invalid",
                        str(error),
                        run_id=run_id,
                        path=manifest.path(task.directory),
                    )
            bundle_path = directory / "collected.json"
            if not generation_done and (marker_exists or bundle_path.exists()):
                finding(
                    "generation_marker_missing",
                    "Collected work exists without the workflow completion marker.",
                    run_id=run_id,
                    path=directory / "generation.done",
                    severity="incomplete",
                )
            try:
                stored = (
                    reporting.run_records(connection, "runs", run_id)
                    if connection is not None
                    else ()
                )
                if not bundle_path.is_file():
                    finding(
                        "bundle_missing",
                        "No collected bundle to compare.",
                        run_id=run_id,
                        path=bundle_path,
                        severity="corrupt" if marker_exists or stored else "incomplete",
                    )
                    continue
                bundle = read_collected(bundle_path)
                digest = staged_digest(bundle_path)
                expected_identity = manifest.to_run_record()
                fields = (
                    "run_id",
                    "name",
                    "tool",
                    "kind",
                    "model_config_id",
                    "output_uri",
                    "container_digest",
                    "code_revision",
                    "created_at",
                )
                if any(
                    getattr(bundle.run, key) != getattr(expected_identity, key) for key in fields
                ):
                    finding(
                        "bundle_identity_mismatch",
                        "Collected run differs from the frozen manifest.",
                        run_id=run_id,
                        path=bundle_path,
                    )
                for artifact in bundle.artifacts:
                    if "://" in artifact.uri:
                        finding(
                            "artifact_not_local",
                            "Remote artifact not verified by offline audit.",
                            run_id=run_id,
                            path=artifact.uri,
                            severity="incomplete",
                        )
                    else:
                        _check_file(
                            manifest.path(artifact.uri),
                            artifact.sha256,
                            artifact.size_bytes,
                            finding,
                            run_id=run_id,
                        )
                if not stored:
                    entry["ingestion"] = (
                        "marker_without_database_run" if marker_exists else "collected_not_ingested"
                    )
                    finding(
                        "database_run_missing",
                        "The actual database has no matching run.",
                        run_id=run_id,
                        severity="corrupt" if marker_exists else "incomplete",
                    )
                else:
                    entry["ingestion"] = "ingested"
                    differences = reporting.bundle_mismatches(
                        connection, bundle, digest, manifest.config
                    )
                    if (
                        manifest.target is not None
                        and reporting.target_digest(connection) != manifest.target.sequence_sha256
                    ):
                        differences.append({"table": "_meta", "field": "target_sha256"})
                    if differences:
                        entry["ingestion"] = "database_mismatch"
                        finding(
                            "database_bundle_mismatch",
                            "Actual database records differ from the collected bundle.",
                            run_id=run_id,
                            differences=differences,
                        )
                    if not marker_exists:
                        finding(
                            "ingested_marker_missing",
                            "Database records exist, but the workflow marker is absent.",
                            run_id=run_id,
                            path=marker_path,
                            severity="incomplete",
                        )
            except (OSError, ValueError, RuntimeError, duckdb.Error) as error:
                finding("bundle_or_database_invalid", str(error), run_id=run_id, path=bundle_path)
    result["integrity"] = (
        "corrupt"
        if any(item["severity"] == "corrupt" for item in findings)
        else "incomplete"
        if findings
        else "verified"
    )
    return result


def _task_output_rows(path: Path) -> int:
    """Validate common native tabular formats without invoking a tool/plugin."""
    with path.open(newline="") as handle:
        if path.suffix.lower() in {".csv", ".tsv"}:
            reader = csv.reader(
                handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",", strict=True
            )
            header = next(reader, [])
            count = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError("task output row width differs from its header")
                count += 1
            return count
        if path.suffix.lower() == ".json":
            value = json.load(handle)
            if not isinstance(value, (dict, list)):
                raise ValueError("task JSON output must be an object or list")
            return len(value)
        count = 0
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if not isinstance(json.loads(line), dict):
                raise TypeError(f"task output row {number} is not a JSON object")
            count += 1
        return count


def _check_file(path, expected_hash, expected_size, finding, *, run_id=None, code="artifact"):
    try:
        if not path.is_file():
            finding(f"{code}_missing", "Expected artifact is absent.", run_id=run_id, path=path)
        elif expected_size is not None and path.stat().st_size != expected_size:
            finding(f"{code}_size_mismatch", "Artifact size changed.", run_id=run_id, path=path)
        elif expected_hash is not None and digest_of(path).sha256 != expected_hash:
            finding(
                f"{code}_hash_mismatch", "Artifact content hash changed.", run_id=run_id, path=path
            )
        elif expected_hash is None:
            finding(
                f"{code}_unhashed",
                "Presence/size checked; no recorded hash to verify content.",
                run_id=run_id,
                path=path,
                severity="incomplete",
            )
    except OSError as error:
        finding(f"{code}_unreadable", str(error), run_id=run_id, path=path)
