"""Read-side campaign summaries, lossless table exports, and bundle comparisons.

All selectors identify the owning run, never an implicit union of evaluator scopes.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from bindocracy.runs.selection import resolve_run_ids
from bindocracy.store.query import select_metrics
from bindocracy.store.records import CollectedRun, ConfigRecord

TABLE_KEYS = {
    "runs": "run_id",
    "designs": "design_id",
    "metrics": "metric_id",
    "decisions": "decision_id",
    "artifacts": "artifact_id",
}
_JSON = {
    "general_config_json",
    "model_config_json",
    "count_details",
    "workflow_metadata",
    "resources",
    "error",
    "metadata",
    "details",
    "reason",
}


def rows(connection, sql: str, params=()) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, params)
    names = [column[0] for column in cursor.description]
    return [
        {
            name: json.loads(value) if name in _JSON and isinstance(value, str) else value
            for name, value in zip(names, row, strict=True)
        }
        for row in cursor.fetchall()
    ]


def selected_runs(connection, selectors=()) -> tuple[dict[str, Any], ...]:
    ids = tuple(dict.fromkeys(resolve_run_ids(connection, tuple(selectors))))
    where = f" WHERE run_id IN ({','.join('?' for _ in ids)})" if selectors else ""
    return tuple(rows(connection, "SELECT * FROM runs" + where + " ORDER BY run_id", ids))


def run_records(connection, table: str, run_id: str) -> tuple[dict[str, Any], ...]:
    if table not in TABLE_KEYS:
        raise ValueError(f"unknown record table {table!r}")
    # Never normalize evidence through write-side models: a lowercase sequence
    # or whitespace in a stored value is a real difference during an audit.
    return tuple(rows(connection, f"SELECT * FROM {table} WHERE run_id = ?", [run_id]))


def summarize_run(connection, run: dict[str, Any]) -> dict:
    metrics = select_metrics(connection, run_ids=[run["run_id"]])
    groups = defaultdict(list)
    for metric in metrics:
        groups[(metric.name, metric.direction)].append(metric)
    summaries = []
    for (name, direction), measurements in sorted(groups.items()):
        finite = [
            m.value
            for m in measurements
            if m.status == "ok" and m.value is not None and math.isfinite(m.value)
        ]
        coverage = defaultdict(lambda: {"observed": [], "finite_ok": []})
        for metric in measurements:
            coverage[metric.design_id]["observed"].append(metric.replicate)
            if metric.status == "ok" and metric.value is not None and math.isfinite(metric.value):
                coverage[metric.design_id]["finite_ok"].append(metric.replicate)
        summaries.append(
            {
                "run_id": run["run_id"],
                "name": name,
                "direction": direction,
                "rows": len(measurements),
                "status_counts": dict(Counter(m.status for m in measurements)),
                "finite_ok_rows": len(finite),
                "nonfinite_rows": sum(
                    m.value is not None and not math.isfinite(m.value) for m in measurements
                ),
                "null_rows": sum(m.value is None for m in measurements),
                "min": min(finite) if finite else None,
                "max": max(finite) if finite else None,
                # Scaling first avoids overflow on otherwise finite large measurements.
                "mean": math.fsum(value / len(finite) for value in finite) if finite else None,
                "replica_coverage": [
                    {
                        "design_id": design_id,
                        **{key: sorted(value) for key, value in counts.items()},
                    }
                    for design_id, counts in sorted(coverage.items())
                ],
            }
        )
    decisions = run_records(connection, "decisions", run["run_id"])
    cohorts = defaultdict(list)
    for decision in decisions:
        cohorts[(decision["kind"], decision["name"], decision["scope_id"])].append(decision)
    design_status_counts = {
        row["status"]: row["count"]
        for row in rows(
            connection,
            "SELECT status, count(*) AS count FROM designs WHERE run_id = ? "
            "GROUP BY status ORDER BY status",
            [run["run_id"]],
        )
    }
    attempted, produced = run["n_attempted"], run["n_produced"]
    units = {
        "generate": "designs",
        "optimize": "requested/attempted parents; produced children",
        "import": "designs",
        "evaluate": "evaluation outputs (not new designs)",
        "filter": "decided candidates",
        "rank": "decided candidates",
        "cluster": "decided candidates",
    }
    return {
        "run": _json_row(run),
        "scientific_outcome": {
            "status": run["status"],
            "count_unit": units[run["kind"]],
            "requested": run["n_requested"],
            "attempted": attempted,
            "produced": produced,
            "attempted_not_produced": attempted - produced
            if run["kind"] != "optimize"
            and attempted is not None
            and produced is not None
            and attempted >= produced
            else None,
            "new_design_rows": sum(design_status_counts.values()),
            "design_status_counts": design_status_counts,
        },
        "workflow_completion": "not inferred from scientific outcome or database presence",
        "metrics": summaries,
        "cohorts": [
            {
                "run_id": run["run_id"],
                "kind": kind,
                "name": name,
                "scope_id": scope,
                "members": [
                    _json_row(item)
                    for item in sorted(
                        members, key=lambda item: (item["rank"] or 0, item["design_id"])
                    )
                ],
            }
            for (kind, name, scope), members in sorted(
                cohorts.items(), key=lambda item: str(item[0])
            )
        ],
    }


def export_query(table: str, run_ids: tuple[str, ...]) -> tuple[str, list[str]]:
    if table not in TABLE_KEYS:
        raise ValueError(f"unknown export table {table!r}; choose {', '.join(TABLE_KEYS)}")
    # Each join follows a primary key: never multiply measurements by decisions/artifacts.
    owner = "t" if table == "runs" else "r"
    joins = "" if table == "runs" else " JOIN runs r ON r.run_id = t.run_id"
    joins += f" JOIN configs c ON c.model_config_id = {owner}.model_config_id"
    provenance = (
        f", {owner}.name AS owner_run_name, {owner}.tool AS owner_tool, "
        f"{owner}.kind AS owner_kind, {owner}.workflow_metadata AS owner_workflow_metadata, "
        f"{owner}.container_digest AS owner_container_digest, {owner}.code_revision AS owner_code_revision, "
        f"{owner}.output_uri AS owner_output_uri, c.general_config_id AS provenance_general_config_id, "
        "c.model_config_id AS provenance_model_config_id, c.general_config_hash, c.model_config_hash, "
        "c.general_config_json, c.model_config_json"
    )
    if table in {"metrics", "decisions", "artifacts"}:
        joins += " LEFT JOIN designs d ON d.design_id = t.design_id"
        provenance += ", d.run_id AS producing_run_id, d.native_id, d.sequence, d.parent_design_id"
    where = f"t.run_id IN ({','.join('?' for _ in run_ids)})" if run_ids else "FALSE"
    key = TABLE_KEYS[table]
    return (
        f"SELECT t.*{provenance} FROM {table} t{joins} WHERE {where} ORDER BY t.run_id, t.{key}",
        list(run_ids),
    )


def bundle_mismatches(
    connection, bundle: CollectedRun, digest: str, config: ConfigRecord
) -> list[dict]:
    """Compare actual typed records, not only the stored digest or row counts."""
    differences = []
    expected_run = bundle.run.model_copy(
        update={
            "workflow_metadata": {**(bundle.run.workflow_metadata or {}), "bundle_sha256": digest}
        }
    )
    for table, key in TABLE_KEYS.items():
        expected = (expected_run,) if table == "runs" else getattr(bundle, table)
        actual = run_records(connection, table, bundle.run.run_id)
        expected_map = {getattr(row, key): row for row in expected}
        actual_map = {row[key]: row for row in actual}
        missing = sorted(expected_map.keys() - actual_map.keys())
        extra = sorted(actual_map.keys() - expected_map.keys())
        changed = sorted(
            identifier
            for identifier in expected_map.keys() & actual_map.keys()
            if _comparable(expected_map[identifier].model_dump())
            != _comparable(actual_map[identifier])
        )
        if missing or extra or changed:
            differences.append(
                {"table": table, "missing": missing, "extra": extra, "changed": changed}
            )
    configs = rows(
        connection, "SELECT * FROM configs WHERE model_config_id = ?", [config.model_config_id]
    )
    # Identical configs may have been ingested earlier, retaining their first timestamp/source.
    fields = (
        "general_config_id",
        "model_config_id",
        "general_config_json",
        "model_config_json",
        "general_config_hash",
        "model_config_hash",
    )
    if not configs or any(configs[0][field] != getattr(config, field) for field in fields):
        differences.append({"table": "configs", "model_config_id": config.model_config_id})
    return differences


def target_digest(connection) -> str | None:
    row = connection.execute("SELECT value FROM _meta WHERE key = 'target_sha256'").fetchone()
    return row[0] if row else None


def _json_row(row: dict) -> dict:
    """Serialize native timestamp columns without normalizing scientific values."""
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in row.items()
    }


def _comparable(value):
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, datetime):
        return value.timestamp()
    return value
