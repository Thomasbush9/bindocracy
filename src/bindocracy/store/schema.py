"""DuckDB schema for one target-design campaign.

One database follows one target through heterogeneous generation, evaluation,
selection, and optional optimization runs. Target and campaign metadata live
in the general configuration; they are intentionally not repeated on every row.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

TABLE_ORDER = ["configs", "runs", "designs", "artifacts", "metrics", "decisions"]

TABLES: dict[str, str] = {
    # Author-facing config files are compiled into immutable, content-addressed
    # records. A run references one resolved config; its payload may list the
    # hashes of the general/filter/model/optimization components used to build it.
    "configs": """
        CREATE TABLE IF NOT EXISTS configs (
            config_hash    VARCHAR PRIMARY KEY,
            kind           VARCHAR NOT NULL CHECK (kind IN (
                               'general', 'filtering', 'model', 'optimization',
                               'resolved'
                           )),
            name           VARCHAR NOT NULL,
            schema_version INTEGER NOT NULL CHECK (schema_version > 0),
            payload        JSON NOT NULL,
            author         VARCHAR,
            rationale      VARCHAR,
            created_at     TIMESTAMPTZ NOT NULL
        )
    """,
    # Every operation is a run. Generation/optimization runs produce designs;
    # evaluation runs produce metrics; filter/cluster/rank runs produce decisions.
    "runs": """
        CREATE TABLE IF NOT EXISTS runs (
            run_id             VARCHAR PRIMARY KEY,
            parent_run_id      VARCHAR REFERENCES runs(run_id),
            name               VARCHAR NOT NULL,
            tool               VARCHAR NOT NULL,
            kind               VARCHAR NOT NULL CHECK (kind IN (
                                   'generate', 'evaluate', 'filter', 'cluster',
                                   'optimize', 'rank', 'import'
                               )),
            config_hash        VARCHAR NOT NULL REFERENCES configs(config_hash),
            status             VARCHAR NOT NULL CHECK (status IN (
                                   'planned', 'running', 'succeeded', 'partial',
                                   'failed', 'cancelled'
                               )),
            n_requested        BIGINT CHECK (n_requested IS NULL OR n_requested >= 0),
            n_attempted        BIGINT CHECK (n_attempted IS NULL OR n_attempted >= 0),
            n_produced         BIGINT CHECK (n_produced IS NULL OR n_produced >= 0),
            n_passed           BIGINT CHECK (n_passed IS NULL OR n_passed >= 0),
            count_details      JSON,
            container_digest   VARCHAR,
            code_revision      VARCHAR,
            workflow_metadata  JSON,
            resources          JSON,
            output_uri         VARCHAR,
            error              JSON,
            created_at         TIMESTAMPTZ NOT NULL,
            started_at         TIMESTAMPTZ,
            finished_at        TIMESTAMPTZ,
            CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
        )
    """,
    # A design is a candidate produced by a run. Backbone-only candidates are
    # valid, so sequence fields are nullable. Optimization children point to the
    # design they were derived from, regardless of which tool produced either.
    "designs": """
        CREATE TABLE IF NOT EXISTS designs (
            design_id        VARCHAR PRIMARY KEY,
            run_id           VARCHAR NOT NULL REFERENCES runs(run_id),
            parent_design_id VARCHAR REFERENCES designs(design_id),
            native_id        VARCHAR NOT NULL,
            candidate_type   VARCHAR NOT NULL CHECK (candidate_type IN (
                                 'sequence', 'backbone', 'complex'
                             )),
            sequence         VARCHAR,
            sequence_hash    VARCHAR,
            length           INTEGER CHECK (length IS NULL OR length > 0),
            seed             BIGINT,
            status           VARCHAR NOT NULL CHECK (status IN (
                                 'produced', 'partial', 'invalid'
                             )),
            metadata         JSON,
            created_at       TIMESTAMPTZ NOT NULL,
            UNIQUE (run_id, native_id),
            CHECK (
                (sequence IS NULL AND sequence_hash IS NULL AND length IS NULL)
                OR
                (sequence IS NOT NULL AND sequence_hash IS NOT NULL AND length IS NOT NULL)
            )
        )
    """,
    # Runs and designs can each have many files. URIs should normally be relative
    # to the campaign root; checksums make relocated archives auditable.
    "artifacts": """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id VARCHAR PRIMARY KEY,
            run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
            design_id   VARCHAR REFERENCES designs(design_id),
            kind        VARCHAR NOT NULL,
            uri         VARCHAR NOT NULL,
            media_type  VARCHAR,
            sha256      VARCHAR,
            size_bytes  BIGINT CHECK (size_bytes IS NULL OR size_bytes >= 0),
            metadata    JSON,
            created_at  TIMESTAMPTZ NOT NULL,
            UNIQUE (run_id, uri)
        )
    """,
    # Long format: one numeric observation per design, evaluator run, metric,
    # and replicate. Evaluator identity/version/parameters belong to the run's
    # resolved config rather than being repeated in every row.
    "metrics": """
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id   VARCHAR PRIMARY KEY,
            run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
            design_id   VARCHAR NOT NULL REFERENCES designs(design_id),
            name        VARCHAR NOT NULL,
            value       DOUBLE,
            unit        VARCHAR,
            direction   VARCHAR NOT NULL CHECK (direction IN ('min', 'max', 'none')),
            replicate   INTEGER NOT NULL DEFAULT 0 CHECK (replicate >= 0),
            status      VARCHAR NOT NULL CHECK (status IN ('ok', 'failed', 'missing')),
            details     JSON,
            measured_at TIMESTAMPTZ NOT NULL,
            UNIQUE (run_id, design_id, name, replicate),
            CHECK ((status = 'ok' AND value IS NOT NULL) OR status <> 'ok')
        )
    """,
    # Non-numeric outcomes remain distinct from metrics. A persisted rank must
    # identify the candidate scope/protocol that made the rank meaningful.
    "decisions": """
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id VARCHAR PRIMARY KEY,
            run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
            design_id   VARCHAR NOT NULL REFERENCES designs(design_id),
            kind        VARCHAR NOT NULL CHECK (kind IN (
                            'filter', 'cluster', 'selection', 'rank', 'label'
                        )),
            name        VARCHAR NOT NULL,
            value       VARCHAR,
            passed      BOOLEAN,
            rank        BIGINT CHECK (rank IS NULL OR rank > 0),
            scope_id    VARCHAR,
            reason      JSON,
            created_at  TIMESTAMPTZ NOT NULL,
            UNIQUE (run_id, design_id, kind, name),
            CHECK (kind <> 'rank' OR (rank IS NOT NULL AND scope_id IS NOT NULL))
        )
    """,
}

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_runs_kind_status ON runs(kind, status)",
    "CREATE INDEX IF NOT EXISTS idx_designs_run ON designs(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_designs_parent ON designs(parent_design_id)",
    "CREATE INDEX IF NOT EXISTS idx_designs_sequence_hash ON designs(sequence_hash)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_design ON artifacts(design_id)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_design ON metrics(design_id)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_design ON decisions(design_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_kind ON decisions(kind, name)",
]

# Stable views only. Metric pivots are deliberately exports, not schema: a
# database must not change columns merely because a new scorer was ingested.
VIEWS: dict[str, str] = {
    "design_history": """
        CREATE OR REPLACE VIEW design_history AS
        SELECT
            d.*,
            r.name AS producing_run_name,
            r.tool AS producing_tool,
            r.kind AS producing_run_kind,
            r.config_hash AS producing_config_hash
        FROM designs d
        JOIN runs r USING (run_id)
    """,
    "metric_history": """
        CREATE OR REPLACE VIEW metric_history AS
        SELECT
            m.*,
            r.name AS evaluator_run_name,
            r.tool AS evaluator_tool,
            r.config_hash AS evaluator_config_hash
        FROM metrics m
        JOIN runs r USING (run_id)
    """,
}

MIGRATIONS: dict[int, list[str]] = {}


def apply(con) -> None:
    """Create or migrate the schema in one transaction."""
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS _meta (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        _check_version(con)
        for name in TABLE_ORDER:
            con.execute(TABLES[name])
        for ddl in INDEXES:
            con.execute(ddl)
        _migrate(con)
        for ddl in VIEWS.values():
            con.execute(ddl)
        _stamp(con)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def _current_version(con) -> int | None:
    row = con.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
    return None if row is None else int(row[0])


def _check_version(con) -> None:
    current = _current_version(con)
    if current is not None and current > SCHEMA_VERSION:
        raise RuntimeError(
            f"database is schema v{current}, this code supports up to v{SCHEMA_VERSION}"
        )


def _migrate(con) -> None:
    current = _current_version(con)
    if current is None:
        return
    for version in range(current + 1, SCHEMA_VERSION + 1):
        for statement in MIGRATIONS.get(version, []):
            con.execute(statement)


def _set_meta(con, key: str, value: str) -> None:
    con.execute("DELETE FROM _meta WHERE key = ?", [key])
    con.execute("INSERT INTO _meta VALUES (?, ?)", [key, value])


def _stamp(con) -> None:
    import duckdb

    _set_meta(con, "schema_version", str(SCHEMA_VERSION))
    _set_meta(con, "duckdb_version", duckdb.__version__)
    con.execute(
        "INSERT INTO _meta "
        "SELECT 'database_id', CAST(uuid() AS VARCHAR) "
        "WHERE NOT EXISTS (SELECT 1 FROM _meta WHERE key = 'database_id')"
    )
    con.execute(
        "INSERT INTO _meta "
        "SELECT 'created_at', CAST(current_timestamp AS VARCHAR) "
        "WHERE NOT EXISTS (SELECT 1 FROM _meta WHERE key = 'created_at')"
    )
