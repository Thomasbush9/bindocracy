"""Two real tools, sharing everything generic and nothing tool-specific.

Mosaic and BoltzGen disagree about almost everything: sequence target versus
structure target, a driver script versus a bound design spec, JSON lines versus
a 237-column CSV, and whether "produced" and "passed" are the same number. If
any of that leaks into the generic machinery, something here fails.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from conftest import design_line, write_boltzgen_task, write_task

from bindocracy.runs import ingest_bundle, write_collected
from bindocracy.store import create_database
from bindocracy.tools import (
    UnknownToolError,
    collect_run,
    load_configs,
    plan,
    plugin_for,
    registered_tools,
    resources,
)


def test_every_built_in_tool_is_registered() -> None:
    assert registered_tools() == (
        "boltzgen", "genie3", "mosaic", "protein_hunter", "pxdesign",
    )


def test_an_unknown_tool_names_the_ones_that_exist() -> None:
    with pytest.raises(UnknownToolError, match="mosaic"):
        plugin_for("alphafold")


def test_the_config_chooses_the_tool(configs, boltzgen_configs) -> None:
    """Nothing passes a tool name; the model YAML's `tool:` key decides."""
    assert load_configs(*configs).tool == "mosaic"
    assert load_configs(*boltzgen_configs).tool == "boltzgen"


def test_each_tool_archives_what_its_own_run_consumes(
    configs, boltzgen_configs, tmp_path: Path
) -> None:
    mosaic = plan(load_configs(*configs), tmp_path / "m")
    boltzgen = plan(load_configs(*boltzgen_configs), tmp_path / "b")

    assert list(mosaic.provenance) == ["driver"]
    assert list(boltzgen.provenance) == ["spec"]
    assert mosaic.path(mosaic.provenance["driver"].path).suffix == ".py"
    assert boltzgen.path(boltzgen.provenance["spec"].path).suffix == ".yaml"


def test_each_tool_names_its_own_per_task_output(
    configs, boltzgen_configs, tmp_path: Path
) -> None:
    mosaic = plan(load_configs(*configs), tmp_path / "m")
    boltzgen = plan(load_configs(*boltzgen_configs), tmp_path / "b")

    assert mosaic.tasks[0].designs.endswith("designs.jsonl")
    assert boltzgen.tasks[0].designs.endswith("final_ranked_designs/all_designs_metrics.csv")
    # The status file is the one output shape they share.
    assert mosaic.tasks[0].status.endswith("status.json")
    assert boltzgen.tasks[0].status.endswith("status.json")


def test_resources_resolve_without_a_manifest_for_both(configs, boltzgen_configs) -> None:
    """Snakemake needs these at DAG time, before any run directory exists."""
    for loaded in (load_configs(*configs), load_configs(*boltzgen_configs)):
        assert set(resources(loaded)) == {
            "slurm_account", "slurm_partition", "gres",
            "cpus_per_task", "mem_mb", "runtime",
        }
    assert resources(load_configs(*boltzgen_configs))["mem_mb"] == 96 * 1024


def test_two_tools_land_in_one_database_without_colliding(
    configs, boltzgen_configs, tmp_path: Path
) -> None:
    """The end of the whole point: one campaign, two generators, one table."""
    database = create_database(tmp_path / "campaign.duckdb")

    mosaic = plan(load_configs(*configs), tmp_path / "m")
    for task_id in (0, 1):
        write_task(mosaic.directory, task_id,
                   [design_line(task_id, i) for i in range(2)], status={})
    boltzgen = plan(load_configs(*boltzgen_configs), tmp_path / "b")
    write_boltzgen_task(boltzgen.directory, 0, status={"n_attempted": 8})

    for manifest in (mosaic, boltzgen):
        bundle = write_collected(
            collect_run(manifest.directory / "run.json"),
            manifest.directory / "collected.json",
        )
        assert ingest_bundle(database, bundle) is True

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute(
        "SELECT producing_tool, count(*) FROM design_history "
        "GROUP BY producing_tool ORDER BY producing_tool"
    ).fetchall() == [("boltzgen", 4), ("mosaic", 4)]
    # One campaign, two generators: the general config is shared and groups
    # both model configs, which is exactly what general_config_id is for.
    assert con.execute("SELECT count(DISTINCT model_config_id) FROM configs").fetchone() == (2,)
    assert con.execute("SELECT count(DISTINCT general_config_id) FROM configs").fetchone() == (1,)
    # Each tool's metrics are namespaced, so a cross-tool query stays honest.
    names = {row[0] for row in con.execute("SELECT DISTINCT name FROM metrics").fetchall()}
    assert "mosaic_ranking_loss" in names
    assert any(name.startswith("boltzgen_") for name in names)
    # n_passed is populated by the tool that filters and null for the one that does not.
    assert con.execute(
        "SELECT tool, n_produced, n_passed FROM runs ORDER BY tool"
    ).fetchall() == [("boltzgen", 4, 2), ("mosaic", 4, None)]
    # The tool that filters records verdicts; the one that does not records none.
    assert con.execute(
        "SELECT r.tool, d.kind, count(*) FROM decisions d JOIN runs r USING (run_id) "
        "GROUP BY 1, 2 ORDER BY 2"
    ).fetchall() == [("boltzgen", "filter", 4), ("boltzgen", "rank", 4)]
    assert con.execute("SELECT count(DISTINCT status) FROM designs").fetchone() == (1,)
    con.close()


def test_launch_reads_the_manifest_not_the_live_config(
    configs, tmp_path: Path, monkeypatch
) -> None:
    """A planned run must launch what it was planned with.

    Editing the authored YAML after planning used to change the command while
    run.json stayed put, so the job ran with values no record described.
    """
    import yaml as _yaml

    from bindocracy.tools import launch_spec

    general_path, model_path = configs
    manifest = plan(load_configs(general_path, model_path), tmp_path / "run")
    before = launch_spec(manifest, 0).argv

    raw = _yaml.safe_load(model_path.read_text())
    raw["sampling"]["binder_length"] = 999
    raw["sampling"]["optimizer"] = {"soft_steps": 7}
    model_path.write_text(_yaml.safe_dump(raw))

    assert launch_spec(manifest, 0).argv == before
    flags = dict(zip(before[3::2], before[4::2], strict=True))
    assert flags["--binder-length"] == "70"
    assert flags["--soft-steps"] == "100"
