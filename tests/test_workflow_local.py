"""Run the real Snakefile end to end with a fake driver and no GPU.

This exercises the parts that unit tests cannot: rule dependencies, wildcard
fan-out, paths, staging, and the single-writer ingestion boundary.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
import yaml

SNAKEFILE = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"


def snakemake(workdir: Path, config_file: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "snakemake",
         "--snakefile", str(SNAKEFILE),
         "--configfile", str(config_file),
         "--cores", "2",
         "--resources", "db_writer=1",
         *extra],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,  # the tests assert on returncode themselves
    )


@pytest.fixture
def campaign(configs, tmp_path: Path) -> tuple[Path, Path, Path]:
    """A workflow config pointing at the fixture configs, plus its run root."""
    general_path, model_path = configs
    named = tmp_path / "config_01.yaml"
    shutil.move(model_path, named)

    database = tmp_path / "campaign.duckdb"
    run_root = tmp_path / "runs"
    config_file = tmp_path / "campaign.yaml"
    config_file.write_text(yaml.safe_dump({
        "database": str(database),
        "run_root": str(run_root),
        "general_config": str(general_path),
        "models": {"mosaic": [str(named)]},
    }, sort_keys=False))
    return config_file, database, run_root


def test_dry_run_plans_one_job_per_task_and_one_ingestion(campaign, tmp_path: Path) -> None:
    config_file, _, _ = campaign

    result = snakemake(tmp_path, config_file, "--dry-run")

    assert result.returncode == 0, result.stderr
    # The "Job stats" table: one line of "<rule> <count>" per planned rule.
    counts = {
        parts[0]: int(parts[1])
        for line in (result.stdout + result.stderr).splitlines()
        if len(parts := line.split()) == 2 and parts[1].isdigit()
    }
    assert counts["generate"] == 2  # sampling.jobs
    assert counts["ingest"] == 1
    assert counts["prepare_run"] == 1


def test_local_end_to_end_fills_the_database(campaign, tmp_path: Path) -> None:
    config_file, database, run_root = campaign

    result = snakemake(tmp_path, config_file)

    assert result.returncode == 0, result.stderr
    run_dir = run_root / "config_01"
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "collected.json").is_file()
    assert (run_dir / "generation.done").is_file()

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM configs").fetchone() == (1,)
    assert con.execute(
        "SELECT status, n_requested, n_produced FROM runs"
    ).fetchone() == ("succeeded", 8, 8)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (8,)
    assert con.execute(
        "SELECT count(*) FROM metrics WHERE name = 'mosaic_ranking_loss'"
    ).fetchone() == (8,)
    con.close()


def test_rerunning_the_workflow_does_not_duplicate_the_run(campaign, tmp_path: Path) -> None:
    config_file, database, run_root = campaign
    assert snakemake(tmp_path, config_file).returncode == 0
    run_id = duckdb.connect(str(database), read_only=True).execute(
        "SELECT run_id FROM runs"
    ).fetchone()[0]

    # Force every rule from collection onwards to run again.
    (run_root / "config_01" / "generation.done").unlink()
    (run_root / "config_01" / "ingested.json").unlink()
    (run_root / "config_01" / "collected.json").unlink()
    result = snakemake(tmp_path, config_file)

    assert result.returncode == 0, result.stderr
    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM runs").fetchone() == (1,)
    assert con.execute("SELECT run_id FROM runs").fetchone() == (run_id,)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (8,)
    con.close()
