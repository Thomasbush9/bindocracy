"""A third tool, defined outside the library, driven through the real workflow.

The claim the whole plugin seam rests on is that adding a tool costs one module
and one registration. A toy adapter exercised in-process does not test that: it
never meets Snakemake, never gets planned, never gets launched, and never
reaches the database.

`tests/toytool/` is a complete foreign tool -- its own config shape, its own
CSV output, its own adapter -- registered through `BINDOCRACY_PLUGINS` without
editing anything under `src/bindocracy/`. If the contract is insufficient,
this is where it shows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
import yaml

SNAKEFILE = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"
REPO = Path(__file__).resolve().parents[1]

# A "tool": writes one CSV of designs into the directory it is given.
TOY_SCRIPT = '''\
import csv, os, sys
out_dir, n, task = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "toy_designs.csv"), "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["name", "seq", "score"])
    for i in range(n):
        writer.writerow([f"task-{task:04d}-toy-{i}", "ACDEFGHIKL", 0.5 + i / 10])
'''


@pytest.fixture
def toy_campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target\nACDEFG\n")
    script = tmp_path / "toy_tool.py"
    script.write_text(TOY_SCRIPT)

    general = tmp_path / "general.yaml"
    general.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "toy-campaign"},
        "target": {"name": "toy-target", "sequence_fasta": str(fasta), "chain_id": "A"},
        "cluster": {"executor": "slurm", "account": "acct", "default_partition": "part"},
    }))
    model = tmp_path / "toy.yaml"
    model.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "toy-v1", "tool": "toy",
        "sampling": {"jobs": 2, "designs_per_job": 3},
        "runtime": {"script": str(script)},
        "resources": {"gpus": 1, "cpus": 2, "memory_gb": 4, "walltime": "01:00:00"},
    }))

    database = tmp_path / "campaign.duckdb"
    index = tmp_path / "index.yaml"
    index.write_text(yaml.safe_dump({
        "database": str(database), "run_root": str(tmp_path / "runs"),
        "general_config": str(general),
        "runs": [{"name": "toy-run-01", "config": str(model)}],
    }))
    return index, database, tmp_path / "runs"


def run_workflow(workdir: Path, index: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "BINDOCRACY_PLUGINS": "tests.toytool", "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, "-m", "snakemake", "--snakefile", str(SNAKEFILE),
         "--configfile", str(index), "--cores", "2", "--resources", "db_writer=1"],
        cwd=workdir, env=env, capture_output=True, text=True, check=False,
    )


def test_a_foreign_tool_runs_end_to_end(toy_campaign, tmp_path: Path) -> None:
    """Planned, fanned out, launched, collected, ingested — no library change."""
    index, database, run_root = toy_campaign

    result = run_workflow(tmp_path, index)

    assert result.returncode == 0, result.stderr
    assert (run_root / "toy-run-01" / "generation.done").is_file()

    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT tool, status, n_produced FROM runs").fetchone() == (
        "toy", "succeeded", 6)
    assert con.execute("SELECT count(*) FROM designs").fetchone() == (6,)
    assert con.execute(
        "SELECT count(*) FROM metrics WHERE name = 'toy_score'").fetchone() == (6,)
    # its own config, stored whole, alongside any other tool's
    assert con.execute("SELECT tool FROM configs").fetchone() == ("toy",)
    con.close()


def test_the_toy_tool_fans_out_like_any_other(toy_campaign, tmp_path: Path) -> None:
    index, database, run_root = toy_campaign

    assert run_workflow(tmp_path, index).returncode == 0

    manifest = json.loads((run_root / "toy-run-01" / "run.json").read_text())
    assert len(manifest["tasks"]) == 2
    assert manifest["tool"] == "toy"
    # the generic machinery archived what the plugin declared, and nothing else
    assert list(manifest["provenance"]) == ["script"]
    assert set(manifest["inputs"]) == {"target_fasta"}
    for task in ("0000", "0001"):
        assert (run_root / "toy-run-01" / "tasks" / task / "toy_designs.csv").is_file()
        assert (run_root / "toy-run-01" / "tasks" / task / "status.json").is_file()


def test_the_built_in_tools_are_unaffected_by_a_third(toy_campaign, tmp_path: Path) -> None:
    index, _, _ = toy_campaign
    run_workflow(tmp_path, index)

    from bindocracy.tools import registered_tools

    assert set(registered_tools()) >= {"boltzgen", "mosaic"}
