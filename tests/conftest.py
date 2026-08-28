"""Shared fixtures: a valid config pair on disk, and a fake Mosaic run.

The fake driver and exec wrapper let the whole workflow run locally without a
GPU or a container, which is the point of the local end-to-end test.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest
import yaml

# Stand-in for singularity/mosaic-exec.sh: drop the "python" argument and run
# the driver with this interpreter, which needs no container.
FAKE_EXEC_WRAPPER = f'#!/usr/bin/env bash\nshift\nexec {sys.executable} "$@"\n'

# Writes the real output contract with no models involved.
FAKE_DRIVER = '''\
import argparse, json, os
from datetime import UTC, datetime

ap = argparse.ArgumentParser()
for flag in ("--target-fasta", "--target-msa", "--save-dir"):
    ap.add_argument(flag, required=True)
for flag in ("--binder-length", "--task-id", "--seed-base", "--n-designs",
             "--soft-steps", "--sharpen-steps", "--final-steps"):
    ap.add_argument(flag, type=int, required=True)
ap.add_argument("--max-runtime", type=float, required=True)
a = ap.parse_args()

save_dir = os.path.abspath(a.save_dir)
os.makedirs(save_dir, exist_ok=True)
now = datetime.now(UTC).isoformat()
with open(os.path.join(save_dir, "designs.jsonl"), "a") as out:
    for index in range(a.n_designs):
        out.write(json.dumps({
            "native_id": f"task-{a.task_id:04d}-design-{index:06d}",
            "sequence": "ACDEFGHIKL"[: max(2, a.binder_length % 10 + 2)],
            "seed": a.seed_base + a.task_id * 100_000 + index,
            "ranking_loss": -0.1 * (index + 1),
            "completed_at": now,
            "seconds": 0.1,
        }) + "\\n")
with open(os.path.join(save_dir, "status.json"), "w") as out:
    json.dump({
        "task_id": a.task_id, "status": "succeeded",
        "started_at": now, "finished_at": now,
        "n_attempted": a.n_designs, "n_produced": a.n_designs,
        "output_file": "designs.jsonl", "error": None,
    }, out)
'''


def write_configs(root: Path, **mosaic_overrides) -> tuple[Path, Path]:
    """Write a valid general + Mosaic config pair and everything they reference."""
    fasta = root / "target.fasta"
    msa = root / "target.a3m"
    driver = root / "hallucinate_binders.py"
    container = root / "mosaic.sif"
    wrapper = root / "mosaic-exec.sh"
    weights = root / "weights"
    scratch = root / "scratch" / "mosaic"

    (weights / "boltz").mkdir(parents=True, exist_ok=True)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    fasta.write_text(">target\nACDEFG\n")
    msa.write_text(">target\nACDEFG\n")
    driver.write_text(FAKE_DRIVER)
    container.write_bytes(b"fixture")
    wrapper.write_text(FAKE_EXEC_WRAPPER)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "msa": str(msa),
            "chain_id": "A",
            "hotspots": [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
            "max_concurrent_jobs": 2,
        },
    }, sort_keys=False))

    mosaic = {
        "schema_version": 1,
        "name": "mosaic-test",
        "tool": "mosaic",
        "driver": {"script": str(driver), "archive": True},
        "sampling": {
            "binder_length": 70,
            "jobs": 2,
            "designs_per_job": 4,
            "max_runtime_hours": 1,
            "seed_base": 0,
        },
        "runtime": {
            "container": str(container),
            "weights": str(weights),
            "exec_wrapper": str(wrapper),
            "scratch": str(scratch),
        },
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "walltime": "02:00:00"},
    }
    for section, values in mosaic_overrides.items():
        mosaic[section].update(values)

    model_path = root / "mosaic.yaml"
    model_path.write_text(yaml.safe_dump(mosaic, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_configs(tmp_path)


def design_line(task_id: int, index: int, sequence: str = "ACDEFG", **overrides) -> str:
    """One valid designs.jsonl record, with fields overridable per test."""
    record = {
        "native_id": f"task-{task_id:04d}-design-{index:06d}",
        "sequence": sequence,
        "seed": task_id * 100_000 + index,
        "ranking_loss": -0.5 - index,
        "completed_at": f"2026-08-28T12:00:{index:02d}+00:00",
        "seconds": 420.0,
    }
    record.update(overrides)
    return json.dumps(record)


def write_task(
    run_dir: Path,
    task_id: int,
    lines: list[str],
    *,
    text: str | None = None,
    status: dict | None = None,
) -> None:
    """Populate one task directory. `status=None` leaves the status file out."""
    task_dir = run_dir / "tasks" / f"{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    body = text if text is not None else "".join(line + "\n" for line in lines)
    if body:
        (task_dir / "designs.jsonl").write_text(body)
    if status is not None:
        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-28T11:00:00+00:00",
            "finished_at": "2026-08-28T12:00:00+00:00",
            "n_attempted": len(lines),
            "n_produced": len(lines),
            "output_file": "designs.jsonl",
            "error": None,
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("fake log\n")
