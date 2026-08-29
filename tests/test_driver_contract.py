"""The connector/driver seam.

Nothing else in the suite covers this: the connector builds an argument vector
on the login node and the driver parses it inside a container, so the two can
drift apart and every unit test still passes. The bug is only visible when a
GPU job has already started. These tests close that gap by parsing the real
connector's argv with the real driver's parser.
"""

from __future__ import annotations

import json
from pathlib import Path

from bindocracy.config.preflight import read_single_fasta
from bindocracy.tools import launch_spec, load_configs, plan
from bindocracy.tools.mosaic.adapter import MosaicOutputAdapter


def parse(driver, argv: tuple[str, ...], monkeypatch):
    """Parse the connector's argv with the driver's own parser."""
    monkeypatch.setattr("sys.argv", ["hallucinate_binders.py", *argv[3:]])
    return driver.parse_args()


def test_the_driver_accepts_exactly_what_the_connector_launches(
    driver, configs, tmp_path: Path, monkeypatch
) -> None:
    loaded = load_configs(*configs)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 1)

    # argparse exits non-zero on an unknown or missing flag, so this failing
    # means the connector would have launched a job that dies immediately.
    parsed = parse(driver, spec.argv, monkeypatch)

    assert parsed.binder_length == loaded.model.sampling.binder_length
    assert parsed.task_id == 1
    assert parsed.n_designs == loaded.model.sampling.designs_per_job
    assert parsed.seed_base == loaded.model.sampling.seed_base
    assert parsed.max_runtime == loaded.model.sampling.max_runtime_hours
    assert parsed.soft_steps == loaded.model.sampling.optimizer.soft_steps
    assert Path(parsed.save_dir) == manifest.directory / "tasks" / "0001"
    assert Path(parsed.target_fasta) == loaded.general.target.sequence_fasta


def test_the_driver_writes_into_the_directory_the_adapter_reads(
    driver, configs, tmp_path: Path, monkeypatch
) -> None:
    """--save-dir and the manifest's task directory must be the same place."""
    loaded = load_configs(*configs)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)
    parsed = parse(driver, spec.argv, monkeypatch)

    save_dir = Path(parsed.save_dir)
    task = next(task for task in manifest.tasks if task.task_id == 0)

    assert save_dir == manifest.path(task.directory)
    assert save_dir / driver.DESIGNS_FILE == manifest.path(task.designs)
    assert save_dir / driver.STATUS_FILE == manifest.path(task.status)
    # And the connector promises Snakemake exactly those two files.
    assert set(spec.outputs) == {manifest.path(task.designs), manifest.path(task.status)}


def test_a_status_file_the_driver_wrote_is_read_back_by_the_adapter(
    driver, configs, tmp_path: Path
) -> None:
    """The driver's writer and the adapter's reader, with no fixture between."""
    loaded = load_configs(*configs)
    manifest = plan(loaded, tmp_path / "run", name="run")
    task_dir = manifest.path("tasks/0000")

    driver.write_status(task_dir, {
        "task_id": 0, "status": "partial",
        "started_at": "2026-08-28T11:00:00+00:00",
        "finished_at": "2026-08-28T12:00:00+00:00",
        "n_attempted": 3, "n_produced": 1,
        "output_file": driver.DESIGNS_FILE, "error": None,
    })
    (task_dir / driver.DESIGNS_FILE).write_text(json.dumps({
        "native_id": "task-0000-design-000000", "sequence": "ACDEFG",
        "seed": 0, "ranking_loss": -0.5,
    }) + "\n")
    (manifest.path("tasks/0001") / driver.DESIGNS_FILE).write_text("")

    collected = MosaicOutputAdapter().collect(manifest.directory, manifest.to_run_record())

    assert collected.run.status == "partial"
    assert collected.run.n_attempted == 3
    assert collected.run.count_details["tasks"]["0000"]["status"] == "partial"
    assert [d.native_id for d in collected.designs] == ["task-0000-design-000000"]


def test_status_is_written_atomically(driver, tmp_path: Path) -> None:
    """A reader must never see a half-written status file, or a stray .tmp."""
    driver.write_status(tmp_path, {"task_id": 0, "status": "succeeded"})

    assert json.loads((tmp_path / driver.STATUS_FILE).read_text())["status"] == "succeeded"
    assert list(tmp_path.glob("*.tmp")) == []


def test_status_write_replaces_rather_than_appends(driver, tmp_path: Path) -> None:
    driver.write_status(tmp_path, {"task_id": 0, "status": "partial"})
    driver.write_status(tmp_path, {"task_id": 0, "status": "succeeded"})

    assert json.loads((tmp_path / driver.STATUS_FILE).read_text())["status"] == "succeeded"


def test_both_fasta_readers_agree(driver, tmp_path: Path) -> None:
    """The driver cannot import bindocracy, so it parses FASTA a second time.

    If the two ever disagree, the binder is designed against a different
    sequence than the one preflight measured and the database recorded, and
    nothing downstream can detect it.
    """
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">dio3 cut construct\nacdefghikl\nMNPQRS\n\n")

    assert driver.read_fasta(str(fasta)) == read_single_fasta(fasta) == "ACDEFGHIKLMNPQRS"


def test_the_driver_names_designs_the_way_the_adapter_expects(
    driver, configs, tmp_path: Path
) -> None:
    """The native_id scheme is a contract between two files that never meet."""
    loaded = load_configs(*configs)
    manifest = plan(loaded, tmp_path / "run")
    task_dir = manifest.path("tasks/0001")

    # Exactly the format hallucinate_binders.py writes, for task 1 design 2.
    native_id = f"task-{1:04d}-design-{2:06d}"
    (task_dir / driver.DESIGNS_FILE).write_text(json.dumps({
        "native_id": native_id, "sequence": "ACDEFG", "seed": 100_002,
        "ranking_loss": -0.5,
    }) + "\n")
    driver.write_status(task_dir, {"task_id": 1, "status": "succeeded",
                                   "n_attempted": 1, "n_produced": 1})

    collected = MosaicOutputAdapter().collect(manifest.directory, manifest.to_run_record())

    assert [d.native_id for d in collected.designs] == [native_id]
    assert collected.run.count_details["tasks"]["0001"]["n_invalid"] == 0
