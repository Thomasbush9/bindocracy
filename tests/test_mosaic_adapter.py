from __future__ import annotations

from pathlib import Path

import pytest
from conftest import design_line, write_task

from bindocracy.adapters import collect_run
from bindocracy.adapters.base import CollectionError
from bindocracy.adapters.mosaic import METRIC_NAME, MosaicOutputAdapter
from bindocracy.config import load_mosaic_configs
from bindocracy.tools import plan as plan_tool_run


def plan(configs: tuple[Path, Path], run_dir: Path, **sampling):
    """Plan a run whose task and design counts the test controls."""
    general_path, model_path = configs
    if sampling:
        import yaml

        raw = yaml.safe_load(model_path.read_text())
        raw["sampling"].update(sampling)
        model_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return plan_tool_run(load_mosaic_configs(general_path, model_path), run_dir)


def collect(manifest):
    return MosaicOutputAdapter().collect(manifest.directory, manifest.to_run_record())


def test_one_task_and_design(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=1)
    write_task(manifest.directory, 0, [design_line(0, 0)], status={})

    collected = collect(manifest)

    assert collected.run.status == "succeeded"
    assert (collected.run.n_requested, collected.run.n_produced) == (1, 1)
    design = collected.designs[0]
    assert design.native_id == "task-0000-design-000000"
    assert design.sequence == "ACDEFG"
    assert design.length == 6
    assert design.seed == 0
    metric = collected.metrics[0]
    assert (metric.name, metric.direction, metric.value) == (METRIC_NAME, "min", -0.5)
    assert metric.design_id == design.design_id
    assert metric.run_id == collected.run.run_id


def test_several_tasks_and_designs(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=2, designs_per_job=3)
    for task_id in (0, 1):
        write_task(
            manifest.directory,
            task_id,
            [design_line(task_id, index) for index in range(3)],
            status={},
        )

    collected = collect(manifest)

    assert collected.run.status == "succeeded"
    assert collected.run.n_produced == 6
    assert len({design.design_id for design in collected.designs}) == 6
    assert len(collected.metrics) == 6


def test_empty_output_is_a_failed_run(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=2)
    write_task(manifest.directory, 0, [], status={"status": "failed", "n_produced": 0,
                                                  "error": "RuntimeError: CUDA OOM"})

    collected = collect(manifest)

    assert collected.run.status == "failed"
    assert collected.designs == ()
    assert collected.run.n_produced == 0


def test_truncated_final_line_keeps_earlier_designs(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=3)
    body = design_line(0, 0) + "\n" + design_line(0, 1) + "\n" + '{"native_id": "task-00'
    write_task(manifest.directory, 0, [], text=body,
               status={"status": "partial", "n_attempted": 3, "n_produced": 2})

    collected = collect(manifest)

    assert collected.run.status == "partial"
    assert [design.native_id for design in collected.designs] == [
        "task-0000-design-000000",
        "task-0000-design-000001",
    ]
    assert collected.run.count_details["tasks"]["0000"]["n_truncated"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"sequence": "ACDEF1G"},
        {"sequence": ""},
        {"ranking_loss": "not-a-number"},
        {"ranking_loss": None},
        {"native_id": "task-0001-design-000001"},  # wrong task for this directory
    ],
    ids=["invalid-sequence", "empty-sequence", "loss-not-numeric", "loss-null",
         "wrong-task"],
)
def test_invalid_records_are_skipped_not_fatal(configs, tmp_path: Path, bad: dict) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=2)
    write_task(
        manifest.directory,
        0,
        [design_line(0, 0), design_line(0, 1, **bad)],
        status={"status": "partial"},
    )

    collected = collect(manifest)

    assert [design.native_id for design in collected.designs] == ["task-0000-design-000000"]
    assert collected.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_duplicate_native_id_is_rejected_once(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=2)
    write_task(manifest.directory, 0, [design_line(0, 0), design_line(0, 0)],
               status={"status": "partial"})

    collected = collect(manifest)

    assert len(collected.designs) == 1
    assert collected.run.count_details["tasks"]["0000"]["n_invalid"] == 1


def test_duplicate_sequence_with_distinct_ids_is_kept(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=2)
    write_task(
        manifest.directory,
        0,
        [design_line(0, 0, sequence="ACDEFG"), design_line(0, 1, sequence="ACDEFG")],
        status={},
    )

    collected = collect(manifest)

    assert len(collected.designs) == 2
    assert len({design.sequence for design in collected.designs}) == 1
    assert len({design.design_id for design in collected.designs}) == 2


def test_missing_task_output_leaves_the_run_partial(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=2, designs_per_job=1)
    write_task(manifest.directory, 0, [design_line(0, 0)], status={})
    # task 1 never ran: no designs.jsonl, no status.json

    collected = collect(manifest)

    assert collected.run.status == "partial"
    assert collected.run.n_produced == 1
    assert collected.run.count_details["tasks"]["0001"]["status"] == "missing"


def test_partial_task_keeps_completed_designs(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=4)
    write_task(manifest.directory, 0, [design_line(0, 0), design_line(0, 1)],
               status={"status": "partial", "n_attempted": 3, "n_produced": 2})

    collected = collect(manifest)

    assert collected.run.status == "partial"
    assert (collected.run.n_attempted, collected.run.n_produced) == (3, 2)


def test_artifacts_cover_outputs_status_logs_and_driver(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=1)
    write_task(manifest.directory, 0, [design_line(0, 0)], status={})

    collected = collect(manifest)

    kinds = {artifact.kind for artifact in collected.artifacts}
    assert kinds == {"native_designs", "task_status", "log", "driver_script"}
    assert all(not Path(artifact.uri).is_absolute() for artifact in collected.artifacts)


def test_recollection_is_deterministic(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=2, designs_per_job=2)
    for task_id in (0, 1):
        write_task(manifest.directory, task_id,
                   [design_line(task_id, index) for index in range(2)], status={})

    first = collect_run(manifest.directory / "run.json")
    second = collect_run(manifest.directory / "run.json")

    assert first.content_hash() == second.content_hash()


def test_collect_requires_a_manifest(configs, tmp_path: Path) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=1)
    (manifest.directory / "run.json").unlink()

    with pytest.raises(CollectionError, match="missing run manifest"):
        collect(manifest)


def test_succeeded_predicate_tracks_the_output_not_the_exit_code(
    configs, tmp_path: Path
) -> None:
    manifest = plan(configs, tmp_path / "run", jobs=1, designs_per_job=2)
    adapter = MosaicOutputAdapter()

    write_task(manifest.directory, 0, [design_line(0, 0)], status={"status": "partial"})
    assert adapter.succeeded(manifest.directory) is False

    write_task(manifest.directory, 0, [design_line(0, 0), design_line(0, 1)], status={})
    assert adapter.succeeded(manifest.directory) is True
