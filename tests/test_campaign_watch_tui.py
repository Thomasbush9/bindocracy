"""Offline Campaign Watch contracts; no collectors, schedulers, or model jobs."""

import asyncio
import copy
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("textual")
from textual.containers import VerticalScroll
from textual.widgets import DataTable

SPEC = importlib.util.spec_from_file_location(
    "campaign_watch_tui", Path(__file__).resolve().parents[1] / "scripts/campaign_watch/tui.py"
)
tui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tui)


def stage(identifier="optimization", **changes):
    record = {
        "id": identifier, "name": identifier, "state": "running", "unit": "designs", "planned": 100,
        "valid": 25, "running_jobs": 1, "pending_jobs": 0, "failed_jobs": 0,
        "scheduler_warning_jobs": 0, "active_gpus": 1, "eta_seconds": 3600,
        "throughput_per_hour": 75, "warnings": [],
    }
    record.update(changes)
    return record


def component(arm="arm00", generator="tool0", **changes):
    record = stage(
        f"{arm}:{generator}", name=generator, arm=arm, generator=generator,
        state="not_submitted", phase="planned", valid=None,
        count_basis="reported", reported_produced=None, coordinate_ready=None,
        tasks_planned=2, tasks_observed=0, counts_reported_tasks=0,
        partial_jobs=0, unknown_jobs=0, failed_by_state={}, job_ids=[], run_ids=[],
        eta_seconds=None, throughput_per_hour=None, running_jobs=0, active_gpus=0,
    )
    record.update(changes)
    return record


def generation_stage(identifier="tier1", components=None):
    return stage(
        identifier, state="not_submitted", phase="planned", valid=None,
        planned=12600, count_basis="reported", reported_produced=None,
        coordinate_ready=None, eta_seconds=None, throughput_per_hour=None,
        running_jobs=0, active_gpus=0,
        components=components if components is not None else [
            component(f"arm{arm:02}", f"tool{tool}") for arm in range(18) for tool in range(7)
        ],
    )


def campaign(identifier="mosaic", stages=None):
    return {"id": identifier, "name": identifier, "root": "/trusted/campaign",
            "observed_at": datetime.now(UTC).isoformat(), "warnings": [],
            "stages": stages if stages is not None else [stage()]}


def snapshot(*campaigns, version=2):
    return {"schema_version": version, "generated_at": datetime.now(UTC).isoformat(),
            "refresh_seconds": 30, "collector": {"host": "cluster", "job_id": "123"},
            "errors": [], "campaigns": list(campaigns)}


def generation_campaign():
    return dict(campaign("generation", [generation_stage(f"tier{i}") for i in range(1, 5)]),
                kind="bindocracy_generation")


def gpu():
    return {"source": "Slurm accounting", "observed_at": datetime.now(UTC).isoformat(),
            "eligible_jobs": 2, "queried_jobs": 2, "sampled_jobs": 1, "low_util_jobs": 0,
            "memory_sampled_jobs": 1, "mean_percent": 50, "min_percent": 50, "max_percent": 50,
            "mean_memory_mib": 1024, "error": None}


def test_mosaic_v1_preserves_progress_runtime_and_telemetry():
    original = campaign()
    original["stages"][0].update(gpu_utilization=gpu(), runtime_minutes_by_arm={"arm": 2.5})
    data = snapshot(original, version=1)
    assert tui.validate_snapshot(data) == data
    record = tui.summary_stage(original)
    assert tui.units(record) == "25/100 designs (unique valid)"
    assert tui.progress(record) == "[##........] 25%"
    assert record["runtime_minutes_by_arm"] == {"arm": 2.5}


def test_generation_unknown_and_overshoot_are_not_validated_counts():
    record = generation_stage(components=[component()])
    assert tui.units(record) == "?/12,600 designs (reported)"
    assert tui.progress(record) == "?"
    assert tui.count_text(record["coordinate_ready"]) == "?"
    record.update(planned=100, reported_produced=125)
    data = snapshot(dict(campaign("generation", [record]), kind="bindocracy_generation"))
    assert tui.validate_snapshot(data)["errors"] == []
    assert tui.progress(record) == "[##########] 125%"
    assert tui.units(record) == "125/100 designs (reported)"
    record["reported_produced"] = 0
    assert tui.progress(record) == "[..........] 0%"
    record["planned"] = 0
    assert tui.progress(record) == "n/a"


def test_recovery_progress_is_bounded_without_hiding_raw_overshoot():
    record = generation_stage(components=[component()])
    record.update(planned=100, reported_produced=125, phase="recovery_complete")
    assert tui.progress(record) == "[##########] 100%"
    assert tui.units(record) == "125/100 designs (reported)"
    record["phase"] = "mixed: recovery_complete, generated"
    assert tui.progress(record) == "[##########] 100%"


def test_arm_aggregate_retains_partial_coverage_and_failure_state():
    known = component("arm", "tool0", reported_produced=120, state="needs_attention", phase="partial",
                      tasks_observed=2, counts_reported_tasks=1, partial_jobs=1,
                      failed_jobs=2, failed_by_state={"OUT_OF_MEMORY": 1, "TIMEOUT": 1},
                      job_ids=["123"], run_ids=["run-a"])
    unknown = component("arm", "tool1", unknown_jobs=1)
    record = tui.arm_records(generation_stage(components=[known, unknown]))[0]
    assert record["reported_produced"] == 120
    assert record["coordinate_ready"] is None
    assert record["state"] == "needs_attention"
    assert record["phase"].startswith("mixed:")
    assert record["partial_jobs"] == record["unknown_jobs"] == 1
    assert record["failed_by_state"] == {"OUT_OF_MEMORY": 1, "TIMEOUT": 1}
    assert record["job_ids"] == ["123"] and record["run_ids"] == ["run-a"]
    assert "1/2 generators" in record["warnings"][0]
    assert tui.arm_records(generation_stage(components=[unknown]))[0]["reported_produced"] is None


@pytest.mark.parametrize("change", [
    {"reported_produced": -1}, {"coordinate_ready": False}, {"valid": -1},
    {"state": "invented"}, {"tasks_observed": "2"}, {"arm": "bad\x1barm"},
    {"id": "x" * 257}, {"name": "x" * 257}, {"count_basis": "validated"},
    {"failed_by_state": {"FAILED": True}}, {"job_ids": [123]},
    {"run_ids": ["run\ncontrol"]}, {"counts_reported_tasks": -1},
    {"eta_sample_count": -1}, {"eta_basis": []},
])
def test_component_validation_isolates_campaign(change):
    bad = generation_campaign()
    bad["stages"][0]["components"][0].update(change)
    good = campaign()
    result = tui.validate_snapshot(snapshot(bad, good))
    assert result["campaigns"] == [good]
    assert len(result["errors"]) == 1


def test_duplicate_components_and_oversize_lists_are_isolated():
    bad = generation_campaign()
    child = bad["stages"][0]["components"][0]
    bad["stages"][0]["components"].append(dict(child, id="different-id"))
    assert tui.validate_snapshot(snapshot(bad))["campaigns"] == []
    bad["stages"][0]["components"] = [child] * 513
    assert tui.validate_snapshot(snapshot(bad))["campaigns"] == []


def test_invalid_campaign_retains_only_its_last_good_record():
    previous = tui.validate_snapshot(snapshot(generation_campaign(), campaign()))
    incoming = copy.deepcopy(previous)
    incoming["campaigns"][0]["stages"][0]["components"][0]["tasks_observed"] = -1
    incoming["campaigns"][1]["stages"][0]["valid"] = 30
    result = tui.validate_snapshot(incoming, previous)
    assert result["campaigns"][0]["stages"] == previous["campaigns"][0]["stages"]
    assert result["campaigns"][0]["observed_at"] == previous["campaigns"][0]["observed_at"]
    assert "Invalid campaign" in result["campaigns"][0]["_invalid"]
    assert result["campaigns"][1]["stages"][0]["valid"] == 30
    assert "_invalid" not in tui.validate_snapshot(previous)["campaigns"][0]


def test_bad_identity_does_not_hide_good_campaigns_or_duplicate_retained_rows():
    good = campaign()
    assert tui.validate_snapshot(snapshot({"id": []}, good))["campaigns"] == [good]
    previous = tui.validate_snapshot(snapshot(good))
    result = tui.validate_snapshot(snapshot(good, copy.deepcopy(good)), previous)
    assert len(result["campaigns"]) == 1
    assert result["campaigns"][0]["_invalid"]


@pytest.mark.parametrize("change", [
    {"schema_version": 3}, {"schema_version": True}, {"generated_at": "yesterday"},
    {"refresh_seconds": 0}, {"refresh_seconds": 86401}, {"errors": "bad"},
    {"collector": {"host": [], "job_id": "123"}}, {"campaigns": {}},
])
def test_envelope_remains_strict(change):
    data = snapshot(campaign())
    data.update(change)
    with pytest.raises(ValueError):
        tui.validate_snapshot(data)


@pytest.mark.parametrize("change", [{"generated_at": 123}, {"collector": []}])
def test_envelope_type_errors_are_explicit(change):
    data = snapshot(campaign())
    data.update(change)
    with pytest.raises(TypeError):
        tui.validate_snapshot(data)


@pytest.mark.parametrize("suffix", ["Z", "+00:00"])
def test_utc_timestamp_spellings_remain_supported(suffix):
    data = snapshot(campaign())
    data["generated_at"] = f"2026-11-05T12:00:00{suffix}"
    data["campaigns"][0]["observed_at"] = data["generated_at"]
    assert tui.validate_snapshot(data) == data


@pytest.mark.parametrize("change", [{"observed_at": 123}, {"gpu_utilization": []}])
def test_campaign_type_errors_retain_last_good_without_hiding_peers(change):
    previous = tui.validate_snapshot(snapshot(campaign("bad"), campaign("good")))
    incoming = copy.deepcopy(previous)
    incoming["campaigns"][0].update(change)
    incoming["campaigns"][1]["stages"][0]["valid"] = 30
    result = tui.validate_snapshot(incoming, previous)
    assert result["campaigns"][0]["stages"] == previous["campaigns"][0]["stages"]
    assert result["campaigns"][0]["observed_at"] == previous["campaigns"][0]["observed_at"]
    assert result["campaigns"][0]["_invalid"]
    assert result["campaigns"][1]["stages"][0]["valid"] == 30
    assert len(result["errors"]) == 1


def test_nonobject_campaign_is_quarantined():
    good = campaign()
    result = tui.validate_snapshot(snapshot([], good))
    assert result["campaigns"] == [good]
    assert "expected object" in result["errors"][0]


def test_campaign_gpu_validation_and_explicit_scope():
    item = generation_campaign()
    item["gpu_utilization"] = gpu()
    data = tui.validate_snapshot(snapshot(item))
    assert not data["errors"]
    app = tui.CampaignWatch("unused", None, 30)
    app.snapshot = data
    assert "[1/2] campaign" in app.gpu_summary(item["stages"][0], item)
    item["gpu_utilization"]["sampled_jobs"] = 3
    assert not tui.validate_snapshot(snapshot(item))["campaigns"]


def test_safe_text_and_bounded_transport(tmp_path):
    assert tui.plain("[bold]literal[/bold]\x1b\n") == "[bold]literal[/bold]  "
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot(campaign(), version=1)))
    assert asyncio.run(tui.fetch_snapshot(str(path)))["schema_version"] == 1
    path.write_bytes(b" " * (tui.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        asyncio.run(tui.fetch_snapshot(str(path)))
    path.write_text('{"schema_version":NaN}')
    with pytest.raises(ValueError, match="non-finite"):
        asyncio.run(tui.fetch_snapshot(str(path)))


class OfflineWatch(tui.CampaignWatch):
    def action_refresh(self):
        pass


@pytest.mark.parametrize("change", [{"generated_at": 123}, {"collector": []}])
def test_refresh_type_errors_keep_last_success_and_report_failure(tmp_path, caplog, change):
    path = tmp_path / "snapshot.json"
    original = snapshot(campaign())
    path.write_text(json.dumps(original))

    async def exercise():
        app = OfflineWatch(str(path), None, 30)
        async with app.run_test(size=(120, 30)) as pilot:
            await app.load_snapshot()
            await pilot.pause()
            last_success = app.last_success
            path.write_text(json.dumps({**original, **change}))
            app.fetching = True
            await app.load_snapshot()
            await pilot.pause()
            assert app.snapshot == original
            assert app.last_success == last_success
            assert app.fetch_error.startswith("TypeError:")
            assert not app.fetching

    asyncio.run(exercise())
    assert any(record.exc_info and record.exc_info[0] is TypeError for record in caplog.records)


def test_unexpected_fetch_fault_is_logged_and_keeps_last_snapshot(monkeypatch, caplog):
    async def failing_fetch(*_args):
        raise RuntimeError("unexpected transport fault")

    async def exercise():
        app = OfflineWatch("unused", None, 30)
        async with app.run_test(size=(120, 30)) as pilot:
            original = tui.validate_snapshot(snapshot(campaign()))
            app.snapshot = original
            app.render_snapshot()
            await pilot.pause()
            monkeypatch.setattr(tui, "fetch_snapshot", failing_fetch)
            app.fetching = True
            await app.load_snapshot()
            await pilot.pause()
            assert app.snapshot is original
            assert app.fetch_error == "RuntimeError: unexpected transport fault"
            assert not app.fetching

    asyncio.run(exercise())
    assert any(record.exc_info and record.exc_info[0] is RuntimeError for record in caplog.records)


def test_navigation_all_arms_tools_and_stable_refresh_selection():
    async def exercise():
        app = OfflineWatch("unused", None, 30)
        async with app.run_test(size=(200, 50)) as pilot:
            app.snapshot = tui.validate_snapshot(snapshot(generation_campaign(), campaign()))
            app.render_snapshot()
            await pilot.pause()
            table = app.query_one("#stages", DataTable)
            assert table.row_count == 4
            await pilot.press("enter", "j", "l")
            assert app.selected_stage == "tier2" and app.level == "arms"
            assert table.row_count == 18
            await pilot.press(*(["j"] * 17))
            assert app.selected_arm == "arm17"
            await pilot.press("enter")
            assert app.level == "generators" and table.row_count == 7
            await pilot.press(*(["j"] * 6))
            assert app.selected_generator == "arm17:tool6"
            selected = (app.selected_campaign, app.selected_stage, app.selected_arm, app.selected_generator)
            refreshed = copy.deepcopy(app.snapshot)
            refreshed["campaigns"].reverse()
            refreshed["campaigns"][1]["stages"].reverse()
            for record in refreshed["campaigns"][1]["stages"]:
                record["components"].reverse()
            app.snapshot = tui.validate_snapshot(refreshed, app.snapshot)
            app.render_snapshot()
            app.update_ages()
            await pilot.pause()
            assert app.level == "generators"
            assert (app.selected_campaign, app.selected_stage, app.selected_arm, app.selected_generator) == selected
            await pilot.press("enter")
            assert isinstance(app.focused, VerticalScroll)
            await pilot.press("escape")
            assert app.focused is table and app.level == "generators"
            await pilot.press("h")
            assert app.level == "arms" and app.selected_arm == "arm17"
            await pilot.press("escape")
            assert app.level == "stages" and app.selected_stage == "tier2"
            await pilot.press("h")
            assert app.focused is app.query_one("#campaigns", DataTable)
            await pilot.press("k", "enter")
            assert app.selected_campaign == "mosaic" and app.level == "stages"
            assert table.row_count == 1
            await pilot.press("enter")
            assert isinstance(app.focused, VerticalScroll)
    asyncio.run(exercise())


def test_metadata_only_envelope_has_no_collector_allocation():
    data = snapshot(generation_campaign())
    data["collector"].update(job_id=None, mode="metadata_only")
    assert tui.validate_snapshot(data) == data
    data["schema_version"] = 1
    with pytest.raises(ValueError, match="job_id"):
        tui.validate_snapshot(data)


def test_detail_pane_separates_evidence_phase_tasks_and_identifiers(monkeypatch):
    async def exercise():
        app = OfflineWatch("unused", None, 30)
        async with app.run_test(size=(200, 50)) as pilot:
            child = component(reported_produced=120, planned=100, state="needs_attention",
                              phase="partial", tasks_observed=2, counts_reported_tasks=1,
                              partial_jobs=1, unknown_jobs=2, failed_jobs=1,
                              failed_by_state={"OUT_OF_MEMORY": 1}, job_ids=["123"], run_ids=["run-abc"])
            child.update(eta_sample_count=0, eta_basis="Failed or partial tasks prevent calibration.")
            item = dict(campaign("generation", [generation_stage(components=[child])]),
                        kind="bindocracy_generation", gpu_utilization=gpu())
            app.snapshot = tui.validate_snapshot(snapshot(item))
            app.render_snapshot()
            await pilot.press("l", "l", "l")
            messages = []
            monkeypatch.setattr(app.query_one("#details"), "update", messages.append)
            app.render_details()
            detail = messages[-1]
            for expected in ("120/100 designs (reported)", "Coordinate-ready: ?", "Phase: partial",
                             "Tasks observed/planned: 2/2", "Tasks with reported counts: 1",
                             "Partial jobs: 1", "Unknown jobs: 2", "OUT_OF_MEMORY=1",
                             "Job IDs: 123", "Run IDs: run-abc", "excludes queue wait",
                             "GPU scope: entire campaign", "samples 1/2",
                             "planned-task execution, not output quota", "ETA calibration samples: 0",
                             "Failed or partial tasks prevent calibration."):
                assert expected in detail
            assert "unique valid" not in detail
            assert "ETA: unavailable" in detail
    asyncio.run(exercise())


def test_independent_failed_unit_remains_visible_beside_running_unit(monkeypatch):
    async def exercise():
        app = OfflineWatch("same-snapshot.json", None, 30)
        async with app.run_test(size=(200, 50)) as pilot:
            failed = component(generator="protein_hunter", state="needs_attention", phase="blocked",
                               warnings=[("Independent unit tier1--arm00--protein_hunter: failed. "
                                          "MissingOutputException: task0146; CPU supervisor 200 "
                                          "excluded from worker/GPU totals.")])
            running = component(generator="boltzgen", state="running", phase="generating",
                                reported_produced=10, running_jobs=1, active_gpus=1,
                                job_ids=["901"], tasks_observed=2, counts_reported_tasks=1)
            tier = generation_stage(components=[failed, running])
            tier.update(state="running", phase="generating", running_jobs=1, active_gpus=1)
            item = dict(campaign("generation", [tier]), kind="bindocracy_generation")
            app.snapshot = tui.validate_snapshot(snapshot(item))
            assert not app.snapshot["errors"]
            app.render_snapshot()
            await pilot.press("l", "l", "l")
            assert app.level == "generators"
            assert [(row["state"], row["active_gpus"]) for row in app.visible_records()] == [
                ("needs_attention", 0), ("running", 1)]
            messages = []
            monkeypatch.setattr(app.query_one("#details"), "update", messages.append)
            app.render_details()
            assert "MissingOutputException: task0146" in messages[-1]
            assert "Phase: blocked" in messages[-1]
            assert "excluded from worker/GPU totals" in messages[-1]
            await pilot.press("j")
            app.render_details()
            assert "Phase: generating" in messages[-1]
            assert "Job IDs: 901" in messages[-1]
            assert "10/100 designs (reported)" in messages[-1]
    asyncio.run(exercise())


def test_generation_task_eta_accepts_calibration_without_inventing_arm_eta():
    child = component(state="running", phase="generating", running_jobs=1, active_gpus=1,
                      eta_seconds=1800, eta_sample_count=5,
                      eta_basis="Five successful same-arm/tool tasks at current running concurrency.",
                      throughput_per_hour=20, reported_produced=10)
    item = dict(campaign("generation", [generation_stage(components=[child])]),
                kind="bindocracy_generation")
    result = tui.validate_snapshot(snapshot(item))
    assert not result["errors"]
    assert child["eta_seconds"] == 1800
    assert tui.arm_records(item["stages"][0])[0]["eta_seconds"] is None
