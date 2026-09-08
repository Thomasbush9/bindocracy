"""The scoring and filtering layer, tested where it is pure.

The engine is deliberately free of database and container dependencies, so
everything that decides a verdict can be exercised with a list of floats. That
is the property worth protecting: a threshold's behaviour should never require
a GPU to check.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bindocracy.adapters.base import CollectionError
from bindocracy.adapters.scoring import metric_records, registered_keys, spec_for
from bindocracy.filters.apply import (
    aggregate_metric,
    apply_filter_set,
    decisions_for_design,
    evaluate_threshold,
    metric_values,
)
from bindocracy.filters.models import FilterRule, FilterSet, Threshold
from bindocracy.runs.designset import (
    DesignSetError,
    build_design_set,
    read_fasta_entries,
    write_design_set,
)
from bindocracy.store.query import DesignQuery, DesignRow, MetricRow
from bindocracy.store.records import MetricStatus

WHEN = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def design_row(index: int, length: int = 80, tool: str = "mosaic") -> DesignRow:
    return DesignRow(
        design_id=f"design-{index:04d}",
        run_id="run-1",
        run_name=f"{tool}-run",
        tool=tool,
        native_id=f"task-0000-design-{index:06d}",
        sequence="A" * length,
        length=length,
        candidate_type="sequence",
        metadata={},
    )


def metric_row(name: str, value: float | None, replicate: int = 0, status: str = "ok"):
    return MetricRow(
        design_id="design-0000",
        run_id="score-1",
        name=name,
        value=value,
        replicate=replicate,
        status=status,
        direction="max",
    )


# --------------------------------------------------------------- design sets


def test_digest_depends_on_members_and_order_only():
    rows = [design_row(i) for i in range(4)]
    query = DesignQuery()
    first = build_design_set(rows, database="a.duckdb", query=query, created_at=WHEN)
    later = build_design_set(
        rows, database="b.duckdb", query=query, created_at=datetime(2027, 1, 1, tzinfo=UTC)
    )
    # A different database path and a different clock must not move the digest;
    # otherwise "the same designs" stops being a stable identity.
    assert first.digest == later.digest

    reordered = build_design_set(rows[::-1], database="a.duckdb", query=query, created_at=WHEN)
    assert reordered.digest != first.digest


def test_empty_design_set_is_refused():
    with pytest.raises(DesignSetError):
        build_design_set([], database="a.duckdb", query=DesignQuery())


def test_shards_are_contiguous_and_cover_the_set_exactly():
    rows = [design_row(i, length=70 + i) for i in range(20)]
    design_set = build_design_set(rows, database="a", query=DesignQuery(), created_at=WHEN)

    for num_shards in (1, 3, 7, 20):
        blocks = [design_set.shard(s, num_shards) for s in range(num_shards)]
        indices = [entry.index for block in blocks for entry in block]
        assert indices == list(range(20)), num_shards
        # Contiguity is what keeps a task's JIT recompilations down, so it is
        # a property worth asserting rather than assuming.
        for block in blocks:
            if block:
                got = [entry.index for entry in block]
                assert got == list(range(got[0], got[0] + len(got)))


def test_shard_index_is_bounds_checked():
    rows = [design_row(i) for i in range(4)]
    design_set = build_design_set(rows, database="a", query=DesignQuery(), created_at=WHEN)
    with pytest.raises(ValueError, match="outside"):
        design_set.shard(3, 3) and design_set.shard(5, 3)


def test_fasta_round_trips_and_leads_with_the_join_key(tmp_path):
    rows = [design_row(i, length=60 + i, tool="boltzgen") for i in range(3)]
    design_set = build_design_set(rows, database="a", query=DesignQuery(), created_at=WHEN)
    fasta_path, manifest_path = write_design_set(design_set, tmp_path)

    entries = read_fasta_entries(fasta_path)
    assert len(entries) == 3
    for entry, parsed in zip(design_set.entries, entries, strict=True):
        header, sequence = parsed
        assert sequence == entry.sequence
        # The driver parses the leading index and nothing else.
        assert int(header.split()[0]) == entry.index
    assert manifest_path.is_file()


def test_rewriting_the_same_set_is_a_no_op(tmp_path):
    rows = [design_row(i) for i in range(3)]
    design_set = build_design_set(rows, database="a", query=DesignQuery(), created_at=WHEN)
    first = write_design_set(design_set, tmp_path)
    assert write_design_set(design_set, tmp_path) == first


# ------------------------------------------------------------------ metrics


def test_failed_measurements_are_stored_not_dropped():
    records = metric_records(
        run_id="score-1",
        design_id="design-0000",
        model="boltz2",
        values={"iptm": 0.7, "iplddt": float("nan"), "bt_pae": None},
        replicate=0,
        measured_at=WHEN,
    )
    by_name = {record.name: record for record in records}
    assert set(by_name) == {"boltz2_iptm", "boltz2_iplddt", "boltz2_bt_pae"}
    assert by_name["boltz2_iptm"].status == MetricStatus.OK
    # NaN and None are evidence that something ran and produced nonsense, which
    # is a different fact from no row at all.
    assert by_name["boltz2_iplddt"].status == MetricStatus.FAILED
    assert by_name["boltz2_iplddt"].value is None
    assert by_name["boltz2_bt_pae"].status == MetricStatus.FAILED


def test_metric_ids_are_stable_across_reparsing():
    kwargs = {
        "run_id": "score-1", "design_id": "design-0000", "model": "boltz2",
        "values": {"iptm": 0.7}, "replicate": 0, "measured_at": WHEN,
    }
    assert metric_records(**kwargs)[0].metric_id == metric_records(**kwargs)[0].metric_id


def test_replicates_get_distinct_ids():
    ids = {
        metric_records(
            run_id="s", design_id="d", model="boltz2", values={"iptm": 0.5},
            replicate=rep, measured_at=WHEN,
        )[0].metric_id
        for rep in range(3)
    }
    assert len(ids) == 3


def test_unregistered_metric_is_refused():
    with pytest.raises(CollectionError, match="unregistered metric"):
        spec_for("interface_vibes")
    assert "iptm" in registered_keys()


def test_stored_name_carries_the_model():
    assert spec_for("iptm").stored_name("esmfold2") == "esmfold2_iptm"
    assert spec_for("iptm").stored_name("boltz2") == "boltz2_iptm"


# ------------------------------------------------------------------ filters


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [("mean", 3.0), ("median", 3.0), ("min", 1.0), ("max", 5.0), ("first", 1.0)],
)
def test_aggregations_are_distinct(aggregate, expected):
    assert aggregate_metric([1.0, 3.0, 5.0], aggregate) == expected


def test_non_ok_replicates_are_dropped_not_zeroed():
    rows = [
        metric_row("boltz2_iptm", 0.8, replicate=0),
        metric_row("boltz2_iptm", None, replicate=1, status="failed"),
        metric_row("boltz2_iptm", 0.9, replicate=2),
    ]
    values = metric_values(rows)
    # Zeroing the failure would give 0.567 and read as a measured bad design.
    assert values["boltz2_iptm"] == [0.8, 0.9]
    assert aggregate_metric(values["boltz2_iptm"], "mean") == pytest.approx(0.85)


def test_a_missing_metric_fails_rather_than_passes():
    threshold = Threshold(metric="boltz2_iptm", op=">=", value=0.6, aggregate="mean")
    result = evaluate_threshold(threshold, {})
    assert result.passed is False
    assert result.missing is True
    assert result.observed is None


def test_insufficient_replicates_fail():
    threshold = Threshold(
        metric="boltz2_iptm", op=">=", value=0.6, aggregate="mean", min_replicates=3
    )
    assert evaluate_threshold(threshold, {"boltz2_iptm": [0.9, 0.9]}).passed is False
    assert evaluate_threshold(threshold, {"boltz2_iptm": [0.9, 0.9, 0.9]}).passed is True


def test_rule_and_set_verdicts_are_both_recorded():
    filter_set = FilterSet(
        name="triage",
        rules=(
            FilterRule(
                name="confident",
                thresholds=(
                    Threshold(metric="boltz2_iptm", op=">=", value=0.6, aggregate="mean"),
                ),
            ),
            FilterRule(
                name="compact",
                thresholds=(
                    Threshold(metric="boltz2_mono_rg", op="<=", value=20.0, aggregate="mean"),
                ),
            ),
        ),
    )
    rows = [metric_row("boltz2_iptm", 0.8), metric_row("boltz2_mono_rg", 25.0)]
    records = decisions_for_design(
        run_id="f1", design_id="design-0000", filter_set=filter_set,
        rows=rows, created_at=WHEN,
    )
    by_name = {record.name: record for record in records}
    assert by_name["confident"].passed is True
    assert by_name["compact"].passed is False
    assert by_name["triage"].passed is False
    # The reason has to name the number that failed, not just that one did.
    failed = [t for t in by_name["compact"].reason["thresholds"] if not t["passed"]]
    assert failed[0]["observed"] == 25.0
    assert failed[0]["threshold"] == 20.0
    assert by_name["triage"].reason["failed_rules"] == ["compact"]


def test_non_gating_rules_annotate_without_excluding():
    filter_set = FilterSet(
        name="triage",
        rules=(
            FilterRule(
                name="confident",
                thresholds=(
                    Threshold(metric="boltz2_iptm", op=">=", value=0.6, aggregate="mean"),
                ),
            ),
            FilterRule(
                name="contacts_epitope",
                thresholds=(
                    Threshold(
                        metric="boltz2_epitope_coverage", op=">=", value=1.0, aggregate="mean"
                    ),
                ),
            ),
        ),
        gating_rules=("confident",),
    )
    records = decisions_for_design(
        run_id="f1", design_id="design-0000", filter_set=filter_set,
        rows=[metric_row("boltz2_iptm", 0.8)], created_at=WHEN,
    )
    by_name = {record.name: record for record in records}
    assert by_name["contacts_epitope"].passed is False
    # Recorded, but not allowed to exclude.
    assert by_name["triage"].passed is True


def test_unscored_designs_get_an_explicit_verdict():
    filter_set = FilterSet(
        name="triage",
        rules=(
            FilterRule(
                name="confident",
                thresholds=(
                    Threshold(metric="boltz2_iptm", op=">=", value=0.6, aggregate="mean"),
                ),
            ),
        ),
    )
    decisions, summary = apply_filter_set(
        run_id="f1",
        filter_set=filter_set,
        design_ids=["design-0000", "never-scored"],
        metrics=[metric_row("boltz2_iptm", 0.8)],
        created_at=WHEN,
    )
    # Absence and rejection look identical in a query and mean opposite things,
    # so every design in the set gets a row either way.
    assert {record.design_id for record in decisions} == {"design-0000", "never-scored"}
    assert summary["n_designs"] == 2
    assert summary["n_passed"] == 1
    assert summary["n_unscored"] == 1


def test_duplicate_thresholds_and_unknown_gating_rules_are_refused():
    with pytest.raises(ValueError, match="more than once"):
        FilterRule(
            name="confident",
            thresholds=(
                Threshold(metric="x", op=">=", value=0.1, aggregate="mean"),
                Threshold(metric="x", op="<=", value=0.9, aggregate="mean"),
            ),
        )
    rule = FilterRule(
        name="confident",
        thresholds=(Threshold(metric="x", op=">=", value=0.1, aggregate="mean"),),
    )
    with pytest.raises(ValueError, match="no such rule"):
        FilterSet(name="s", rules=(rule,), gating_rules=("nope",))
