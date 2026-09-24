"""Selecting designs out of the database, and filtering them into verdicts.

The half of the scoring stage that was DRAFT: `designset.py` could freeze rows
and `store/query.py` could select them, but nothing joined the two, so every
design set was built by hand. An optimization run reads what a filter chose, so
this is now load-bearing.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
import yaml

from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig
from bindocracy.filters.config import FilterConfig
from bindocracy.filters.run import FilterRunError, run_filter
from bindocracy.runs import ingest_collected
from bindocracy.runs.selection import SelectionError, build, read_only, resolve_run_ids
from bindocracy.store import CampaignStore, create_database
from bindocracy.store.query import DesignQuery, select_designs
from bindocracy.store.records import (
    ArtifactRecord,
    CandidateType,
    CollectedRun,
    ConfigRecord,
    DesignRecord,
    MetricDirection,
    MetricRecord,
    MetricStatus,
    RunKind,
    RunRecord,
    RunStatus,
)

SEQUENCES = ["MKTVLIWAFG", "MKTAAAGSDE", "GGGGSGGGGS", "MKTVLIWAFA"]


def _config_record(general: GeneralConfig, name: str = "fixture") -> ConfigRecord:
    return ConfigRecord(
        general_name=general.campaign.name,
        general_schema_version=general.schema_version,
        general_config_json=general.model_dump(mode="json"),
        model_name=name,
        tool="mosaic",
        model_schema_version=1,
        model_config_json={"name": name, "tool": "mosaic"},
    )


@pytest.fixture
def campaign(tmp_path, optimize_config_files):
    """A database with four designs, two scoring runs, and saved poses.

    Built by hand rather than by running anything: the point is to exercise
    selection and filtering, and a real scoring run would need a GPU.
    """
    general_path, _ = optimize_config_files
    general = load_yaml(general_path, GeneralConfig)
    database = create_database(tmp_path / "campaign.duckdb")
    config = _config_record(general)

    generate = RunRecord(
        run_id="gen", name="mosaic-run", tool="mosaic", kind=RunKind.GENERATE,
        model_config_id=config.model_config_id, status=RunStatus.SUCCEEDED,
        n_requested=4, n_attempted=4, n_produced=4, output_uri=str(tmp_path / "gen"),
    )
    designs = tuple(
        DesignRecord(
            design_id=f"d{index}", run_id="gen", native_id=f"n{index}",
            candidate_type=CandidateType.SEQUENCE, sequence=sequence,
        )
        for index, sequence in enumerate(SEQUENCES)
    )

    # Poses on disk, pointed at by run-dir-relative URIs -- the convention.
    poses = tmp_path / "scorerun"
    (poses / "tasks" / "0000").mkdir(parents=True)
    artifacts = []
    for index in range(len(SEQUENCES)):
        relative = f"tasks/0000/pose-{index}.pdb"
        (poses / relative).write_text("ATOM\nEND\n")
        artifacts.append(ArtifactRecord(
            artifact_id=f"a{index}", run_id="score", design_id=f"d{index}",
            kind="predicted_structure", uri=relative,
            metadata={"model": "boltz2", "condition": "complex", "replicate": 0},
        ))

    score = RunRecord(
        run_id="score", name="score-boltz2", tool="scorer", kind=RunKind.EVALUATE,
        model_config_id=config.model_config_id, status=RunStatus.SUCCEEDED,
        n_requested=4, n_attempted=4, n_produced=4, output_uri=str(poses),
    )
    metrics = tuple(
        MetricRecord(
            metric_id=f"m{index}", run_id="score", design_id=f"d{index}",
            name="boltz2_iptm", value=value, replicate=0,
            status=MetricStatus.OK, direction=MetricDirection.MAX,
        )
        for index, value in enumerate([0.9, 0.7, 0.2, 0.85])
    )

    with CampaignStore(database) as store:
        store.ingest(CollectedRun(run=generate, designs=designs), configs=[config])
        store.ingest(CollectedRun(run=score, artifacts=tuple(artifacts), metrics=metrics))
    return database, general, general_path


def _filter_yaml(path: Path, manifest: Path, *, threshold: float, name="worth_it") -> Path:
    path.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "policy", "tool": "filter",
        "design_set": str(manifest),
        "evaluator_runs": ["score-boltz2"],
        "filter_set": {
            "name": name,
            "rules": [{
                "name": "confident",
                "thresholds": [{
                    "metric": "boltz2_iptm", "op": ">=",
                    "value": threshold, "aggregate": "mean",
                }],
            }],
        },
    }, sort_keys=False))
    return path


def _apply(database, general, general_path, filter_path, out: Path):
    config = load_yaml(filter_path, FilterConfig)
    collected, record = run_filter(
        database=database, general=general, config=config,
        output_dir=out, general_source=general_path,
    )
    ingest_collected(database, collected, config=record)
    return collected


# --- freezing a query -------------------------------------------------------


def test_a_query_freezes_into_a_content_addressed_set(campaign, tmp_path) -> None:
    database, _, _ = campaign
    design_set, fasta, manifest = build(database, DesignQuery(), tmp_path / "sets")
    assert design_set.n_designs == 4
    assert fasta.is_file() and manifest.is_file()
    assert fasta.stem == manifest.stem == design_set.digest.removeprefix("sha256:")[:16]


def test_the_same_query_twice_is_the_same_set(campaign, tmp_path) -> None:
    """The digest covers the members and their order and nothing else, so
    re-running a query a minute later is recognisably the same set rather than
    a new one."""
    database, _, _ = campaign
    first, _, _ = build(database, DesignQuery(), tmp_path / "sets")
    second, _, _ = build(database, DesignQuery(), tmp_path / "sets")
    assert first.digest == second.digest
    assert first.scope_id == second.scope_id


def test_an_empty_result_is_refused_and_names_the_query(campaign, tmp_path) -> None:
    """A set of zero designs is never what anybody meant, and the message has
    to say which clause emptied it."""
    database, _, _ = campaign
    with pytest.raises(SelectionError, match="min_length"):
        build(database, DesignQuery(min_length=500), tmp_path / "sets")


def test_a_missing_database_is_refused_rather_than_created(tmp_path) -> None:
    """DuckDB creates a file when asked to open a missing one, so without this
    check a typo'd path yields an empty database and a query that legitimately
    returns nothing -- which reads as 'no designs passed the filter'."""
    with pytest.raises(SelectionError, match="no campaign database"):
        build(tmp_path / "absent.duckdb", DesignQuery(), tmp_path / "sets")


# --- filtering --------------------------------------------------------------


def test_a_filter_writes_one_verdict_per_design_plus_the_set(campaign, tmp_path) -> None:
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    collected = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.8),
        tmp_path / "run",
    )
    assert collected.run.n_requested == 4
    assert collected.run.n_passed == 2  # 0.9 and 0.85
    # One row per rule, plus one for the set as a whole.
    assert len(collected.decisions) == 4 * 2


def test_an_unscored_design_fails_rather_than_passes(campaign, tmp_path) -> None:
    """Absence of evidence is not evidence. The alternative silently admits
    every design a scorer never reached."""
    database, general, general_path = campaign
    with CampaignStore(database) as store:
        config = _config_record(general, "extra")
        store.ingest(
            CollectedRun(
                run=RunRecord(
                    run_id="gen2", name="mosaic-run-2", tool="mosaic",
                    kind=RunKind.GENERATE, model_config_id=config.model_config_id,
                    status=RunStatus.SUCCEEDED, n_requested=1, n_attempted=1,
                    n_produced=1,
                ),
                designs=(DesignRecord(
                    design_id="dx", run_id="gen2", native_id="nx",
                    candidate_type=CandidateType.SEQUENCE, sequence="WWWWWWWWWW",
                ),),
            ),
            configs=[config],
        )
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    collected = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.0),
        tmp_path / "run",
    )
    assert collected.run.count_details["n_unscored"] == 1
    verdicts = {
        record.design_id: record.passed
        for record in collected.decisions
        if record.name == "worth_it"
    }
    assert verdicts["dx"] is False


def test_a_threshold_on_an_absent_metric_is_refused(campaign, tmp_path) -> None:
    """It would fail every design, which reads as a strict filter rather than
    a typo. The suggestion is what makes the message useful."""
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    path = _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.5)
    path.write_text(path.read_text().replace("boltz2_iptm", "iptm"))
    with pytest.raises(FilterRunError, match="did you mean"):
        _apply(database, general, general_path, path, tmp_path / "run")


def test_an_evaluator_run_that_does_not_exist_is_refused(campaign, tmp_path) -> None:
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    path = _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.5)
    path.write_text(path.read_text().replace("score-boltz2", "score-typo"))
    with pytest.raises(FilterRunError, match="no evaluate run named"):
        _apply(database, general, general_path, path, tmp_path / "run")


def test_re_applying_the_same_policy_is_a_no_op(campaign, tmp_path) -> None:
    """A filter has no run directory to key restart-safety on, so its run_id is
    derived from the policy, the set and the evaluators. Without that, a second
    `filter apply` writes a second set of identical verdicts under a new run --
    and a design set built from 'passed worth_it' would then depend on which
    duplicate a query happened to see."""
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    path = _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.8)

    first = _apply(database, general, general_path, path, tmp_path / "r1")
    config = load_yaml(path, FilterConfig)
    second, record = run_filter(
        database=database, general=general, config=config, output_dir=tmp_path / "r2"
    )
    assert second.run.run_id == first.run.run_id
    # Timestamps differ between the two; the verdicts do not, and it is the
    # verdicts that identity is about.
    assert second.content_hash() != first.content_hash()
    assert second.verdict_hash() == first.verdict_hash()
    assert ingest_collected(database, second, config=record) is False


def test_a_changed_threshold_is_a_new_run_beside_the_old_one(campaign, tmp_path) -> None:
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    loose = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "a.yaml", manifest, threshold=0.5), tmp_path / "r1",
    )
    strict = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "b.yaml", manifest, threshold=0.8), tmp_path / "r2",
    )
    assert loose.run.run_id != strict.run.run_id
    assert loose.run.n_passed == 3 and strict.run.n_passed == 2


# --- selecting what a filter chose ------------------------------------------


def test_designs_can_be_selected_by_what_a_filter_passed(campaign, tmp_path) -> None:
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    collected = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.8), tmp_path / "run",
    )
    chosen, _, _ = build(
        database,
        DesignQuery(passed_filter=("worth_it",), filter_runs=(collected.run.run_id,)),
        tmp_path / "sets",
    )
    assert chosen.n_designs == 2
    assert {entry.sequence for entry in chosen.entries} == {"MKTVLIWAFG", "MKTVLIWAFA"}


def test_a_filter_selection_must_name_the_filter_run() -> None:
    """A rule name alone is ambiguous the moment a policy has been revised."""
    with pytest.raises(ValueError, match="passed_filter requires filter_runs"):
        DesignQuery(passed_filter=("worth_it",))


def test_naming_a_filter_run_without_a_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="narrows nothing"):
        DesignQuery(filter_runs=("some-run",))


def test_filter_run_identity_keeps_revised_policies_separate(campaign, tmp_path) -> None:
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    loose = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "a.yaml", manifest, threshold=0.5), tmp_path / "r1",
    )
    strict = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "b.yaml", manifest, threshold=0.8), tmp_path / "r2",
    )

    with pytest.raises(SelectionError):
        build(
            database,
            DesignQuery(passed_filter=("worth_it",), filter_runs=("policy",)),
            tmp_path / "sets",
        )
    for result, expected in ((loose, {"d0", "d1", "d3"}), (strict, {"d0", "d3"})):
        chosen, _, _ = build(
            database,
            DesignQuery(passed_filter=("worth_it",), filter_runs=(result.run.run_id,)),
            tmp_path / result.run.run_id,
        )
        assert {entry.design_id for entry in chosen.entries} == expected


def test_the_resolved_run_id_is_what_the_set_records(campaign, tmp_path) -> None:
    """A manifest naming a run by a name two runs share does not describe a
    reproducible selection."""
    database, general, general_path = campaign
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    collected = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.8), tmp_path / "run",
    )
    chosen, _, chosen_manifest = build(
        database,
        DesignQuery(passed_filter=("worth_it",), filter_runs=("policy",)),
        tmp_path / "sets",
    )
    recorded = json.loads(chosen_manifest.read_text())["query"]["filter_runs"]
    assert recorded == [collected.run.run_id]
    assert chosen.query["filter_runs"] == [collected.run.run_id]


def test_a_design_with_no_verdict_is_not_selected(campaign, tmp_path) -> None:
    """`filters/models.py` rule 2 -- a missing verdict is not a pass -- applied
    at selection time as well as at filter time."""
    database, general, general_path = campaign
    _, _, manifest = build(
        database, DesignQuery(min_length=10, max_length=10), tmp_path / "sets"
    )
    collected = _apply(
        database, general, general_path,
        _filter_yaml(tmp_path / "f.yaml", manifest, threshold=0.0), tmp_path / "run",
    )
    with read_only(database) as connection:
        everything = select_designs(connection, DesignQuery())
        passed = select_designs(connection, DesignQuery(
            passed_filter=("worth_it",), filter_runs=(collected.run.run_id,)
        ))
    assert len(passed) == len(everything) == 4


def test_run_ids_resolve_as_well_as_names(campaign) -> None:
    database, _, _ = campaign
    with read_only(database) as connection:
        assert resolve_run_ids(connection, ("score-boltz2",), kind="evaluate") == ("score",)
        assert resolve_run_ids(connection, ("score",), kind="evaluate") == ("score",)


# --- the parent poses an optimizer reads ------------------------------------


def test_parent_structures_resolve_against_the_producing_runs_directory(
    campaign, tmp_path, optimize_config_files
) -> None:
    """`artifacts.uri` is RUN-DIRECTORY-RELATIVE for every tool here, which is
    what lets a run directory be moved. Reading the column as a path finds
    nothing; it has to be joined to the producing run's `output_uri`."""
    from bindocracy.tools.optimize.config import OptimizeConfig
    from bindocracy.tools.optimize.preflight import preflight_optimize

    database, general, _ = campaign
    _, model_path = optimize_config_files
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")

    model = load_yaml(model_path, OptimizeConfig).model_copy(update={
        "design_set": manifest,
        "inputs": ("sequence", "structure"),
        "structures_from": "boltz2",
    })
    pre = preflight_optimize(general, model)
    assert len(pre.structures) == 4
    assert pre.n_unfolded == 0
    for path in pre.structures.values():
        assert Path(path).is_absolute() and Path(path).is_file()


def test_parents_with_no_pose_refuse_the_run_by_default(
    campaign, tmp_path, optimize_config_files
) -> None:
    """A run that silently optimizes a subset reports a smaller number with
    nothing saying why."""
    from bindocracy.tools.optimize.config import OptimizeConfig
    from bindocracy.tools.optimize.preflight import (
        OptimizePreflightError,
        preflight_optimize,
    )

    database, general, _ = campaign
    _, model_path = optimize_config_files
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    model = load_yaml(model_path, OptimizeConfig).model_copy(update={
        "design_set": manifest,
        "inputs": ("sequence", "structure"),
        "structures_from": "chai1",  # nothing scored with it
    })
    with pytest.raises(OptimizePreflightError, match="have no 'chai1' pose"):
        preflight_optimize(general, model)

    allowed = model.model_copy(update={"allow_unfolded": True})
    pre = preflight_optimize(general, allowed)
    assert pre.structures == {} and pre.n_unfolded == 4


def test_parent_metrics_are_averaged_over_replicates(
    campaign, tmp_path, optimize_config_files
) -> None:
    from bindocracy.tools.optimize.config import OptimizeConfig
    from bindocracy.tools.optimize.preflight import preflight_optimize

    database, general, _ = campaign
    _, model_path = optimize_config_files
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    model = load_yaml(model_path, OptimizeConfig).model_copy(update={
        "design_set": manifest,
        "inputs": ("sequence", "metrics"),
        "metric_inputs": ("boltz2_iptm",),
    })
    pre = preflight_optimize(general, model)
    assert len(pre.metrics) == 4
    assert sorted(row["boltz2_iptm"] for row in pre.metrics.values()) == [
        0.2, 0.7, 0.85, 0.9
    ]


def test_a_metric_input_that_does_not_exist_is_refused(
    campaign, tmp_path, optimize_config_files
) -> None:
    from bindocracy.tools.optimize.config import OptimizeConfig
    from bindocracy.tools.optimize.preflight import (
        OptimizePreflightError,
        preflight_optimize,
    )

    database, general, _ = campaign
    _, model_path = optimize_config_files
    _, _, manifest = build(database, DesignQuery(), tmp_path / "sets")
    model = load_yaml(model_path, OptimizeConfig).model_copy(update={
        "design_set": manifest,
        "inputs": ("sequence", "metrics"),
        "metric_inputs": ("iptm",),
    })
    with pytest.raises(OptimizePreflightError, match="no rows for"):
        preflight_optimize(general, model)


def test_selection_uses_a_read_only_connection(campaign) -> None:
    """A selection must not be able to modify the campaign it selects from, and
    a read-only connection is a cheaper guarantee than a code review."""
    database, _, _ = campaign
    with read_only(database) as connection, pytest.raises(duckdb.Error):
        connection.execute("DELETE FROM designs")
