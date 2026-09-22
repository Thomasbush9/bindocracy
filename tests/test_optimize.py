"""The custom-optimization seam: contract, config, preflight, driver, adapter.

Most of what is pinned here is a refusal. An optimizer writes to the `designs`
table, so the interesting behaviour is not what it stores but what it declines
to store and still reports honestly.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig
from bindocracy.tools import plugin_for
from bindocracy.tools.optimize.config import OptimizeConfig
from bindocracy.tools.optimize.contract import (
    ContractError,
    normalize_sequence,
    read_outputs,
)
from bindocracy.tools.optimize.preflight import OptimizePreflightError, preflight_optimize

DRIVER = Path(__file__).resolve().parents[1] / "drivers" / "optimize" / "run_optimizer.py"
DECLARED = {"loss": "min", "n_mutations": "none"}


@pytest.fixture
def optimize_driver():
    spec = importlib.util.spec_from_file_location("run_optimizer", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path: Path, *lines: dict) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def parse(path: Path, *, n_parents: int = 3, max_children: int = 2):
    return read_outputs(
        path, n_parents=n_parents, max_children=max_children, declared=DECLARED
    )


# --- the contract: a bad row is rejected, not fatal -------------------------


def test_a_valid_child_is_accepted(tmp_path) -> None:
    parsed, counts = parse(rows(
        tmp_path / "c.jsonl",
        {"parent_index": 0, "child": 0, "sequence": "MKTV", "metrics": {"loss": 0.3}},
    ))
    assert counts["n_children"] == 1
    assert parsed[0].sequence == "MKTV"
    assert parsed[0].metrics == {"loss": 0.3}


def test_one_bad_row_does_not_lose_the_good_ones(tmp_path) -> None:
    """The behaviour the whole design turns on.

    One non-canonical sequence out of many is a bug in one branch of a script.
    Aborting the file would cost a GPU-day to punish a typo; storing it anyway
    would put an unorderable sequence in `designs` forever. So: reject the row,
    count why, keep the rest.
    """
    parsed, counts = parse(rows(
        tmp_path / "c.jsonl",
        {"parent_index": 0, "sequence": "MKTV"},
        {"parent_index": 1, "sequence": "MKXV"},
        {"parent_index": 2, "sequence": "MKTA"},
    ))
    assert counts["n_children"] == 2
    assert counts["rejected"] == {"sequence_not_canonical": 1}
    assert [row.parent_index for row in parsed] == [0, 2]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({"parent_index": 0, "sequence": "MKXV"}, "sequence_not_canonical"),
        ({"parent_index": 0, "sequence": ""}, "sequence_not_canonical"),
        ({"parent_index": 0, "sequence": 42}, "sequence_not_canonical"),
        ({"parent_index": 9, "sequence": "MKTV"}, "parent_index_out_of_range"),
        ({"parent_index": -1, "sequence": "MKTV"}, "parent_index_out_of_range"),
        ({"parent_index": "0", "sequence": "MKTV"}, "parent_index_not_an_integer"),
        ({"parent_index": 0, "child": 2, "sequence": "MKTV"}, "child_beyond_max_children"),
        ({"parent_index": 0, "child": -1, "sequence": "MKTV"},
         "child_not_a_non_negative_integer"),
        ({"parent_index": 0, "sequence": "MKTV", "metrics": {"nope": 1.0}},
         "undeclared_metric"),
        ({"parent_index": 0, "sequence": "MKTV", "metrics": {"loss": "low"}},
         "metric_not_a_number"),
        ({"parent_index": 0, "sequence": "MKTV", "metrics": {"loss": True}},
         "metric_not_a_number"),
        ({"parent_index": 0, "sequence": "MKTV", "structure": "/abs/p.pdb"},
         "path_not_relative_to_structure_dir"),
        ({"parent_index": 0, "sequence": "MKTV", "structure": "../escape.pdb"},
         "path_not_relative_to_structure_dir"),
    ],
)
def test_each_violation_is_named(tmp_path, row, reason) -> None:
    _, counts = parse(rows(tmp_path / "c.jsonl", row))
    assert counts["rejected"] == {reason: 1}, counts
    assert counts["n_children"] == 0


def test_a_duplicate_parent_child_is_rejected(tmp_path) -> None:
    """One of the two would be lost, and which one depends on file order."""
    _, counts = parse(rows(
        tmp_path / "c.jsonl",
        {"parent_index": 0, "child": 0, "sequence": "MKTV"},
        {"parent_index": 0, "child": 0, "sequence": "MKTA"},
    ))
    assert counts["n_children"] == 1
    assert counts["rejected"] == {"duplicate_parent_child": 1}


def test_max_children_is_keyed_on_the_ordinal_not_arrival_order(tmp_path) -> None:
    """So which rows count as excess does not depend on how the file was
    written. Two children at ordinals 0 and 1 are fine under max_children=2
    however they are ordered; one at ordinal 5 is not, even if it arrives
    first."""
    _, counts = parse(rows(
        tmp_path / "c.jsonl",
        {"parent_index": 0, "child": 1, "sequence": "MKTV"},
        {"parent_index": 0, "child": 0, "sequence": "MKTA"},
    ))
    assert counts["n_children"] == 2
    assert not counts["rejected"]


def test_a_failure_row_needs_no_sequence(tmp_path) -> None:
    parsed, counts = parse(rows(
        tmp_path / "c.jsonl", {"parent_index": 1, "failed": "did not converge"}
    ))
    assert counts["n_failed"] == 1
    assert counts["n_children"] == 0
    assert parsed[0].is_failure and parsed[0].failed == "did not converge"


def test_a_torn_final_line_keeps_the_rest(tmp_path) -> None:
    """What a killed job leaves behind, and the one case where salvaging is not
    a judgement about the script."""
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"parent_index": 0, "sequence": "MKTV"}) + "\n"
        + '{"parent_index": 1, "sequ'
    )
    parsed, counts = parse(path)
    assert counts["n_children"] == 1 and counts["n_torn_lines"] == 1
    assert len(parsed) == 1


def test_a_missing_output_file_is_fatal(tmp_path) -> None:
    """Nothing to salvage and nothing true to report."""
    with pytest.raises(ContractError, match="wrote no output file"):
        parse(tmp_path / "absent.jsonl")


def test_sequences_are_upper_cased_and_stripped() -> None:
    assert normalize_sequence(" mkt v\n", where="x") == "MKTV"


def test_the_rejected_alphabet_is_named_in_the_error() -> None:
    with pytest.raises(ContractError, match=r"\['B', 'X'\]"):
        normalize_sequence("MKTVXB", where="x")


# --- the config -------------------------------------------------------------


def test_loss_models_is_required(tmp_path) -> None:
    """The field a later selection needs to avoid measuring its own optimizer.
    An omission is refused; `[]` is how you claim the loss saw no model."""
    from conftest import write_optimize_configs

    _, model_path = write_optimize_configs(tmp_path)
    payload = json.loads(OptimizeConfig.model_validate(
        load_yaml(model_path, OptimizeConfig).model_dump(mode="json")
    ).model_dump_json())
    payload.pop("loss_models")
    with pytest.raises(ValidationError, match="loss_models"):
        OptimizeConfig.model_validate(payload)


def test_a_declared_metric_may_not_shadow_a_registered_one(optimize_configs) -> None:
    """`iptm` means one thing. An optimizer's internal estimate under that name
    would put two definitions in one column."""
    _, model = optimize_configs
    with pytest.raises(ValidationError, match="already registered"):
        model.model_copy()  # no-op; validate a fresh payload instead
        OptimizeConfig.model_validate(
            {**model.model_dump(mode="json"), "metrics": {"iptm": {"direction": "max"}}}
        )


def test_a_metric_without_a_direction_is_refused(optimize_configs) -> None:
    _, model = optimize_configs
    with pytest.raises(ValidationError):
        OptimizeConfig.model_validate(
            {**model.model_dump(mode="json"), "metrics": {"thing": {}}}
        )


def test_asking_for_metrics_means_naming_them(optimize_configs) -> None:
    """Otherwise the same run silently does something different next month,
    depending on which scoring runs have landed."""
    _, model = optimize_configs
    with pytest.raises(ValidationError, match="metric_inputs names none"):
        OptimizeConfig.model_validate(
            {**model.model_dump(mode="json"), "inputs": ["sequence", "metrics"]}
        )


def test_naming_metrics_without_asking_for_them_is_refused(optimize_configs) -> None:
    _, model = optimize_configs
    with pytest.raises(ValidationError, match="does not include"):
        OptimizeConfig.model_validate(
            {**model.model_dump(mode="json"), "metric_inputs": ["boltz2_iptm"]}
        )


def test_the_metric_prefix_is_the_optimizers_name(optimize_configs) -> None:
    _, model = optimize_configs
    assert model.metric_prefix == "refine_test"


def test_the_protocol_hash_excludes_the_design_set(optimize_configs, write_design_set) -> None:
    """So optimizing more parents later extends a measurement rather than
    starting a new one -- the choice `scorer/config.py` already makes."""
    _, model = optimize_configs
    other = model.model_copy(update={"design_set": write_design_set(["MKTV", "MKTA"])})
    assert other.protocol_hash == model.protocol_hash


def test_the_protocol_hash_moves_when_the_loss_models_change(optimize_configs) -> None:
    _, model = optimize_configs
    changed = model.model_copy(update={"loss_models": ("boltz2",)})
    assert changed.protocol_hash != model.protocol_hash


# --- preflight --------------------------------------------------------------


def test_a_missing_design_set_says_how_to_build_one(optimize_configs, tmp_path) -> None:
    general, model = optimize_configs
    model = model.model_copy(update={"design_set": tmp_path / "absent.json"})
    with pytest.raises(OptimizePreflightError, match="designset build"):
        preflight_optimize(general, model)


def test_an_unknown_loss_model_is_refused(optimize_configs) -> None:
    """A misspelled name would exclude nothing later while looking like it
    had, which is worse than not having the field."""
    general, model = optimize_configs
    model = model.model_copy(update={"loss_models": ("boltz3",)})
    with pytest.raises(OptimizePreflightError, match="not a model this campaign"):
        preflight_optimize(general, model)


def test_structure_inputs_must_say_which_models_poses(optimize_configs) -> None:
    """Two models disagree about where the binder sits by a median 22.7 A, so
    'the structure' of a design is not one thing."""
    general, model = optimize_configs
    model = model.model_copy(update={"inputs": ("sequence", "structure")})
    with pytest.raises(OptimizePreflightError, match="structures_from must name"):
        preflight_optimize(general, model)


def test_the_script_bytes_are_hashed(optimize_configs) -> None:
    """A script edited between two runs must not be able to make them look
    comparable."""
    general, model = optimize_configs
    first = preflight_optimize(general, model).script_sha256
    Path(model.script).write_text(Path(model.script).read_text() + "\n# edited\n")
    assert preflight_optimize(general, model).script_sha256 != first


def test_hotspots_are_resolved_to_fasta_positions(optimize_config_files) -> None:
    """Author numbering with a chain letter in, 1-based FASTA positions out --
    the convention the epitope function settled on, because residue ids in a
    predicted pose are positional and carry no author numbering."""
    general_path, model_path = optimize_config_files
    general = load_yaml(general_path, GeneralConfig)
    general = general.model_copy(
        update={"target": general.target.model_copy(update={"hotspots": ("A12", "A3")})}
    )
    pre = preflight_optimize(general, load_yaml(model_path, OptimizeConfig))
    assert pre.hotspots == (3, 12)


def test_a_hotspot_past_the_end_of_the_target_is_refused(optimize_config_files) -> None:
    """It proves the author numbering is not FASTA position for this target.
    An epitope measured on the wrong numbering reads as a real miss."""
    general_path, model_path = optimize_config_files
    general = load_yaml(general_path, GeneralConfig)
    general = general.model_copy(
        update={"target": general.target.model_copy(update={"hotspots": ("A9999",)})}
    )
    with pytest.raises(OptimizePreflightError, match="outside the target's"):
        preflight_optimize(general, load_yaml(model_path, OptimizeConfig))


def test_a_hotspot_on_another_chain_is_refused(optimize_config_files) -> None:
    general_path, model_path = optimize_config_files
    general = load_yaml(general_path, GeneralConfig)
    general = general.model_copy(
        update={"target": general.target.model_copy(update={"hotspots": ("B12",)})}
    )
    with pytest.raises(OptimizePreflightError, match="not this target's epitope"):
        preflight_optimize(general, load_yaml(model_path, OptimizeConfig))


# --- the driver -------------------------------------------------------------


def test_the_shard_is_contiguous_not_strided(optimize_driver) -> None:
    """Entries are length-sorted, so a contiguous block spans few binder
    lengths and therefore few JAX recompilations."""
    entries = [{"index": index} for index in range(10)]
    assert [e["index"] for e in optimize_driver.contiguous_shard(entries, 0, 3)] == [0, 1, 2, 3]
    assert [e["index"] for e in optimize_driver.contiguous_shard(entries, 1, 3)] == [4, 5, 6, 7]
    assert [e["index"] for e in optimize_driver.contiguous_shard(entries, 2, 3)] == [8, 9]


def test_shard_local_indices_are_translated_to_design_set_indices(
    optimize_driver, tmp_path
) -> None:
    """A script only ever sees 0..n-1 for its own shard. Translating that back
    is the driver's job, and the reason it is the one range check that lives
    there."""
    shard = [{"index": 7}, {"index": 8}]
    raw = rows(
        tmp_path / "raw.jsonl",
        {"parent_index": 0, "sequence": "MKTV"},
        {"parent_index": 1, "sequence": "MKTA"},
    )
    children, counts = optimize_driver.translate(raw, shard)
    assert [row["parent_index"] for row in children] == [7, 8]
    assert [row["shard_parent_index"] for row in children] == [0, 1]
    assert counts["n_children"] == 2 and counts["n_unmappable"] == 0


def test_a_row_naming_a_parent_outside_the_shard_is_dropped(
    optimize_driver, tmp_path
) -> None:
    """It cannot be stored: the index would attach the child to a different
    design, and a table where some children have the wrong parent is worse than
    one missing a few."""
    raw = rows(tmp_path / "raw.jsonl", {"parent_index": 5, "sequence": "MKTV"})
    children, counts = optimize_driver.translate(raw, [{"index": 0}])
    assert children == [] and counts["n_unmappable"] == 1


def test_an_empty_shard_succeeds(optimize_loaded, tmp_path) -> None:
    """Legitimate when jobs exceed designs. Not a failing run."""
    general, model = optimize_loaded.general, optimize_loaded.model
    save = tmp_path / "empty-task"
    result = subprocess.run(
        [sys.executable, str(DRIVER),
         "--design-set", str(optimize_loaded.preflight.fasta_path),
         "--script", str(model.script),
         "--target-fasta", str(general.target.sequence_fasta),
         "--work-dir", str(save / "work"), "--structure-dir", str(save / "s"),
         "--save-dir", str(save), "--shard", "5", "--num-shards", "6"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((save / "status.json").read_text())["status"] == "succeeded"
    assert (save / "children.jsonl").read_text() == ""


def test_a_script_that_writes_nothing_is_a_failure(optimize_loaded, tmp_path) -> None:
    """A run that succeeds and produces nothing is reported as failed rather
    than as an optimization that improved nothing."""
    silent = tmp_path / "silent.py"
    silent.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "for f in ('--inputs', '--outputs', '--context'): p.add_argument(f)\n"
        "a = p.parse_args()\n"
        "pathlib.Path(a.outputs).write_text('')\n"
    )
    result = _run_driver(optimize_loaded, tmp_path / "t", script=silent)
    assert result.returncode == 1
    assert json.loads((tmp_path / "t" / "status.json").read_text())["status"] == "failed"


def test_a_crashing_script_records_its_exit_code(optimize_loaded, tmp_path) -> None:
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.exit(3)\n")
    result = _run_driver(optimize_loaded, tmp_path / "t", script=crash)
    assert result.returncode == 1
    status = json.loads((tmp_path / "t" / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["details"]["exit_code"] == 3


def test_the_script_is_handed_only_the_inputs_it_declared(
    optimize_loaded, tmp_path
) -> None:
    """So it cannot come to depend on an input it did not ask for -- which is
    what lets the harness know it can run over unfolded designs."""
    _run_driver(optimize_loaded, tmp_path / "t")
    handed = [
        json.loads(line)
        for line in (tmp_path / "t" / "work" / "inputs.jsonl").read_text().splitlines()
    ]
    assert handed
    for row in handed:
        assert set(row) == {"index", "length", "sequence"}


def test_the_context_carries_the_run_level_fields(optimize_loaded, tmp_path) -> None:
    _run_driver(optimize_loaded, tmp_path / "t")
    context = json.loads((tmp_path / "t" / "work" / "context.json").read_text())
    assert context["seed"] == 7
    assert context["n_parents"] == 3
    assert context["hotspots"] == []
    assert context["target_sequence"].startswith("DDNRLCTLA")
    assert Path(context["structure_dir"]).is_dir()


def _run_driver(loaded, save: Path, *, script: Path | None = None):
    general, model = loaded.general, loaded.model
    return subprocess.run(
        [sys.executable, str(DRIVER),
         "--design-set", str(loaded.preflight.fasta_path),
         "--design-set-manifest", str(model.design_set),
         "--script", str(script or model.script),
         "--target-fasta", str(general.target.sequence_fasta),
         "--work-dir", str(save / "work"), "--structure-dir", str(save / "structures"),
         "--save-dir", str(save), "--shard", "0", "--num-shards", "1",
         "--seed", str(model.seed), "--max-children", str(model.max_children),
         "--inputs-wanted", ",".join(model.inputs),
         "--script-args", *model.args],
        check=False, capture_output=True, text=True,
    )


# --- the plan ---------------------------------------------------------------


def test_the_plan_records_which_models_the_loss_saw(optimize_loaded) -> None:
    plan = plugin_for("optimize").tool_plan(optimize_loaded)
    assert plan.workflow["loss_models"] == []
    assert plan.workflow["script_sha256"]
    assert plan.kind.value == "optimize"


def test_a_host_run_optimizer_records_the_script_as_its_container(
    optimize_loaded,
) -> None:
    """`ToolPlan.container` is what determined the result. With no image, that
    is the script."""
    plan = plugin_for("optimize").tool_plan(optimize_loaded)
    assert Path(plan.container) == Path(optimize_loaded.model.script)


def test_the_plan_carries_the_design_set_it_was_planned_with(optimize_loaded) -> None:
    plan = plugin_for("optimize").tool_plan(optimize_loaded)
    assert plan.workflow["n_designs"] == 3
    assert plan.workflow["design_set_digest"] == optimize_loaded.preflight.design_set.digest
    assert plan.designs_file == "children.jsonl"


# --- the round trip: plan, run, collect, ingest -----------------------------


def _complete_run(loaded, run_dir: Path):
    """Plan a run, execute every task's real command, and collect it."""
    import subprocess as sp

    from bindocracy.tools import collect_run, launch_spec, plan

    manifest = plan(loaded, run_dir)
    for task in manifest.tasks:
        spec = launch_spec(manifest, task.task_id)
        for directory in spec.mkdirs:
            directory.mkdir(parents=True, exist_ok=True)
        spec.log.parent.mkdir(parents=True, exist_ok=True)
        result = sp.run(spec.argv, check=False, capture_output=True, text=True)
        spec.log.write_text(result.stdout + result.stderr)
    return manifest, collect_run(manifest.directory / "run.json")


def test_children_become_designs_with_parents(optimize_loaded, tmp_path) -> None:
    """The lineage is a column, not an archaeology exercise."""
    _, collected = _complete_run(optimize_loaded, tmp_path / "run")
    assert collected.designs
    parents = {entry.design_id for entry in optimize_loaded.preflight.design_set.entries}
    for design in collected.designs:
        assert design.parent_design_id in parents
        assert design.candidate_type == "sequence"


def test_the_native_id_carries_the_lineage(optimize_loaded, tmp_path) -> None:
    """So a person reading a FASTA sees it without a join, and so re-collecting
    the same run produces the same design_id."""
    _, collected = _complete_run(optimize_loaded, tmp_path / "run")
    assert all(".refine_test" in design.native_id for design in collected.designs)


def test_recollecting_produces_identical_design_ids(optimize_loaded, tmp_path) -> None:
    """Ingestion is idempotent only if identity is content-derived. A random
    design_id here would make every re-collect a new set of designs."""
    from bindocracy.tools import collect_run

    manifest, first = _complete_run(optimize_loaded, tmp_path / "run")
    second = collect_run(manifest.directory / "run.json")
    assert [d.design_id for d in first.designs] == [d.design_id for d in second.designs]


def test_every_child_records_which_models_the_loss_saw(
    optimize_config_files, tmp_path
) -> None:
    """On the child, not only on the run: a design outlives the query that
    found it, and this is what stops a later ranking measuring the optimizer."""
    from bindocracy.config.load import LoadedConfigs
    from bindocracy.tools.optimize.config import OptimizeConfig as OC
    from bindocracy.tools.optimize.preflight import preflight_optimize as pre

    general_path, model_path = optimize_config_files
    general = load_yaml(general_path, GeneralConfig)
    model = load_yaml(model_path, OC).model_copy(update={"loss_models": ("boltz2",)})
    loaded = LoadedConfigs(
        general=general, model=model, preflight=pre(general, model),
        general_path=general_path, model_path=model_path,
    )
    _, collected = _complete_run(loaded, tmp_path / "run")
    assert collected.designs
    for design in collected.designs:
        assert design.metadata["loss_models"] == ["boltz2"]
    assert collected.run.count_details["loss_models"] == ["boltz2"]


def test_the_distance_from_the_parent_is_recorded(optimize_loaded, tmp_path) -> None:
    """A child identical to its parent is a real result -- the optimizer found
    nothing -- and worth counting rather than meeting later as a duplicate."""
    _, collected = _complete_run(optimize_loaded, tmp_path / "run")
    for design in collected.designs:
        assert design.metadata["n_substitutions"] >= 0
        assert design.metadata["length_delta"] == 0
    assert "n_unchanged" in collected.run.count_details


def test_metrics_are_prefixed_with_the_optimizers_name(optimize_loaded, tmp_path) -> None:
    """`refine_test_loss` is the optimizer's OPINION of a child, never a
    measurement of it, and the two must not be joinable by accident."""
    _, collected = _complete_run(optimize_loaded, tmp_path / "run")
    names = {record.name for record in collected.metrics}
    assert names == {"refine_test_loss", "refine_test_n_mutations"}
    directions = {record.name: record.direction for record in collected.metrics}
    assert directions["refine_test_loss"] == "min"
    assert directions["refine_test_n_mutations"] == "none"


def test_a_refused_parent_is_counted_not_dropped(optimize_loaded, tmp_path) -> None:
    """The third design in the fixture set has no hydrophobic residue, so the
    example optimizer declines it and says so."""
    _, collected = _complete_run(optimize_loaded, tmp_path / "run")
    assert collected.run.count_details["n_failed"] == 1
    assert collected.run.n_attempted == 3
    assert collected.run.n_produced == len(collected.designs)


def test_the_trajectory_is_pointed_at_from_the_database(optimize_loaded, tmp_path) -> None:
    """So a later look at how a child was reached is a read, not a re-run."""
    _, collected = _complete_run(optimize_loaded, tmp_path / "run")
    kinds = {record.kind for record in collected.artifacts}
    assert "optimization_trajectory" in kinds
    trajectories = [a for a in collected.artifacts if a.kind == "optimization_trajectory"]
    assert all(a.design_id is not None for a in trajectories)
    # Stored RUN-DIRECTORY-RELATIVE, the convention `adapters/common.py` sets
    # for every tool, so that a run directory can be moved.
    run_dir = Path(collected.run.output_uri)
    assert all(not Path(a.uri).is_absolute() for a in trajectories)
    assert all((run_dir / a.uri).is_file() for a in trajectories)


def test_the_children_reach_a_real_database(optimize_loaded, tmp_path) -> None:
    """The whole path: plan, run, collect, ingest, query the lineage back."""
    import duckdb

    from bindocracy.runs import ingest_bundle, write_collected
    from bindocracy.store import CampaignStore, create_database

    manifest, collected = _complete_run(optimize_loaded, tmp_path / "run")
    database = create_database(tmp_path / "campaign.duckdb")

    # The parents have to exist before their children can reference them.
    from bindocracy.store.records import (
        CandidateType,
        CollectedRun,
        DesignRecord,
        RunKind,
        RunRecord,
        RunStatus,
    )
    parent_run = RunRecord(
        run_id="parents", name="parents", tool="mosaic", kind=RunKind.GENERATE,
        model_config_id=manifest.config.model_config_id, status=RunStatus.SUCCEEDED,
        n_requested=3, n_attempted=3, n_produced=3,
    )
    parents = tuple(
        DesignRecord(
            design_id=entry.design_id, run_id="parents", native_id=entry.native_id,
            candidate_type=CandidateType.SEQUENCE, sequence=entry.sequence,
        )
        for entry in optimize_loaded.preflight.design_set.entries
    )
    with CampaignStore(database) as store:
        store.ingest(
            CollectedRun(run=parent_run, designs=parents), configs=[manifest.config]
        )

    bundle = write_collected(collected, tmp_path / "collected.json")
    assert ingest_bundle(database, bundle) is True
    # Restart-safe: the same bundle again is a no-op, not a second copy.
    assert ingest_bundle(database, bundle) is False

    con = duckdb.connect(str(database), read_only=True)
    n_children, n_with_parent = con.execute(
        "SELECT count(*), count(parent_design_id) FROM designs WHERE run_id = ?",
        [collected.run.run_id],
    ).fetchone()
    assert n_children == len(collected.designs) and n_with_parent == n_children
    # The lineage is queryable, which is the point of the whole exercise.
    joined = con.execute(
        "SELECT c.native_id, p.native_id FROM designs c JOIN designs p "
        "ON p.design_id = c.parent_design_id WHERE c.run_id = ?",
        [collected.run.run_id],
    ).fetchall()
    assert len(joined) == n_children
    assert all(child.startswith(parent) for child, parent in joined)
    con.close()
