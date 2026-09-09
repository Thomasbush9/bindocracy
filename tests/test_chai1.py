"""Chai-1 plugin: the checks that would otherwise fail silently or expensively."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bindocracy.adapters.scoring import registered_keys, spec_for
from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.store.records import MetricDirection
from bindocracy.tools import plugin_for
from bindocracy.tools.chai1.config import Chai1Config
from bindocracy.tools.chai1.preflight import expected_pqt_basename, preflight_chai1

CHAI1_DRIVER = (
    Path(__file__).resolve().parents[1] / "drivers" / "chai1" / "score_chai1.py"
)

# The DIO3 target the campaign actually uses, and the basename the container
# computed for it on 2026-09-09. `expected_pqt_basename` reimplements Chai's
# rule on the host so preflight can run without the image; this vector is what
# stops the two drifting apart unnoticed. Regenerate with:
#   singularity exec chai1.sif /opt/chai-venv/bin/python -c \
#     "from chai_lab.data.parsing.msas.aligned_pqt import expected_basename; ..."
DIO3_SEQUENCE = (
    "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQR"
    "LVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERL"
    "YVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
)
DIO3_PQT = "cbd2ba0f12cec856871a12ec282d381cd343b6a719711179bf69776c46316e49.aligned.pqt"


def with_protocol(model, **overrides):
    """A copy with protocol knobs changed. Configs are frozen on purpose -- a
    loaded config is the record of what a run was planned with."""
    return model.model_copy(
        update={"protocol": model.protocol.model_copy(update=overrides)}
    )


def with_runtime(model, **overrides):
    return model.model_copy(
        update={"runtime": model.runtime.model_copy(update=overrides)}
    )


@pytest.fixture
def chai1_driver():
    """The driver imports only stdlib at module scope; torch and chai_lab are
    reached inside functions, so it loads outside the container."""
    spec = importlib.util.spec_from_file_location("score_chai1", CHAI1_DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the MSA, which is the whole point of the preflight ---------------------


def test_pqt_basename_matches_the_container_rule() -> None:
    assert expected_pqt_basename(DIO3_SEQUENCE) == DIO3_PQT


def test_pqt_basename_ignores_case_like_chai_does() -> None:
    assert expected_pqt_basename("acdef") == expected_pqt_basename("ACDEF")


def test_a_missing_alignment_is_refused_not_warned(chai1_configs) -> None:
    """Chai logs a warning and folds single-sequence. Preflight must not."""
    general, model = chai1_configs
    model.runtime.msa_directory.joinpath(DIO3_PQT).unlink()
    with pytest.raises(ConfigPreflightError) as error:
        preflight_chai1(general, model)
    message = str(error.value)
    assert DIO3_PQT in message
    assert "single-sequence" in message
    assert "a3m-to-pqt" in message


def test_msa_directory_required_when_the_protocol_says_so(chai1_configs) -> None:
    general, model = chai1_configs
    with pytest.raises(ConfigPreflightError, match="msa_directory is unset"):
        preflight_chai1(general, with_runtime(model, msa_directory=None))


def test_single_sequence_is_allowed_when_chosen_explicitly(chai1_configs) -> None:
    general, model = chai1_configs
    model = with_runtime(with_protocol(model, use_target_msa=False), msa_directory=None)
    assert preflight_chai1(general, model).msa_path is None


def test_the_resolved_alignment_is_recorded_as_a_run_input(chai1_configs) -> None:
    general, model = chai1_configs
    pre = preflight_chai1(general, model)
    assert pre.msa_path is not None and pre.msa_path.name == DIO3_PQT


# --- refusals that would otherwise cost a GPU allocation --------------------


def test_unencodable_residues_are_refused(chai1_configs, write_design_set) -> None:
    general, model = chai1_configs
    model = model.model_copy(
        update={"design_set": write_design_set(["ACDEFGHIKL", "ACDEFGHIKX"])}
    )
    with pytest.raises(ConfigPreflightError, match="cannot encode"):
        preflight_chai1(general, model)


def test_sharding_wider_than_the_set_is_refused(chai1_configs) -> None:
    general, model = chai1_configs
    model = model.model_copy(
        update={"sharding": model.sharding.model_copy(update={"jobs": 99})}
    )
    with pytest.raises(ConfigPreflightError, match="exceeds"):
        preflight_chai1(general, model)


# --- the protocol hash ------------------------------------------------------


def test_protocol_hash_excludes_the_design_set(chai1_configs, write_design_set) -> None:
    """Scoring more designs later extends a measurement, it does not start a
    new one -- which is what makes incremental scoring safe."""
    _, model = chai1_configs
    before = model.protocol_hash
    moved = model.model_copy(
        update={"design_set": write_design_set(["ACDEFGHIKL", "MNPQRSTVWY"])}
    )
    assert moved.protocol_hash == before


def test_protocol_hash_excludes_low_memory(chai1_configs) -> None:
    """A memory/speed tradeoff that does not change the prediction."""
    _, model = chai1_configs
    before = model.protocol_hash
    flipped = with_protocol(model, low_memory=not model.protocol.low_memory)
    assert flipped.protocol_hash == before


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_trunk_recycles", 5),
        ("num_diffn_timesteps", 50),
        ("num_diffn_samples", 2),
        ("recycle_msa_subsample", 512),
        ("use_esm_embeddings", False),
        ("use_target_msa", False),
    ],
)
def test_protocol_hash_moves_with_every_knob_that_changes_the_number(
    chai1_configs, field, value
) -> None:
    _, model = chai1_configs
    before = model.protocol_hash
    changed = with_protocol(model, **{field: value})
    assert changed.protocol_hash != before, f"{field} does not affect the protocol hash"


def test_protocol_knobs_have_no_defaults() -> None:
    """An unset default is an experiment nobody chose."""
    with pytest.raises(ValueError):
        Chai1Config.model_validate(
            {
                "schema_version": 1, "name": "x", "tool": "chai1",
                "design_set": "/tmp/x.json", "driver_script": "/tmp/d.py",
                "protocol": {},  # every knob omitted
                "sharding": {"jobs": 1},
                "runtime": {"container": "/tmp/c.sif", "scratch": "/tmp/s"},
                "resources": {"gpus": 1, "cpus": 4, "memory_gb": 32, "walltime": "1:00:00"},
            }
        )


def test_a_run_with_no_readers_measures_nothing(chai1_configs) -> None:
    _, model = chai1_configs
    with pytest.raises(ValueError, match="measure nothing"):
        model.readers.__class__(complex=False)


# --- the plan ---------------------------------------------------------------


def test_the_plan_is_an_evaluate_run_that_makes_no_designs(chai1_loaded) -> None:
    plan = plugin_for("chai1").tool_plan(chai1_loaded)
    assert plan.kind.value == "evaluate"
    assert plan.designs_file == "metrics.jsonl"
    assert plan.workflow["metric_prefix"] == "chai1"
    assert plan.workflow["samples_per_design"] == 5
    assert plan.workflow["scope_id"]
    # The alignment is digested as a run input, so the run records the bytes it
    # folded against rather than the directory it looked in.
    assert "target_msa_pqt" in plan.inputs


def test_the_adapter_is_the_scorer_seam_under_another_name() -> None:
    from bindocracy.tools.scorer.adapter import ScorerOutputAdapter

    adapter = plugin_for("chai1").adapter()
    assert isinstance(adapter, ScorerOutputAdapter)
    assert adapter.tool == "chai1"


# --- metrics ----------------------------------------------------------------


def test_every_metric_the_driver_emits_is_registered(chai1_driver) -> None:
    """`spec_for` refuses unregistered names at collection time, i.e. after the
    GPU hours are spent. This asserts it at test time instead."""
    emitted = {
        "aggregate_score", "complex_ptm", "iptm", "bt_iptm", "tb_iptm",
        "iptm_min", "binder_ptm", "complex_plddt", "binder_plddt",
        "bt_pae", "tb_pae",
        "has_clashes", "n_clashing_chain_pairs", "binder_intra_clashes",
    }
    assert emitted <= set(registered_keys())
    for key in emitted:
        spec_for(key)


def test_pae_and_clash_metrics_point_the_right_way() -> None:
    """A metric stored with the wrong direction sorts backwards and nothing in
    the row says so."""
    for key in ("bt_pae", "tb_pae", "has_clashes", "n_clashing_chain_pairs",
                "binder_intra_clashes"):
        assert spec_for(key).direction is MetricDirection.MIN
    for key in ("iptm", "complex_ptm", "aggregate_score", "binder_plddt",
                "bt_iptm", "tb_iptm", "iptm_min"):
        assert spec_for(key).direction is MetricDirection.MAX


# --- the driver's pure functions --------------------------------------------


def test_driver_reads_the_real_design_set_header(chai1_driver, tmp_path) -> None:
    """The header is `>{index} tool=... run=... native=... len=...` and only the
    index may be parsed -- `designset.py:_render_fasta` says the rest is for the
    human, and a design_id never crosses into the container."""
    fasta = tmp_path / "set.fasta"
    fasta.write_text(
        ">000000 tool=genie3 run=run19 native=task-0000-x len=10\nACDEF\nGHIKL\n"
        ">000001 tool=freebindcraft run=run19 native=y len=5\nMNPQR\n"
    )
    assert chai1_driver.read_fasta_entries(fasta) == [
        (0, "ACDEFGHIKL"),
        (1, "MNPQR"),
    ]


def test_driver_shards_contiguously(chai1_driver) -> None:
    """Contiguous, not strided: the host's DesignSet.shard does the same, and
    the two must agree or a design is scored twice or not at all."""
    entries = [(i, "A") for i in range(10)]
    shards = [chai1_driver.shard_of(entries, s, 3) for s in range(3)]
    assert [len(s) for s in shards] == [4, 4, 2]
    assert [e[0] for s in shards for e in s] == list(range(10))


def test_driver_writes_target_as_chain_a(chai1_driver, tmp_path) -> None:
    """Chai names asym units in input order, and the campaign's target is
    chain A."""
    path = tmp_path / "in.fasta"
    chai1_driver.write_chai_fasta(path, "TARGET", "BINDER")
    text = path.read_text()
    assert text.index("name=target") < text.index("name=binder")
    assert text == ">protein|name=target\nTARGET\n>protein|name=binder\nBINDER\n"


def test_driver_status_is_failed_when_nothing_was_measured(
    chai1_driver, tmp_path, monkeypatch
) -> None:
    """Reporting success with an empty reader is the AF2 silent-success bug."""
    design_set = tmp_path / "set.fasta"
    design_set.write_text(">000000 tool=t run=r native=n len=10\nACDEFGHIKL\n")
    target = tmp_path / "target.fasta"
    target.write_text(">t\nMKTAYIAKQR\n")

    def explode(**kwargs):
        raise RuntimeError("no GPU here")

    monkeypatch.setattr(chai1_driver, "install_offline_policy", lambda: None)
    monkeypatch.setitem(
        __import__("sys").modules, "chai_lab.chai1",
        type("m", (), {"run_inference": staticmethod(explode)}),
    )
    monkeypatch.setattr(
        __import__("sys"), "argv",
        ["score_chai1.py",
         "--design-set", str(design_set), "--target-fasta", str(target),
         "--work-dir", str(tmp_path / "w"), "--save-dir", str(tmp_path / "s"),
         "--shard", "0", "--num-shards", "1", "--task-id", "0",
         "--num-trunk-recycles", "3", "--num-diffn-timesteps", "10",
         "--num-diffn-samples", "1", "--num-trunk-samples", "1",
         "--recycle-msa-subsample", "0", "--seed", "0"],
    )
    assert chai1_driver.main() == 1
    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["n_produced"] == 0
    assert status["produced_by_reader"] == {"complex": 0}
    rows = [
        json.loads(line)
        for line in (tmp_path / "s" / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[0]["failed"].startswith("RuntimeError")


# --- the connector/driver contract ------------------------------------------


def test_the_driver_accepts_exactly_what_the_connector_launches(
    chai1_driver, chai1_config_files, tmp_path
) -> None:
    """argparse exits non-zero on an unknown or missing flag, so this failing
    means the connector would launch a GPU job that dies immediately. The
    parser here is the driver's own, not a copy."""
    from bindocracy.tools import launch_spec, load_configs, plan

    loaded = load_configs(*chai1_config_files)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 1)

    index = next(i for i, a in enumerate(spec.argv) if a.endswith("score_chai1.py"))
    parsed = chai1_driver.build_parser().parse_args(list(spec.argv[index + 1 :]))

    assert parsed.task_id == 1
    assert parsed.shard == 1 and parsed.num_shards == 2
    assert parsed.num_diffn_samples == 5
    assert parsed.num_trunk_recycles == 3
    assert parsed.low_memory is True
    assert Path(parsed.save_dir) == manifest.directory / "tasks" / "0001"
    assert Path(parsed.target_fasta) == loaded.general.target.sequence_fasta
    assert parsed.msa_directory == loaded.model.runtime.msa_directory


def test_the_launch_runs_the_images_own_interpreter(
    chai1_config_files, tmp_path
) -> None:
    """`singularity exec ... python` resolves against whatever PATH survives,
    and the image ships a system python3 beside the venv that has torch."""
    from bindocracy.tools import launch_spec, load_configs, plan

    loaded = load_configs(*chai1_config_files)
    spec = launch_spec(plan(loaded, tmp_path / "run"), 0)
    assert "/opt/chai-venv/bin/python" in spec.argv
    assert spec.argv[0] == "singularity"
    assert "--nv" in spec.argv and "--cleanenv" in spec.argv


def test_cleanenv_does_not_lose_the_allocated_gpu(
    chai1_config_files, tmp_path, monkeypatch
) -> None:
    """known-issues.md section 2.6: --cleanenv discards the variable SLURM uses
    to name the GPU, and the container then sees every device on the node."""
    from bindocracy.tools import launch_spec, load_configs, plan

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    loaded = load_configs(*chai1_config_files)
    spec = launch_spec(plan(loaded, tmp_path / "run"), 0)
    assert spec.env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "3"
    assert spec.env["SINGULARITYENV_CHAI_RUNTIME_DIR"].endswith("0000")


def test_each_task_folds_into_its_own_tree(chai1_config_files, tmp_path) -> None:
    """run_inference asserts an empty output directory, so two tasks sharing
    one would fail the second one into a confusing assertion."""
    from bindocracy.tools import launch_spec, load_configs, plan

    loaded = load_configs(*chai1_config_files)
    manifest = plan(loaded, tmp_path / "run")
    work = [
        launch_spec(manifest, task).argv[
            launch_spec(manifest, task).argv.index("--work-dir") + 1
        ]
        for task in (0, 1)
    ]
    assert work[0] != work[1]


def test_clash_counting_ignores_the_intra_chain_diagonal(chai1_driver) -> None:
    """`chain_chain_clashes` holds INTRA-chain clashes on the diagonal.

    Observed on a real fold as [[17, 0], [0, 11]] beside
    has_inter_chain_clashes=False -- counting nonzero entries without masking
    the diagonal reported a design's clashes with itself as an interface
    problem, in every row, for every design.

    Needs torch, so it runs inside the image rather than on the login node.
    """
    torch = pytest.importorskip("torch")

    class _PTM:
        complex_ptm = property(lambda self: torch.tensor([0.85]))

    ptm = type(
        "P", (), {
            "complex_ptm": torch.tensor([0.85]),
            "interface_ptm": torch.tensor([0.47]),
            "per_chain_ptm": torch.tensor([[0.92, 0.83]]),
            "per_chain_pair_iptm": torch.tensor([[[0.92, 0.395], [0.47, 0.83]]]),
        },
    )()
    clashes = type(
        "C", (), {
            "has_inter_chain_clashes": torch.tensor([False]),
            "chain_chain_clashes": torch.tensor([[[17, 0], [0, 11]]]),
        },
    )()
    ranking = type(
        "R", (), {
            "aggregate_score": torch.tensor([0.55]),
            "ptm_scores": ptm,
            "clash_scores": clashes,
        },
    )()
    candidates = type(
        "S", (), {
            "ranking_data": [ranking],
            "plddt": torch.rand(1, 6),
            "pae": torch.rand(1, 6, 6),
        },
    )()

    values = chai1_driver.metrics_from_candidate(candidates, 0, 4, 2)
    assert values["n_clashing_chain_pairs"] == 0.0, "diagonal counted as an interface clash"
    assert values["binder_intra_clashes"] == 11.0
    assert values["has_clashes"] == 0.0
    # ipTM is directional and Chai's own number is the max of the two.
    assert values["bt_iptm"] == pytest.approx(0.47, abs=1e-4)
    assert values["tb_iptm"] == pytest.approx(0.395, abs=1e-4)
    assert values["iptm_min"] == pytest.approx(0.395, abs=1e-4)


def test_token_count_mismatch_fails_the_design_not_the_metrics(chai1_driver) -> None:
    """Per-chain slices are positional; a changed tokenization must not
    silently produce a metric for the wrong residues."""
    torch = pytest.importorskip("torch")

    ptm = type("P", (), {
        "complex_ptm": torch.tensor([0.8]), "interface_ptm": torch.tensor([0.4]),
        "per_chain_ptm": torch.tensor([[0.9, 0.8]]),
        "per_chain_pair_iptm": torch.tensor([[[0.9, 0.3], [0.4, 0.8]]]),
    })()
    clashes = type("C", (), {
        "has_inter_chain_clashes": torch.tensor([False]),
        "chain_chain_clashes": torch.tensor([[[0, 0], [0, 0]]]),
    })()
    ranking = type("R", (), {
        "aggregate_score": torch.tensor([0.5]), "ptm_scores": ptm, "clash_scores": clashes,
    })()
    candidates = type("S", (), {
        "ranking_data": [ranking], "plddt": torch.rand(1, 6), "pae": torch.rand(1, 6, 6),
    })()

    with pytest.raises(ValueError, match="token count"):
        chai1_driver.metrics_from_candidate(candidates, 0, 4, 99)


# --- the adapter, against real GPU output -----------------------------------

CHAI1_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chai1" / "metrics.jsonl"


def test_the_adapter_turns_real_chai_output_into_metric_rows(tmp_path) -> None:
    """The rows here are unmodified output of the 2026-09-09 smoke run on an
    H100 (three designs, two diffusion samples each), with only the absolute
    structure paths shortened. Synthetic rows would prove the parser agrees
    with the test author, not with Chai."""
    from datetime import UTC, datetime

    from bindocracy.tools.scorer.adapter import _read_metrics

    records, counts = _read_metrics(
        CHAI1_FIXTURE,
        index_to_design={0: "design-aaa", 1: "design-bbb"},
        prefix="chai1",
        run_id="run-1",
        fallback_time=datetime(2026, 9, 9, tzinfo=UTC),
    )

    assert counts["n_lines"] == 4
    assert not counts["failures"]
    assert counts["indices"] == {0, 1}

    names = {record.name for record in records}
    assert "chai1_iptm" in names and "chai1_binder_intra_clashes" in names
    # Fourteen metrics per candidate, four candidates.
    assert len(records) == 14 * 4

    # Replicates are stored as rows, never collapsed at write time.
    iptm = sorted(
        (r.replicate, r.value) for r in records
        if r.name == "chai1_iptm" and r.design_id == "design-aaa"
    )
    assert [rep for rep, _ in iptm] == [0, 1]
    assert iptm[0][1] != iptm[1][1], "two diffusion samples produced one number"


def test_directional_iptm_disagrees_in_real_output() -> None:
    """The reason `tb_iptm` and `iptm_min` exist: Chai's `iptm` is the max of
    the two directions, so alone it keeps the flattering half."""
    rows = [
        json.loads(line) for line in CHAI1_FIXTURE.read_text().splitlines() if line.strip()
    ]
    for row in rows:
        values = row["metrics"]
        assert values["iptm"] == pytest.approx(
            max(values["bt_iptm"], values["tb_iptm"]), abs=1e-6
        )
        assert values["iptm_min"] == pytest.approx(
            min(values["bt_iptm"], values["tb_iptm"]), abs=1e-6
        )
    gaps = [r["metrics"]["bt_iptm"] - r["metrics"]["tb_iptm"] for r in rows]
    assert max(gaps) > 0.05, "the two directions were expected to differ materially"


def test_no_design_in_real_output_has_a_phantom_interface_clash() -> None:
    """The first smoke run reported n_clashing_chain_pairs=1.0 for every design
    while has_clashes was 0.0, because the diagonal holds intra-chain counts."""
    rows = [
        json.loads(line) for line in CHAI1_FIXTURE.read_text().splitlines() if line.strip()
    ]
    for row in rows:
        values = row["metrics"]
        assert values["has_clashes"] == 0.0
        assert values["n_clashing_chain_pairs"] == 0.0
        # ...while the binder genuinely does clash with itself.
        assert values["binder_intra_clashes"] > 0
