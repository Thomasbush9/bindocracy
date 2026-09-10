"""Upstream OpenFold3 plugin.

Most of these pin things that were learned by running it, not by reading it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import plugin_for
from bindocracy.tools.of3_upstream.config import (
    ACCEPTED_MSA_BASENAMES,
    OF3Protocol,
)
from bindocracy.tools.of3_upstream.preflight import preflight_of3_upstream

DRIVER = (
    Path(__file__).resolve().parents[1] / "drivers" / "of3_upstream" / "score_of3.py"
)


@pytest.fixture
def of3_driver():
    spec = importlib.util.spec_from_file_location("score_of3", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the filename trap -------------------------------------------------------


def test_the_alignment_is_staged_under_a_name_openfold3_accepts(
    of3_driver, tmp_path
) -> None:
    """`parse_msas_direct` keeps only files whose *basename* is a key of
    `max_seq_counts` and silently `continue`s past the rest. A correctly
    formatted a3m named anything else is dropped, the MSA dict comes back
    empty, and the run dies frames later on `sorted(...)[0]` with an IndexError
    naming nothing relevant. Observed 2026-09-10."""
    a3m = tmp_path / "DIO3.a3m"
    a3m.write_text(">t\nACDEF\n>h\nACDEF\n")
    staged = of3_driver.stage_msa(a3m, tmp_path / "work")
    assert staged.stem in ACCEPTED_MSA_BASENAMES
    assert staged.name == "colabfold_main.a3m"
    assert staged.read_text() == a3m.read_text()


def test_the_accepted_names_match_the_containers_own_list() -> None:
    """Transcribed from the config OpenFold3 wrote for a real run. If the image
    changes this set, the driver's staged name must still be in it."""
    assert "colabfold_main" in ACCEPTED_MSA_BASENAMES
    assert "uniref90_hits" in ACCEPTED_MSA_BASENAMES
    assert "DIO3" not in ACCEPTED_MSA_BASENAMES


# --- the MSA server, which must stay off ------------------------------------


def test_the_msa_server_is_refused() -> None:
    """Letting OpenFold3 search its own alignment folds the target against a
    different MSA from every other scorer -- the confound the MSA work
    removed."""
    with pytest.raises(ValueError, match="change of MSA"):
        OF3Protocol.model_validate(
            {"num_diffusion_samples": 1, "seed": 0,
             "use_target_msa": True, "use_msa_server": True}
        )


def test_use_target_msa_without_an_alignment_is_refused(of3_configs) -> None:
    general, model = of3_configs
    general = general.model_copy(
        update={"target": general.target.model_copy(update={"msa": None})}
    )
    with pytest.raises(ConfigPreflightError, match="nothing to search"):
        preflight_of3_upstream(general, model)


# --- the weights, which are not in the image --------------------------------


def test_a_missing_checkpoint_is_refused_before_the_gpu(of3_configs) -> None:
    general, model = of3_configs
    Path(model.runtime.checkpoint).unlink()
    with pytest.raises(ConfigPreflightError, match="ships no weights"):
        preflight_of3_upstream(general, model)


# --- the prefix, which is the whole point -----------------------------------


def test_it_is_a_different_model_from_mosaics_of3(of3_loaded) -> None:
    """On GFP, mosaic's OF3 gives pLDDT 38.5 at 24 A from consensus and this
    one gives 88.7 at 3.9 A; they differ from each other by 24.6 A. One
    `of3_iptm` column holding both would average two implementations."""
    plan = plugin_for("of3_upstream").tool_plan(of3_loaded)
    assert plan.workflow["metric_prefix"] == "of3_upstream"
    assert plan.workflow["metric_prefix"] != "of3"
    assert plan.workflow["msa_server"] is False
    # The weights are outside the image, so the container digest alone does not
    # determine the result.
    assert plan.workflow["checkpoint"]


def test_the_plan_is_an_evaluate_run(of3_loaded) -> None:
    plan = plugin_for("of3_upstream").tool_plan(of3_loaded)
    assert plan.kind.value == "evaluate"
    assert plan.designs_file == "metrics.jsonl"
    assert plan.workflow["scope_id"]


def test_the_adapter_is_the_shared_scorer_seam() -> None:
    from bindocracy.tools.scorer.adapter import ScorerOutputAdapter

    adapter = plugin_for("of3_upstream").adapter()
    assert isinstance(adapter, ScorerOutputAdapter)
    assert adapter.tool == "of3_upstream"


# --- the query OpenFold3 is handed ------------------------------------------


def test_the_target_is_chain_a_and_the_binder_has_no_alignment(of3_driver) -> None:
    """Chain A for the target matches general.target.chain_id and both other
    co-folding drivers, so all three agree which chain is which."""
    query = of3_driver.build_query(
        [(0, "BINDERSEQ")], "TARGETSEQ", Path("/tmp/colabfold_main.a3m"), 42, False
    )
    chains = query["queries"]["design-000000"]["chains"]
    assert [c["chain_ids"] for c in chains] == [["A"], ["B"]]
    assert chains[0]["sequence"] == "TARGETSEQ"
    assert "main_msa_file_paths" in chains[0]
    # A de novo binder has no homologs; every other scorer here assumes that too.
    assert "main_msa_file_paths" not in chains[1]
    assert query["queries"]["design-000000"]["use_paired_msas"] is False
    assert query["seeds"] == [42]


def test_the_monomer_query_has_one_chain(of3_driver) -> None:
    query = of3_driver.build_query([(0, "SEQ")], "TARGET", None, 7, True)
    chains = query["queries"]["design-000000"]["chains"]
    assert len(chains) == 1 and chains[0]["sequence"] == "SEQ"


# --- metrics, mapped from real observed output ------------------------------


def test_metrics_come_from_the_real_confidence_schema(of3_driver) -> None:
    """These key names are from a confidences_aggregated.json OpenFold3 wrote
    on 2026-09-10, not from the source."""
    observed = {
        "avg_plddt": 84.9, "gpde": 0.657, "iptm": 0.374, "ptm": 0.780,
        "disorder": 0.031, "has_clash": 0.0, "sample_ranking_score": 0.471,
        "chain_ptm": {"A": 0.892, "B": 0.791},
        "chain_pair_iptm": {"(A, B)": 0.374},
    }
    values = of3_driver.metrics_from_confidences(observed, 84.9, 79.1, False)
    assert values["complex_ptm"] == 0.780
    assert values["iptm"] == 0.374
    assert values["binder_ptm"] == 0.791
    assert values["aggregate_score"] == 0.471
    assert values["has_clashes"] == 0.0


def test_a_monomer_emits_no_interface_metrics(of3_driver) -> None:
    """A single-chain fold has no interface. Emitting ipTM as zero would read
    as a measured bad interface rather than an absent one."""
    values = of3_driver.metrics_from_confidences(
        {"ptm": 0.89, "avg_plddt": 88.7}, 88.7, float("nan"), True
    )
    assert set(values) == {"mono_plddt", "mono_ptm"}
    assert "iptm" not in values


def test_every_metric_the_driver_emits_is_registered(of3_driver) -> None:
    from bindocracy.adapters.scoring import registered_keys

    emitted = {
        "complex_ptm", "iptm", "bt_iptm", "binder_ptm", "complex_plddt",
        "binder_plddt", "aggregate_score", "has_clashes", "mono_plddt", "mono_ptm",
    }
    assert emitted <= set(registered_keys())


def test_the_driver_reads_the_real_design_set_header(of3_driver, tmp_path) -> None:
    fasta = tmp_path / "set.fasta"
    fasta.write_text(">000004 tool=genie3 run=r native=n len=5\nACDEF\n")
    assert of3_driver.read_fasta_entries(fasta) == [(4, "ACDEF")]
