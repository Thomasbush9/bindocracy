"""AlphaFold 3 plugin: the checks that would otherwise fail after allocation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bindocracy.adapters.scoring import registered_keys
from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import plugin_for
from bindocracy.tools.af3.preflight import WEIGHTS_FILE, preflight_af3

AF3_DRIVER = Path(__file__).resolve().parents[1] / "drivers" / "af3" / "score_af3.py"


@pytest.fixture
def af3_driver():
    spec = importlib.util.spec_from_file_location("score_af3", AF3_DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the weights, which are deliberately not in the image -------------------


def test_missing_weights_are_refused_before_the_gpu(af3_configs) -> None:
    """They are outside the image on purpose -- the terms restrict
    distribution -- so their absence is a config error, and it should surface
    here rather than deep inside run_alphafold."""
    general, model = af3_configs
    (model.runtime.model_dir / WEIGHTS_FILE).unlink()
    with pytest.raises(ConfigPreflightError) as error:
        preflight_af3(general, model)
    assert WEIGHTS_FILE in str(error.value)
    assert "restricts distribution" in str(error.value)


def test_the_weights_location_is_recorded_in_the_run(af3_loaded) -> None:
    """Unlike Chai-1, the container digest alone does NOT determine an AF3
    result, because the weights are bound rather than embedded. The run has to
    say where they came from."""
    plan = plugin_for("af3").tool_plan(af3_loaded)
    assert plan.workflow["model_dir"]
    assert plan.workflow["data_pipeline"] == "disabled"


# --- the MSA ---------------------------------------------------------------


def test_use_target_msa_without_an_alignment_is_refused(af3_configs) -> None:
    """AF3 runs with --norun_data_pipeline and has no database to search, so a
    missing alignment means silently folding single-sequence."""
    general, model = af3_configs
    general = general.model_copy(
        update={"target": general.target.model_copy(update={"msa": None})}
    )
    with pytest.raises(ConfigPreflightError, match="single-sequence"):
        preflight_af3(general, model)


def test_an_alignment_for_another_protein_is_refused(af3_configs, tmp_path) -> None:
    """An a3m whose query row is a different protein loads fine and changes
    every number."""
    general, model = af3_configs
    wrong = tmp_path / "wrong.a3m"
    wrong.write_text(">other\nMKTAYIAKQRQISFVKSHFSRQ\n")
    general = general.model_copy(
        update={"target": general.target.model_copy(update={"msa": wrong})}
    )
    with pytest.raises(ConfigPreflightError):
        preflight_af3(general, model)


def test_single_sequence_is_allowed_when_chosen(af3_configs) -> None:
    general, model = af3_configs
    model = model.model_copy(
        update={"protocol": model.protocol.model_copy(update={"use_target_msa": False})}
    )
    assert preflight_af3(general, model).msa_path is None


# --- the plan --------------------------------------------------------------


def test_the_plan_is_an_evaluate_run_that_makes_no_designs(af3_loaded) -> None:
    plan = plugin_for("af3").tool_plan(af3_loaded)
    assert plan.kind.value == "evaluate"
    assert plan.designs_file == "metrics.jsonl"
    assert plan.workflow["metric_prefix"] == "af3"
    assert plan.workflow["scope_id"]


def test_the_adapter_is_the_shared_scorer_seam() -> None:
    from bindocracy.tools.scorer.adapter import ScorerOutputAdapter

    adapter = plugin_for("af3").adapter()
    assert isinstance(adapter, ScorerOutputAdapter)
    assert adapter.tool == "af3"


def test_protocol_hash_excludes_the_design_set(af3_configs, write_design_set) -> None:
    _, model = af3_configs
    before = model.protocol_hash
    moved = model.model_copy(
        update={"design_set": write_design_set(["ACDEFGHIKL", "MNPQRSTVWY"])}
    )
    assert moved.protocol_hash == before


@pytest.mark.parametrize(
    "field, value",
    [("num_trunk_recycles", 5), ("num_diffn_timesteps", 50),
     ("num_diffn_samples", 1), ("use_target_msa", False)],
)
def test_protocol_hash_moves_with_every_knob(af3_configs, field, value) -> None:
    _, model = af3_configs
    before = model.protocol_hash
    changed = model.model_copy(
        update={"protocol": model.protocol.model_copy(update={field: value})}
    )
    assert changed.protocol_hash != before, f"{field} does not affect the protocol hash"


# --- the driver ------------------------------------------------------------


def test_the_driver_accepts_what_the_connector_launches(
    af3_driver, af3_config_files, tmp_path
) -> None:
    from bindocracy.tools import launch_spec, load_configs, plan

    loaded = load_configs(*af3_config_files)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)
    index = next(i for i, a in enumerate(spec.argv) if a.endswith("score_af3.py"))
    # The driver builds its parser inside main(); parse with the same flags.
    argv = list(spec.argv[index + 1 :])
    assert "--model-dir" in argv and "--norun" not in " ".join(argv)
    assert "/models" in argv  # bound path, not a host path
    assert any(a.endswith(".fasta") for a in argv)


def test_the_weights_are_bound_read_only(af3_config_files, tmp_path) -> None:
    from bindocracy.tools import launch_spec, load_configs, plan

    loaded = load_configs(*af3_config_files)
    spec = launch_spec(plan(loaded, tmp_path / "run"), 0)
    binds = [a for a in spec.argv if ":/models:ro" in a]
    assert binds, "AF3 weights must be bound read-only, never embedded"


def test_the_driver_folds_a_lone_chain_when_asked(af3_driver) -> None:
    """The monomer path: one protein, no target and no alignment."""
    spec = af3_driver.fold_input_monomer("gfp", "ACDEFGHIKL", 42)
    assert len(spec["sequences"]) == 1
    protein = spec["sequences"][0]["protein"]
    assert protein["id"] == "A"
    assert protein["unpairedMsa"] == "" and protein["pairedMsa"] == ""


def test_the_complex_input_puts_the_target_in_chain_a(af3_driver) -> None:
    """Chain A for the target matches general.target.chain_id and the Chai-1
    driver, so both co-folding scorers agree on which chain is which."""
    spec = af3_driver.fold_input("d", "TARGETSEQ", "BINDERSEQ", ">t\nTARGETSEQ\n", 0)
    ids = [s["protein"]["id"] for s in spec["sequences"]]
    seqs = [s["protein"]["sequence"] for s in spec["sequences"]]
    assert ids == ["A", "B"]
    assert seqs == ["TARGETSEQ", "BINDERSEQ"]
    # The binder is MSA-free, as everywhere else here.
    assert spec["sequences"][1]["protein"]["unpairedMsa"] == ""
    assert spec["sequences"][0]["protein"]["unpairedMsa"].startswith(">t")


def test_every_metric_the_driver_emits_is_registered(af3_driver) -> None:
    emitted = {
        "complex_ptm", "iptm", "aggregate_score", "bt_iptm", "tb_iptm",
        "iptm_min", "binder_ptm", "complex_plddt", "binder_plddt",
        "bt_pae", "tb_pae", "has_clashes", "mono_plddt", "mono_ptm",
    }
    assert emitted <= set(registered_keys())


def test_af3_does_not_inherit_mosaics_jax_cache(af3_config_files, tmp_path) -> None:
    """A workflow that runs mosaic and AF3 exports
    JAX_COMPILATION_CACHE_DIR=/jax_cache, which exists only inside mosaic.sif.
    AF3 then dies with `NOT_FOUND: /jax_cache/...`, naming a directory nobody
    configured. Observed 2026-09-09; the fix also earns the cache, since 434
    designs of varying length otherwise recompile per length."""
    from bindocracy.tools import launch_spec, load_configs, plan

    loaded = load_configs(*af3_config_files)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)
    cache = spec.env["SINGULARITYENV_JAX_COMPILATION_CACHE_DIR"]
    assert cache != "/jax_cache"
    assert cache.endswith("jax_cache")
    # And it must exist before the container starts.
    assert any(str(d).endswith("jax_cache") for d in spec.mkdirs)
