"""Saved structures, model variants, and the flags that reach the container.

The point of saving a pose is that a later epitope, contact or clash pass can
read it instead of folding everything again. That only works if the pointer
survives into the database, so these tests follow it from the driver's JSONL to
an artifact row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools.scorer.adapter import _read_metrics
from bindocracy.tools.scorer.config import ACCEPTS_TARGET_MSA, HAS_SAMPLER

WHEN = __import__("datetime").datetime(2026, 9, 9, tzinfo=__import__("datetime").UTC)


def _row(index: int, **over) -> str:
    row = {
        "index": index, "condition": "complex", "replicate": 0,
        "metrics": {"iptm": 0.7}, "seconds": 1.0, "failed": None,
        "structure": f"structures/complex/design-{index:06d}_s0.pdb",
    }
    row.update(over)
    return json.dumps(row)


def test_a_saved_pose_becomes_a_pointer_the_adapter_can_use(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(_row(0) + "\n" + _row(1) + "\n")
    _, counts = _read_metrics(
        path, index_to_design={0: "d-a", 1: "d-b"}, prefix="boltz2",
        run_id="r1", fallback_time=WHEN,
    )
    assert counts["structures"] == [
        ("structures/complex/design-000000_s0.pdb", "d-a", "complex", 0),
        ("structures/complex/design-000001_s0.pdb", "d-b", "complex", 0),
    ]


def test_a_pose_is_recorded_even_when_the_metrics_are_empty(tmp_path) -> None:
    """A structure that exists is worth pointing at whether or not the numbers
    came out; the two are separate facts about the same fold."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(_row(0, metrics={}) + "\n")
    records, counts = _read_metrics(
        path, index_to_design={0: "d-a"}, prefix="boltz2",
        run_id="r1", fallback_time=WHEN,
    )
    assert records == []
    assert len(counts["structures"]) == 1


def test_a_fold_that_saved_no_pose_contributes_nothing(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(_row(0, structure=None) + "\n")
    _, counts = _read_metrics(
        path, index_to_design={0: "d-a"}, prefix="boltz2",
        run_id="r1", fallback_time=WHEN,
    )
    assert counts["structures"] == []


def test_the_condition_travels_with_the_pose(tmp_path) -> None:
    """A complex pose and the monomer pose of the same design are different
    structures; an artifact row that lost the condition could not say which."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        _row(0) + "\n"
        + _row(0, condition="monomer",
               structure="structures/monomer/design-000000_s0.pdb") + "\n"
    )
    _, counts = _read_metrics(
        path, index_to_design={0: "d-a"}, prefix="boltz2",
        run_id="r1", fallback_time=WHEN,
    )
    assert {c for _, _, c, _ in counts["structures"]} == {"complex", "monomer"}


# --- what the container is told -------------------------------------------


def test_saving_a_file_does_not_change_the_protocol(scorer_configs) -> None:
    """`save_structures` is outside `protocol` on purpose: two runs differing
    only in whether they kept the pose measured the same thing."""
    _, model = scorer_configs
    before = model.protocol_hash
    flipped = model.model_copy(update={"save_structures": not model.save_structures})
    assert flipped.protocol_hash == before


def test_the_protenix_variant_is_part_of_the_metric_name(scorer_configs) -> None:
    """mini and base are different weights. One `protenix_iptm` column holding
    both would silently average a 2-step sampler with a 20-step one."""
    _, model = scorer_configs
    mini = model.model_copy(
        update={"model": model.model.model_copy(update={"name": "protenix", "variant": "mini"})}
    )
    base = model.model_copy(
        update={"model": model.model.model_copy(update={"name": "protenix", "variant": "base"})}
    )
    assert mini.metric_prefix == "protenix_mini"
    assert base.metric_prefix == "protenix_base"
    assert mini.protocol_hash != base.protocol_hash


def test_only_checkpoints_on_disk_are_offered() -> None:
    """`tiny` has a loader in mosaic and no weights here; naming it would
    reach for a download that offline mode turns into an opaque crash."""
    from bindocracy.tools.scorer.config import ScoringModel

    with pytest.raises(ValueError):
        ScoringModel.model_validate({
            "name": "protenix", "recycling_steps": 3, "sampling_steps": 20,
            "num_samples": 1, "variant": "tiny", "use_target_msa": True,
        })


def test_af2_now_accepts_a_target_msa() -> None:
    """Verified by running it: 3/3 designs across both readers, 2026-09-09.
    True only against the dev source or a rebuilt image -- the shipped
    mosaic.sif still asserts at models/af2.py:391."""
    assert ACCEPTS_TARGET_MSA["af2"] is True
    assert HAS_SAMPLER["af2"] is False


def test_promera_cannot_take_the_campaign_alignment() -> None:
    """`models/promera.py:77-85` raises NotImplementedError on any chain with an
    msa_path -- Promera resolves alignments through tinyprot's sequence-keyed
    cache and cannot be pointed at an a3m. Observed 2026-09-09 after this flag
    was set True from reading its imports; the run is what settled it.

    Failing loudly is the right behaviour and worth pinning: the alternative,
    silently substituting a ColabFold search, is the bug that made OpenFold3 and
    Protenix incomparable.
    """
    assert ACCEPTS_TARGET_MSA["promera"] is False
    assert HAS_SAMPLER["promera"] is True


def test_promera_needs_the_binder_feature_path(chai1_driver=None) -> None:
    """Its `model_output` applies the PSSM through `apply_binder_sequence`,
    which asserts on a target-only feature pack. The path is not cosmetic --
    `binder_features` stubs the binder's sidechains -- so it is recorded per
    run rather than inferred."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "drivers" / "scorer" / "score_designs.py"
    spec = importlib.util.spec_from_file_location("score_designs_fp", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.BINDER_FEATURE_PATH == {"promera"}


def test_the_driver_accepts_the_flags_the_connector_adds(scorer_driver) -> None:
    """A flag that grows on the connector side and not the driver's launches a
    GPU job that dies in argparse. Parsed here with the driver's own parser."""
    import sys

    argv = [
        "score_designs.py",
        "--design-set", "/tmp/set.fasta", "--target-fasta", "/tmp/t.fasta",
        "--model", "protenix", "--recycling", "3", "--num-samples", "1",
        "--seed", "0", "--readers", "complex", "--shard", "0",
        "--num-shards", "1", "--task-id", "0", "--save-dir", "/tmp/out",
        "--sampling-steps", "20", "--variant", "base", "--save-structures",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        parsed = scorer_driver.parse_args()
    finally:
        sys.argv = saved
    assert parsed.variant == "base"
    assert parsed.save_structures is True


@pytest.fixture
def scorer_driver():
    """The mosaic driver's module. Its container-only imports all sit inside
    functions, so it loads on the host."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "drivers" / "scorer" / "score_designs.py"
    spec = importlib.util.spec_from_file_location("score_designs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




# --- deprecation -------------------------------------------------------------


def test_mosaics_of3_is_refused_for_new_runs(scorer_configs) -> None:
    """It does not fold correctly: GFP pLDDT 38.5 at 24 A from consensus where
    the official image gives 88.7 at 3.9 A, and 0.597 on Nipah-G, below the
    sequence-only control bar."""
    from bindocracy.tools.scorer.preflight import preflight_scorer

    general, model = scorer_configs
    broken = model.model_copy(
        update={"model": model.model.model_copy(
            update={"name": "of3", "sampling_steps": 25})}
    )
    with pytest.raises(ConfigPreflightError) as error:
        preflight_scorer(general, broken)
    message = str(error.value)
    assert "deprecated" in message
    assert "of3_upstream" in message, "the refusal must name the replacement"


def test_a_historical_run_still_loads_from_its_manifest() -> None:
    """The literal stays valid on purpose. `configs_of()` reads a manifest's
    stored config with a bare model_validate and never calls preflight, so a
    run made before the deprecation is still relaunchable and collectable.
    Refusing at the validator instead would strand every past of3 run."""
    from bindocracy.tools.scorer.config import ScoringModel

    archived = ScoringModel.model_validate({
        "name": "of3", "recycling_steps": 3, "sampling_steps": 25,
        "num_samples": 1, "use_target_msa": True,
    })
    assert archived.name == "of3"


def test_it_can_be_chosen_deliberately(scorer_configs) -> None:
    """One honest reason to: reproducing a historical comparison. The config
    then says out loud that a known-bad scorer was picked on purpose."""
    from bindocracy.tools.scorer.preflight import preflight_scorer

    general, model = scorer_configs
    deliberate = model.model_copy(update={
        "model": model.model.model_copy(update={"name": "of3", "sampling_steps": 25}),
        "allow_deprecated": True,
    })
    # Gets past the deprecation gate; fails later on the fixture's absent
    # design set, which is the next check rather than this one.
    with pytest.raises(ConfigPreflightError) as error:
        preflight_scorer(general, deliberate)
    assert "deprecated" not in str(error.value)


def test_deprecation_does_not_change_the_protocol_hash(scorer_configs) -> None:
    """`allow_deprecated` is an operator's decision, not a measurement: two
    runs differing only there measured the same thing."""
    _, model = scorer_configs
    before = model.protocol_hash
    assert model.model_copy(update={"allow_deprecated": True}).protocol_hash == before
