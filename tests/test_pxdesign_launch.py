"""PXDesign's config, its preflight, and the command one task runs.

Most of these guard a CLI default that is wrong rather than absent: a preset
that configures no filters, an eta schedule the CLI overwrites, and a seed that
comes from the clock. None of them fails loudly, so all of them are decided in
the config and asserted here.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from conftest import (
    PXDESIGN_TASK_NAME,
    pxdesign_spec,
    write_pxdesign_configs,
    write_pxdesign_msa,
)
from pydantic import ValidationError

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import launch_spec, load_configs, plan, resources
from bindocracy.tools.pxdesign.config import PXDesignConfig


def flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def flags(argv: tuple[str, ...], name: str) -> list[str]:
    return [value for option, value in pairwise(argv) if option == name]


# --- configuration ---------------------------------------------------------


def test_a_pxdesign_config_loads_and_keeps_its_spec(pxdesign_configs) -> None:
    loaded = load_configs(*pxdesign_configs)

    assert isinstance(loaded.model, PXDesignConfig)
    assert loaded.tool == "pxdesign"
    # The spec is folded in, so the stored config says what the run asked for
    # rather than where the question was written down.
    assert loaded.model.spec.contents == loaded.preflight.spec
    assert loaded.model.spec.contents["binder_length"] == 80


def test_a_recovered_config_carrying_its_spec_still_validates(pxdesign_configs) -> None:
    loaded = load_configs(*pxdesign_configs)
    stored = loaded.model.model_dump(mode="json")

    assert PXDesignConfig.model_validate(stored).spec.contents is not None


def test_the_preset_that_configures_no_filters_cannot_be_asked_for(tmp_path: Path) -> None:
    """`custom` is the CLI's own default, and it runs with no filters at all."""
    general, model = write_pxdesign_configs(tmp_path)
    document = yaml.safe_load(model.read_text())
    document["sampling"]["preset"] = "custom"
    model.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError, match="preview|extended"):
        load_configs(general, model)


def test_the_preset_has_no_default(tmp_path: Path) -> None:
    """Leaving it out must fail, not silently inherit the CLI's `custom`."""
    general, model = write_pxdesign_configs(tmp_path)
    document = yaml.safe_load(model.read_text())
    del document["sampling"]["preset"]
    model.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError, match="preset"):
        load_configs(general, model)


def test_the_eta_schedule_defaults_to_the_containers_intent(pxdesign_configs) -> None:
    """Not the CLI's const / 2.5 / 2.5, which silently overwrites the config."""
    sampling = load_configs(*pxdesign_configs).model.sampling

    assert (sampling.eta_type, sampling.eta_min, sampling.eta_max) == (
        "piecewise_65", 1.0, 2.5,
    )


# --- preflight -------------------------------------------------------------


def test_a_spec_for_another_structure_is_refused(tmp_path: Path) -> None:
    """The silent-wrong-answer case: everything else about the run is fine."""
    elsewhere = tmp_path / "other.cif"
    elsewhere.write_text("data_other\n#\n")
    spec = pxdesign_spec(elsewhere, write_pxdesign_msa(tmp_path))
    general, model = write_pxdesign_configs(tmp_path, spec=spec)

    with pytest.raises(ConfigPreflightError, match="not the campaign target"):
        load_configs(general, model)


def test_a_spec_with_no_task_name_is_refused(tmp_path: Path) -> None:
    """Without it the results directory is named after the spec's filename."""
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path))
    del spec["task_name"]
    general, model = write_pxdesign_configs(tmp_path, spec=spec)

    with pytest.raises(ConfigPreflightError, match="task_name"):
        load_configs(general, model)


@pytest.mark.parametrize("missing", ["non_pairing.a3m", "pairing.a3m"])
def test_an_incomplete_msa_directory_is_refused(tmp_path: Path, missing: str) -> None:
    """It is what keeps the Protenix stage from calling an MSA service."""
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    msa = write_pxdesign_msa(tmp_path)
    (msa / missing).unlink()
    general, model = write_pxdesign_configs(tmp_path, spec=pxdesign_spec(cif, msa))

    with pytest.raises(ConfigPreflightError, match="cannot reach"):
        load_configs(general, model)


def test_a_chain_with_no_msa_at_all_is_refused(tmp_path: Path) -> None:
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path))
    del spec["target"]["chains"]["A"]["msa"]
    general, model = write_pxdesign_configs(tmp_path, spec=spec)

    with pytest.raises(ConfigPreflightError, match="names no `msa` directory"):
        load_configs(general, model)


def test_a_spec_with_no_binder_length_is_refused(tmp_path: Path) -> None:
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path))
    del spec["binder_length"]
    general, model = write_pxdesign_configs(tmp_path, spec=spec)

    with pytest.raises(ConfigPreflightError, match="binder_length"):
        load_configs(general, model)


def test_a_spec_conditioning_on_another_epitope_is_refused(tmp_path: Path) -> None:
    """PXDesign hotspots are label_seq_id integers, so numbers are comparable."""
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path), hotspots=[40, 99, 107])
    general, model = write_pxdesign_configs(tmp_path, spec=spec)
    document = yaml.safe_load(general.read_text())
    document["target"]["hotspots"] = ["A10", "A12"]
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="conditions on residues"):
        load_configs(general, model)


def test_a_relative_target_path_is_refused(tmp_path: Path) -> None:
    """Preflight and the container resolve it against different directories."""
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path))
    spec["target"]["file"] = "target.cif"
    general, model = write_pxdesign_configs(tmp_path, spec=spec)

    with pytest.raises(ConfigPreflightError, match="absolute"):
        load_configs(general, model)


def test_a_relative_msa_path_is_refused(tmp_path: Path) -> None:
    cif = tmp_path / "target.cif"
    cif.write_text("data_target\n#\n")
    spec = pxdesign_spec(cif, write_pxdesign_msa(tmp_path))
    spec["target"]["chains"]["A"]["msa"] = "msa/A"
    general, model = write_pxdesign_configs(tmp_path, spec=spec)

    with pytest.raises(ConfigPreflightError, match="absolute path"):
        load_configs(general, model)


def test_a_missing_target_structure_is_refused(tmp_path: Path) -> None:
    general, model = write_pxdesign_configs(tmp_path)
    document = yaml.safe_load(general.read_text())
    del document["target"]["structure_cif"]
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="structure_cif"):
        load_configs(general, model)


# --- planning --------------------------------------------------------------


def test_the_plan_asks_for_exactly_what_the_table_will_hold(tmp_path: Path) -> None:
    """PXDesign pads summary.csv to --N_sample rather than returning fewer."""
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 2, "designs_per_job": 6, "seed_base": 0, "preset": "extended"},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert manifest.designs_per_task == 6
    assert manifest.to_run_record().n_requested == 12
    # No separate "generated" count: unlike BoltzGen there is no pool to trim.
    assert manifest.tasks[0].n_generated is None


def test_the_plan_names_the_table_the_adapter_reads(pxdesign_configs, tmp_path) -> None:
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert manifest.tasks[0].designs == (
        f"tasks/0000/design_outputs/{PXDESIGN_TASK_NAME}/summary.csv"
    )


def test_the_plan_records_what_only_the_spec_knows(pxdesign_configs, tmp_path) -> None:
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert manifest.workflow["binder_length"] == 80
    assert manifest.workflow["task_name"] == PXDESIGN_TASK_NAME
    assert manifest.workflow["preset"] == "extended"
    # The schedule that actually ran, which is neither default.
    assert manifest.workflow["eta"] == {"type": "piecewise_65", "min": 1.0, "max": 2.5}


def test_the_run_digests_the_geometry_and_the_alignments(pxdesign_configs, tmp_path) -> None:
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert set(manifest.inputs) == {"spec_target", "msa_A_non_pairing", "msa_A_pairing"}
    manifest.verify_inputs()


def test_editing_an_alignment_stops_a_planned_run(pxdesign_configs, tmp_path) -> None:
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")
    alignment = Path(manifest.inputs["msa_A_pairing"].uri)
    alignment.write_text(">target\nWWWWWW\n")

    with pytest.raises(Exception, match="no longer match"):
        manifest.verify_inputs()


def test_the_spec_is_archived(pxdesign_configs, tmp_path) -> None:
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert set(manifest.provenance) == {"spec"}
    assert manifest.path(manifest.provenance["spec"].path).is_file()


# --- launch ----------------------------------------------------------------


def test_one_task_runs_the_archived_spec(pxdesign_configs, tmp_path) -> None:
    loaded = load_configs(*pxdesign_configs)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)

    assert spec.argv[:4] == ("singularity", "run", "--cleanenv", "--nv")
    assert "pipeline" in spec.argv
    assert flag(spec.argv, "-i") == str(manifest.path(manifest.provenance["spec"].path))
    assert flag(spec.argv, "-o") == str(manifest.path("tasks/0000"))
    assert flag(spec.argv, "--N_sample") == "4"


def test_the_preset_is_always_passed(pxdesign_configs, tmp_path) -> None:
    """Omitting it would run `custom`, which configures no filters."""
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert flag(launch_spec(manifest, 0).argv, "--preset") == "extended"


def test_the_eta_schedule_is_passed_back_explicitly(pxdesign_configs, tmp_path) -> None:
    """The CLI emits every shared option, so silence means const / 2.5 / 2.5."""
    argv = launch_spec(plan(load_configs(*pxdesign_configs), tmp_path / "run"), 0).argv

    assert flag(argv, "--eta_type") == "piecewise_65"
    assert flag(argv, "--eta_min") == "1.0"
    assert flag(argv, "--eta_max") == "2.5"


def test_each_task_samples_from_its_own_seed(tmp_path: Path) -> None:
    """Without --seeds PXDesign seeds from the clock, so nothing is repeatable."""
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 3, "designs_per_job": 2, "seed_base": 100, "preset": "extended"},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    seeds = [flag(launch_spec(manifest, task).argv, "--seeds") for task in range(3)]
    assert seeds == ["100", "101", "102"]
    # --N_max_runs 1 is what makes exactly one seed the right number.
    assert flag(launch_spec(manifest, 0).argv, "--N_max_runs") == "1"


def test_the_working_directory_is_the_tasks_own(tmp_path: Path) -> None:
    """PXDesign resolves ./msa_cache against it, and two tasks must not share."""
    general, model = write_pxdesign_configs(
        tmp_path,
        sampling={"jobs": 2, "designs_per_job": 2, "seed_base": 0, "preset": "extended"},
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert flag(launch_spec(manifest, 0).argv, "--pwd") == str(manifest.path("tasks/0000"))
    assert flag(launch_spec(manifest, 1).argv, "--pwd") == str(manifest.path("tasks/0001"))


def test_every_jit_cache_is_node_local(pxdesign_configs, tmp_path) -> None:
    """TMPDIR on Lustre kills any Triton-JIT tool with Errno 39."""
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")
    spec = launch_spec(manifest, 0)
    node_tmp = spec.env["TMPDIR"]

    assert manifest.run_id[:8] in node_tmp
    for name in ("SINGULARITYENV_TMPDIR", "SINGULARITYENV_PXDESIGN_CACHE",
                 "SINGULARITYENV_TRITON_CACHE_DIR"):
        assert spec.env[name].startswith(node_tmp)
    assert Path(node_tmp) in spec.mkdirs


def test_the_boolean_pass_through_flags_are_strings(pxdesign_configs, tmp_path) -> None:
    """They reach an argparse that reads them as values, not as store_true."""
    argv = launch_spec(plan(load_configs(*pxdesign_configs), tmp_path / "run"), 0).argv

    assert flag(argv, "--use_fast_ln") == "True"
    assert flag(argv, "--use_deepspeed_evo_attention") == "True"


def test_turning_a_fused_kernel_off_says_so(tmp_path: Path) -> None:
    general, model = write_pxdesign_configs(
        tmp_path, runtime={"use_fast_ln": False, "dtype": "fp32"}
    )
    argv = launch_spec(plan(load_configs(general, model), tmp_path / "run"), 0).argv

    assert flag(argv, "--use_fast_ln") == "False"
    assert flag(argv, "--dtype") == "fp32"


def test_the_allocated_device_survives_cleanenv(pxdesign_configs, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert launch_spec(manifest, 0).env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "2"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert "SINGULARITYENV_CUDA_VISIBLE_DEVICES" not in launch_spec(manifest, 0).env


def test_resources_come_from_the_config(pxdesign_configs) -> None:
    loaded = load_configs(*pxdesign_configs)

    assert resources(loaded) == {
        "slurm_account": "test-account",
        "slurm_partition": "test-gpu",
        "gres": "gpu:1",
        "cpus_per_task": 16,
        "mem_mb": 96 * 1024,
        "runtime": 10 * 60,
    }


def test_the_launch_reads_the_manifest_not_the_edited_yaml(pxdesign_configs, tmp_path) -> None:
    loaded = load_configs(*pxdesign_configs)
    manifest = plan(loaded, tmp_path / "run")
    document = yaml.safe_load(Path(pxdesign_configs[1]).read_text())
    document["sampling"]["designs_per_job"] = 999
    Path(pxdesign_configs[1]).write_text(yaml.safe_dump(document))

    assert flag(launch_spec(manifest, 0).argv, "--N_sample") == "4"


def test_nothing_is_bound_that_the_site_already_binds(pxdesign_configs, tmp_path) -> None:
    """`/n` is a system bind path here, so naming it again would be noise."""
    manifest = plan(load_configs(*pxdesign_configs), tmp_path / "run")

    assert flags(launch_spec(manifest, 0).argv, "--bind") == []
