"""BoltzGen preflight and launch.

The preflight test that matters is the wrong-target one. A BoltzGen spec
carries its own structure path, so it can drift away from the campaign target
while every other check still passes — and the run then designs against the
wrong protein and looks completely normal (harness-design section 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import write_boltzgen_configs

from bindocracy.config import ConfigPreflightError
from bindocracy.tools import load_configs, plan
from bindocracy.tools.boltzgen.launch import boltzgen_launch_spec


@pytest.fixture
def planned(boltzgen_configs, tmp_path: Path):
    loaded = load_configs(*boltzgen_configs)
    return loaded, plan(loaded, tmp_path / "run")


def test_the_spec_is_archived_and_bound_not_the_authored_one(planned) -> None:
    loaded, manifest = planned

    spec = Path(boltzgen_launch_spec(loaded, manifest, 0).argv[6])

    assert list(manifest.provenance) == ["spec"]
    assert spec == manifest.path("provenance/binder_spec.yaml")
    assert spec != loaded.model.spec.template
    assert spec.is_file()


def test_argv_is_a_singularity_run_with_the_configured_counts(planned) -> None:
    loaded, manifest = planned

    argv = boltzgen_launch_spec(loaded, manifest, 0).argv

    assert argv[:5] == ("singularity", "run", "--cleanenv", "--nv",
                        str(loaded.model.runtime.container))
    assert argv[5] == "run"
    flags = dict(zip(argv[7::2], argv[8::2], strict=True))
    assert flags["--output"] == str(manifest.path("tasks/0000"))
    assert flags["--num_designs"] == "8"
    assert flags["--budget"] == "4"
    assert flags["--protocol"] == "protein-anything"
    assert flags["--filter_biased"] == "false"
    assert "--diffusion_batch_size" not in flags


def test_diffusion_batch_size_is_passed_when_set(boltzgen_configs, tmp_path: Path) -> None:
    general_path, model_path = boltzgen_configs
    raw = yaml.safe_load(model_path.read_text())
    raw["sampling"]["diffusion_batch_size"] = 10
    model_path.write_text(yaml.safe_dump(raw))
    loaded = load_configs(general_path, model_path)

    argv = boltzgen_launch_spec(loaded, plan(loaded, tmp_path / "run"), 0).argv

    assert "--diffusion_batch_size" in argv
    assert argv[argv.index("--diffusion_batch_size") + 1] == "10"


def test_tmpdir_is_node_local_and_unique_per_task(planned) -> None:
    """Lustre TMPDIR kills any Triton-JIT tool with Errno 39."""
    loaded, manifest = planned

    first = boltzgen_launch_spec(loaded, manifest, 0).env["TMPDIR"]

    assert first.startswith(str(loaded.model.runtime.node_tmp_root))
    assert manifest.run_id[:8] in first
    assert first.endswith("0000")
    env = boltzgen_launch_spec(loaded, manifest, 0).env
    assert env["SINGULARITYENV_TMPDIR"] == first
    assert env["SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE"].startswith(first)
    assert env["SINGULARITYENV_SSL_CERT_FILE"].endswith("ca-certificates.crt")


def test_the_expected_output_is_the_metrics_table(planned) -> None:
    loaded, manifest = planned

    spec = boltzgen_launch_spec(loaded, manifest, 0)

    assert spec.outputs == (
        manifest.path("tasks/0000/final_ranked_designs/all_designs_metrics.csv"),
    )
    assert spec.log == manifest.path("logs/task-0000.log")


def test_a_spec_aimed_at_another_structure_is_refused(tmp_path: Path) -> None:
    """The silent-wrong-answer case: everything else about this run is valid."""
    general_path, model_path = write_boltzgen_configs(tmp_path)
    decoy = tmp_path / "some_other_protein.cif"
    decoy.write_text("data_decoy\n#\n")
    spec = tmp_path / "binder_spec.yaml"
    spec.write_text(yaml.safe_dump({
        "entities": [
            {"protein": {"id": "B", "sequence": "70..90"}},
            {"file": {"path": str(decoy), "include": [{"chain": {"id": "A"}}]}},
        ]
    }))

    with pytest.raises(ConfigPreflightError, match="does not reference the campaign target"):
        load_configs(general_path, model_path)


def test_a_spec_referencing_a_missing_file_is_refused(tmp_path: Path) -> None:
    general_path, model_path = write_boltzgen_configs(tmp_path)
    spec = tmp_path / "binder_spec.yaml"
    spec.write_text(yaml.safe_dump({
        "entities": [{"file": {"path": str(tmp_path / "gone.cif")}}]
    }))

    with pytest.raises(ConfigPreflightError, match="references files that do not exist"):
        load_configs(general_path, model_path)


def test_boltzgen_requires_a_target_structure(tmp_path: Path) -> None:
    """Mosaic folds the target from sequence; BoltzGen needs the geometry."""
    general_path, model_path = write_boltzgen_configs(tmp_path)
    raw = yaml.safe_load(general_path.read_text())
    del raw["target"]["structure_cif"]
    general_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigPreflightError, match="needs target.structure_cif"):
        load_configs(general_path, model_path)


def test_budget_cannot_exceed_what_is_generated(tmp_path: Path) -> None:
    general_path, model_path = write_boltzgen_configs(tmp_path)
    raw = yaml.safe_load(model_path.read_text())
    raw["sampling"]["budget"] = 99
    model_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="budget cannot exceed num_designs"):
        load_configs(general_path, model_path)
