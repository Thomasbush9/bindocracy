"""Protein-Hunter's config, its preflight, and the command one task runs.

This is the only tool whose target reaches the container as a string rather
than a file, so the checks that matter are the ones that make sure the string
is the campaign's target and that the run does not quietly reach the network
for an alignment.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from conftest import write_protein_hunter_configs
from pydantic import ValidationError

from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.tools import launch_spec, load_configs, plan, resources
from bindocracy.tools.protein_hunter.config import ProteinHunterConfig
from bindocracy.tools.protein_hunter.plugin import design_name


def flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def flags(argv: tuple[str, ...], name: str) -> list[str]:
    return [value for option, value in pairwise(argv) if option == name]


# --- configuration ---------------------------------------------------------


def test_a_protein_hunter_config_loads(protein_hunter_configs) -> None:
    loaded = load_configs(*protein_hunter_configs)

    assert isinstance(loaded.model, ProteinHunterConfig)
    assert loaded.tool == "protein_hunter"
    # Nothing to fold in: this tool references no external scientific file, so
    # its stored config is already the whole answer.
    assert loaded.model.sampling.cycles == 5


def test_the_msa_mode_has_no_default(tmp_path: Path) -> None:
    """`single` and `mmseqs` are different experiments; the author picks."""
    general, model = write_protein_hunter_configs(tmp_path)
    document = yaml.safe_load(model.read_text())
    del document["msa"]["mode"]
    model.write_text(yaml.safe_dump(document))

    with pytest.raises(ValidationError, match="mode"):
        load_configs(general, model)


def test_an_inverted_length_range_is_refused(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(
        tmp_path, sampling={"min_binder_length": 120, "max_binder_length": 65}
    )

    with pytest.raises(ValidationError, match="max_binder_length"):
        load_configs(general, model)


# --- preflight -------------------------------------------------------------


def test_mmseqs_without_an_alignment_is_refused(tmp_path: Path) -> None:
    """It would call api.colabfold.com from a node with no route to it."""
    general, model = write_protein_hunter_configs(tmp_path)
    document = yaml.safe_load(general.read_text())
    del document["target"]["msa"]
    general.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigPreflightError, match="api.colabfold.com"):
        load_configs(general, model)


def test_single_mode_needs_no_alignment(tmp_path: Path) -> None:
    """Folding the target with no MSA is a choice, not an error."""
    general, model = write_protein_hunter_configs(tmp_path, msa={"mode": "single"})
    document = yaml.safe_load(general.read_text())
    del document["target"]["msa"]
    general.write_text(yaml.safe_dump(document))

    loaded = load_configs(general, model)
    assert loaded.preflight.msa is None
    assert set(loaded.preflight.input_files) == {"target_fasta"}


def test_an_empty_alignment_is_refused(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(tmp_path)
    (tmp_path / "target.a3m").write_text("")

    with pytest.raises(ConfigPreflightError, match="no sequences"):
        load_configs(general, model)


def test_a_missing_alignment_is_refused(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(tmp_path)
    (tmp_path / "target.a3m").unlink()

    with pytest.raises(ConfigPreflightError, match="does not exist"):
        load_configs(general, model)


# --- planning --------------------------------------------------------------


def test_designs_are_trajectories_times_cycles(tmp_path: Path) -> None:
    """The counting trap: 4 trajectories at 5 cycles is 20 designs, not 4."""
    general, model = write_protein_hunter_configs(
        tmp_path, sampling={"jobs": 2, "trajectories_per_job": 4, "cycles": 5}
    )
    manifest = plan(load_configs(general, model), tmp_path / "run")

    assert manifest.designs_per_task == 20
    assert manifest.to_run_record().n_requested == 40
    assert manifest.workflow["trajectories_per_task"] == 4
    assert manifest.workflow["cycles"] == 5


def test_the_plan_records_that_the_run_is_not_reproducible(
    protein_hunter_configs, tmp_path
) -> None:
    """The pipeline has no seed flag at all, and every other tool here has one."""
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")

    assert manifest.workflow["reproducible"] is False


def test_the_plan_records_the_thresholds_that_define_passing(
    protein_hunter_configs, tmp_path
) -> None:
    """They are not in the output, and they decide what n_passed means."""
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")

    assert manifest.workflow["high_iptm_threshold"] == 0.7
    assert manifest.workflow["high_plddt_threshold"] == 0.7


def test_the_run_digests_the_sequence_and_the_alignment(
    protein_hunter_configs, tmp_path
) -> None:
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")

    assert set(manifest.inputs) == {"target_fasta", "target_msa"}
    manifest.verify_inputs()


def test_the_design_name_is_derived_from_the_campaign_target() -> None:
    """Not authored, so it cannot drift; word characters only, as chai demands."""
    assert design_name("dio3-cut") == "dio3_cut"
    assert design_name("DIO3 cut v2") == "DIO3_cut_v2"


# --- launch ----------------------------------------------------------------


def test_the_target_is_passed_as_a_sequence_not_a_path(
    protein_hunter_configs, tmp_path
) -> None:
    """Unique among these tools: the target is an argv value."""
    loaded = load_configs(*protein_hunter_configs)
    manifest = plan(loaded, tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    expected = read_single_fasta(loaded.general.target.sequence_fasta)
    assert flag(argv, "--protein-seqs") == expected
    assert not Path(flag(argv, "--protein-seqs")).exists()


def test_a_substituted_target_stops_the_launch(protein_hunter_configs, tmp_path) -> None:
    """Nothing downstream would notice: there is no path to check afterwards."""
    loaded = load_configs(*protein_hunter_configs)
    manifest = plan(loaded, tmp_path / "run")
    loaded.general.target.sequence_fasta.write_text(">target\nWWWWWWWWWW\n")

    with pytest.raises(ValueError, match="designs against"):
        launch_spec(manifest, 0)


def test_the_alignment_reaches_the_driver_only_in_mmseqs_mode(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(tmp_path)
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert flag(argv, "--a3m") == str(tmp_path / "target.a3m")
    assert flag(argv, "--msa-mode") == "mmseqs"


def test_single_mode_passes_no_alignment(tmp_path: Path) -> None:
    general, model = write_protein_hunter_configs(tmp_path, msa={"mode": "single"})
    manifest = plan(load_configs(general, model), tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    assert flag(argv, "--msa-mode") == "single"
    assert flags(argv, "--a3m") == []


def test_one_task_runs_the_archived_driver(protein_hunter_configs, tmp_path) -> None:
    loaded = load_configs(*protein_hunter_configs)
    manifest = plan(loaded, tmp_path / "run")
    spec = launch_spec(manifest, 0)

    assert spec.argv[:4] == ("singularity", "run", "--cleanenv", "--nv")
    assert "python" in spec.argv
    assert spec.argv[spec.argv.index("python") + 1] == str(
        manifest.path(manifest.provenance["driver"].path)
    )
    assert flag(spec.argv, "--save-dir") == str(manifest.path("tasks/0000"))
    assert flag(spec.argv, "--num-designs") == "3"
    assert flag(spec.argv, "--num-cycles") == "5"


def test_the_template_flag_is_never_passed(protein_hunter_configs, tmp_path) -> None:
    """A value that is not an existing file fetches from RCSB and hangs."""
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")

    assert "--template_path" not in launch_spec(manifest, 0).argv
    assert "--template-path" not in launch_spec(manifest, 0).argv


def test_every_cache_the_image_derives_is_node_local(
    protein_hunter_configs, tmp_path
) -> None:
    """XDG_CACHE_HOME, HF_HOME and TORCH_HOME all hang off TMPDIR in the image."""
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")
    spec = launch_spec(manifest, 0)
    node_tmp = spec.env["TMPDIR"]

    assert manifest.run_id[:8] in node_tmp
    assert spec.env["SINGULARITYENV_TMPDIR"] == node_tmp
    assert spec.env["SINGULARITYENV_PROTEIN_HUNTER_RUNTIME_CACHE"].startswith(node_tmp)
    # The runscript mkdirs under TMPDIR on its first line, under `set -eu`.
    assert Path(node_tmp) in spec.mkdirs


def test_the_allocated_device_survives_cleanenv(
    protein_hunter_configs, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")

    assert launch_spec(manifest, 0).env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "1"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert "SINGULARITYENV_CUDA_VISIBLE_DEVICES" not in launch_spec(manifest, 0).env


def test_resources_come_from_the_config(protein_hunter_configs) -> None:
    loaded = load_configs(*protein_hunter_configs)

    assert resources(loaded) == {
        "slurm_account": "test-account",
        "slurm_partition": "test-gpu",
        "gres": "gpu:1",
        "cpus_per_task": 8,
        "mem_mb": 64 * 1024,
        "runtime": 8 * 60,
    }


def test_the_launch_reads_the_manifest_not_the_edited_yaml(
    protein_hunter_configs, tmp_path
) -> None:
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")
    document = yaml.safe_load(Path(protein_hunter_configs[1]).read_text())
    document["sampling"]["trajectories_per_job"] = 999
    Path(protein_hunter_configs[1]).write_text(yaml.safe_dump(document))

    assert flag(launch_spec(manifest, 0).argv, "--num-designs") == "3"


# --- the driver seam -------------------------------------------------------


def test_the_driver_accepts_exactly_what_the_connector_launches(
    protein_hunter_driver, protein_hunter_configs, tmp_path
) -> None:
    loaded = load_configs(*protein_hunter_configs)
    manifest = plan(loaded, tmp_path / "run")
    argv = launch_spec(manifest, 0).argv

    # argparse exits non-zero on an unknown or missing flag, so this failing
    # means the connector would launch a job that dies immediately.
    parsed = protein_hunter_driver.parse_args(
        list(argv[argv.index("python") + 2 :])
    )

    assert Path(parsed.save_dir) == manifest.path("tasks/0000")
    assert parsed.num_designs == loaded.model.sampling.trajectories_per_job
    assert parsed.num_cycles == loaded.model.sampling.cycles
    assert parsed.msa_mode == "mmseqs"
    assert parsed.max_seqs == 512


def test_the_driver_writes_where_the_adapter_reads(
    protein_hunter_driver, protein_hunter_configs, tmp_path
) -> None:
    manifest = plan(load_configs(*protein_hunter_configs), tmp_path / "run")
    task = manifest.tasks[0]
    argv = launch_spec(manifest, 0).argv
    parsed = protein_hunter_driver.parse_args(list(argv[argv.index("python") + 2 :]))

    assert Path(parsed.save_dir) / "summary_all_runs.csv" == manifest.path(task.designs)


def test_seeding_rewrites_only_the_first_header(
    protein_hunter_driver, tmp_path: Path
) -> None:
    """The parser does int(line[1:]) on the first header and nothing else."""
    a3m = tmp_path / "in.a3m"
    a3m.write_text(">DIO3 query\nACDEFG\n>hit_1 something\nACDE-G\n")
    env_dir = protein_hunter_driver.seed_msa_cache(a3m, tmp_path / "B_env", 512)

    written = (env_dir / "uniref.a3m").read_text().splitlines()
    assert written[0] == ">101"
    assert written[2] == ">hit_1 something"
    # Presence of these two is what stops the HTTP call and the untar.
    assert (env_dir / "out.tar.gz").exists()
    assert (env_dir / "bfd.mgnify30.metaeuk30.smag30.a3m").exists()


def test_seeding_subsamples_to_max_seqs(protein_hunter_driver, tmp_path: Path) -> None:
    """Downstream hardcodes 4096, so the whole alignment would be carried."""
    a3m = tmp_path / "in.a3m"
    a3m.write_text("".join(f">hit_{i}\nACDEFG\n" for i in range(50)))
    env_dir = protein_hunter_driver.seed_msa_cache(a3m, tmp_path / "B_env", 10)

    assert (env_dir / "uniref.a3m").read_text().count(">") == 10


def test_an_empty_alignment_stops_the_driver(
    protein_hunter_driver, tmp_path: Path
) -> None:
    a3m = tmp_path / "in.a3m"
    a3m.write_text("")

    with pytest.raises(SystemExit, match="no sequences"):
        protein_hunter_driver.seed_msa_cache(a3m, tmp_path / "B_env", 512)
