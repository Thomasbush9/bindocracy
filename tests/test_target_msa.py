from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import write_configs

from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig, ResourceConfig
from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.runs.msa import prepare_target_msa


@pytest.fixture
def preparation(tmp_path):
    general_path, _ = write_configs(tmp_path)
    general = load_yaml(general_path, GeneralConfig)
    general.target.msa.unlink()
    script = tmp_path / "msa-search.sbatch"
    script.write_text("#!/bin/bash\n")
    image = tmp_path / "msa.sif"
    image.touch()
    database = tmp_path / "database"
    database.mkdir()
    return general, {
        "script": script,
        "image": image,
        "database": database,
        "resources": ResourceConfig(gpus=1, cpus=8, memory_gb=64, walltime="08:00:00"),
    }


def test_existing_wrong_target_is_not_replaced(preparation, monkeypatch):
    general, options = preparation
    general.target.msa.write_text(">wrong-target\nWWWWWW\n")

    def unexpected_submission(*args, **kwargs):
        pytest.fail("an existing alignment must not trigger a replacement search")

    monkeypatch.setattr("bindocracy.runs.msa.subprocess.run", unexpected_submission)
    with pytest.raises(ConfigPreflightError, match="not an alignment"):
        prepare_target_msa(general, **options)
    assert general.target.msa.read_text() == ">wrong-target\nWWWWWW\n"


@pytest.mark.parametrize(
    "outcome", ["failed_job", "missing_output", "wrong_target", "changed_fasta"]
)
def test_unsuccessful_preparation_never_publishes(preparation, monkeypatch, outcome):
    general, options = preparation

    def search(argv, *, env, **kwargs):
        output = Path(env["MSA_OUT"]) / "target.a3m"
        if outcome != "missing_output":
            output.write_text(
                ">target\n" + ("WWWWWW" if outcome == "wrong_target" else "ACDEFG") + "\n"
            )
        if outcome == "changed_fasta":
            general.target.sequence_fasta.write_text(">changed\nWWWWWW\n")
        return subprocess.CompletedProcess(argv, 1 if outcome == "failed_job" else 0, "123", "")

    monkeypatch.setattr("bindocracy.runs.msa.subprocess.run", search)
    with pytest.raises(ConfigPreflightError):
        prepare_target_msa(general, **options)
    assert not general.target.msa.exists()


def test_external_alignment_is_not_clobbered(preparation, monkeypatch):
    general, options = preparation

    def search(argv, *, env, **kwargs):
        (Path(env["MSA_OUT"]) / "target.a3m").write_text(">target\nACDEFG\n")
        general.target.msa.write_text(">externally-created\nACDEFG\n>homolog\nACDEFA\n")
        return subprocess.CompletedProcess(argv, 0, "123", "")

    monkeypatch.setattr("bindocracy.runs.msa.subprocess.run", search)
    with pytest.raises(FileExistsError):
        prepare_target_msa(general, **options)
    assert general.target.msa.read_text() == ">externally-created\nACDEFG\n>homolog\nACDEFA\n"
