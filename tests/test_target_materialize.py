from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.runs.materialize import materialize_target
from bindocracy.tools.chai1.preflight import expected_pqt_basename


@pytest.fixture
def target(tmp_path):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">target\nACD\n")
    msa = tmp_path / "uniref90.a3m"
    msa.write_text(">query\nACD\n>homolog description\nAaC-\n")
    return GeneralConfig.model_validate(
        {
            "schema_version": 1,
            "campaign": {"name": "fixture"},
            "target": {"name": "fixture", "sequence_fasta": fasta, "msa": msa, "chain_id": "A"},
            "cluster": {"executor": "slurm", "account": "fixture", "default_partition": "fixture"},
        }
    )


def structure_target(target, tmp_path):
    import biotite.structure as struc
    import numpy as np
    from biotite.structure.io.pdb import PDBFile

    atoms = struc.AtomArray(4)
    atoms.chain_id = ["A", "A", "A", "B"]
    atoms.res_id = [7, 7, 11, 101]
    atoms.ins_code = ["", "A", "", ""]
    atoms.res_name = ["ALA", "CYS", "ASP", "GLY"]
    atoms.atom_name = ["CA"] * 4
    atoms.element = ["C"] * 4
    atoms.coord = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    atoms.set_annotation("atom_id", [17, 18, 24, 80])
    atoms.set_annotation("occupancy", [1.0] * 4)
    atoms.set_annotation("b_factor", [20.0] * 4)
    atoms.set_annotation("charge", [0] * 4)
    path = tmp_path / "input.pdb"
    document = PDBFile()
    document.set_structure(atoms)
    document.write(path)
    return target.model_copy(
        update={"target": target.target.model_copy(update={"structure_pdb": path})}
    )


def test_alignment_handoffs_retain_insertions_and_record_hashes(target, tmp_path):
    import pyarrow.parquet as pq

    directory = tmp_path / "materialized"
    result = materialize_target(target, output_dir=directory, formats=["chai", "pxdesign"])
    parquet = directory / "chai" / expected_pqt_basename("ACD")
    assert pq.read_table(parquet).to_pylist() == [
        {"sequence": "ACD", "source_database": "query", "pairing_key": "", "comment": "query"},
        {
            "sequence": "AaC-",
            "source_database": "uniref90",
            "pairing_key": "",
            "comment": "homolog description",
        },
    ]
    assert (directory / "pxdesign/non_pairing.a3m").read_text() == target.target.msa.read_text()
    assert (directory / "pxdesign/pairing.a3m").read_text() == ">query\nACD\n"
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert (
        manifest["sources"]["msa"]["sha256"]
        == hashlib.sha256(target.target.msa.read_bytes()).hexdigest()
    )
    for name, record in manifest["outputs"].items():
        assert record["sha256"] == hashlib.sha256((directory / name).read_bytes()).hexdigest()


def test_repeated_materialization_is_unchanged(target, tmp_path):
    directory = tmp_path / "materialized"
    first = materialize_target(target, output_dir=directory, formats=["chai", "pxdesign"])
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }
    second = materialize_target(target, output_dir=directory, formats=["pxdesign", "chai", "chai"])
    assert not first["reused"] and second["reused"]
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize(
    "bad",
    [
        ">query\nACE\n",  # unrelated query
        ">query\nAaCD\n",  # inserted query cannot define columns
        ">query\nACD\n>hit\nAC\n",  # short row
        ">query\nACD\n>hit\nAC.D\n",  # incompatible dot semantics
        ">query\nACD\n>empty\n",  # empty record
        "#3,1\n>query\nACD\n",  # unsupported multimer metadata
        ">query\nACD\n>hit\nAJD\n",  # not a shared consumer token
        ">query\nACD\n>hit\nA" + "a" * 256 + "CD\n",  # Chai count overflow
    ],
)
def test_bad_alignment_never_publishes(target, tmp_path, bad):
    target.target.msa.write_text(bad)
    directory = tmp_path / "materialized"
    with pytest.raises(ConfigPreflightError):
        materialize_target(target, output_dir=directory, formats=["chai", "pxdesign"])
    assert not directory.exists()
    assert not list(tmp_path.glob(".materialized-*"))


@pytest.mark.parametrize("change", ["output", "manifest", "source", "extra", "symlink"])
def test_materialization_refuses_tampering(target, tmp_path, change):
    directory = tmp_path / "materialized"
    materialize_target(target, output_dir=directory, formats=["pxdesign"])
    output = directory / "pxdesign/non_pairing.a3m"
    if change == "output":
        output.write_text(">wrong\nWWW\n")
    elif change == "manifest":
        (directory / "materialization.json").write_text("{}")
    elif change == "source":
        target.target.msa.write_text(">query\nACD\n>new\nAC-\n")
    elif change == "extra":
        (directory / "external.txt").write_text("not ours")
    else:
        output.unlink()
        output.symlink_to(target.target.msa)
    original = output.read_bytes()
    with pytest.raises(ConfigPreflightError):
        materialize_target(target, output_dir=directory, formats=["pxdesign"])
    assert output.read_bytes() == original


def test_unrelated_directory_is_never_adopted(target, tmp_path):
    directory = tmp_path / "materialized"
    directory.mkdir()
    with pytest.raises(ConfigPreflightError):
        materialize_target(target, output_dir=directory, formats=["pxdesign"])
    assert list(directory.iterdir()) == []


def test_materialization_supports_filesystems_without_rename_noreplace(
    target, tmp_path, monkeypatch
):
    def unsupported(*_args):
        ctypes.set_errno(errno.EINVAL)
        return -1

    monkeypatch.setattr(
        ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(renameat2=unsupported)
    )
    directory = tmp_path / "materialized"
    materialize_target(target, output_dir=directory, formats=["pxdesign"])
    assert (directory / "pxdesign/non_pairing.a3m").read_text() == target.target.msa.read_text()


def test_concurrent_empty_directory_is_not_overwritten(target, tmp_path, monkeypatch):
    from bindocracy.runs import materialize

    publish = materialize._publish
    reserved = []

    def another_publisher(stage, destination):
        destination.mkdir()
        reserved.append(destination.stat().st_ino)
        publish(stage, destination)

    monkeypatch.setattr(materialize, "_publish", another_publisher)
    directory = tmp_path / "materialized"
    with pytest.raises(ConfigPreflightError):
        materialize_target(target, output_dir=directory, formats=["pxdesign"])
    assert directory.stat().st_ino == reserved[0]
    assert list(directory.iterdir()) == []


def test_failed_publication_removes_empty_reservation(target, tmp_path, monkeypatch):
    def no_space(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "publication failure")

    monkeypatch.setattr(os, "rename", no_space)
    directory = tmp_path / "materialized"
    with pytest.raises(OSError):
        materialize_target(target, output_dir=directory, formats=["pxdesign"])
    assert not directory.exists()


def test_chai_requires_known_database_provenance(target, tmp_path):
    unnamed = tmp_path / "target.a3m"
    unnamed.write_bytes(target.target.msa.read_bytes())
    target = target.model_copy(update={"target": target.target.model_copy(update={"msa": unnamed})})
    with pytest.raises(ConfigPreflightError):
        materialize_target(target, output_dir=tmp_path / "no", formats=["chai"])
    result = materialize_target(
        target, output_dir=tmp_path / "yes", formats=["chai"], msa_source="mgnify"
    )
    assert result["provenance"]["msa_source"] == "mgnify"


def test_structure_roundtrip_retains_all_chains_residue_numbers_and_insertions(target, tmp_path):
    from biotite.structure.io import pdb, pdbx

    target = structure_target(target, tmp_path)
    directory = tmp_path / "converted"
    materialize_target(target, output_dir=directory, formats=["cif"])
    atoms = pdbx.get_structure(pdbx.CIFFile.read(directory / "target.cif"), model=1)
    assert atoms.chain_id.tolist() == ["A", "A", "A", "B"]
    assert atoms.res_id.tolist() == [7, 7, 11, 101]
    assert atoms.ins_code.tolist() == ["", "A", "", ""]
    assert atoms.res_name.tolist() == ["ALA", "CYS", "ASP", "GLY"]
    target = target.model_copy(
        update={
            "target": target.target.model_copy(
                update={"structure_pdb": None, "structure_cif": directory / "target.cif"}
            )
        }
    )
    materialize_target(target, output_dir=tmp_path / "returned", formats=["pdb"])
    restored = pdb.PDBFile.read(tmp_path / "returned/target.pdb").get_structure(model=1)
    assert restored.res_id.tolist() == [7, 7, 11, 101]
    assert restored.ins_code.tolist() == ["", "A", "", ""]
    assert restored.coord.tolist() == atoms.coord.tolist()


def test_failure_after_alignment_conversion_publishes_nothing(target, tmp_path):
    target = structure_target(target, tmp_path)
    target.target.structure_pdb.write_text(
        target.target.structure_pdb.read_text().replace("ASP", "GLU")
    )
    with pytest.raises(ConfigPreflightError):
        materialize_target(
            target, output_dir=tmp_path / "materialized", formats=["chai", "pxdesign", "cif"]
        )
    assert not (tmp_path / "materialized").exists()
    assert not list(tmp_path.glob(".materialized-*"))


@pytest.mark.parametrize("loss", ["numbering", "chain", "precision", "entities", "metadata"])
def test_lossy_cif_to_pdb_refused(target, tmp_path, loss):
    from biotite.structure.io import pdbx

    target = structure_target(target, tmp_path)
    materialize_target(target, output_dir=tmp_path / "cif", formats=["cif"])
    path = tmp_path / "cif/target.cif"
    document = pdbx.CIFFile.read(path)
    site = document.block["atom_site"]
    if loss == "numbering":
        site["label_seq_id"] = [1, 2, 3, 4]
    elif loss == "chain":
        site["label_asym_id"] = ["LONG"] * 4
    elif loss == "precision":
        site["Cartn_x"] = ["1.00000001", "4", "7", "10"]
    elif loss == "entities":
        site["label_entity_id"] = ["1", "1", "2", "3"]
    else:
        document.block["struct"] = pdbx.CIFCategory({"title": ["important metadata"]})
    document.write(path)
    target = target.model_copy(
        update={
            "target": target.target.model_copy(
                update={"structure_pdb": None, "structure_cif": path}
            )
        }
    )
    with pytest.raises(ConfigPreflightError):
        materialize_target(target, output_dir=tmp_path / "pdb", formats=["pdb"])
    assert not (tmp_path / "pdb").exists()


@pytest.mark.skipif(
    not os.environ.get("CHAI_TEST_IMAGE"), reason="opt-in CPU-only container parser check"
)
def test_real_chai_parser_reads_materialized_alignment(target, tmp_path):
    directory = tmp_path / "materialized"
    materialize_target(target, output_dir=directory, formats=["chai"])
    path = directory / "chai" / expected_pqt_basename("ACD")
    script = """
import sys
from pathlib import Path
from chai_lab.data.parsing.msas.aligned_pqt import parse_aligned_pqt_to_msa_context
context = parse_aligned_pqt_to_msa_context(Path(sys.argv[1]), quota_sizes=None)
assert tuple(context.tokens.shape) == (2, 3)
assert context.deletion_matrix.tolist() == [[0, 0, 0], [0, 1, 0]]
"""
    subprocess.run(
        [
            "singularity",
            "exec",
            "--cleanenv",
            "--bind",
            f"{tmp_path}:{tmp_path}",
            os.environ["CHAI_TEST_IMAGE"],
            "/opt/chai-venv/bin/python",
            "-B",
            "-c",
            script,
            str(path),
        ],
        check=True,
        timeout=120,
    )


@pytest.mark.skipif(
    not os.environ.get("PXDESIGN_TEST_IMAGE"), reason="opt-in CPU-only container parser check"
)
def test_real_px_parser_reads_both_handoffs(target, tmp_path):
    directory = tmp_path / "materialized"
    materialize_target(target, output_dir=directory, formats=["pxdesign"])
    script = """
import sys
from pathlib import Path
from protenix.data.msa_utils import parse_prot_msa_data
root = Path(sys.argv[1])
paths = [str(root / name) for name in ('non_pairing.a3m', 'pairing.a3m')]
msas = parse_prot_msa_data(paths, [-1, -1])
assert msas[paths[0]].sequences == ['ACD', 'AC-']
assert msas[paths[0]].deletion_matrix == [[0, 0, 0], [0, 1, 0]]
assert msas[paths[1]].sequences == ['ACD']
"""
    subprocess.run(
        [
            "singularity",
            "exec",
            "--cleanenv",
            "--bind",
            f"{tmp_path}:{tmp_path}",
            os.environ["PXDESIGN_TEST_IMAGE"],
            "/opt/conda/envs/pxdesign/bin/python",
            "-B",
            "-c",
            script,
            str(directory / "pxdesign"),
        ],
        check=True,
        timeout=120,
    )
