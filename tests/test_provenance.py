"""What a run records about the inputs it actually consumed.

Config IDs hash the configuration document, and that document contains paths.
Replacing a target FASTA or a design spec at the same path therefore leaves
every ID unchanged while the science changes underneath. These tests cover the
three places that is now caught: the run manifest digests its inputs, reuse
refuses a tampered archive, and the database refuses a different target.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    design_line,
    write_boltzgen_configs,
    write_boltzgen_task,
    write_task,
)

from bindocracy.runs import ManifestError, ingest_bundle, write_collected
from bindocracy.runs.inputs import digest_of
from bindocracy.store import CampaignStore, TargetMismatchError, create_database
from bindocracy.tools import collect_run, load_configs, plan

# ---- digests --------------------------------------------------------------

def test_a_referenced_input_is_hashed(tmp_path: Path) -> None:
    path = tmp_path / "target.fasta"
    path.write_text(">t\nACDEFG\n")

    digest = digest_of(path)

    assert digest.sha256.startswith("sha256:")
    assert digest.size_bytes == path.stat().st_size
    assert digest.matches(path)


def test_editing_a_file_in_place_changes_its_digest(tmp_path: Path) -> None:
    """The whole point: the path is unchanged, the bytes are not."""
    path = tmp_path / "target.fasta"
    path.write_text(">t\nACDEFG\n")
    before = digest_of(path)

    path.write_text(">t\nWWWWWW\n")

    assert not before.matches(path)


# ---- the manifest ---------------------------------------------------------

def test_a_run_records_every_input_it_consumed(configs, tmp_path: Path) -> None:
    """The plugin declares them, because only it knows what it reads."""
    manifest = plan(load_configs(*configs), tmp_path / "run")

    assert set(manifest.inputs) == {"target_fasta", "target_msa", "exec_wrapper"}
    assert all(d.size_bytes > 0 for d in manifest.inputs.values())
    assert manifest.target.name == "test-target"
    assert manifest.target.length == 6
    assert manifest.target.sequence_sha256.startswith("sha256:")


def test_a_boltzgen_run_records_the_spec_contents_not_just_its_path(
    boltzgen_configs, tmp_path: Path
) -> None:
    """`model_config_json` names the spec file; a path does not say what was designed."""
    manifest = plan(load_configs(*boltzgen_configs), tmp_path / "run")

    assert manifest.workflow["spec"]["entities"][0]["protein"]["sequence"] == "70..90"
    # BoltzGen reads geometry, never the FASTA or the MSA.
    assert set(manifest.inputs) == {"spec_structure_0"}


def test_reuse_refuses_an_archive_that_was_edited(configs, tmp_path: Path) -> None:
    """The archive is what the run executes, so it cannot be edited in place."""
    loaded = load_configs(*configs)
    manifest = plan(loaded, tmp_path / "run")
    archived = manifest.path(manifest.provenance["driver"].path)
    archived.write_text("# someone edited the archived driver\n")

    with pytest.raises(ManifestError, match="archived inputs that have since changed"):
        plan(load_configs(*configs), tmp_path / "run")


def test_reuse_accepts_an_untouched_run(configs, tmp_path: Path) -> None:
    first = plan(load_configs(*configs), tmp_path / "run")

    second = plan(load_configs(*configs), tmp_path / "run")

    assert second.run_id == first.run_id
    assert second.inputs == first.inputs


# ---- the database ---------------------------------------------------------

def test_a_database_is_stamped_with_its_target(tmp_path: Path) -> None:
    database = create_database(tmp_path / "campaign.duckdb")

    with CampaignStore(database) as store:
        assert store.assert_target("dio3", "sha256:aaa") is True
        assert store.assert_target("dio3", "sha256:aaa") is False


def test_a_database_refuses_a_different_target(tmp_path: Path) -> None:
    """One campaign database is one target; nothing used to enforce it."""
    database = create_database(tmp_path / "campaign.duckdb")

    with CampaignStore(database) as store:
        store.assert_target("dio3", "sha256:aaa")
        with pytest.raises(TargetMismatchError, match="follows target dio3"):
            store.assert_target("something-else", "sha256:bbb")


def test_ingesting_another_target_into_a_campaign_is_refused(
    configs, boltzgen_configs, tmp_path: Path
) -> None:
    """End to end: a run against a replaced target cannot join the campaign."""
    database = create_database(tmp_path / "campaign.duckdb")
    manifest = plan(load_configs(*configs), tmp_path / "m")
    for task_id in (0, 1):
        write_task(manifest.directory, task_id, [design_line(task_id, 0)], status={})
    bundle = write_collected(collect_run(manifest.directory / "run.json"),
                             manifest.directory / "collected.json")
    assert ingest_bundle(database, bundle) is True

    # A second run whose general config names a different protein.
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = write_boltzgen_configs(other_root)
    (other_root / "target.fasta").write_text(">target\nWWWWWWWWWW\n")
    second = plan(load_configs(*other), tmp_path / "b")
    write_boltzgen_task(second.directory, 0, status={})
    second_bundle = write_collected(collect_run(second.directory / "run.json"),
                                    second.directory / "collected.json")

    with pytest.raises(TargetMismatchError):
        ingest_bundle(database, second_bundle)

    import duckdb
    con = duckdb.connect(str(database), read_only=True)
    assert con.execute("SELECT count(*) FROM runs").fetchone() == (1,)
    con.close()


# ---- enforcement ----------------------------------------------------------

def test_a_launch_refuses_an_input_that_changed_since_planning(
    configs, tmp_path: Path
) -> None:
    """Recording a digest is only worth anything if something checks it."""
    general_path, model_path = configs
    manifest = plan(load_configs(general_path, model_path), tmp_path / "run")
    manifest.verify_inputs()  # clean to begin with

    (tmp_path / "target.a3m").write_text(">target\nWWWWWW\n")

    with pytest.raises(ManifestError, match="no longer match what it"):
        manifest.verify_inputs()


def test_a_launch_refuses_an_edited_archive(configs, tmp_path: Path) -> None:
    manifest = plan(load_configs(*configs), tmp_path / "run")

    manifest.path(manifest.provenance["driver"].path).write_text("# edited\n")

    with pytest.raises(ManifestError, match="no longer match what it"):
        manifest.verify_inputs()


def test_verify_inputs_names_what_changed(configs, tmp_path: Path) -> None:
    manifest = plan(load_configs(*configs), tmp_path / "run")
    (tmp_path / "target.fasta").write_text(">target\nWWWWWW\n")

    with pytest.raises(ManifestError) as raised:
        manifest.verify_inputs()

    assert "target_fasta" in str(raised.value)
