"""Execution identity cannot be rebound, and extension archives stay closed."""

import pytest

from bindocracy.runs.manifest import ManifestError, RunManifest, ToolPlan, plan_run
from bindocracy.tools import load_configs


def test_run_directory_cannot_be_reused_by_another_attempt(configs, tmp_path):
    loaded = load_configs(*configs)
    script = tmp_path / "script.py"
    script.write_text("print('example')\n")
    tool = ToolPlan(1, 1, "output.jsonl", {"script": script}, script, {})
    first = plan_run(loaded, tool, tmp_path / "run", identity={"attempt_id": "one"})
    with pytest.raises(ManifestError, match="different benchmark attempt"):
        plan_run(loaded, tool, tmp_path / "run", identity={"attempt_id": "two"})
    assert RunManifest.read(first.path("run.json")).attempt_id == "one"


def test_archive_preserves_distinct_same_named_inputs(configs, tmp_path):
    loaded = load_configs(*configs)
    files = {}
    for name, content in (("first", "first pose"), ("second", "second pose")):
        directory = tmp_path / name
        directory.mkdir()
        files[name] = directory / "pose.dat"
        files[name].write_text(content)
    tool = ToolPlan(
        1, 1, "output.jsonl", files, files["first"], {},
        archive_paths={name: f"structures/{name}/pose.dat" for name in files},
    )
    run = plan_run(loaded, tool, tmp_path / "run", identity={})
    files["first"].write_text("changed authoring copy")
    assert run.path(run.provenance["first"].path).read_text() == "first pose"
    assert run.path(run.provenance["second"].path).read_text() == "second pose"
    run.verify_inputs()


@pytest.mark.parametrize("destination", ["../outside", "/absolute", "a/../../outside"])
def test_archive_destinations_cannot_escape(configs, tmp_path, destination):
    loaded = load_configs(*configs)
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    tool = ToolPlan(
        1, 1, "out.jsonl", {"script": source}, source, {},
        archive_paths={"script": destination},
    )
    with pytest.raises(ManifestError, match="unsafe archive"):
        plan_run(loaded, tool, tmp_path / "run", identity={})
    assert not (tmp_path / "outside").exists()
