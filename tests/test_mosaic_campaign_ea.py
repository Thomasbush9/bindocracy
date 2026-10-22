"""Campaign persistence/identity tests; no Mosaic imports or model execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def campaign(monkeypatch):
    drivers = Path(__file__).resolve().parents[1] / "drivers" / "mosaic"
    monkeypatch.syspath_prepend(str(drivers))
    spec = importlib.util.spec_from_file_location("campaign_under_test", drivers / "campaign_ea.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings():
    return {"population": 4, "microbatch": 4, "cycles": 3,
            "soft_steps": 2, "sharp_steps": 1,
            "soft_stepsize": .05, "sharp_stepsize": .025,
            "soft_momentum": .7, "sharp_momentum": .5,
            "soft_scale": 1., "sharp_scale": 2., "num_samples": 1,
            "init_noise": .15, "cull_fraction": .5,
            "noise_low": .1, "noise_high": .2, "seed_base": 2026101400}


def root_record(sequence="AC" + "A" * 73):
    return {"design_id": "root-1", "sequence": sequence, "arm": "pocket", "tool": "generator",
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}


def test_initialization_preserves_mapping_and_independent_noise(campaign):
    sequence = "ACD" + "W" * 72
    hard, states = campaign.initial_states(sequence, np.random.default_rng(9), .15)
    assert hard[0, 2].argmax() == campaign.SEARCH_ALPHABET.index("D")
    assert hard[0, 3].argmax() == campaign.SEARCH_ALPHABET.index("W")
    assert np.allclose(hard[:, 1], 1 / 19)
    assert not np.array_equal(states[0], states[1])
    campaign.validate_states(states)
    np.testing.assert_array_equal(states, campaign.initial_states(sequence, np.random.default_rng(9), .15)[1])


def test_elite_archive_follows_parents_not_replaced_slots(campaign):
    states = np.arange(4 * 75 * 19).reshape(4, 75, 19)
    archive = campaign.archive_update(None, states, [1., 8., 3., 9.], list("abcd"), "cycle-1.npz")
    parents = np.array([0, 2, 2, 0])
    inherited = campaign.gather_archive(archive, parents)
    assert inherited["nodes"] == ["a", "c", "c", "a"]
    next_archive = campaign.archive_update(inherited, states + 1, [2., 4., 2., .5], list("efgh"), "cycle-2.npz")
    assert next_archive["nodes"] == ["a", "c", "g", "h"]
    assert next_archive["checkpoints"] == ["cycle-1.npz", "cycle-1.npz", "cycle-2.npz", "cycle-2.npz"]
    np.testing.assert_array_equal(next_archive["states"][1], states[2])


def test_smoke_plan_uses_separate_tree_and_settings(campaign, settings, tmp_path):
    root = root_record()
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps([root]))
    manifest = {"root": str(tmp_path), "roots": str(roots), "settings": {**settings, "soft_steps": 90},
                "chunks": {"pocket": [[root["design_id"]]]},
                "canary": {"root_ids": {"pocket": [root["design_id"]]},
                           "settings": {"soft_steps": 2}}}
    smoke = campaign.plan(manifest, "pocket", 0, True)
    production = campaign.plan(manifest, "pocket", 0, False)
    assert smoke[0]["soft_steps"] == 2
    assert production[0]["soft_steps"] == 90
    assert smoke[2] == tmp_path / "canary" / "pocket"
    assert production[2] == tmp_path / "results" / "pocket"
    assert smoke[3] == tmp_path / "canary" / "control" / "pocket-0.json"


class FakeRandom:
    @staticmethod
    def key(seed):
        return np.array([0, seed], dtype=np.uint32)

    @staticmethod
    def split(key, num=2):
        return np.stack([key + np.array([i + 1, 0], dtype=np.uint32) for i in range(num)])

    @staticmethod
    def key_data(key):
        return key


class FakeRuntime:
    """Deterministic test double only at the real model-owning seam."""
    jax = SimpleNamespace(random=FakeRandom)
    jnp = np

    def __init__(self, campaign):
        self.campaign = campaign
        self.evaluations = 0
        self.rank_calls = []

    def constrain(self, sequence):
        self.sequence = sequence

    def decode(self, states):
        sequences = []
        for state in states:
            letters = [self.campaign.SEARCH_ALPHABET[i] for i in state.argmax(-1)]
            for index, aa in enumerate(self.sequence):
                if aa == "C":
                    letters[index] = "C"
            sequences.append("".join(letters))
        return sequences

    def evaluate_states(self, states, keys):
        self.evaluations += 1
        values = np.array([4., 1., 3., 2.]) + self.evaluations
        return values, [{"term": float(value)} for value in values]

    def optimize(self, states, key, settings, label):
        return states.copy(), states.copy(), states.copy()

    def rank(self, sequence):
        self.rank_calls.append(sequence)
        return [3., 2., 1., 4.][len(self.rank_calls) - 1]


def test_full_root_records_all_cycles_and_exact_winner(campaign, settings, tmp_path):
    root = root_record()
    manifest = {"phase_id": "phase", "targets": {"pocket": {"fasta": "target.fa", "msa": "target.a3m", "epitope_idx": [0]}}}
    runtime = FakeRuntime(campaign)
    campaign.run_root(tmp_path, root, settings, manifest, "manifest-hash", runtime, lambda *a, **k: None)
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(result["final_members"]) == 4
    assert len(runtime.rank_calls) == 4
    assert result["selected_winner"] == result["final_members"][2]
    cycles = [json.loads(line) for line in (tmp_path / "cycles.jsonl").read_text().splitlines()]
    assert [row["cycle"] for row in cycles] == [0, 1, 2, 3]
    assert cycles[0]["members"][0]["sequence"] == root["sequence"]
    assert "parents" in cycles[1] and "parents" in cycles[2] and "parents" not in cycles[3]
    all_nodes = {member["candidate_id"] for row in cycles for member in row["members"]}
    assert len(all_nodes) == 16
    for row, previous in zip(cycles[2:], cycles[1:-1], strict=True):
        for slot, member in enumerate(row["members"]):
            assert member["parent_id"] == previous["members"][previous["parents"][slot]]["candidate_id"]
    assert campaign.validated_complete(tmp_path, root, settings, "manifest-hash")
    before = (tmp_path / "result.json").read_bytes()
    assert campaign.validated_complete(tmp_path, root, settings, "manifest-hash")
    assert (tmp_path / "result.json").read_bytes() == before
    with (tmp_path / "cycle-2.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        campaign.validated_complete(tmp_path, root, settings, "manifest-hash")


def test_nonfinite_ranking_never_commits_complete_result(campaign, settings, tmp_path):
    runtime = FakeRuntime(campaign)
    runtime.rank = lambda sequence: float("nan")
    root = root_record("A" * 75)
    manifest = {"phase_id": "phase", "targets": {"pocket": {"fasta": "target.fa", "msa": "target.a3m", "epitope_idx": [0]}}}
    with pytest.raises(ValueError, match="nonfinite"):
        campaign.run_root(tmp_path, root, settings, manifest, "hash", runtime, lambda *a, **k: None)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("sequence", ["A" * 75, "ACC" + "A" * 72])
def test_constraint_validation_rejects_new_or_removed_cysteine(campaign, sequence):
    campaign.validate_sequence(sequence, sequence)
    changed = ("C" if sequence[0] != "C" else "A") + sequence[1:]
    with pytest.raises(ValueError, match="cysteine"):
        campaign.validate_sequence(changed, sequence)
