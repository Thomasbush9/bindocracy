"""CPU-only state-machine tests at the model seam; no production objective mocks."""
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
    spec = importlib.util.spec_from_file_location("gated_under_test", drivers / "campaign_gated.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumericalKeys:
    @staticmethod
    def key(seed):
        return np.asarray([0, seed], dtype=np.uint32)

    @staticmethod
    def split(key, num=2):
        return np.stack([key + np.asarray([i + 1, 0], dtype=np.uint32) for i in range(num)])

    @staticmethod
    def fold_in(key, value):
        return key + np.asarray([1, value], dtype=np.uint32)

    @staticmethod
    def key_data(key):
        return key

    @staticmethod
    def wrap_key_data(key):
        return key


class QuadraticRuntime:
    """Real differentiable quadratic with a slot offset, replacing only the model.

    Its changing gradient exercises extrapolation, clipping, projection and
    momentum. The offset makes gate25/75 accept and gate50 reject without a
    constant fake optimizer or dependence on how many calls preceded a resume.
    """
    jax = SimpleNamespace(random=NumericalKeys)

    def __init__(self, campaign):
        self.campaign = campaign
        self.evaluations = []
        self.ranked = []

    @staticmethod
    def project(states):
        flat = np.asarray(states, dtype=np.float64).reshape(-1, 19)
        ordered = np.sort(flat, axis=1)[:, ::-1]
        cumsum = np.cumsum(ordered, axis=1) - 1
        rho = np.count_nonzero(ordered - cumsum / np.arange(1, 20) > 0, axis=1)
        theta = cumsum[np.arange(len(flat)), rho - 1] / rho
        return np.maximum(flat - theta[:, None], 0).reshape(states.shape).astype(np.float32)

    def evaluate_states(self, states, masks, keys, gradient=False):
        states = np.asarray(states)
        self.evaluations.append((gradient, len(states)))
        desired = np.linspace(0.01, 0.09, 19, dtype=np.float32)
        delta = states - desired
        values = (delta ** 2).sum(axis=(-2, -1)) + np.arange(len(states)) * 1000
        components = [{"quadratic_plus_offset": float(value)} for value in values]
        if gradient:
            grad = 2 * delta
            grad -= grad.mean(axis=-1, keepdims=True)
            return values, components, grad
        return values, components

    def decode(self, states, masks):
        return self.campaign.Runtime.decode(self, states, masks)

    def rank(self, sequence):
        self.ranked.append(sequence)
        return float(sum(ord(letter) for letter in sequence))


@pytest.fixture
def example(tmp_path, campaign):
    settings = {"population": 4, "microbatch": 4, "cycles": 3, "soft_steps": 25, "sharp_steps": 1,
                "soft_stepsize": .01, "sharp_stepsize": .01, "soft_momentum": .7, "sharp_momentum": .5,
                "soft_scale": 1., "sharp_scale": 2., "num_samples": 1, "init_noise": .15,
                "noise_low": .1, "noise_high": .2, "seed_base": 42, "cull_interval": 25}
    roots = []
    for slot in range(4):
        sequence = "A" * slot + "C" + "A" * (74 - slot)
        roots.append({"design_id": f"original-{slot}", "arm": "pocket", "tool": "generator",
                      "sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()})
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"metric": campaign.METRIC,
                                           "by_arm": {"pocket": {"25": 0., "50": 1e9, "75": 0.}}}))
    manifest = {"phase_id": "phase", "calibration": str(calibration_path), "targets": {"pocket": {}}}
    batch = {"arm": "pocket", "batch_id": "batch", "root_ids": [root["design_id"] for root in roots]}
    return settings, roots, manifest, batch


def run(campaign, example, directory, count=4, status=None):
    settings, roots, manifest, batch = example
    directory.mkdir(parents=True, exist_ok=True)
    runtime = QuadraticRuntime(campaign)
    campaign.run_batch(directory, roots[:count], batch, settings, manifest, "fingerprint", runtime,
                       status or (lambda *args, **kwargs: None))
    return runtime, json.loads((directory / "result.json").read_text())


def test_distinct_roots_gate_state_masks_and_actual_endpoint(campaign, example, tmp_path):
    directory = tmp_path / "batch"
    runtime, result = run(campaign, example, directory)
    assert len(result["final_members"]) == 4
    assert runtime.evaluations.count((False, 4)) == 2  # one baseline and one endpoint, no gate reevaluation
    assert runtime.evaluations.count((True, 4)) == 78
    gates = campaign.records(directory / "gates.jsonl")
    assert [gate["accepted"] for gate in gates] == [True, False, True]
    assert [gate["step"] for gate in gates] == [25, 50, 75]
    for gate in gates:
        step = gate["step"]
        with np.load(directory / f"trajectory-{step:04d}.npz") as trajectory, np.load(directory / f"checkpoint-{step:04d}.npz") as checkpoint:
            survivors = [slot for slot in range(4) if not gate["accepted"] or slot != gate["worst_slot"]]
            np.testing.assert_array_equal(checkpoint["x"][survivors], trajectory["updated_x"][survivors])
            np.testing.assert_array_equal(checkpoint["x_prev"][survivors], trajectory["x_before"][survivors])
            np.testing.assert_array_equal(checkpoint["evaluated_state"], trajectory["evaluated_state"])
            if gate["accepted"]:
                worst, parent = gate["worst_slot"], gate["parent_slot"]
                np.testing.assert_array_equal(checkpoint["x"][worst], checkpoint["x_prev"][worst])
                np.testing.assert_array_equal(checkpoint["cysteines"][worst], trajectory["cysteines"][parent])
    assert any(member["slot_original_id"] != member["lineage_root_id"] for member in result["final_members"])
    assert campaign.validated_complete(directory, example[1], example[0], "fingerprint")


def test_partial_population_is_not_expanded_or_culled(campaign, example, tmp_path):
    directory = tmp_path / "batch"
    runtime, result = run(campaign, example, directory, count=2)
    assert len(result["final_members"]) == 2
    assert result["active_count"] == 2 and result["padding_count"] == 2
    assert all(count == 2 for _, count in runtime.evaluations)
    assert all(gate["reason"] == "partial_group_no_cull" and not gate["accepted"]
               for gate in campaign.records(directory / "gates.jsonl"))
    assert campaign.validated_complete(directory, example[1][:2], example[0], "fingerprint")


def test_preemption_replays_from_exact_optimizer_and_rng_state(campaign, example, tmp_path):
    clean = tmp_path / "clean" / "batch"
    resumed = tmp_path / "resumed" / "batch"
    run(campaign, example, clean)

    def interrupt(stage, **progress):
        if stage == "gate" and progress["step"] == 50:
            raise RuntimeError("simulated preemption before gate50 checkpoint commit")

    with pytest.raises(RuntimeError, match="simulated preemption"):
        run(campaign, example, resumed, status=interrupt)
    run(campaign, example, resumed)
    assert campaign.records(clean / "steps.jsonl") == campaign.records(resumed / "steps.jsonl")
    assert campaign.records(clean / "gates.jsonl") == campaign.records(resumed / "gates.jsonl")
    with np.load(clean / "final.npz") as expected, np.load(resumed / "final.npz") as actual:
        for field in expected.files:
            np.testing.assert_array_equal(actual[field], expected[field])
    assert campaign.validated_complete(resumed, example[1], example[0], "fingerprint")


def test_validator_rejects_modified_lineage_and_corrupt_artifact(campaign, example, tmp_path):
    directory = tmp_path / "batch"
    _, result = run(campaign, example, directory)
    original = json.dumps(result)
    result["final_members"][0]["lineage_root_id"] = "not-an-original"
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="lineage"):
        campaign.validated_complete(directory, example[1], example[0], "fingerprint")
    (directory / "result.json").write_text(original)
    with (directory / "checkpoint-0025.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        campaign.validated_complete(directory, example[1], example[0], "fingerprint")


def test_gate_strict_threshold_and_smallest_index_tie(campaign):
    worst, median, gap, accepted = campaign.gate_decision([3., 3., 1., 1.], 2.)
    assert (worst, median, gap, accepted) == (0, 1., 2., False)
    assert campaign.gate_decision([3., 3., 1., 1.], 1.99)[3]
