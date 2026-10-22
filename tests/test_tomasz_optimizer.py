"""CPU-only behavior tests for the faithful wrapper and auxiliary instrumentation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

KIT = Path(__file__).resolve().parents[1] / "examples/kits/tomasz-optimization"


def load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def optimizer():
    return load("optimizer")


@pytest.fixture
def trace():
    return load("component_trace")


def args():
    return SimpleNamespace(models="boltz2", cycles=3, hard_sharp=False)


def test_sorted_trajectories_preserve_parent_identity_and_cysteine(optimizer, tmp_path):
    (tmp_path / "trajectories").mkdir()
    parents = [{"index": 12, "sequence": "ACAA"}, {"index": 7, "sequence": "VVVV"}]
    payload = {"results": [
        {"trajectory": 1, "sequence": "VVVA", "loss": -3},
        {"trajectory": 0, "sequence": "AAAA", "loss": -2},
    ], "config": {"models": [{"name": "boltz2"}, {"name": "of3"}]}}
    rows = optimizer._rows(parents, 1, 4, payload, None, 2,
                           {"structure_dir": str(tmp_path)}, args(), {"components.jsonl": "trace"})
    assert [r["parent_index"] for r in rows] == [7, 12]
    assert rows[1]["metrics"]["n_substitutions"] == 1
    assert rows[1]["metrics"]["lm_of3"] == 1
    artifact = json.loads((tmp_path / rows[1]["trajectory"]).read_text())
    assert artifact["parent_sequence"] == "ACAA"
    assert artifact["artifacts"]["components.jsonl"] == "trace"


def test_duplicate_trajectory_is_not_silently_a_missing_parent(optimizer, tmp_path):
    result = {"trajectory": 0, "sequence": "AAAA", "loss": 2}
    with pytest.raises(ValueError, match="every trajectory"):
        optimizer._rows([{"index": 0, "sequence": "AAAA"}], 1, 1,
                        {"results": [result, result]}, None, 0,
                        {"structure_dir": str(tmp_path)}, args())


def test_best_across_cycles_not_last_cycle(optimizer):
    best = {"trajectory": 0, "sequence": "AAAA", "loss": 1.0}
    last = {"trajectory": 0, "sequence": "VVVV", "loss": 4.0}
    cycles = {"cycles": [{"designs": [best]}, {"designs": [last]}]}
    optimizer.validate_selection({"results": [best]}, cycles)
    with pytest.raises(ValueError, match="best across cycles"):
        optimizer.validate_selection({"results": [last]}, cycles)


def test_recipe_preserves_all_scientific_parameters(optimizer, tmp_path):
    recipe = {"target": {"fasta": "old", "copies": 1},
              "binder": {"length": 75, "init_noise": .15, "no_cys": True},
              "run": {"batch": 4, "seed": 0},
              "models": [{"name": "of3", "weight": 1,
                          "params": {"num_samples": 1, "sampling_steps": 20}}],
              "losses": [{"name": "BinderTargetContact", "weight": 2,
                          "params": {"contact_distance": 20, "epitope_idx": [4]}}],
              "optimizer": {"soft": {"stepsize": .05, "momentum": .7}},
              "mpnn": {"terms": [{"params": {"num_samples": 16, "jacobi_iterations": 10}}]}}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    config = optimizer.design_config(75, tmp_path / "parent.fasta", 1, 9, [2],
                                     {"target_sequence": "AAAA", "target_msa": "target.a3m"},
                                     SimpleNamespace(recipe_config=path, esmc_checkpoint=""))
    for key in ("models", "optimizer", "mpnn"):
        assert config[key] == recipe[key]
    assert config["binder"]["no_cys"] is True
    assert config["binder"]["init_noise"] == .15
    assert config["target"]["copies"] == 1
    assert config["losses"][0]["params"]["epitope_idx"] == [2]
    assert config["run"] == {"batch": 1, "seed": 9}


class Combination:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, sequence, key):
        r = 0.0
        aux_values = []
        for w, loss in self.terms:
            v, a = loss(sequence, key=key)
            key += 1
            r += w * v
            aux_values.append(a)
        return r, aux_values


def test_instrumentation_preserves_values_calls_and_random_key_order(trace):
    calls = []
    def loss(sequence, *, key):
        calls.append((sequence, key))
        return -key, {"positive_diagnostic": key}
    original = Combination([(2., loss), (.5, loss)])
    expected, diagnostics = original("parent", key=3)
    expected_calls = list(calls)
    calls.clear()
    instrumented, _ = trace.instrument_combination(Combination.__call__)
    actual, aux = instrumented(original, "parent", key=3)
    assert actual == expected
    assert calls == expected_calls
    assert [r["aux"] for r in aux] == diagnostics
    assert [r["value"] for r in aux] == [-3, -4]
    assert [r["weight"] for r in aux] == [2., .5]


def test_components_report_actual_losses_not_diagnostic_signs(trace):
    cfg = {"models": [{"name": "boltz2"}, {"name": "of3"}],
           "losses": [{"name": "PLDDTLoss"}], "sequence_models": [{"name": "esmc"}]}
    aux = [{"value": -1.6, "weight": 1., "aux": {"boltz2": [
        {"value": -.8, "weight": 2., "aux": {"plddt": .8}}]}},
        {"value": -.7, "weight": .5, "aux": [
            {"value": [-.7], "weight": [1.], "aux": {"plddt": [.7]}}]},
        {"value": 2., "weight": .5, "aux": {"clipped": 2., "": {"esmc_pll": -1.8}}}]
    result = trace.components(aux, cfg)
    assert result["boltz2/0:PLDDTLoss"]["weighted_value"] == -1.6
    assert result["of3/0:PLDDTLoss"]["weighted_value"] == -.35
    assert result["esmc"]["weighted_value"] == 1.


@pytest.mark.parametrize("bad,total", [
    (float("nan"), -.8), (float("inf"), -.8), (1_000_000., -.8), (.8, -.7),
])
def test_recording_refuses_invalid_or_inconsistent_objective(trace, tmp_path, bad, total):
    # Lists here represent auxiliary pytree structure, not a batch dimension.
    # A lightweight array-shaped leaf exercises the same recorder slicing path.
    class Leaf:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, index):
            assert index == 0
            return self.value
    cfg = {"models": [{"name": "boltz2"}], "losses": [{"name": "PLDDTLoss"}]}
    aux = [{"value": Leaf(-.8), "weight": Leaf(1.), "aux": {"boltz2": [
        {"value": Leaf(-.8), "weight": Leaf(1.), "aux": {"plddt": Leaf(bad)}}]}}]
    recorder = trace.Recorder(tmp_path, cfg, [{"parent_index": 12}], 9)
    try:
        with pytest.raises(FloatingPointError):
            recorder.record([total], aux, cycle=1, stage="soft", step=0, sharp_replicate=None)
    finally:
        recorder.sink.close()
    row = json.loads((tmp_path / "components.jsonl").read_text())
    assert row["finite"] is False
    assert row["parent_index"] == 12
    assert row["invalid_paths"]


def gradient_case(trace, raw, weights, values, *, observed_gradient=None, total_gradient=None):
    raw = np.asarray(raw, dtype=float)
    weights = np.asarray(weights, dtype=float)
    centered = raw - raw.mean(axis=-1, keepdims=True)
    total = (centered * weights[:, None, None]).sum(axis=0)
    names = [f"model/{i}:term" for i in range(len(weights))]
    observed = {name: {"value": value, "model_weight": 1., "term_weight": weight,
                       "weighted_value": value * weight}
                for name, value, weight in zip(names, values, weights)}
    return trace.gradient_metrics(
        names, values, weights, raw,
        total if total_gradient is None else total_gradient,
        total if observed_gradient is None else observed_gradient,
        observed, float(np.dot(values, weights)), float(np.dot(values, weights)),
    )


def test_gradient_dominance_is_not_scalar_loss_magnitude(trace):
    result = gradient_case(trace, [[[10, -10]], [[1, -1]]], [1, 1], [.01, 100])
    a, b = result["components"].values()
    assert result["finite"]
    assert a["weighted_value"] < b["weighted_value"]
    assert a["norm_share"] == pytest.approx(10 / 11)
    assert b["norm_share"] == pytest.approx(1 / 11)
    assert a["raw_gradient_norm"] == pytest.approx(np.sqrt(200))
    assert result["cancellation_ratio"] == pytest.approx(1)
    assert result["pairwise_cosines"] == pytest.approx(np.ones((2, 2)))
    assert result["gradient_reconstruction_max_abs_error"] == 0


def test_conflicting_gradients_have_signed_projection_and_cancellation(trace):
    result = gradient_case(trace, [[[2, -2]], [[-1, 1]]], [1, 1], [3, 4])
    a, b = result["components"].values()
    assert result["finite"]
    assert a["projection_share"] == pytest.approx(2)
    assert b["projection_share"] == pytest.approx(-1)
    assert b["cosine_with_total"] == pytest.approx(-1)
    assert result["pairwise_cosines"][0][1] == pytest.approx(-1)
    assert result["cancellation_ratio"] == pytest.approx(1 / 3)


def test_weighting_and_amino_acid_centering_precede_norms(trace):
    # Common-mode offsets disappear; a signed coefficient reverses alignment.
    result = gradient_case(trace, [[[12, 8]], [[4, 2]]], [.5, -2], [6, 7])
    a, b = result["components"].values()
    assert result["finite"]
    assert a["raw_gradient_norm"] == pytest.approx(np.sqrt(8))
    assert a["weighted_gradient_norm"] == pytest.approx(np.sqrt(2))
    assert b["weighted_gradient_norm"] == pytest.approx(np.sqrt(8))
    assert b["coefficient"] == -2
    assert b["weighted_value"] == -14
    assert a["norm_share"] == pytest.approx(1 / 3)
    assert a["projection_share"] == pytest.approx(-1)
    assert b["projection_share"] == pytest.approx(2)


def test_zero_weight_and_zero_norm_are_defined_without_fake_cosines(trace):
    result = gradient_case(trace, [[[2, -2]], [[8, 8]]], [0, 2], [3, 4])
    a, b = result["components"].values()
    assert result["finite"]
    assert a["raw_gradient_norm"] > 0
    assert a["weighted_gradient_norm"] == b["weighted_gradient_norm"] == 0
    assert a["norm_share"] == b["norm_share"] == 0
    assert a["projection_share"] is b["projection_share"] is None
    assert a["cosine_with_total"] is b["cosine_with_total"] is None
    assert result["pairwise_cosines"] == [[None, None], [None, None]]
    assert result["cancellation_ratio"] is None


def test_complete_cancellation_keeps_nonzero_component_norms(trace):
    result = gradient_case(trace, [[[2, -2]], [[-2, 2]]], [1, 1], [1, 2])
    assert result["finite"]
    assert result["total_gradient_norm"] == 0
    assert result["sum_component_norms"] > 0
    assert result["cancellation_ratio"] == 0
    assert result["components"]["model/0:term"]["projection_share"] is None
    assert result["pairwise_cosines"][0][1] == pytest.approx(-1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_unweighted_gradient_is_rejected_even_with_zero_weight(trace, tmp_path, bad):
    metrics = gradient_case(trace, [[[bad, 1]]], [0], [1],
                            total_gradient=[[0, 0]], observed_gradient=[[0, 0]])
    assert not metrics["finite"]
    assert "/raw_gradients" in metrics["invalid_paths"]
    recorder = trace.Recorder(tmp_path, {}, [{"parent_index": 12}], 9, 90, 1)
    try:
        with pytest.raises(FloatingPointError, match="component gradients"):
            recorder.record_gradient(
                metrics, trajectory=0, evaluation=8,
                location={"cycle": 1, "stage": "soft", "step": 0, "sharp_replicate": None},
                random_key=[1, 2], runtime=.25, batch_size=4,
            )
    finally:
        recorder.close()
    row = json.loads((tmp_path / "gradients.jsonl").read_text())
    assert not row["finite"]
    assert row["parent_index"] == 12 and row["evaluation"] == 8
    assert row["coverage"]["selected_trajectories"] == [0]
    assert row["schema_version"] == 3
    assert row["evaluation_and_diagnostic_runtime_seconds"] == .25
    assert row["coverage"]["execution"] == "same_evaluation_direct_loss_first_order_vjp"


def test_reconstruction_checks_optimizer_gradient_not_only_component_sum(trace):
    result = gradient_case(trace, [[[2, -2]]], [1], [1],
                           observed_gradient=[[1, -1]])
    assert not result["finite"]
    assert result["gradient_reconstruction_max_abs_error"] == 1
    assert "/gradient_reconstruction" in result["invalid_paths"]
    assert "/total_gradient_centering" in result["invalid_paths"]
    close = gradient_case(trace, [[[2, -2]]], [1], [1],
                          observed_gradient=[[2.00001, -2.00001]])
    assert close["finite"]


def test_unsanitized_total_gradient_and_scalar_reconciliation_are_checked(trace):
    result = gradient_case(trace, [[[1, -1]]], [1], [1],
                           total_gradient=[[float("nan"), 0]], observed_gradient=[[1, -1]])
    assert not result["finite"]
    assert "/total_gradient" in result["invalid_paths"]
    result = trace.gradient_metrics(
        ["term"], [2], [3], [[[1, -1]]], [[3, -3]], [[3, -3]],
        {"term": {"value": 2, "model_weight": 1, "term_weight": 4, "weighted_value": 8}},
        6, 8,
    )
    assert not result["finite"]
    assert "/components/term/value_reconstruction" in result["invalid_paths"]
    assert "/total_value_reconstruction" in result["invalid_paths"]


@pytest.mark.parametrize("steps,expected", [(1, [0]), (2, [0, 1]), (181, [0, 90, 180]),
                                          (180, [0, 90, 179]), (30, [0, 29])])
def test_sampling_includes_first_last_and_interval(trace, steps, expected):
    assert [s for s in range(steps) if trace.sample_gradient_step(90, s, steps)] == expected
    assert trace.sample_gradient_step(90, None, steps)
    assert not trace.sample_gradient_step(0, None, steps)
    assert not trace.sample_gradient_step(0, 0, steps)




def test_first_order_same_evaluation_matches_scalar_value_and_grad(trace):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    eqx = pytest.importorskip("equinox")
    config = {"models": [], "losses": [],
              "sequence_models": [{"name": "linear"}, {"name": "quadratic"}]}

    def loss(x, *, key):
        random = jax.random.uniform(key, ())
        first = jnp.sum(x * jnp.array([2., -1.])) + random
        second = jnp.sum(x**2) - random
        return 3 * first - .5 * second, [
            {"value": first, "weight": 3., "aux": {}},
            {"value": [second], "weight": [-.5], "aux": {}},
        ]

    def scalar_evaluation(x, key):
        (value, aux), gradient = eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)
        return (jnp.nan_to_num(value, nan=1e6), aux,
                jnp.nan_to_num(gradient - gradient.mean(axis=-1, keepdims=True)))

    evaluate = trace.gradient_evaluator(
        SimpleNamespace(jax=jax, jnp=jnp, eqx=eqx), config,
    )
    xs = jnp.array([[[.2, .8]], [[.7, .3]]])
    keys = jax.random.split(jax.random.key(42), 2)
    result, (batch_values, batch_coefficients, batch_gradients) = evaluate(loss, xs, keys)
    expected = jax.vmap(scalar_evaluation)(xs, keys)
    for observed_leaf, expected_leaf in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        assert np.asarray(observed_leaf) == pytest.approx(np.asarray(expected_leaf))
    assert np.asarray(batch_values[:, -1]) == pytest.approx(np.asarray(result[0]))
    expected_centered = 3 * jnp.array([2., -1.]) - xs
    expected_centered -= expected_centered.mean(axis=-1, keepdims=True)
    assert np.asarray(result[2]) == pytest.approx(np.asarray(expected_centered))
    for trajectory in range(2):
        values, coefficients, gradients = (
            v[trajectory] for v in (batch_values, batch_coefficients, batch_gradients)
        )
        aux = jax.tree.map(lambda leaf, row=trajectory: leaf[row], result[1])
        assert np.asarray(gradients[0]) == pytest.approx(np.array([[2., -1.]]))
        assert np.asarray(gradients[1]) == pytest.approx(2 * np.asarray(xs[trajectory]))
        random = jax.random.uniform(keys[trajectory], ())
        expected_values = [jnp.sum(xs[trajectory] * jnp.array([2., -1.])) + random,
                           jnp.sum(xs[trajectory]**2) - random]
        assert np.asarray(values[:-1]) == pytest.approx(np.asarray(expected_values))
        metrics = trace.gradient_metrics(
            ["linear", "quadratic"], values[:-1], coefficients, gradients[:-1], gradients[-1],
            result[2][trajectory], trace.components(aux, config),
            float(values[-1]), float(result[0][trajectory]),
        )
        assert metrics["finite"]
        assert metrics["gradient_reconstruction_max_abs_error"] < 1e-5
        assert metrics["total_gradient_centering_max_abs_error"] < 1e-5


def test_centering_uses_full_nineteen_amino_acid_axis(trace):
    raw = np.arange(38, dtype=float).reshape(1, 2, 19)
    expected = raw - raw.mean(axis=-1, keepdims=True)
    result = gradient_case(trace, raw, [2], [7])
    assert result["finite"]
    term = result["components"]["model/0:term"]
    assert term["raw_gradient_norm"] == pytest.approx(np.linalg.norm(expected))
    assert term["weighted_gradient_norm"] == pytest.approx(2 * np.linalg.norm(expected))
    assert result["gradient_reconstruction_max_abs_error"] == 0
