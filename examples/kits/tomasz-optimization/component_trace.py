"""Observe Tomasz's signed objective and sampled same-evaluation gradients.

LinearCombination's auxiliary append is AST-instrumented, leaving its evaluation,
fold_in, and summation statements intact. Sampled evaluations use one first-order
VJP of the unchanged loss's component values and total, followed by the original
total-gradient centering/sanitization. Unsampled evaluations use batched_eval.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
import time

SENTINEL = 1_000_000.0


def instrument_combination(function):
    """Compile the original function with exactly one auxiliary-only AST edit."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "aux_values" and node.func.attr == "append"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "a"):
            node.args[0] = ast.parse(
                "{'value': v, 'weight': w, 'aux': a}", mode="eval"
            ).body
            replacements += 1
    if replacements != 1:
        raise RuntimeError("Unsupported LinearCombination: expected one auxiliary append")
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(function), "exec"), function.__globals__, namespace)
    return namespace[function.__name__], source


def sequential_evaluator(function):
    """Keep the original scalar evaluator; serialize independent batch rows."""
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and ast.unparse(node.value) == "jax.vmap(single)(xs, keys)":
            node.value = ast.parse(
                "jax.lax.map(lambda pair: single(*pair), (xs, keys))", mode="eval"
            ).body
            replacements += 1
    if replacements != 1:
        raise RuntimeError("Unsupported batched_eval: expected one independent-row vmap")
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), "exec"), original.__globals__, namespace)
    return namespace[original.__name__], ast.unparse(tree)


def json_value(value):
    """Strict JSON, retaining evidence of nonfinite values as strings."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def invalid_values(value, path=""):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in invalid_values(v, f"{path}/{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in invalid_values(v, f"{path}/{i}")]
    if hasattr(value, "tolist"):
        return invalid_values(value.tolist(), path)
    if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) >= SENTINEL):
        return [path]
    return []


def scalar(value):
    """Single-sample OF3 returns scalar auxiliaries as one-element lists."""
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return float(value)


def component_records(aux, config):
    """Differentiable leaves and actual coefficients, in objective order."""
    outer = config["models"] + config.get("sequence_models", [])
    if len(aux) != len(outer):
        raise ValueError("Objective auxiliary/model count mismatch")
    inner_specs = config["losses"] + config.get("mpnn", {}).get("terms", [])
    for index, (record, spec) in enumerate(zip(aux, outer)):
        weight = record["weight"]
        name = spec["name"]
        if index < len(config["models"]):
            inner = record["aux"]
            if isinstance(inner, dict):
                inner = inner[name]
            if len(inner) != len(inner_specs):
                raise ValueError(f"Objective auxiliary/term count mismatch for {name}")
            for term_index, (term, term_spec) in enumerate(zip(inner, inner_specs)):
                yield (f"{name}/{term_index}:{term_spec['name']}",
                       term["value"], weight, term["weight"])
        else:
            yield name, record["value"], 1.0, weight


def components(aux, config):
    """Extract exact additive leaf values; keep original diagnostics separately."""
    result = {}
    for name, value, model_weight, term_weight in component_records(aux, config):
        raw, model, term = scalar(value), scalar(model_weight), scalar(term_weight)
        result[name] = {"value": raw, "model_weight": model, "term_weight": term,
                        "weighted_value": raw * model * term}
    return result


GRADIENT_ATOL = 1e-5
GRADIENT_RTOL = 1e-4


def gradient_metrics(names, raw_values, coefficients, raw_gradients, total_gradient,
                     observed_gradient, observed_components, total_value, observed_total):
    """CPU numeric seam: center BEFORE weighting/norms, never sanitize gradients.

    Reconstruction passes iff max absolute error <= 1e-5 + 1e-4 * max(abs(g)).
    g is the centered, pre-clipping optimizer gradient derived from the total VJP.
    Projection shares are signed dot products divided by ||g||², not attribution
    percentages. Cancellation ratio is ||g|| / sum(||weighted component g||).
    Invalid results are returned as evidence; Recorder persists then rejects them.
    """
    import numpy as np

    raw = np.asarray(raw_gradients, dtype=np.float64)
    original = np.asarray(observed_gradient, dtype=np.float64)
    uncentered_total = np.asarray(total_gradient, dtype=np.float64)
    values = np.asarray(raw_values, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[0] != len(names) or raw.shape[1:] != original.shape:
        raise ValueError("Component gradients must have shape [components, residues, vocabulary]")
    if uncentered_total.shape != original.shape or values.shape != weights.shape or values.shape != (len(names),):
        raise ValueError("Gradient/value/coefficient shape mismatch")
    invalid = []
    for path, array in (("raw_gradients", raw), ("total_gradient", uncentered_total),
                        ("observed_gradient", original), ("raw_values", values),
                        ("coefficients", weights)):
        if not np.isfinite(array).all():
            invalid.append("/" + path)
    if not math.isfinite(total_value) or not math.isfinite(observed_total):
        invalid.append("/total_value")
    # A nonfinite *unweighted* gradient remains invalid even at coefficient zero.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        centered = raw - raw.mean(axis=-1, keepdims=True)
        weighted = centered * weights[:, None, None]
        total = uncentered_total - uncentered_total.mean(axis=-1, keepdims=True)
        raw_norms = np.linalg.norm(centered.reshape(len(names), -1), axis=1)
        norms = np.linalg.norm(weighted.reshape(len(names), -1), axis=1)
        total_norm = float(np.linalg.norm(original))
        norm_sum = float(norms.sum())
        reconstructed = weighted.sum(axis=0)
        error = float(np.max(np.abs(reconstructed - original)))
        centering_error = float(np.max(np.abs(total - original)))
        tolerance = GRADIENT_ATOL + GRADIENT_RTOL * float(np.max(np.abs(original)))
        flat = weighted.reshape(len(names), -1)
        dots = flat @ original.reshape(-1)
        pair_dots = flat @ flat.T
    if not math.isfinite(error) or error > tolerance:
        invalid.append("/gradient_reconstruction")
    if not math.isfinite(centering_error) or centering_error > tolerance:
        invalid.append("/total_gradient_centering")
    terms = {}
    for i, name in enumerate(names):
        value, coefficient = float(values[i]), float(weights[i])
        weighted_value = value * coefficient
        observed = observed_components[name]
        checks = ((value, observed["value"]),
                  (coefficient, observed["model_weight"] * observed["term_weight"]),
                  (weighted_value, observed["weighted_value"]))
        if not all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4) for a, b in checks):
            invalid.append(f"/components/{name}/value_reconstruction")
        terms[name] = {
            "coefficient": coefficient, "raw_value": value, "weighted_value": weighted_value,
            "raw_gradient_norm": float(raw_norms[i]), "weighted_gradient_norm": float(norms[i]),
            "norm_share": float(norms[i] / norm_sum) if norm_sum else 0.0,
            "cosine_with_total": float(dots[i] / (norms[i] * total_norm))
            if norms[i] and total_norm else None,
            "projection_share": float(dots[i] / total_norm**2) if total_norm else None,
        }
    if not (math.isclose(total_value, observed_total, rel_tol=1e-5, abs_tol=1e-4)
            and math.isclose(sum(t["weighted_value"] for t in terms.values()),
                             observed_total, rel_tol=1e-5, abs_tol=1e-4)):
        invalid.append("/total_value_reconstruction")
    pairwise = [[float(pair_dots[i, j] / (norms[i] * norms[j]))
                 if norms[i] and norms[j] else None for j in range(len(names))]
                for i in range(len(names))]
    result = {
        "components": terms, "component_order": list(names), "pairwise_cosines": pairwise,
        "total_gradient_norm": total_norm, "sum_component_norms": norm_sum,
        "total_gradient_max_abs": float(np.max(np.abs(original))),
        "cancellation_ratio": total_norm / norm_sum if norm_sum else None,
        "gradient_reconstruction_max_abs_error": error,
        "total_gradient_centering_max_abs_error": centering_error,
        "gradient_reconstruction_tolerance": tolerance,
        "total": total_value, "observed_total": observed_total,
    }
    # Unlike scalar loss sentinels, large finite gradient norms are legitimate.
    def nonfinite(value, path=""):
        if isinstance(value, dict):
            return [p for k, v in value.items() for p in nonfinite(v, f"{path}/{k}")]
        if isinstance(value, list):
            return [p for i, v in enumerate(value) for p in nonfinite(v, f"{path}/{i}")]
        return [path] if isinstance(value, float) and not math.isfinite(value) else []
    invalid.extend(nonfinite(result))
    return {**result, "finite": not invalid, "invalid_paths": invalid}


def sample_gradient_step(interval, step, n_steps):
    return interval > 0 and (step is None or step == 0 or step == n_steps - 1
                             or step % interval == 0)


def gradient_evaluator(optimizers, config):
    """One first-order loss VJP, with original total-gradient processing.

    No differentiation of value_and_grad and no separate forward replay. The
    actual total, not the reconstructed component sum, supplies the optimizer
    gradient. This preserves the objective and gradient-processing formulas,
    not the compiled identity or bitwise numerics of original batched_eval.
    """
    jax, jnp, eqx = optimizers.jax, optimizers.jnp, optimizers.eqx

    @eqx.filter_jit
    def evaluate(loss, xs, keys):
        def single(x, key):
            def vector(sequence):
                total, aux = loss(sequence, key=key)
                records = list(component_records(aux, config))
                raw = jnp.stack([jnp.asarray(r[1]).reshape(()) for r in records])
                coefficients = jnp.stack([
                    jnp.asarray(r[2]).reshape(()) * jnp.asarray(r[3]).reshape(())
                    for r in records
                ])
                return jnp.concatenate((raw, jnp.asarray(total).reshape(1))), (coefficients, aux)

            values, pullback, (coefficients, aux) = jax.vjp(vector, x, has_aux=True)
            gradients = jax.lax.map(
                lambda index: pullback(jax.nn.one_hot(
                    index, values.shape[0], dtype=values.dtype,
                ))[0],
                jnp.arange(values.shape[0]),
            )
            total_gradient = gradients[-1]
            # Match original batched_eval exactly; never substitute the component
            # gradient sum, which remains an independent reconstruction check.
            gradient = jnp.nan_to_num(
                total_gradient - total_gradient.mean(axis=-1, keepdims=True),
            )
            result = (jnp.nan_to_num(values[-1], nan=1e6), aux, gradient)
            return result, (values, coefficients, gradients)

        return jax.lax.map(lambda pair: single(*pair), (xs, keys))
    return evaluate


class Recorder:
    def __init__(self, directory, config, identities, seed, gradient_interval=0,
                 gradient_trajectories=1):
        if gradient_interval < 0 or gradient_trajectories < 1:
            raise ValueError("Gradient interval must be nonnegative and trajectories positive")
        self.directory = Path(directory)
        self.config = config
        self.identities = identities
        self.seed = seed
        self.evaluation = 0
        self.sink = (self.directory / "components.jsonl").open("w", buffering=1)
        self.gradient_interval = gradient_interval
        self.gradient_trajectories = gradient_trajectories
        self.gradient_sink = ((self.directory / "gradients.jsonl").open("w", buffering=1)
                              if gradient_interval else None)

    def close(self):
        self.sink.close()
        if self.gradient_sink is not None:
            self.gradient_sink.close()

    def record_gradient(self, metrics, *, trajectory, evaluation, location, random_key,
                        runtime, batch_size):
        row = {"schema_version": 3, "evaluation": evaluation,
               "seed": self.seed, "trajectory": trajectory, **self.identities[trajectory],
               **location, **metrics, "random_key": random_key,
               "evaluation_and_diagnostic_runtime_seconds": runtime,
               "coverage": {"selection": "first_n_trajectories",
                            "requested_trajectories": self.gradient_trajectories,
                            "selected_trajectories": list(range(min(batch_size, self.gradient_trajectories))),
                            "batch_size": batch_size, "gradient_interval": self.gradient_interval,
                            "schedule": "first_last_interval_and_cycle_rank",
                            "component_count": len(metrics["component_order"]),
                            "execution": "same_evaluation_direct_loss_first_order_vjp",
                            "vocabulary_axis": -1, "before_total_clipping": True}}
        self.gradient_sink.write(json.dumps(json_value(row), allow_nan=False) + "\n")
        if not metrics["finite"]:
            raise FloatingPointError("Invalid or unreconciled component gradients: " +
                                     "; ".join(metrics["invalid_paths"]))

    def record(self, values, auxiliary, *, cycle, stage, step, sharp_replicate, random_keys=None):
        # Transfer only already-returned scalars/auxiliary, not sequences/gradients.
        errors = []
        for trajectory, identity in enumerate(self.identities):
            def take(tree):
                if isinstance(tree, dict):
                    return {k: take(v) for k, v in tree.items()}
                if isinstance(tree, (list, tuple)):
                    return [take(v) for v in tree]
                return tree[trajectory]
            aux = take(auxiliary)
            total = float(values[trajectory])
            invalid = invalid_values({"total": total, "aux": aux})
            terms = components(aux, self.config)
            invalid += invalid_values(terms, "/components")
            reconstructed = sum(term["weighted_value"] for term in terms.values())
            residual = total - reconstructed
            if not invalid and not math.isclose(total, reconstructed, rel_tol=1e-5, abs_tol=1e-4):
                invalid.append("/components/reconstruction")
            row = {"schema_version": 1, "evaluation": self.evaluation,
                   "seed": self.seed, "trajectory": trajectory, **identity,
                   "cycle": cycle, "stage": stage, "step": step,
                   "sharp_replicate": sharp_replicate,
                   "total": total, "components": terms, "aux": aux,
                   "reconstructed_total": reconstructed, "reconstruction_residual": residual,
                   "finite": not invalid, "invalid_paths": invalid}
            if random_keys is not None:
                row["random_key"] = random_keys[trajectory]
            self.sink.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            errors.extend(f"trajectory {trajectory}: {p}" for p in invalid)
        self.evaluation += 1
        if errors:
            raise FloatingPointError("Invalid/sentinel or unreconciled objective: " + "; ".join(errors))

    def install(self, optimizers, combination):
        original = optimizers.batched_eval
        combination.__call__, source = instrument_combination(combination.__call__)
        (self.directory / "instrumented_combination.py").write_text(source + "\n")
        if self.gradient_interval:
            original, source = sequential_evaluator(original)
            (self.directory / "instrumented_batched_eval.py").write_text(source + "\n")

        diagnose = gradient_evaluator(optimizers, self.config) if self.gradient_interval else None
        original_optimizer = optimizers.batched_simplex_APGM
        stage_calls = 0
        active = None

        def observed_optimizer(*args, **kwargs):
            nonlocal stage_calls, active
            state = sys._getframe(1).f_locals
            if "cyc" not in state or state["args"].resume_soft:
                raise RuntimeError("Unsupported run_design optimizer callsite/resume")
            stage_index = stage_calls % (1 + state["n_rep"])
            active = {"cycle": int(state["cyc"]) + 1,
                      "stage": "soft" if stage_index == 0 else "sharp",
                      "sharp_replicate": None if stage_index == 0 else stage_index - 1}
            stage_calls += 1
            try:
                return original_optimizer(*args, **kwargs)
            finally:
                active = None

        def observed(*args, **kwargs):
            caller = sys._getframe(1)
            state = caller.f_locals
            if active is not None:
                location = {**active, "step": int(state["_iter"])}
            elif caller.f_code.co_name == "main" and "cyc" in state:
                location = {"cycle": int(state["cyc"]) + 1, "stage": "cycle_rank",
                            "step": None, "sharp_replicate": int(state["rep"])}
            else:
                raise RuntimeError("Unrecognized run_design evaluation callsite")
            bound = inspect.signature(original).bind(*args, **kwargs).arguments
            keys = bound["keys"]
            sampled = sample_gradient_step(
                self.gradient_interval, location["step"], state.get("n_steps", 0),
            )
            if sampled:
                started = time.perf_counter()
                result, batch = diagnose(bound["loss"], bound["xs"], keys)
            else:
                result = original(*args, **kwargs)
            evaluation = self.evaluation
            self.record(result[0], result[1],
                        random_keys=optimizers.jax.random.key_data(keys), **location)
            if sampled:
                selected = min(len(self.identities), self.gradient_trajectories)
                # Keep the original batch shape on device; transfer only requested
                # diagnostic trajectories from this same evaluation.
                batch = optimizers.jax.device_get(tuple(value[:selected] for value in batch))
                runtime = (time.perf_counter() - started) / selected
                for trajectory in range(selected):
                    values, weights, gradients = (value[trajectory] for value in batch)
                    aux = optimizers.jax.tree.map(lambda leaf: leaf[trajectory], result[1])
                    observed = components(aux, self.config)
                    metrics = gradient_metrics(
                        list(observed), values[:-1], weights, gradients[:-1], gradients[-1],
                        result[2][trajectory], observed, float(values[-1]),
                        float(result[0][trajectory]),
                    )
                    self.record_gradient(
                        metrics, trajectory=trajectory, evaluation=evaluation, location=location,
                        random_key=optimizers.jax.random.key_data(keys[trajectory]),
                        runtime=runtime, batch_size=len(self.identities),
                    )
            return result
        optimizers.batched_eval = observed
        optimizers.batched_simplex_APGM = observed_optimizer


def source_manifest(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in paths}
