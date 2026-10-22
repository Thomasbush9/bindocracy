#!/usr/bin/env python3
"""Run frozen run_design.py with signed-loss traces and same-evaluation gradients."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import sys

from component_trace import GRADIENT_ATOL, GRADIENT_RTOL, Recorder, source_manifest


def main():
    # Only arguments after -- reach the frozen collaborator script.
    separator = sys.argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("identities", type=Path)
    parser.add_argument("--gradient-interval", type=int, default=0)
    parser.add_argument("--gradient-trajectories", type=int, default=1)
    settings = parser.parse_args(sys.argv[1:separator])
    if settings.gradient_interval < 0 or settings.gradient_trajectories < 1:
        parser.error("gradient interval must be nonnegative and trajectories positive")
    if settings.gradient_interval:
        flag = "--xla_gpu_exclude_nondeterministic_ops=true"
        flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
        if any(value.startswith("--xla_gpu_exclude_nondeterministic_ops") and value != flag
               for value in flags):
            raise ValueError("Gradient diagnostics require deterministic GPU operations")
        if flag not in flags:
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flag).strip()
        # Reduced-precision matmuls break strict component-gradient additivity
        # even with deterministic kernels. Set before importing JAX/models.
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    script = settings.script.resolve()
    identities_path = settings.identities.resolve()
    arguments = sys.argv[separator + 1:]
    config_path = Path(arguments[arguments.index("--config") + 1])
    directory = Path(arguments[arguments.index("--out") + 1])
    config = json.loads(config_path.read_text())
    identities = json.loads(identities_path.read_text())
    sys.path.insert(0, str(script.parent))

    if any(m.get("params", {}).get("num_samples", 1) != 1 for m in config["models"]):
        raise ValueError("Component recording requires the historical single-sample recipe")
    import mosaic
    import mosaic.common as common
    import mosaic.optimizers as optimizers

    package = Path(mosaic.__file__).resolve().parent
    sources = [script, script.with_name("design_config.py"), Path(__file__),
               Path(__file__).with_name("component_trace.py"), *sorted(package.rglob("*.py"))]
    dependencies = {}
    if any(m["name"] == "of3" for m in config["models"]):
        # This is the differentiable OpenFold3 source recipe, never a scoring
        # driver's model-name replacement. Missing dependencies are fatal.
        import jopenfold3
        import mosaic.models.of3 as of3
        if of3.OF3.__module__ != "mosaic.models.of3":
            raise RuntimeError("OF3 has been replaced by a different model")
        dependency = Path(jopenfold3.__file__).resolve().parent
        sources.extend(sorted(dependency.rglob("*.py")))
        dependencies["jopenfold3"] = str(dependency)
    manifest = {"schema_version": 1, "argv": [str(script), *arguments],
                "sources_sha256": source_manifest(sources), "dependencies": dependencies,
                "identity": identities, "config": config,
                "gradients": {
                    "schema_version": 3,
                    "enabled": settings.gradient_interval > 0,
                    "interval": settings.gradient_interval,
                    "requested_trajectories": settings.gradient_trajectories,
                    "selected_trajectories": list(range(min(len(identities), settings.gradient_trajectories)))
                    if settings.gradient_interval else [],
                    "batch_size": len(identities),
                    "cycles": int(arguments[arguments.index("--soft-sharp-cycles") + 1]),
                    "sharp_replicates": int(arguments[arguments.index("--sharp-replicates") + 1]),
                    "schedule": "first_last_interval_and_cycle_rank",
                    "stage_steps": {stage: config["optimizer"][stage]["n_steps"]
                                    for stage in ("soft", "sharp")},
                    "reconstruction_atol": GRADIENT_ATOL,
                    "reconstruction_rtol": GRADIENT_RTOL,
                    "reconstruction_rule": "max_abs_error <= atol + rtol * max(abs(optimizer_gradient))",
                    "centering": "subtract amino-acid-axis mean before weighting and total clipping",
                    "norm_share": "weighted_gradient_norm / sum_component_norms; zero if all zero",
                    "projection_share": "dot(weighted_component, original_total) / total_norm**2; signed, not percentage attribution; null if total zero",
                    "cancellation_ratio": "total_gradient_norm / sum_component_norms; null if all zero",
                    "pairwise_cosines": "matrix in component_order; null if either weighted norm zero",
                    "runtime": "full sampled evaluation, diagnostic compilation, recording and synchronized transfer, divided across recorded trajectories; not incremental overhead",
                    "nonfinite": "unweighted component and actual-total pullbacks checked before diagnostic sanitization, including zero-weight components; optimizer total uses original nan_to_num formula and scalar sentinels are rejected",
                    "execution": "sampled path: one direct unchanged loss evaluation per row, sequential lax.map over the original batch and component pullbacks; unsampled path serializes original independent-row evaluator",
                    "optimizer_gradient": "actual-total pullback followed by original amino-acid-axis centering and nan_to_num; not a component-sum replacement; sampled path does not call original compiled batched_eval or preserve its return identity or bitwise numerics",
                    "derivative_order": "first-order only; no nested value_and_grad differentiation",
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                    "matmul_precision": optimizers.jax.config.jax_default_matmul_precision,
                    "determinism": "exclude nondeterministic GPU operations; serialize batch rows to bound deterministic-kernel memory; batch membership, keys and update schedule unchanged",
                    "total_gradient_check": "centering-consistency check against optimizer gradient derived from this same total pullback, not an independent total-gradient comparison; weighted component-sum reconstruction remains independent",
                },
                "semantics": {
                    "value": "actual signed term return, after configured clipping",
                    "weighted_value": "value * model_weight * term_weight",
                    "aux": "all original diagnostics plus LinearCombination value/weight records",
                    "total": "unchanged objective total with original nan_to_num formula; sentinel rejected; sampled path uses direct loss VJP, unsampled path uses original scalar evaluator with sequential batch execution when gradients enabled",
                    "step": "zero-based stage step; null for original cycle-ranking evaluation",
                    "selection": "unchanged source best sharp candidate across all cycles",
                    "nonfinite": "any nonfinite or absolute value >= 1e6 fails the whole chunk",
                }}
    (directory / "trace_manifest.json").write_text(json.dumps(manifest, indent=2))
    recorder = Recorder(directory, config, identities, config["run"]["seed"],
                        settings.gradient_interval, settings.gradient_trajectories)
    recorder.install(optimizers, common.LinearCombination)
    sys.argv = [str(script), *arguments]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
