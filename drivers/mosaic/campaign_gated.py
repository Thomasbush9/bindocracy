"""Distinct-original, local gated Mosaic populations with resumable APGM state.

Each batch has at most four originals, one initialized row per original. Gates
use the loss evaluated at APGM's extrapolated v, NOT the unevaluated updated x.
Replacement perturbs a survivor's current updated simplex x with evolution.perturb;
only offspring momentum is reset at gates. Stage boundaries reset momentum; between
cycles evolution.perturb commits argmax then re-softens all active members.
Final refolds use actual surviving endpoints, never a killed lineage's archive.

CLI: --manifest PATH --batch-indices 0,1 [--smoke]. A process owns one target/model;
smoke uses canary.settings and writes exclusively below smoke/results.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import traceback
from pathlib import Path

import numpy as np

from campaign_ea import (
    ALPHABET, SEARCH_ALPHABET, Runtime as BaseRuntime, atomic_json, atomic_text,
    checkpoint, cpu, digest, initial_states, now, root_seed, validate_sequence,
)

METRIC = "relative_gap_to_survivor_median"


def effective_settings(manifest, smoke=False):
    settings = dict(manifest["settings"])
    if smoke:
        settings.update(manifest["canary"]["settings"])
    for field in ("population", "microbatch", "cycles", "soft_steps", "sharp_steps", "num_samples", "cull_interval"):
        if type(settings[field]) is not int or settings[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if settings["population"] != 4 or settings["microbatch"] != 4:
        raise ValueError("gated populations and compiled microbatches must be four")
    if settings["cycles"] != 3 or settings["cull_interval"] != 25:
        raise ValueError("three cycles and 25-step cumulative gates are required")
    if not smoke and (settings["soft_steps"], settings["sharp_steps"]) != (90, 30):
        raise ValueError("production requires three 90-soft + 30-sharp cycles")
    if not 0 <= settings["init_noise"] <= 1:
        raise ValueError("invalid initialization noise")
    if not 0.1 <= settings["noise_low"] <= settings["noise_high"] <= 0.2:
        raise ValueError("replacement noise must remain in [0.1,0.2]")
    for stage in ("soft", "sharp"):
        for name in ("stepsize", "scale"):
            if settings[f"{stage}_{name}"] <= 0:
                raise ValueError(f"invalid {stage} {name}")
        if not 0 <= settings[f"{stage}_momentum"] < 1:
            raise ValueError(f"invalid {stage} momentum")
    cpu(settings)
    return settings


def stages(settings):
    return [(cycle, stage, settings[f"{stage}_steps"])
            for cycle in range(1, settings["cycles"] + 1) for stage in ("soft", "sharp")]


def snapshot_steps(settings):
    ends, total = set(), 0
    for _, _, length in stages(settings):
        total += length
        ends.add(total)
    return sorted(ends | set(range(25, total + 1, 25)))


def check_states(states, count=4):
    states = np.asarray(states)
    if states.shape != (count, 75, 19) or not np.isfinite(states).all():
        raise ValueError("invalid simplex state shape or values")
    if np.any(states < -1e-6) or not np.allclose(states.sum(-1), 1, atol=1e-4):
        raise ValueError("state left the probability simplex")
    return states


def padded(array):
    array = np.asarray(array)
    return np.concatenate([array, np.repeat(array[-1:], 4 - len(array), axis=0)])


def json_bytes(value):
    return np.frombuffer(json.dumps(cpu(value), allow_nan=False).encode(), dtype=np.uint8)


def from_bytes(value):
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes())


def append_record(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(cpu(record), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def gate_decision(losses, threshold):
    losses = np.asarray(losses)
    cpu(losses)
    worst = int(np.argmax(losses))  # deterministic smallest-index tie break
    median = float(np.median(np.delete(losses, worst)))
    gap = float((losses[worst] - median) / max(abs(median), 1e-6))
    return worst, median, gap, bool(gap > threshold)


class Runtime(BaseRuntime):
    """Map only small constraint leaves, with all model weights shared in_axes=None."""

    def __init__(self, target, settings):
        super().__init__(target, settings)
        import equinox as eqx
        from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
        from mosaic.optimizers import projection_simplex
        jax = self.jax
        self.projection = projection_simplex

        def biases(tree):
            terms = jax.tree_util.tree_leaves(
                tree.loss, is_leaf=lambda node: isinstance(node, InverseFoldingSequenceRecovery))
            return tuple(term.bias for term in terms if isinstance(term, InverseFoldingSequenceRecovery))

        if len(biases(self.template)) != 1:
            raise ValueError("expected exactly one MPNN recovery term")

        def constrained(loss, mask, bias):
            loss = eqx.tree_at(biases, loss, (bias,))
            return eqx.tree_at(lambda tree: tree.cysteines, loss, mask)

        @eqx.filter_jit
        def gradients(loss, xs, keys, masks, bias):
            def single(x, key, mask, b):
                (value, aux), grad = eqx.filter_value_and_grad(
                    constrained(loss, mask, b), has_aux=True)(x, key=key)
                # Upstream centers the gradient before clipping. Nonfinite values
                # are deliberately rejected on host instead of nan_to_num hiding them.
                return value, aux, grad - grad.mean(axis=-1, keepdims=True)
            return jax.vmap(single)(xs, keys, masks, bias)

        @eqx.filter_jit
        def evaluate(loss, xs, keys, masks, bias):
            return jax.vmap(lambda x, k, m, b: constrained(loss, m, b)(x, key=k))(
                xs, keys, masks, bias)

        self.gradients, self.evaluate_group = gradients, evaluate

    def constraint_arrays(self, masks):
        jnp = self.jnp
        masks = jnp.asarray(padded(masks))
        allowed = jnp.where(
            masks[:, :, None],
            jnp.arange(20)[None, None, :] == ALPHABET.index("C"),
            jnp.arange(20)[None, None, :] != ALPHABET.index("C"))
        return masks, jnp.where(allowed, 0.0, -1e6)

    def evaluate_states(self, states, masks, keys, gradient=False):
        count = len(states)
        masks, bias = self.constraint_arrays(masks)
        fn = self.gradients if gradient else self.evaluate_group
        result = fn(self.template, self.jnp.asarray(padded(states)), keys, masks, bias)
        values = np.asarray(result[0])[:count]
        components = [cpu(self.jax.tree.map(lambda x: x[i], result[1])) for i in range(count)]
        cpu(values)
        if gradient:
            grads = np.asarray(result[2])[:count]
            cpu(grads)
            return values, components, grads
        return values, components

    def project(self, states):
        return self.projection(np.asarray(states).reshape(-1, 19)).reshape(states.shape).astype(np.float32)

    def decode(self, states, masks):
        sequences = []
        for state, mask in zip(states, masks, strict=True):
            sequences.append("".join("C" if mask[i] else SEARCH_ALPHABET[token]
                                     for i, token in enumerate(state.argmax(-1))))
        return sequences


def run_batch(directory, roots, batch, settings, manifest, fingerprint, runtime, status):
    from evolution import perturb
    directory = Path(directory)
    count = len(roots)
    arm, batch_id = batch["arm"], batch["batch_id"]
    phase = manifest["phase_id"]
    prefix = f"{phase}:{arm}:{batch_id}"
    ids = [root["design_id"] for root in roots]
    seed = root_seed(settings["seed_base"], phase, arm, batch_id)
    mutation_seed = root_seed(seed, phase, arm, "mutation")
    cycle_seed = root_seed(seed, phase, arm, "cycle_restart")
    eval_seed = root_seed(seed, phase, arm, "evaluation")
    optimization_seed = root_seed(seed, phase, arm, "optimization")
    initial_seeds = [root_seed(settings["seed_base"], phase, arm, identity) for identity in ids]
    rng = np.random.default_rng(mutation_seed)
    cycle_rng = np.random.default_rng(cycle_seed)
    jax = runtime.jax
    key = jax.random.key(optimization_seed)
    eval_keys = jax.random.split(jax.random.key(eval_seed), 4)
    masks = np.asarray([[aa == "C" for aa in root["sequence"]] for root in roots])
    calibration = json.loads(Path(manifest["calibration"]).read_text())
    calibration_hash = digest(manifest["calibration"])
    if calibration["metric"] != METRIC:
        raise ValueError("unsupported gate calibration metric")
    total = sum(length for _, _, length in stages(settings))
    thresholds = {step: float(calibration["by_arm"][arm][str(step)])
                  for step in range(25, total + 1, 25)}
    cpu(thresholds)
    snapshots = set(snapshot_steps(settings))
    artifacts = ["input.json", "loss_metadata.json", "baseline.npz", "steps.jsonl", "gates.jsonl", "cycles.jsonl", "final.npz"]
    artifacts += [f"cycle-restart-{cycle}.npz" for cycle in range(1, settings["cycles"])]
    artifacts += [f"trajectory-{step:04d}.npz" for step in range(1, total + 1)]
    artifacts += [f"checkpoint-{step:04d}.npz" for step in snapshot_steps(settings)]
    seeds = {"batch": seed, "initial": initial_seeds, "mutation": mutation_seed,
             "optimization": optimization_seed, "evaluation": eval_seed, "cycle_restart": cycle_seed, "ranking": 0}
    identity = {"phase_id": phase, "arm": arm, "batch_id": batch_id, "roots": roots,
                "settings": settings, "seeds": seeds, "manifest_sha256": fingerprint,
                "calibration_sha256": calibration_hash, "active_count": count, "padding_count": 4 - count}
    completed = 0
    lineages = list(ids)
    nodes = [f"{prefix}:initial:{slot}" for slot in range(count)]
    ancestry = {node: ids[slot] for slot, node in enumerate(nodes)}
    resume_path = directory / "resume.json"
    if resume_path.exists():
        if json.loads((directory / "input.json").read_text()) != cpu(identity):
            raise ValueError("resume identity/settings/calibration mismatch")
        pointer = json.loads(resume_path.read_text())
        name = pointer["checkpoint"]
        if name not in artifacts or digest(directory / name) != pointer["sha256"]:
            raise ValueError("invalid resume checkpoint")
        with np.load(directory / name, allow_pickle=False) as saved:
            x, x_prev = saved["x"].copy(), saved["x_prev"].copy()
            masks = saved["cysteines"].astype(bool)
            key = jax.random.wrap_key_data(saved["next_key"])
            meta = from_bytes(saved["metadata"])
        completed = meta["completed_steps"]
        nodes, lineages, ancestry = meta["nodes"], meta["lineage_roots"], meta["ancestry"]
        rng.bit_generator.state = meta["mutation_rng"]
        cycle_rng.bit_generator.state = meta["cycle_rng"]
        # Discard only the uncommitted journal suffix after a preemption. Raw
        # trailing trajectory files are deterministically overwritten on resume.
        width = settings["soft_steps"] + settings["sharp_steps"]
        for journal, limit in (("steps.jsonl", completed), ("gates.jsonl", completed // 25),
                               ("cycles.jsonl", min(completed // width, settings["cycles"] - 1))):
            with (directory / journal).open() as stream:
                committed = [json.loads(next(stream)) for _ in range(limit)]
            atomic_text(directory / journal, "".join(json.dumps(row) + "\n" for row in committed))
        check_states(x, count)
        check_states(x_prev, count)
        status("resuming", completed_steps=completed)
    else:
        atomic_json(directory / "input.json", identity)
        atomic_json(directory / "loss_metadata.json", {
            "objective": "cycle_binders.build", "metric": METRIC,
            "calibration": calibration, "calibration_sha256": calibration_hash,
            "gate_loss_state": "extrapolated v before APGM update; not updated x",
            "replacement_parent_state": "survivor updated x at the same step",
            "final_policy": "actual surviving final updated x, independently evaluated; no best archive",
            "stage_policy": "reset x_prev=x at soft/sharp boundaries; evolution.perturb all active updated endpoints between cycles, no cycle culling",
            "gradient_policy": "upstream centered gradient, sqrt(75) norm clipping, float64 CPU simplex projection then float32; nonfinite values fail",
            "padding_policy": "fixed B4 compute only; padding excluded from metrics, selection, ancestry, output",
            "in_loop_recycles": 1, "in_loop_samples": settings["num_samples"],
            "ranking_recycles": 3, "ranking_samples": 6, "ranking_key": 0,
            "target": manifest["targets"][arm], "gate_steps": sorted(thresholds),
        })
        for name in ("steps.jsonl", "gates.jsonl", "cycles.jsonl"):
            atomic_text(directory / name, "")
        hard, initialized = [], []
        for root, initial_seed in zip(roots, initial_seeds, strict=True):
            original, noisy = initial_states(root["sequence"], np.random.default_rng(initial_seed), settings["init_noise"])
            hard.append(original[0])
            initialized.append(noisy[0])  # ONE row per distinct original, not four.
        x = runtime.project(np.asarray(initialized))
        x_prev = x.copy()
        status("baseline", active_count=count, padding_count=4 - count)
        baseline_losses, baseline_components = runtime.evaluate_states(np.asarray(hard), masks, eval_keys)
        checkpoint(directory / "baseline.npz", states=hard, initial_states=x, losses=baseline_losses,
                   cysteines=masks, eval_keys=jax.random.key_data(eval_keys),
                   metadata=json_bytes({"root_ids": ids, "components": baseline_components, "seeds": seeds}))

    offset = 0
    for cycle, stage, length in stages(settings):
        stage_end = offset + length
        if completed >= stage_end:
            offset = stage_end
            continue
        start = max(0, completed - offset)
        if start == 0:
            x = runtime.project(x)
            x_prev = x.copy()  # the intentional upstream APGM stage reset
            key, _ = jax.random.split(key)
        status("optimizing", cycle=cycle, stage_name=stage, completed_steps=completed)
        for local_step in range(start, length):
            step = offset + local_step + 1
            before_nodes, before_lineages = list(nodes), list(lineages)
            before_masks = masks.copy()
            previous_x, previous_prev = x, x_prev
            v = np.asarray(x + settings[f"{stage}_momentum"] * (x - x_prev), dtype=np.float32)
            evaluation_keys = jax.random.split(key, 4)
            values, components, grads = runtime.evaluate_states(v, masks, evaluation_keys, gradient=True)
            norms = np.sqrt((grads ** 2).sum(axis=(-2, -1)))
            clips = np.ones_like(norms)
            np.divide(np.sqrt(75), norms, out=clips, where=norms > np.sqrt(75))
            clipped = grads * clips[:, None, None]
            x_new = runtime.project(settings[f"{stage}_scale"] * (v - settings[f"{stage}_stepsize"] * clipped))
            x_prev, x = previous_x, x_new
            key = jax.random.fold_in(key, 0)
            trajectory = f"trajectory-{step:04d}.npz"
            # This immutable record precedes selection. Losses belong only to v.
            checkpoint(directory / trajectory, evaluated_state=v, losses=values,
                       x_before=previous_x, x_prev_before=previous_prev, updated_x=x_new,
                       gradients=grads, gradient_norms=norms, gradient_clips=clips,
                       cysteines=before_masks, evaluation_keys=jax.random.key_data(evaluation_keys))
            append_record(directory / "steps.jsonl", {
                "step": step, "cycle": cycle, "stage": stage, "stage_step": local_step + 1,
                "active_count": count, "padding_count": 4 - count, "trajectory": trajectory,
                "loss_state": "evaluated_state", "optimizer_update_state": "updated_x",
                "members": [{"slot": slot, "slot_original_id": ids[slot],
                             "node_id": before_nodes[slot], "lineage_root_id": before_lineages[slot],
                             "loss": values[slot], "components": components[slot],
                             "gradient_norm": norms[slot], "gradient_clip": clips[slot]}
                            for slot in range(count)],
            })
            if step % 25 == 0:
                gate = {"step": step, "cycle": cycle, "stage": stage, "metric": METRIC,
                        "threshold": thresholds[step], "accepted": False, "active_count": count,
                        "losses": values, "pre_nodes": before_nodes, "pre_lineage_roots": before_lineages,
                        "loss_state": f"{trajectory}:evaluated_state", "parent_state": f"{trajectory}:updated_x"}
                if count < 4:
                    gate.update(reason="partial_group_no_cull", replacement_count=0)
                else:
                    worst, median, gap, accept = gate_decision(values, thresholds[step])
                    gate.update(worst_slot=worst, survivor_median=median, gap=gap, accepted=accept,
                                reason="gap_exceeds_threshold" if accept else "gap_does_not_exceed_threshold",
                                replacement_count=int(accept))
                    if accept:
                        parent = int(rng.choice([slot for slot in range(count) if slot != worst]))
                        mutation_before = rng.bit_generator.state
                        replay_rng = np.random.default_rng()
                        replay_rng.bit_generator.state = mutation_before
                        noise = float(replay_rng.uniform(settings["noise_low"], settings["noise_high"]))
                        offspring = perturb(x[parent:parent + 1], settings["noise_low"],
                                            settings["noise_high"], rng)[0].astype(np.float32)
                        # Copy-on-write: previous_x belongs to the immutable trajectory.
                        x, x_prev = x.copy(), x_prev.copy()
                        x[worst], x_prev[worst] = offspring, offspring
                        masks[worst] = masks[parent]
                        node = f"{prefix}:gate:{step}:slot:{worst}"
                        ancestry[node] = nodes[parent]
                        gate.update(parent_slot=parent, parent_node_id=nodes[parent],
                                    killed_node_id=nodes[worst], offspring_node_id=node, noise=noise,
                                    mutation="evolution.perturb: argmax then Dirichlet re-softening",
                                    mutation_rng_before=mutation_before, mutation_rng_after=rng.bit_generator.state)
                        nodes[worst], lineages[worst] = node, lineages[parent]
                gate.update(post_nodes=list(nodes), post_lineage_roots=list(lineages))
                append_record(directory / "gates.jsonl", gate)
                status("gate", step=step, accepted=gate["accepted"], reason=gate["reason"])
            if stage == "sharp" and local_step + 1 == length and cycle < settings["cycles"]:
                restart_name = f"cycle-restart-{cycle}.npz"
                cycle_before = cycle_rng.bit_generator.state
                restart = perturb(x, settings["noise_low"], settings["noise_high"], cycle_rng).astype(np.float32)
                checkpoint(directory / restart_name, before=x, x_prev_before=x_prev,
                           after=restart, cysteines=masks,
                           metadata=json_bytes({"cycle": cycle, "step": step, "nodes": nodes,
                                                "lineage_roots": lineages, "rng_before": cycle_before,
                                                "rng_after": cycle_rng.bit_generator.state}))
                append_record(directory / "cycles.jsonl", {
                    "cycle": cycle, "step": step, "event": "all_active_members_resoften",
                    "method": "evolution.perturb", "noise_low": settings["noise_low"],
                    "noise_high": settings["noise_high"], "active_count": count,
                    "checkpoint": restart_name, "nodes": list(nodes), "lineage_roots": list(lineages),
                })
                x = restart
                x_prev = x.copy()
            check_states(x, count)
            completed = step
            if step in snapshots:
                name = f"checkpoint-{step:04d}.npz"
                checkpoint(directory / name, x=x, x_prev=x_prev, cysteines=masks,
                           evaluated_state=v, evaluated_losses=values, evaluated_cysteines=before_masks,
                           updated_x_before_gate=x_new, evaluation_keys=jax.random.key_data(evaluation_keys),
                           next_key=jax.random.key_data(key), eval_keys=jax.random.key_data(eval_keys),
                           metadata=json_bytes({"completed_steps": step, "cycle": cycle, "stage": stage,
                                                "stage_step": local_step + 1, "nodes": nodes,
                                                "lineage_roots": lineages, "ancestry": ancestry,
                                                "evaluated_nodes": before_nodes, "mutation_rng": rng.bit_generator.state,
                                                "cycle_rng": cycle_rng.bit_generator.state,
                                                "manifest_sha256": fingerprint, "calibration_sha256": calibration_hash}))
                atomic_json(resume_path, {"checkpoint": name, "sha256": digest(directory / name)})
        offset = stage_end

    status("ranking", completed_steps=total)
    losses, components = runtime.evaluate_states(x, masks, eval_keys)
    sequences = runtime.decode(x, masks)
    checkpoint(directory / "final.npz", states=x, x_prev=x_prev, cysteines=masks, losses=losses,
               eval_keys=jax.random.key_data(eval_keys), next_key=jax.random.key_data(key),
               metadata=json_bytes({"nodes": nodes, "lineage_roots": lineages, "ancestry": ancestry,
                                    "components": components, "mutation_rng": rng.bit_generator.state,
                                    "cycle_rng": cycle_rng.bit_generator.state}))
    members = []
    for slot, sequence in enumerate(sequences):
        original = roots[ids.index(lineages[slot])]["sequence"]
        validate_sequence(sequence, original)
        source = nodes[slot]
        lineage = [source]
        while lineage[-1] in ancestry:
            lineage.append(ancestry[lineage[-1]])
        candidate = f"{prefix}:final:{slot}"
        ranking = runtime.rank(sequence)
        members.append({"slot": slot, "slot_original_id": ids[slot], "lineage_root_id": lineages[slot],
                        "candidate_id": candidate, "source_node_id": source, "parent_id": source,
                        "lineage": list(reversed(lineage)) + [candidate], "sequence": sequence,
                        "optimization_loss": float(losses[slot]), "optimization_components": components[slot],
                        "ranking_loss": ranking, "winning_checkpoint_id": "final.npz"})
        # Each completed refold is recoverable without waiting for the group result.
        atomic_json(directory / "ranked_members.json", {"members": members, "complete": len(members) == count})
        status("ranking", ranked_members=slot + 1)
    artifacts.append("ranked_members.json")
    gates = records(directory / "gates.jsonl")
    accepted = sum(gate["accepted"] for gate in gates)
    result = {"phase_id": phase, "arm": arm, "batch_id": batch_id, "root_ids": ids, "root_identities": roots,
              "active_count": count, "padding_count": 4 - count, "complete": True,
              "settings": settings, "manifest_sha256": fingerprint, "calibration_sha256": calibration_hash,
              "seeds": seeds, "final_members": members, "ancestry": ancestry,
              "gate_counts": {"checks": len(gates), "accepted": accepted, "rejected": len(gates) - accepted,
                              "member_replacements": accepted, "member_opportunities": len(gates) * count},
              "selected_winner": min(members, key=lambda member: (member["ranking_loss"], member["candidate_id"])),
              "artifacts": {name: digest(directory / name) for name in artifacts}, "completed_at": now()}
    atomic_json(directory / "result.json", result)


def validated_complete(directory, roots, settings, fingerprint):
    """Validate committed population identity, complete artifacts and live ancestry.

    Absent result returns False. Any malformed committed result raises rather
    than silently overwriting scientific output. This function needs only NumPy,
    not JAX, a GPU, or the model dependencies.
    """
    from evolution import perturb
    directory = Path(directory)
    path = directory / "result.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    count = len(roots)
    ids = [root["design_id"] for root in roots]
    if not 1 <= count <= 4 or len(set(ids)) != count:
        raise ValueError("expected one to four distinct roots")
    if (result.get("complete") is not True or result.get("manifest_sha256") != fingerprint
            or result.get("settings") != settings or result.get("root_identities") != roots
            or result.get("root_ids") != ids or result.get("active_count") != count
            or result.get("padding_count") != 4 - count or result.get("batch_id") != directory.name
            or any(root["arm"] != result.get("arm") for root in roots)):
        raise ValueError("completed batch identity/settings mismatch")
    members = result["final_members"]
    if len(members) != count or [member["slot"] for member in members] != list(range(count)):
        raise ValueError("completed result population/slots mismatch")
    total = sum(length for _, _, length in stages(settings))
    expected = {"input.json", "loss_metadata.json", "baseline.npz", "steps.jsonl",
                "gates.jsonl", "cycles.jsonl", "final.npz", "ranked_members.json"}
    expected.update(f"cycle-restart-{cycle}.npz" for cycle in range(1, settings["cycles"]))
    expected.update(f"trajectory-{step:04d}.npz" for step in range(1, total + 1))
    expected.update(f"checkpoint-{step:04d}.npz" for step in snapshot_steps(settings))
    if set(result["artifacts"]) != expected:
        raise ValueError("incomplete or unexpected artifact inventory")
    for name, expected_hash in result["artifacts"].items():
        if digest(directory / name) != expected_hash:
            raise ValueError(f"artifact integrity mismatch: {name}")
        if name.endswith(".npz"):
            with np.load(directory / name, allow_pickle=False) as arrays:
                for field in arrays.files:
                    value = arrays[field]
                    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                        raise ValueError(f"nonfinite/non-numeric checkpoint {name}:{field}")
    input_record = json.loads((directory / "input.json").read_text())
    if (input_record["roots"] != roots or input_record["settings"] != settings
            or input_record["manifest_sha256"] != fingerprint
            or input_record["phase_id"] != result["phase_id"]
            or input_record["batch_id"] != result["batch_id"]
            or input_record["seeds"] != result["seeds"]
            or input_record["calibration_sha256"] != result["calibration_sha256"]):
        raise ValueError("input provenance mismatch")
    metadata = json.loads((directory / "loss_metadata.json").read_text())
    if metadata["calibration_sha256"] != result["calibration_sha256"] or metadata["metric"] != METRIC:
        raise ValueError("calibration provenance mismatch")
    calibration = metadata["calibration"]
    thresholds = calibration["by_arm"][result["arm"]]
    if calibration["metric"] != METRIC:
        raise ValueError("wrong calibration objective")
    steps = records(directory / "steps.jsonl")
    gates = records(directory / "gates.jsonl")
    cycles = records(directory / "cycles.jsonl")
    if [row["step"] for row in steps] != list(range(1, total + 1)):
        raise ValueError("missing or duplicate optimization steps")
    if [row["step"] for row in gates] != list(range(25, total + 1, 25)):
        raise ValueError("missing or duplicate gate checks")
    width = settings["soft_steps"] + settings["sharp_steps"]
    if [(row["cycle"], row["step"]) for row in cycles] != [
            (cycle, cycle * width) for cycle in range(1, settings["cycles"])]:
        raise ValueError("missing cycle restarts")
    gate_by_step = {row["step"]: row for row in gates}
    prefix = f"{result['phase_id']}:{result['arm']}:{result['batch_id']}"
    nodes = [f"{prefix}:initial:{slot}" for slot in range(count)]
    ancestry = {node: ids[slot] for slot, node in enumerate(nodes)}
    lineages = list(ids)
    selection_rng = np.random.default_rng(result["seeds"]["mutation"])
    root_masks = {root["design_id"]: np.asarray([aa == "C" for aa in root["sequence"]]) for root in roots}
    with np.load(directory / "baseline.npz", allow_pickle=False) as baseline:
        check_states(baseline["states"], count)
        check_states(baseline["initial_states"], count)
        last_x = baseline["initial_states"].copy()
        last_prev = last_x.copy()
        if (baseline["losses"].shape != (count,)
                or from_bytes(baseline["metadata"])["root_ids"] != ids
                or not np.array_equal(baseline["cysteines"], [root_masks[identity] for identity in ids])):
            raise ValueError("baseline originals/masks mismatch")
    snapshots = set(snapshot_steps(settings))
    for row in steps:
        step = row["step"]
        cycle = (step - 1) // width + 1
        position = (step - 1) % width + 1
        stage = "soft" if position <= settings["soft_steps"] else "sharp"
        stage_step = position if stage == "soft" else position - settings["soft_steps"]
        if (row["cycle"], row["stage"], row["stage_step"]) != (cycle, stage, stage_step):
            raise ValueError("step schedule mismatch")
        if (row["active_count"] != count or row["padding_count"] != 4 - count
                or row["trajectory"] != f"trajectory-{step:04d}.npz" or len(row["members"]) != count):
            raise ValueError("step population/trajectory mismatch")
        with np.load(directory / row["trajectory"], allow_pickle=False) as trajectory:
            updated = check_states(trajectory["updated_x"], count).copy()
            previous = check_states(trajectory["x_before"], count).copy()
            check_states(trajectory["x_prev_before"], count)
            if stage_step == 1:
                if not np.allclose(previous, last_x, rtol=0, atol=1e-7) or not np.array_equal(trajectory["x_prev_before"], previous):
                    raise ValueError("incorrect intentional stage reset")
            elif not np.array_equal(previous, last_x) or not np.array_equal(trajectory["x_prev_before"], last_prev):
                raise ValueError("optimizer state was reset or changed within a stage")
            values = trajectory["losses"].copy()
            expected_v = np.asarray(previous + settings[f"{stage}_momentum"] * (
                previous - trajectory["x_prev_before"]), dtype=np.float32)
            if not np.array_equal(trajectory["evaluated_state"], expected_v):
                raise ValueError("evaluated loss/state correspondence mismatch")
            masks = np.asarray([root_masks[identity] for identity in lineages])
            if not np.array_equal(trajectory["cysteines"], masks):
                raise ValueError("trajectory constraints differ from selected ancestry")
            for slot, member in enumerate(row["members"]):
                if (member["slot"] != slot or member["slot_original_id"] != ids[slot]
                        or member["node_id"] != nodes[slot] or member["lineage_root_id"] != lineages[slot]
                        or member["loss"] != float(values[slot])):
                    raise ValueError("step member/loss/lineage mismatch")
                cpu(member["components"])
        if step in gate_by_step:
            gate = gate_by_step[step]
            if (gate["pre_nodes"] != nodes or gate["pre_lineage_roots"] != lineages
                    or gate["threshold"] != thresholds[str(step)] or gate["active_count"] != count
                    or gate["metric"] != METRIC or not np.array_equal(gate["losses"], values)):
                raise ValueError("gate input/provenance mismatch")
            if count < 4:
                if gate["accepted"] or gate["replacement_count"] != 0 or gate["reason"] != "partial_group_no_cull":
                    raise ValueError("partial group was culled")
            else:
                worst, median, gap, accepted = gate_decision(values, gate["threshold"])
                if (gate["worst_slot"] != worst or gate["survivor_median"] != median or gate["gap"] != gap
                        or gate["accepted"] != accepted or gate["replacement_count"] != int(accepted)):
                    raise ValueError("incorrect calibrated gate decision")
                if accepted:
                    parent = gate["parent_slot"]
                    if type(parent) is not int or parent not in set(range(count)) - {worst}:
                        raise ValueError("offspring parent is not a surviving member")
                    expected_parent = int(selection_rng.choice([slot for slot in range(count) if slot != worst]))
                    if parent != expected_parent or selection_rng.bit_generator.state != gate["mutation_rng_before"]:
                        raise ValueError("replacement parent/RNG is not the deterministic uniform survivor draw")
                    node = f"{prefix}:gate:{step}:slot:{worst}"
                    if (gate["parent_node_id"] != nodes[parent] or gate["killed_node_id"] != nodes[worst]
                            or gate["offspring_node_id"] != node or node in ancestry):
                        raise ValueError("replacement ancestry mismatch")
                    offspring = perturb(updated[parent:parent + 1], settings["noise_low"],
                                        settings["noise_high"], selection_rng)[0].astype(np.float32)
                    if selection_rng.bit_generator.state != gate["mutation_rng_after"]:
                        raise ValueError("replacement RNG progression mismatch")
                    updated[worst], previous[worst] = offspring, offspring
                    masks[worst] = masks[parent]
                    ancestry[node] = nodes[parent]
                    nodes[worst], lineages[worst] = node, lineages[parent]
            if gate["post_nodes"] != nodes or gate["post_lineage_roots"] != lineages:
                raise ValueError("post-gate lineage mismatch")
        if position == width and cycle < settings["cycles"]:
            with np.load(directory / f"cycle-restart-{cycle}.npz", allow_pickle=False) as restart:
                if not np.array_equal(restart["before"], updated) or not np.array_equal(restart["x_prev_before"], previous):
                    raise ValueError("cycle restart did not use surviving updated endpoints")
                meta = from_bytes(restart["metadata"])
                replay = np.random.default_rng()
                replay.bit_generator.state = meta["rng_before"]
                restarted = perturb(updated, settings["noise_low"], settings["noise_high"], replay).astype(np.float32)
                if (not np.array_equal(restarted, restart["after"])
                        or replay.bit_generator.state != meta["rng_after"]
                        or meta["nodes"] != nodes or meta["lineage_roots"] != lineages):
                    raise ValueError("cycle restart state/ancestry/RNG mismatch")
                updated, previous = restarted, restarted.copy()
        last_x, last_prev = updated, previous
        if step in snapshots:
            with np.load(directory / f"checkpoint-{step:04d}.npz", allow_pickle=False) as saved:
                check_states(saved["x"], count)
                check_states(saved["x_prev"], count)
                meta = from_bytes(saved["metadata"])
                if (not np.array_equal(saved["x"], updated) or not np.array_equal(saved["x_prev"], previous)
                        or not np.array_equal(saved["cysteines"], masks)
                        or not np.array_equal(saved["evaluated_losses"], values)
                        or meta["nodes"] != nodes or meta["lineage_roots"] != lineages
                        or meta["ancestry"] != ancestry or meta["completed_steps"] != step
                        or meta["manifest_sha256"] != fingerprint):
                    raise ValueError("checkpoint optimizer/lineage state mismatch")
    if result["ancestry"] != ancestry:
        raise ValueError("result ancestry differs from actual gate history")
    with np.load(directory / "final.npz", allow_pickle=False) as final:
        check_states(final["states"], count)
        if not np.array_equal(final["states"], updated) or not np.array_equal(final["cysteines"], masks):
            raise ValueError("final outputs are not actual surviving endpoints")
        meta = from_bytes(final["metadata"])
        if meta["nodes"] != nodes or meta["lineage_roots"] != lineages or meta["ancestry"] != ancestry:
            raise ValueError("final checkpoint lineage mismatch")
        for slot, member in enumerate(members):
            candidate = f"{prefix}:final:{slot}"
            lineage = [nodes[slot]]
            while lineage[-1] in ancestry:
                lineage.append(ancestry[lineage[-1]])
            expected_lineage = list(reversed(lineage)) + [candidate]
            decoded = "".join("C" if masks[slot, pos] else SEARCH_ALPHABET[token]
                              for pos, token in enumerate(final["states"][slot].argmax(-1)))
            if (member["candidate_id"] != candidate or member["slot_original_id"] != ids[slot]
                    or member["lineage_root_id"] != lineages[slot] or member["source_node_id"] != nodes[slot]
                    or member["parent_id"] != nodes[slot] or member["lineage"] != expected_lineage
                    or member["sequence"] != decoded or member["optimization_loss"] != float(final["losses"][slot])
                    or member["winning_checkpoint_id"] != "final.npz"):
                raise ValueError("final member is inconsistent with surviving lineage")
            validate_sequence(member["sequence"], roots[ids.index(lineages[slot])]["sequence"])
            cpu([member["optimization_loss"], member["ranking_loss"], member["optimization_components"]])
    if result["selected_winner"] != min(members, key=lambda member: (member["ranking_loss"], member["candidate_id"])):
        raise ValueError("selected winner is not the exact best refold")
    ranked = json.loads((directory / "ranked_members.json").read_text())
    if ranked != {"members": members, "complete": True}:
        raise ValueError("incremental refolds differ from committed result")
    accepted = sum(gate["accepted"] for gate in gates)
    if result["gate_counts"] != {"checks": len(gates), "accepted": accepted, "rejected": len(gates) - accepted,
                                 "member_replacements": accepted, "member_opportunities": len(gates) * count}:
        raise ValueError("gate summary mismatch")
    return True


def plan(manifest, indices, smoke):
    import hashlib
    settings = effective_settings(manifest, smoke)
    root_records = json.loads(Path(manifest["roots"]).read_text())
    roots = {}
    for root in root_records:
        identity = root["design_id"]
        if not isinstance(identity, str) or not identity or Path(identity).name != identity or identity in (".", ".."):
            raise ValueError("unsafe root identity")
        if identity in roots:
            raise ValueError("duplicate original identity")
        validate_sequence(root["sequence"], root["sequence"])
        if hashlib.sha256(root["sequence"].encode()).hexdigest() != root["sequence_sha256"]:
            raise ValueError("original sequence hash mismatch")
        roots[identity] = root
    seen, names = set(), set()
    for batch in manifest["batches"]:
        ids = batch["root_ids"]
        name, arm = batch["batch_id"], batch["arm"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("unsafe batch identity")
        if arm not in ("pocket", "y84") or (arm, name) in names:
            raise ValueError("invalid or duplicate arm/batch")
        names.add((arm, name))
        if not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids):
            raise ValueError("a local batch must contain one to four distinct originals")
        if any(identity not in roots or roots[identity]["arm"] != arm for identity in ids):
            raise ValueError("batch mixes arms or refers to an unknown original")
        if seen.intersection(ids):
            raise ValueError("original occurs in multiple batches")
        seen.update(ids)
    if seen != set(roots):
        raise ValueError("batches must cover the manifest original population exactly")
    if not indices or len(indices) != len(set(indices)) or any(index < 0 or index >= len(manifest["batches"]) for index in indices):
        raise ValueError("batch indices must be unique and in range")
    selected = [manifest["batches"][index] for index in indices]
    if len({batch["arm"] for batch in selected}) != 1:
        raise ValueError("a model-owning process may optimize only one target arm")
    return settings, [(batch, [roots[identity] for identity in batch["root_ids"]]) for batch in selected]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-indices", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    worker_path = args.manifest.parent / "gated-failed-status.json"
    current_directory = None
    state = {"status": "starting", "started_at": now(), "ended_at": None,
             "job_id": os.environ.get("SLURM_JOB_ID"), "completed_count": 0, "progress": {}}
    try:
        manifest = json.loads(args.manifest.read_text())
        indices = [int(value) for value in args.batch_indices.split(",")]
        settings, selected = plan(manifest, indices, args.smoke)
        output = Path(manifest["root"])
        if args.smoke:
            output = output / "smoke"
        worker_path = output / "control" / "workers" / f"gated-{','.join(map(str, indices))}.json"
        worker_path.parent.mkdir(parents=True, exist_ok=True)
        state.update(phase_id=manifest["phase_id"], batch_indices=indices, smoke=args.smoke,
                     arm=selected[0][0]["arm"], status="running")
        fingerprint = digest(args.manifest)
        with worker_path.with_suffix(".lock").open("a") as worker_lock:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(worker_path, state)
            runtime = None
            for batch, roots in selected:
                directory = output / "results" / batch["arm"] / batch["batch_id"]
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / ".lock").open("a") as batch_lock:
                    fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    current_directory = directory
                    state["current_batch"] = batch["batch_id"]

                    def status(stage, **progress):
                        print(f"{batch['arm']}/{batch['batch_id']}: {stage} {progress}", flush=True)
                        state["progress"] = {"stage": stage, **progress}
                        atomic_json(worker_path, state)
                        atomic_json(directory / "status.json", {
                            "phase_id": manifest["phase_id"], "arm": batch["arm"], "batch_id": batch["batch_id"],
                            "status": stage, "job_id": state["job_id"], "started_at": state["started_at"],
                            "ended_at": now() if stage == "complete" else None, "progress": progress})

                    if validated_complete(directory, roots, settings, fingerprint):
                        status("complete", recovered_after_result_commit=True)
                    else:
                        status("loading" if runtime is None else "starting")
                        if runtime is None:
                            runtime = Runtime(manifest["targets"][batch["arm"]], settings)
                        run_batch(directory, roots, batch, settings, manifest, fingerprint, runtime, status)
                        status("complete")
                    state["completed_count"] += 1
                    atomic_json(worker_path, state)
                    current_directory = None
            state.update(status="complete", ended_at=now(), current_batch=None)
            atomic_json(worker_path, state)
        return 0
    except BaseException as error:
        details = {"error": str(error), "traceback": traceback.format_exc()}
        state.update(status="failed", ended_at=now(), **details)
        atomic_json(worker_path, state)
        if current_directory is not None:
            atomic_json(current_directory / "status.json", {
                **state, "batch_id": state["current_batch"], "status": "failed"})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
