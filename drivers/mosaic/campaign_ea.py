"""Persistent, manifest-driven four-descendant Mosaic EA runner.

Run adjacent to frozen cycle_binders.py and evolution.py with --manifest,
--arm and --chunk-index. --smoke exclusively uses the canary output tree.
Checkpoints are cycle-boundary snapshots; interrupted roots restart from their
recorded deterministic seed, while validated complete roots are never rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ALPHABET = "ARNDCQEGHILKMFPSTWYV"
SEARCH_ALPHABET = ALPHABET.replace("C", "")


def now():
    return datetime.now(UTC).isoformat()


def cpu(value):
    """JSON-safe numeric trees, rejecting nonfinite results rather than hiding them."""
    if isinstance(value, dict):
        return {str(k): cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"unsupported result dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite result")
    return array.item() if array.ndim == 0 else array.tolist()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_text(path, json.dumps(cpu(payload), indent=2, allow_nan=False) + "\n")


def checkpoint(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    converted = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"invalid numeric checkpoint field {name}")
        converted[name] = array
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **converted)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_sequence(sequence, original):
    if len(sequence) != 75 or any(aa not in ALPHABET for aa in sequence):
        raise ValueError("expected a canonical 75-residue binder")
    if [i for i, aa in enumerate(sequence) if aa == "C"] != [
        i for i, aa in enumerate(original) if aa == "C"
    ]:
        raise ValueError("cysteine constraints were violated")


def validate_settings(settings):
    for field in ("population", "microbatch", "cycles", "soft_steps", "sharp_steps", "num_samples"):
        if type(settings[field]) is not int or settings[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if settings["population"] != 4 or 4 % settings["microbatch"]:
        raise ValueError("logical population must be four; microbatch must divide four")
    if not 0 <= settings["init_noise"] <= 1:
        raise ValueError("init_noise must be in [0,1]")
    if not 0 <= settings["noise_low"] <= settings["noise_high"] <= 1:
        raise ValueError("invalid mutation noise range")
    if not 0 <= settings["cull_fraction"] < 1:
        raise ValueError("invalid cull_fraction")
    for stage in ("soft", "sharp"):
        for field in ("stepsize", "scale"):
            if settings[f"{stage}_{field}"] <= 0:
                raise ValueError(f"{stage}_{field} must be positive")
        if not 0 <= settings[f"{stage}_momentum"] < 1:
            raise ValueError(f"{stage}_momentum must be in [0,1)")
    cpu(settings)


def plan(manifest, arm, chunk_index, smoke):
    settings = dict(manifest["settings"])
    if smoke:
        settings.update(manifest["canary"]["settings"])
        if chunk_index != 0:
            raise ValueError("canary has one chunk per arm (index zero)")
        ids = manifest["canary"]["root_ids"][arm]
        output = Path(manifest["root"]) / "canary"
        worker = output / "control" / f"{arm}-{chunk_index}.json"
    else:
        chunks = manifest["chunks"][arm]
        if not 0 <= chunk_index < len(chunks):
            raise ValueError("chunk index out of range")
        ids = chunks[chunk_index]
        output = Path(manifest["root"]) / "results"
        worker = Path(manifest["root"]) / "control" / "workers" / f"{arm}-{chunk_index}.json"
    validate_settings(settings)
    records = json.loads(Path(manifest["roots"]).read_text())
    roots = {}
    for record in records:
        identity = record["design_id"]
        if identity in roots:
            raise ValueError(f"duplicate root {identity}")
        if not identity or Path(identity).name != identity or identity in (".", ".."):
            raise ValueError("unsafe root identity")
        validate_sequence(record["sequence"], record["sequence"])
        if hashlib.sha256(record["sequence"].encode()).hexdigest() != record["sequence_sha256"]:
            raise ValueError(f"root sequence hash mismatch: {identity}")
        roots[identity] = record
    if not smoke:
        all_ids = [identity for chunk in manifest["chunks"][arm] for identity in chunk]
        expected_ids = {identity for identity, root in roots.items() if root["arm"] == arm}
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != expected_ids:
            raise ValueError("production chunks must cover every arm root exactly once")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("chunk roots must be nonempty and unique")
    selected = [roots[identity] for identity in ids]
    if any(root["arm"] != arm for root in selected):
        raise ValueError("chunk mixes target arms")
    return settings, selected, output / arm, worker


def root_seed(seed_base, phase_id, arm, design_id):
    message = json.dumps([seed_base, phase_id, arm, design_id], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], "little")


def initial_states(sequence, rng, noise):
    # C sites have an irrelevant uniform latent row; the loss maps them to exact C.
    latent = np.full((75, 19), 1 / 19, dtype=np.float32)
    for position, aa in enumerate(sequence):
        if aa != "C":
            latent[position] = np.eye(19, dtype=np.float32)[SEARCH_ALPHABET.index(aa)]
    hard = np.repeat(latent[None], 4, axis=0)
    draws = rng.dirichlet(np.ones(19), size=(4, 75))
    return hard, ((1 - noise) * hard + noise * draws).astype(np.float32)


def validate_states(states):
    states = np.asarray(states)
    if states.shape != (4, 75, 19) or not np.isfinite(states).all():
        raise ValueError("invalid population state")
    if np.any(states < -1e-6) or not np.allclose(states.sum(-1), 1, atol=1e-4):
        raise ValueError("population left the probability simplex")
    return states


def archive_update(archive, states, losses, node_ids, checkpoint_id):
    """Archive follows selected ancestry, never the previous occupant of a slot."""
    if archive is None:
        return {"states": np.asarray(states).copy(), "losses": np.asarray(losses).copy(),
                "nodes": list(node_ids), "checkpoints": [checkpoint_id] * 4}
    take = np.asarray(losses) < archive["losses"]
    return {
        "states": np.where(take[:, None, None], states, archive["states"]),
        "losses": np.where(take, losses, archive["losses"]),
        "nodes": [node_ids[i] if take[i] else archive["nodes"][i] for i in range(4)],
        "checkpoints": [checkpoint_id if take[i] else archive["checkpoints"][i] for i in range(4)],
    }


def gather_archive(archive, parents):
    return {"states": archive["states"][parents].copy(),
            "losses": archive["losses"][parents].copy(),
            "nodes": [archive["nodes"][i] for i in parents],
            "checkpoints": [archive["checkpoints"][i] for i in parents]}


def validated_complete(directory, root, settings, fingerprint):
    path = directory / "result.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if not result.get("complete") or result.get("manifest_sha256") != fingerprint:
        raise ValueError(f"existing result is incomplete or belongs to a different manifest: {path}")
    if (result.get("design_id") != root["design_id"] or result.get("arm") != root["arm"]
            or result.get("original_sequence") != root["sequence"] or result.get("settings") != settings):
        raise ValueError(f"existing result identity/settings mismatch: {path}")
    members = result["final_members"]
    if len(members) != 4 or len({m["candidate_id"] for m in members}) != 4:
        raise ValueError("completed result does not have four distinct offspring")
    for member in members:
        validate_sequence(member["sequence"], root["sequence"])
        cpu([member["ranking_loss"], member["optimization_loss"]])
        if not member["source_node_id"] or not member["lineage"]:
            raise ValueError("missing member lineage")
    winner = min(members, key=lambda m: (m["ranking_loss"], m["candidate_id"]))
    if result["selected_winner"] != winner:
        raise ValueError("selected winner is not the exact best ranked offspring")
    expected = {"input.json", "loss_metadata.json", "cycles.jsonl", "baseline.npz", "final.npz"}
    expected.update(f"cycle-{cycle}.npz" for cycle in range(1, settings["cycles"] + 1))
    if set(result["artifacts"]) != expected:
        raise ValueError("incomplete result artifact inventory")
    for name, expected_hash in result["artifacts"].items():
        if digest(directory / name) != expected_hash:
            raise ValueError(f"checkpoint integrity mismatch: {name}")
        if name.endswith(".npz"):
            with np.load(directory / name, allow_pickle=False) as arrays:
                for field in arrays.files:
                    value = arrays[field]
                    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                        raise ValueError(f"invalid checkpoint {name}:{field}")
                validate_states(arrays["states"])
    cycles = [json.loads(line) for line in (directory / "cycles.jsonl").read_text().splitlines()]
    if [entry["cycle"] for entry in cycles] != list(range(settings["cycles"] + 1)):
        raise ValueError("missing or duplicate cycle records")
    nodes = {}
    for entry in cycles:
        if len(entry["members"]) != 4:
            raise ValueError("missing cycle members")
        for member in entry["members"]:
            validate_sequence(member["sequence"], root["sequence"])
            cpu([member["loss"], member["components"]])
            if member["candidate_id"] in nodes:
                raise ValueError("duplicate immutable node identity")
            if member["parent_id"] != root["design_id"] and member["parent_id"] not in nodes:
                raise ValueError("unknown parent node")
            nodes[member["candidate_id"]] = member
    for member in members:
        source = nodes.get(member["source_node_id"])
        if source is None:
            raise ValueError("unknown elite source node")
        if source["sequence"] != member["sequence"] or source["loss"] != member["optimization_loss"]:
            raise ValueError("final member differs from its claimed winning source")
        path = [member["candidate_id"], source["candidate_id"]]
        node = source
        while node["parent_id"] in nodes:
            node = nodes[node["parent_id"]]
            path.append(node["candidate_id"])
        path.append(root["design_id"])
        if member["lineage"] != list(reversed(path)):
            raise ValueError("final member lineage differs from its actual source ancestry")
    return True


class Runtime:
    """Model-owning module: loaded once, root-specific data remain array leaves."""

    def __init__(self, target, settings):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import cycle_binders as cycle
        self.jax, self.jnp, self.cycle = jax, jnp, cycle
        cycle.preflight()
        self.target = cycle.read_fasta(target["fasta"])
        self.msa = target["msa"]
        epitope = cycle.parse_epitope(",".join(map(str, target["epitope_idx"])), len(self.target))
        self.folder, self.template = cycle.build(
            self.target, self.msa, 75, epitope_idx=epitope,
            num_samples=settings["num_samples"], root_constraints=True,
        )
        # Unlike mosaic.optimizers.batched_eval, this does not nan_to_num losses.
        @eqx.filter_jit
        def evaluate(loss, states, keys):
            return jax.vmap(lambda state, key: loss(state, key=key))(states, keys)
        self.evaluate = evaluate
        self.microbatch = settings["microbatch"]

    def constrain(self, sequence):
        self.loss = self.template.for_root(sequence)
        self.original = sequence

    def decode(self, states):
        full = np.asarray(self.jax.vmap(self.loss.sequence)(self.jnp.asarray(states)))
        sequences = ["".join(ALPHABET[i] for i in row) for row in full.argmax(-1)]
        for sequence in sequences:
            validate_sequence(sequence, self.original)
        return sequences

    def evaluate_states(self, states, keys):
        states = validate_states(states)
        values, components = [], []
        for start in range(0, 4, self.microbatch):
            stop = start + self.microbatch
            value, aux = self.evaluate(self.loss, self.jnp.asarray(states[start:stop]), keys[start:stop])
            values.extend(cpu(value))
            for local in range(self.microbatch):
                components.append(cpu(self.jax.tree.map(lambda a: a[local], aux)))
        return np.asarray(values), components

    def optimize(self, states, key, settings, label):
        endpoints = [[], [], []]
        for start in range(0, 4, self.microbatch):
            # A fixed microbatch is part of the frozen experiment settings.
            batch_key = key if self.microbatch == 4 else self.jax.random.fold_in(key, start)
            batch = self.cycle.optimize_cycle(
                self.loss, self.jnp.asarray(states[start:start + self.microbatch]),
                batch_key, settings, label=f"{label} members {start}:{start + self.microbatch}",
            )
            for collection, value in zip(endpoints, batch, strict=True):
                collection.append(np.asarray(value))
        return tuple(validate_states(np.concatenate(parts)) for parts in endpoints)

    def rank(self, sequence):
        value = self.cycle.refold_and_score(self.folder, sequence, self.target, self.msa)
        return cpu(value)


def run_root(directory, root, settings, manifest, fingerprint, runtime, status):
    from evolution import cull_steps, make_hooks
    jax = runtime.jax
    seed = root_seed(settings["seed_base"], manifest["phase_id"], root["arm"], root["design_id"])
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    key, eval_key = jax.random.split(key)
    eval_keys = jax.random.split(eval_key, 1)
    eval_keys = runtime.jnp.repeat(eval_keys, 4, axis=0)
    runtime.constrain(root["sequence"])
    hard, states = initial_states(root["sequence"], rng, settings["init_noise"])
    select, mutate = make_hooks(settings["cull_fraction"], settings["noise_low"], settings["noise_high"])
    width = settings["soft_steps"] + settings["sharp_steps"]
    selection_steps = set(cull_steps(np.arange(1, settings["cycles"] + 1) * width, 1).tolist())
    journal, ancestry, archive = [], {}, None
    parent_ids = [root["design_id"]] * 4
    artifacts = ["input.json", "loss_metadata.json", "cycles.jsonl", "baseline.npz", "final.npz"]
    atomic_json(directory / "input.json", {"root": root, "settings": settings, "seed": seed,
                "manifest_sha256": fingerprint, "phase_id": manifest["phase_id"]})
    atomic_json(directory / "loss_metadata.json", {
        "objective": "cycle_binders.build", "alphabet": ALPHABET, "search_alphabet": SEARCH_ALPHABET,
        "fixed_cysteine_positions": [i for i, aa in enumerate(root["sequence"]) if aa == "C"],
        "target": manifest["targets"][root["arm"]], "in_loop_recycles": 1,
        "in_loop_samples": settings["num_samples"], "ranking_recycles": 3, "ranking_samples": 6,
        "ranking_key": 0, "ranking_objective": "IPTMLoss + .5 TargetBinderIPSAE + .5 BinderTargetIPSAE",
        "selection_steps": sorted(selection_steps), "evaluation_keys_shared_across_members": True,
        "archive_policy": "best evaluated sharp-cycle state along selected ancestry; baseline excluded",
        "objective_terms": {"BinderTargetContact": 2, "DistogramIPTMProxy": 2,
                            "WithinBinderContact": 1, "HelixLoss": 1, "DistogramRadiusOfGyration": 1,
                            "PLDDTLoss": 1, "IPSAE_min": 1, "InverseFoldingSequenceRecovery": 1},
    })

    def persist(stage, cycle, evaluated, losses, components, nodes, parents, extra=None):
        decoded = runtime.decode(evaluated)
        members = []
        for slot in range(4):
            ancestry[nodes[slot]] = parents[slot]
            members.append({"slot": slot, "candidate_id": nodes[slot], "parent_id": parents[slot],
                            "sequence": decoded[slot], "loss": float(losses[slot]),
                            "components": components[slot]})
        record = {"stage": stage, "cycle": cycle, "cumulative_steps": cycle * width,
                  "members": members, **(extra or {})}
        journal.append(record)
        atomic_text(directory / "cycles.jsonl", "".join(json.dumps(cpu(row), allow_nan=False) + "\n" for row in journal))

    def rng_bytes():
        return np.frombuffer(json.dumps(rng.bit_generator.state).encode(), dtype=np.uint8)

    status("baseline", cycle=0)
    baseline_values, baseline_components = runtime.evaluate_states(hard, eval_keys)
    baseline_nodes = [f"{root['design_id']}:baseline:{i}" for i in range(4)]
    persist("baseline", 0, hard, baseline_values, baseline_components, baseline_nodes, parent_ids)
    checkpoint(directory / "baseline.npz", states=hard, initial_states=states, losses=baseline_values,
               eval_keys=jax.random.key_data(eval_keys), next_key=jax.random.key_data(key),
               rng_state=rng_bytes())
    parent_ids = baseline_nodes
    for cycle in range(1, settings["cycles"] + 1):
        status("optimizing", cycle=cycle)
        key, cycle_key = jax.random.split(key)
        soft, final, endpoint = runtime.optimize(states, cycle_key, settings, f"{root['design_id']} cycle {cycle}")
        losses, components = runtime.evaluate_states(endpoint, eval_keys)
        nodes = [f"{root['design_id']}:cycle:{cycle}:member:{i}" for i in range(4)]
        name = f"cycle-{cycle}.npz"
        archive = archive_update(archive, endpoint, losses, nodes, name)
        parents = np.arange(4, dtype=np.int64)
        next_states = endpoint
        extra = {"checkpoint_id": name, "elite_source_node_ids": archive["nodes"],
                 "elite_checkpoint_ids": archive["checkpoints"]}
        # Record evaluated preselection identities, not the newly gathered slots.
        if cycle * width in selection_steps:
            parents = np.asarray(select(losses, rng))
            if parents.shape != (4,) or parents.dtype.kind not in "iu" or np.any((parents < 0) | (parents >= 4)):
                raise ValueError("selection returned invalid parent indices")
            next_states = validate_states(mutate(endpoint[parents], parents, losses, rng)).astype(np.float32)
            extra.update(parents=parents.tolist(),
                         replacement_mapping=[{"slot": i, "parent_slot": int(parents[i]),
                                               "parent_node_id": nodes[parents[i]]} for i in range(4)],
                         noise={"low": settings["noise_low"], "high": settings["noise_high"]})
        persist("cycle", cycle, endpoint, losses, components, nodes, parent_ids, extra)
        checkpoint(directory / name, states=endpoint, soft_endpoint=soft, final_endpoint=final,
                   next_states=next_states, losses=losses, parents=parents,
                   elite_states=archive["states"], elite_losses=archive["losses"],
                   eval_keys=jax.random.key_data(eval_keys), cycle_key=jax.random.key_data(cycle_key),
                   next_key=jax.random.key_data(key), rng_state=rng_bytes())
        artifacts.append(name)
        if cycle < settings["cycles"]:
            archive = gather_archive(archive, parents)
            parent_ids = [nodes[i] for i in parents]
            states = next_states
    status("ranking", cycle=settings["cycles"])
    sequences = runtime.decode(archive["states"])
    checkpoint(directory / "final.npz", states=archive["states"], losses=archive["losses"],
               last_cycle_states=endpoint, eval_keys=jax.random.key_data(eval_keys),
               next_key=jax.random.key_data(key), rng_state=rng_bytes())

    def lineage(node):
        path = [node]
        while node in ancestry:
            node = ancestry[node]
            path.append(node)
        return list(reversed(path))

    members = []
    for slot, sequence in enumerate(sequences):
        print(f"{root['design_id']}: ranking offspring {slot + 1}/4 (3 recycles, 6 samples)", flush=True)
        ranking = cpu(runtime.rank(sequence))
        candidate_id = f"{root['design_id']}:offspring:{slot}"
        source = archive["nodes"][slot]
        members.append({"slot": slot, "candidate_id": candidate_id, "sequence": sequence,
                        "ranking_loss": ranking, "optimization_loss": float(archive["losses"][slot]),
                        "source_node_id": source, "parent_id": source,
                        "winning_checkpoint_id": archive["checkpoints"][slot],
                        "lineage": lineage(source) + [candidate_id],
                        "endpoint_node_id": nodes[slot], "optimization_lineage": lineage(nodes[slot])})
        status("ranking", ranked_members=slot + 1)
    result = {"phase_id": manifest["phase_id"], "design_id": root["design_id"], "arm": root["arm"],
              "tool": root["tool"], "root_identity": root, "original_sequence": root["sequence"],
              "complete": True, "settings": settings, "seed": seed,
              "seeds": {"root": seed, "ranking": 0, "eval_keys": jax.random.key_data(eval_keys)},
              "manifest_sha256": fingerprint, "final_members": members,
              "selected_winner": min(members, key=lambda m: (m["ranking_loss"], m["candidate_id"])),
              "artifacts": {name: digest(directory / name) for name in artifacts}, "completed_at": now()}
    atomic_json(directory / "result.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("pocket", "y84"))
    parser.add_argument("--chunk-index", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    settings, roots, output, worker_path = plan(manifest, args.arm, args.chunk_index, args.smoke)
    fingerprint = digest(args.manifest)
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    # Kernel-released advisory lock permits safe restart after preemption.
    with worker_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"phase_id": manifest["phase_id"], "job_id": os.environ.get("SLURM_JOB_ID"),
                 "arm": args.arm, "chunk_index": args.chunk_index, "smoke": args.smoke,
                 "status": "running", "started_at": now(), "ended_at": None,
                 "planned_roots": [root["design_id"] for root in roots], "completed_count": 0,
                 "current_root": None, "progress": {}}
        atomic_json(worker_path, state)
        runtime = None
        current_directory = None
        try:
            for root in roots:
                directory = output / root["design_id"]
                directory.mkdir(parents=True, exist_ok=True)
                current_directory = directory
                state["current_root"] = root["design_id"]
                state["progress"] = {}
                root_started = now()

                def status(stage, **progress):
                    print(f"{root['design_id']}: {stage} {progress}", flush=True)
                    state["progress"] = {"stage": stage, **progress}
                    atomic_json(worker_path, state)
                    atomic_json(directory / "status.json", {
                        "phase_id": manifest["phase_id"], "job_id": state["job_id"],
                        "arm": args.arm, "chunk_index": args.chunk_index,
                        "design_id": root["design_id"], "status": stage,
                        "started_at": root_started, "ended_at": now() if stage == "complete" else None,
                        "progress": progress,
                    })

                if validated_complete(directory, root, settings, fingerprint):
                    print(f"{root['design_id']}: skipping validated complete result", flush=True)
                    root_status = directory / "status.json"
                    if not root_status.exists() or json.loads(root_status.read_text()).get("status") != "complete":
                        status("complete", recovered_after_result_commit=True)
                else:
                    status("loading" if runtime is None else "starting")
                    if runtime is None:
                        runtime = Runtime(manifest["targets"][args.arm], settings)
                    run_root(directory, root, settings, manifest, fingerprint, runtime, status)
                    status("complete")
                state["completed_count"] += 1
                atomic_json(worker_path, state)
            state.update(status="complete", ended_at=now(), current_root=None)
            atomic_json(worker_path, state)
            return 0
        except BaseException as error:
            details = {"error": str(error), "traceback": traceback.format_exc()}
            state.update(status="failed", ended_at=now(), **details)
            atomic_json(worker_path, state)
            if current_directory is not None:
                atomic_json(current_directory / "status.json", {
                    "phase_id": manifest["phase_id"], "arm": args.arm, "chunk_index": args.chunk_index,
                    "design_id": state["current_root"], "status": "failed",
                    "job_id": state["job_id"], "started_at": root_started, "ended_at": now(),
                    "progress": state["progress"], **details,
                })
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
