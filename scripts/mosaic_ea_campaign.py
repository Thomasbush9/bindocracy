#!/usr/bin/env python3
"""Freeze, qualify, submit and collect the two-arm Mosaic EA campaign.

No database writer or GPU model runs in this controller. Native workers use the
frozen driver and source overlay. Two arrays each have an eight-GPU throttle;
canaries must be terminal and independently validated before either is submitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

ARMS = ("pocket", "y84")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
SETTINGS = {"population": 4, "microbatch": 4, "cycles": 3, "soft_steps": 90, "sharp_steps": 30,
                "soft_stepsize": 0.05, "sharp_stepsize": 0.025, "soft_momentum": 0.7,
                "sharp_momentum": 0.5, "soft_scale": 1.0, "sharp_scale": 2.0,
                "num_samples": 1, "init_noise": 0.15, "cull_fraction": 0.5,
                "noise_low": 0.10, "noise_high": 0.20, "seed_base": 2026101400}


def now():
    return datetime.now(UTC).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=90, check=False)
    require(result.returncode == 0, f"Command failed: {shlex.join(args)}\n{result.stderr}")
    return result.stdout


def prepare(root, source, repo):
    require(not root.exists(), f"Refusing to overwrite phase {root}")
    candidates = load(source / "reports/candidates.json")["scored"]
    require(len(candidates) == 2100 and len({r['design_id'] for r in candidates}) == 2100,
            "Expected exact balanced cohort of 2100 unique input identities")
    require(all(len(r['sequence']) == 75 and set(r['sequence']) <= set('ACDEFGHIKLMNPQRSTVWY')
                for r in candidates), "Invalid seed sequences")
    counts = Counter((r['arm'], r['tool']) for r in candidates)
    require(len(counts) == 14 and set(counts.values()) == {150}, "Unbalanced generator/target cohort")
    campaign = Path(load(source / 'phase.json')['campaign_root'])
    for directory in ('provenance/driver', 'inputs', 'logs', 'control', 'reports'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    provenance = root / 'provenance'
    driver = provenance / 'driver'
    for name in ('campaign_ea.py', 'cycle_binders.py', 'evolution.py'):
        shutil.copy2(repo / 'drivers/mosaic' / name, driver / name)
    shutil.copy2(Path(__file__), provenance / 'mosaic_ea_campaign.py')
    shutil.copy2(source / 'scripts/watch_phase.py', provenance / 'watch_phase.py')
    mosaic = repo.parent / 'mosaic_setup/mosaic'
    shutil.copytree(mosaic / 'src', provenance / 'mosaic-src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.agents', '.codex'))
    shutil.copy2(mosaic / 'singularity/mosaic-exec.sh', provenance / 'mosaic-exec.sh')
    shutil.copy2(source / 'reports/candidates.json', root / 'inputs/baseline-candidates.json')
    shutil.copy2(source / 'reports/predictions.json', root / 'inputs/baseline-predictions.json')
    shutil.copy2(source / 'cohort.json', root / 'inputs/source-cohort.json')
    roots_path = root / 'inputs/roots.json'
    atomic(roots_path, candidates)
    targets = {}
    for arm in ARMS:
        dest = root / 'inputs' / arm
        dest.mkdir()
        for name in ('target.fasta', 'target.a3m'):
            shutil.copy2(campaign / 'data' / arm / name, dest / name)
        targets[arm] = {"fasta": str(dest / 'target.fasta'), "msa": str(dest / 'target.a3m'),
                            "epitope_idx": [67, 98, 154] if arm == 'pocket' else [61]}
    # Interleave generator/decile strata so early progress is not one-generator-only.
    chunks, canary_ids = {}, {}
    for arm in ARMS:
        members = sorted((r for r in candidates if r['arm'] == arm),
                         key=lambda r: (r.get('sampling_decile', 0), r['tool'], r['design_id']))
        groups = {}
        for row in members:
            groups.setdefault(row['tool'], []).append(row)
        ordered = [groups[tool][i] for i in range(150) for tool in sorted(groups)]
        chunks[arm] = [[r['design_id'] for r in ordered[i:i + 4]] for i in range(0, len(ordered), 4)]
        canary_ids[arm] = [next(r['design_id'] for r in members if ('C' in r['sequence']) == has_c)
                          for has_c in (False, True)]
    runtime = {"image": str(repo.parent / 'images/mosaic.sif'),
                   "weights": str(repo.parent / 'mosaic_setup/weights'),
                   "wrapper": str(provenance / 'mosaic-exec.sh'),
                   "dev_source": str(provenance / 'mosaic-src'), "driver": str(driver / 'campaign_ea.py')}
    # Hash the executable science and core checkpoint, not unrelated model caches.
    core_weights = Path(runtime['weights']) / 'boltz/boltz2_conf.ckpt'
    require(core_weights.is_file() and core_weights.stat().st_size > 0, 'Missing Boltz2 weights')
    inventory = {str(p): sha(p) for p in provenance.rglob('*') if p.is_file()}
    inventory.update({str(p): sha(p) for p in (roots_path, root / 'inputs/baseline-candidates.json',
                                               root / 'inputs/baseline-predictions.json', core_weights,
                                               Path(runtime['image']))})
    for target in targets.values():
        for key in ('fasta', 'msa'):
            inventory[target[key]] = sha(target[key])
    atomic(provenance / 'sha256.json', inventory)
    manifest = {"phase_id": root.name, "root": str(root), "source_candidates": str(source / 'reports/candidates.json'),
                    "roots": str(roots_path), "targets": targets, "chunks": chunks, "settings": SETTINGS,
                    "canary": {"root_ids": canary_ids, "settings": {"cycles": 3, "soft_steps": 2, "sharp_steps": 1}},
                    "runtime": runtime, "account": 'kempner_bsabatini_lab', "partition": 'kempner_h100',
                    "resources": {"max_concurrent_gpus": 16, "concurrency_per_arm": 8, "gpus_per_job": 1,
                                   "cpus_per_job": 8, "memory_gb": 96, "walltime": '12:00:00', "canary_walltime": '02:00:00'},
                    "created_at": now(), "approval": 'User authorized both-arm full cohort optimization and launch; budget16H100 interpreted as concurrent GPU cap, not an unbounded retry budget.',
                    "evaluation": 'Boltz2 internal ranking only. ESMFold and Protenix held out; not used for selection or launched here.',
                    "cysteine_policy": 'Preserve original C positions exactly; forbid new C; never silently substitute A.',
                    "source_inventory": str(provenance / 'sha256.json')}
    atomic(root / 'phase.json', manifest)
    print(json.dumps({"phase": str(root), "inputs": len(candidates), "chunks": {a:len(chunks[a]) for a in ARMS},
                          "cysteine_inputs": sum('C' in r['sequence'] for r in candidates)}, indent=2))


def verify_frozen(manifest):
    # Image/checkpoint are large immutable installation assets, already hashed at freeze.
    # Source, inputs and controller are cheap enough to verify at every submission.
    for name, expected in load(manifest['source_inventory']).items():
        if str(Path(manifest['root'])) in name:
            require(sha(name) == expected, f'Frozen source/input changed: {name}')


def journal(root):
    path = root / 'control/submission.json'
    return load(path) if path.exists() else {"phase_id": root.name, "jobs": []}


def submit(root, smoke):
    manifest = load(root / 'phase.json')
    verify_frozen(manifest)
    record = journal(root)
    require(not any(j['smoke'] == smoke for j in record['jobs']), 'This execution has already been submitted')
    if not smoke:
        proof = load(root / 'reports/canary-validation.json')
        require(proof['complete'] and proof['phase_sha256'] == sha(root / 'phase.json'), 'Missing/stale canary proof')
        rows = scheduler([j for j in record['jobs'] if j['smoke']])
        require(len(rows) == 2 and all(r['state'] == 'COMPLETED' and r['exit_code'] == '0:0' for r in rows),
                f'Canaries not successfully terminal: {rows}')
    claim = root / 'control' / ('canary-submit.claim' if smoke else 'production-submit.claim')
    with claim.open('x') as stream:
        json.dump({"at": now(), "pid": os.getpid(), "manifest_sha256": sha(root / 'phase.json')}, stream)
    for arm in ARMS:
        role = ('canary-' if smoke else '') + arm
        runtime = manifest['runtime']
        lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
                 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1',
                 f"export MOSAIC_SIF={shlex.quote(runtime['image'])}",
                 f"export MOSAIC_WEIGHTS={shlex.quote(runtime['weights'])}",
                 f"export MOSAIC_DEV_SRC={shlex.quote(runtime['dev_source'])}",
                 f'export MOSAIC_SCRATCH={shlex.quote(str(root / "scratch" / role))}/"${{SLURM_ARRAY_TASK_ID:-0}}"',
                 'export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$MOSAIC_SCRATCH/jax_cache"',
                 'export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false',
                 'export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1',
                 'export SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=1',
                 'export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1',
                 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',
                 'exec ' + shlex.join([runtime['wrapper'], 'python', '-u', runtime['driver'],
                                       '--manifest', str(root / 'phase.json'), '--arm', arm,
                                       '--chunk-index']) + ' "${SLURM_ARRAY_TASK_ID:-0}"' + (' --smoke' if smoke else '')]
        script = root / 'control' / f'{role}.sbatch'
        script.write_text('\n'.join(lines) + '\n')
        args = ['sbatch', '--parsable', '--account=' + manifest['account'], '--partition=' + manifest['partition'],
                '--gpus=1', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=96G',
                '--time=' + manifest['resources']['canary_walltime' if smoke else 'walltime'],
                '--job-name=' + root.name + '-' + role, '--comment=' + root.name,
                '--output=' + str(root / 'logs' / (role + '-%A_%a.log'))]
        if not smoke:
            args.append(f'--array=0-{len(manifest["chunks"][arm])-1}%8')
        args.append(str(script))
        # Intent persists before submission; a transport failure is not an automatic retry.
        atomic(root / 'control' / f'{role}-submission-intent.json', {"at": now(), "command": args})
        job_id = command(args).strip().split(';')[0]
        require(job_id.isdigit(), f'Ambiguous sbatch response: {job_id!r}')
        record['jobs'].append({"job_id": job_id, "arm": arm, "smoke": smoke, "at": now(), "command": args})
        atomic(root / 'control/submission.json', record)
        print(f'SUBMITTED {role} {job_id}', flush=True)
    if not smoke:
        # This dependent collector survives a login-host/session failure. It starts
        # only after BOTH arrays release their GPUs, so the 16-GPU cap still holds.
        dependency = ':'.join(j['job_id'] for j in record['jobs'] if not j['smoke'])
        invocation = [sys.executable, str(root / 'provenance/mosaic_ea_campaign.py'),
                      '--phase', str(root), 'collect']
        args = ['sbatch', '--parsable', '--account=' + manifest['account'],
                '--partition=' + manifest['partition'], '--gpus=1', '--nodes=1',
                '--ntasks=1', '--cpus-per-task=2', '--mem=8G', '--time=00:30:00',
                '--dependency=afterany:' + dependency, '--job-name=' + root.name + '-collect',
                '--comment=' + root.name, '--output=' + str(root / 'logs/collect-%j.log'),
                '--wrap=' + shlex.join(invocation)]
        atomic(root / 'control/collector-intent.json', {'at': now(), 'command': args})
        job_id = command(args).strip().split(';')[0]
        require(job_id.isdigit(), f'Ambiguous collector sbatch response: {job_id!r}')
        atomic(root / 'control/collector.json', {'at': now(), 'job_id': job_id, 'command': args})
        print(f'SUBMITTED collector {job_id}', flush=True)


def scheduler(jobs):
    if not jobs:
        return []
    raw = command(['sacct', '-n', '-P', '-X', '--array', '-j', ','.join(j['job_id'] for j in jobs),
                   '--format=JobID%80,State%40,ExitCode,ElapsedRaw,AllocTRES%180,Start,End'])
    rows = []
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) >= 7 and fields[0]:
            rows.append({"job_id": fields[0].strip(), "state": fields[1].split()[0].rstrip('+'),
                             "exit_code": fields[2], "elapsed_seconds": int(fields[3] or 0),
                             "alloc_tres": fields[4], "start": fields[5], "end": fields[6]})
    return rows


@cache
def result_validator(root):
    path = root / 'provenance/driver/campaign_ea.py'
    spec = importlib.util.spec_from_file_location('frozen_ea_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validated_complete


def validate_result(path, original, settings):
    result = load(path)
    require(result.get('complete') is True, f'Incomplete root {path}')
    root = path.parents[3]
    require(result_validator(root)(path.parent, original, settings, sha(root / 'phase.json')),
            f'Invalid artifact integrity or lineage: {path}')
    require(result['design_id'] == original['design_id'] and result['arm'] == original['arm'], f'Wrong root identity: {path}')
    require(result['original_sequence'] == original['sequence'], f'Changed input sequence: {path}')
    require(all(result['settings'][k] == v for k, v in settings.items()), f'Wrong scientific parameters: {path}')
    members = result['final_members']
    require(len(members) == settings['population'], f'Missing final members: {path}')
    require(len({m['candidate_id'] for m in members}) == len(members), f'Duplicate candidate IDs: {path}')
    for member in members:
        seq = member['sequence']
        require(len(seq) == len(original['sequence']) and set(seq) <= set('ACDEFGHIKLMNPQRSTVWY'), f'Invalid sequence: {path}')
        require([i for i,c in enumerate(seq) if c == 'C'] == [i for i,c in enumerate(original['sequence']) if c == 'C'],
                f'Cysteine constraint violated: {path}')
        require(all(math.isfinite(member[key]) for key in ('ranking_loss', 'optimization_loss')), f'Nonfinite scores: {path}')
        require(member['source_node_id'], f'Missing checkpoint ancestry: {path}')
    winner = result['selected_winner']
    require(winner in members and winner['ranking_loss'] == min(m['ranking_loss'] for m in members), f'Invalid winner: {path}')
    cycles = [json.loads(line) for line in (path.parent / 'cycles.jsonl').read_text().splitlines()]
    require(len(cycles) >= settings['cycles'] + 1, f'Missing baseline/cycle history: {path}')
    require(len(list(path.parent.glob('*.npz'))) >= settings['cycles'] + 1, f'Missing numeric checkpoints: {path}')
    return result


def validate_canaries(root):
    manifest = load(root / 'phase.json')
    verify_frozen(manifest)
    originals = {r['design_id']:r for r in load(manifest['roots'])}
    settings = {**manifest['settings'], **manifest['canary']['settings']}
    results = []
    for arm in ARMS:
        for design_id in manifest['canary']['root_ids'][arm]:
            path = root / 'canary' / arm / design_id / 'result.json'
            value = validate_result(path, originals[design_id], settings)
            results.append({"design_id": design_id, "arm": arm, "cysteine": 'C' in originals[design_id]['sequence'],
                                "result": str(path), "sha256": sha(path), "offspring": len(value['final_members'])})
    proof = {"phase_id": root.name, "complete": True, "at": now(), "phase_sha256": sha(root / 'phase.json'),
                 "inputs": len(results), "offspring": sum(r['offspring'] for r in results), "results": results}
    atomic(root / 'reports/canary-validation.json', proof)
    print(json.dumps(proof, indent=2))


def collect(root, scheduler_rows):
    manifest = load(root / 'phase.json')
    originals = load(manifest['roots'])
    results, winners, offspring = [], [], []
    for original in originals:
        path = root / 'results' / original['arm'] / original['design_id'] / 'result.json'
        result = validate_result(path, original, manifest['settings'])
        results.append({"design_id": original['design_id'], "arm": original['arm'], "result": str(path), "sha256": sha(path)})
        common = {"root_design_id": original['design_id'], "arm": original['arm'], "generator": original['tool'],
                      "original_sequence": original['sequence'], "native_decile": original.get('sampling_decile'),
                      "backbone_lineage_id": original.get('backbone_lineage_id'),
                      "trajectory_lineage_id": original.get('trajectory_lineage_id')}
        for member in result['final_members']:
            offspring.append({**common, **member})
        winners.append({**common, **result['selected_winner']})
    atomic(root / 'reports/offspring.json', offspring)
    atomic(root / 'reports/winners.json', winners)
    columns = ('root_design_id', 'arm', 'generator', 'native_decile', 'original_sequence', 'candidate_id',
               'sequence', 'optimization_loss', 'ranking_loss', 'source_node_id', 'backbone_lineage_id', 'trajectory_lineage_id')
    for name, records in (('winners', winners), ('offspring', offspring)):
        with (root / 'reports' / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        with (root / 'reports' / f'{name}.fasta').open('w') as stream:
            for record in records:
                stream.write(f'>{record["candidate_id"]} root={record["root_design_id"]} arm={record["arm"]} generator={record["generator"]}\n{record["sequence"]}\n')
    report = {"phase_id": root.name, "complete": True, "finished_at": now(), "input_roots": len(results),
                  "offspring": len(offspring), "selected_winners": len(winners), "results": results,
                  "gpu_hours": sum(r['elapsed_seconds'] for r in scheduler_rows) / 3600,
                  "scheduler": scheduler_rows, "settings": manifest['settings'],
                  "interpretation": 'Internal Boltz2 ranking; no held-out score or experimental binding claim. Baseline and all cycle/member lineage retained.'}
    atomic(root / 'reports/final-report.json', report)
    return report


def monitor(root, once=False):
    print(f'MONITOR_READY phase={root.name} gpu=0 database_access=none', flush=True)
    while True:
        manifest = load(root / 'phase.json')
        jobs = [j for j in journal(root)['jobs'] if not j['smoke']]
        require(len(jobs) == 2, 'Both production arrays must be journaled before monitoring')
        rows = scheduler(jobs)
        failures = [r for r in rows if (r['state'] in TERMINAL and
                    (r['state'] != 'COMPLETED' or r['exit_code'] != '0:0'))]
        active = sum(r['state'] in {'RUNNING', 'COMPLETING'} for r in rows)
        issues = [f'Failed allocation {r["job_id"]}: {r["state"]} {r["exit_code"]}' for r in failures]
        if active > 16:
            issues.append(f'Concurrent GPU cap exceeded: {active}')
        for row in rows:
            if row['state'] != 'RUNNING' or row['elapsed_seconds'] <= 3600:
                continue
            parent_id = row['job_id'].split('_')[0]
            owner = next((job for job in jobs if job['job_id'] == parent_id), None)
            if owner is None:
                issues.append(f'Unrecognized running array task: {row["job_id"]}')
                continue
            log = root / 'logs' / f'{owner["arm"]}-{row["job_id"]}.log'
            if not log.exists() or time.time() - log.stat().st_mtime > 3600:
                issues.append(f'No native log progress for one hour: {row["job_id"]}')
        completed = {arm:len(list((root / 'results' / arm).glob('*/result.json'))) for arm in ARMS}
        expected_tasks = sum(len(manifest['chunks'][a]) for a in ARMS)
        terminal = len(rows) == expected_tasks and all(r['state'] in TERMINAL for r in rows)
        if terminal and sum(completed.values()) != 2100:
            issues.append(f'Terminal arrays but incomplete root coverage: {completed}')
        snapshot = {"phase_id": root.name, "at": now(), "status": 'blocked' if issues else 'running', "complete": False,
                        "issues": issues, "completed_roots": completed, "expected_roots": 2100, "active_gpus": active,
                        "observed_gpu_hours": sum(r['elapsed_seconds'] for r in rows) / 3600,
                        "scheduler": rows}
        atomic(root / 'reports/monitor-status.json', snapshot)
        if issues:
            # Stop further spending on a scientifically/operationally broken array.
            args = ['scancel'] + [j['job_id'] for j in jobs]
            result = subprocess.run(args, capture_output=True, text=True, timeout=90, check=False)
            atomic(root / 'reports/failure-report.json', {**snapshot, 'cancellation':{"command": args,
                   "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}})
            print('PHASE_BLOCKED ' + json.dumps(issues), flush=True)
            return 2
        if terminal:
            final_path = root / 'reports/final-report.json'
            if final_path.exists():
                report = load(final_path)
                require(report.get('complete') is True, 'Collector published an incomplete report')
                atomic(root / 'reports/monitor-status.json', {**snapshot, 'status':'complete', 'complete':True,
                                                             'final_report':str(final_path)})
                print(f'PHASE_COMPLETE roots={report["input_roots"]} offspring={report["offspring"]} report={final_path}', flush=True)
                return 0
            collector_path = root / 'control/collector.json'
            require(collector_path.exists(), 'No dependent collector is registered')
            collector_rows = scheduler([load(collector_path)])
            require(not any(r['state'] in TERMINAL for r in collector_rows),
                    f'Collector terminated without a complete final report: {collector_rows}')
        if once:
            print(json.dumps({k:v for k,v in snapshot.items() if k != 'scheduler'}, indent=2))
            return 0
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--repo', type=Path, required=True)
    sub.add_parser('submit-canaries')
    sub.add_parser('validate-canaries')
    sub.add_parser('submit')
    sub.add_parser('collect')
    mon = sub.add_parser('monitor')
    mon.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.phase.resolve()
    if args.command == 'prepare':
        prepare(root, args.source.resolve(), args.repo.resolve())
    elif args.command in ('submit', 'submit-canaries'):
        submit(root, args.command == 'submit-canaries')
    elif args.command == 'validate-canaries':
        validate_canaries(root)
    elif args.command == 'collect':
        manifest = load(root / 'phase.json')
        rows = scheduler([j for j in journal(root)['jobs'] if not j['smoke']])
        require(len(rows) == sum(len(manifest['chunks'][arm]) for arm in ARMS)
                and all(r['state'] == 'COMPLETED' and r['exit_code'] == '0:0' for r in rows),
                'Cannot collect unsuccessful or incompletely accounted arrays')
        report = collect(root, rows)
        print(f'COLLECTION_COMPLETE roots={report["input_roots"]} offspring={report["offspring"]}', flush=True)
    elif args.command == 'monitor':
        return monitor(root, args.once)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
