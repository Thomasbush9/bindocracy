#!/usr/bin/env python3
"""Freeze, qualify, launch and incrementally collect locally gated optimization."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mosaic_ea_campaign import TERMINAL, atomic, command, load, now, require, scheduler, sha

ARMS = ('pocket', 'y84')


def prepare(root, previous, calibration, repo):
    require(not root.exists(), f'Refusing to overwrite {root}')
    old = load(previous / 'phase.json')
    retained = load(previous / 'control/comparison-cohort.json')
    excluded = set(retained['design_ids'])
    originals = load(old['roots'])
    members = [r for r in originals if r['design_id'] not in excluded]
    require(len(members) + len(excluded) == len(originals), 'Incorrect retained subset')
    require(len({r['design_id'] for r in members}) == len(members), 'Duplicate input identities')
    gate = load(calibration)
    require(gate['metric'] == 'relative_gap_to_survivor_median', 'Wrong gate metric')
    for arm in ARMS:
        require(all(str(step) in gate['by_arm'][arm] for step in range(25, 360, 25)), 'Incomplete thresholds')
        rows = [r for r in members if r['arm'] == arm]
        require(len({r['sequence'] for r in rows}) == len(rows), 'Duplicate starting sequences')
    for name in ('provenance/driver', 'inputs', 'control', 'logs', 'reports'):
        (root / name).mkdir(parents=True, exist_ok=True)
    provenance = root / 'provenance'
    for name in ('campaign_gated.py', 'campaign_ea.py', 'cycle_binders.py', 'evolution.py'):
        shutil.copy2(repo / 'drivers/mosaic' / name, provenance / 'driver' / name)
    for name in ('mosaic_gated_campaign.py', 'mosaic_ea_campaign.py',
                 'mosaic_gate_calibration.py', 'mosaic_pipeline_traces.py'):
        shutil.copy2(repo / 'scripts' / name, provenance / name)
    shutil.copytree(Path(old['runtime']['dev_source']), provenance / 'mosaic-src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(old['runtime']['wrapper'], provenance / 'mosaic-exec.sh')
    require(sha(gate['report']['path']) == gate['report']['sha256'], 'Calibration report changed')
    report_path = root / 'inputs/calibration-report.json'
    shutil.copy2(gate['report']['path'], report_path)
    gate['report']['path'] = str(report_path)
    atomic(root / 'inputs/calibration.json', gate)
    shutil.copy2(previous / 'control/comparison-cohort.json', root / 'inputs/excluded-comparison.json')
    shutil.copy2(previous / 'phase.json', root / 'inputs/previous-phase.json')
    shutil.copy2(old['source_inventory'], root / 'inputs/previous-source-inventory.json')
    for name in ('baseline-candidates.json', 'baseline-predictions.json'):
        shutil.copy2(previous / 'inputs' / name, root / 'inputs' / name)
    atomic(root / 'inputs/roots.json', members)
    targets = {}
    for arm in ARMS:
        folder = root / 'inputs' / arm
        folder.mkdir()
        targets[arm] = {'epitope_idx': old['targets'][arm]['epitope_idx']}
        for field, name in (('fasta', 'target.fasta'), ('msa', 'target.a3m')):
            shutil.copy2(old['targets'][arm][field], folder / name)
            targets[arm][field] = str(folder / name)
    rng = random.Random(2026101425)
    batches, grouped = [], {}
    for arm in ARMS:
        strata = {}
        for row in members:
            if row['arm'] == arm:
                strata.setdefault((row['tool'], row.get('sampling_decile')), []).append(row)
        keys = sorted(strata)
        rng.shuffle(keys)
        for rows in strata.values():
            rng.shuffle(rows)
        ordered = [strata[key][i] for i in range(max(map(len, strata.values())))
                   for key in keys if i < len(strata[key])]
        grouped[arm] = []
        for start in range(0, len(ordered), 4):
            grouped[arm].append(len(batches))
            batches.append({'arm': arm, 'batch_id': f'{arm}-{start // 4:04d}',
                                'root_ids': [r['design_id'] for r in ordered[start:start + 4]]})
    by_id = {r['design_id']: r for r in members}
    canaries = {}
    for arm, indices in grouped.items():
        full = next(i for i in indices if len(batches[i]['root_ids']) == 4 and
                    len({'C' in by_id[r]['sequence'] for r in batches[i]['root_ids']}) == 2)
        partial = [i for i in indices if len(batches[i]['root_ids']) < 4]
        canaries[arm] = [full, *partial]
    workers = []
    for start in range(0, max(map(len, grouped.values())), 4):
        for arm in ARMS:
            indices = grouped[arm][start:start + 4]
            if indices:
                workers.append({'arm': arm, 'batch_indices': indices})
    settings = {k: v for k, v in old['settings'].items() if k != 'cull_fraction'}
    settings.update(cull_interval=25)
    runtime = {**old['runtime'], 'driver': str(provenance / 'driver/campaign_gated.py'),
               'dev_source': str(provenance / 'mosaic-src'), 'wrapper': str(provenance / 'mosaic-exec.sh')}
    inventory = {str(p): sha(p) for directory in (provenance, root / 'inputs')
                 for p in directory.rglob('*') if p.is_file()}
    atomic(provenance / 'sha256.json', inventory)
    old_jobs = [j['job_id'] for j in load(previous / 'control/submission.json')['jobs'] if not j['smoke']]
    manifest = {'phase_id': root.name, 'root': str(root), 'roots': str(root / 'inputs/roots.json'),
                    'previous_phase': str(previous), 'previous_array_ids': old_jobs,
                    'excluded_inputs': len(excluded), 'targets': targets, 'batches': batches, 'workers': workers,
                    'settings': settings, 'calibration': str(root / 'inputs/calibration.json'),
                    'canary': {'batch_indices': canaries, 'settings': {'cycles': 3, 'soft_steps': 25, 'sharp_steps': 1}},
                    'runtime': runtime, 'account': old['account'], 'partition': old['partition'],
                    'resources': {'max_concurrent_gpus': 16, 'gpus_per_job': 1, 'cpus_per_job': 8,
                                   'memory_gb': 96, 'walltime': '04:00:00', 'canary_walltime': '01:00:00'},
                    'created_at': now(), 'source_inventory': str(provenance / 'sha256.json'),
                    'protocol': 'Four distinct inputs; worst candidate every25steps; historical relative-gap gate; uniform surviving parent plus0.1–0.2 noise; partial groups not culled.',
                    'calibration_caveat': 'Historical dual-backend objective differs; thresholds are a relative-gap proxy targeting12.5%historically, not a guarantee for this population.',
                    'approval': 'User authorized remaining designs now, maximum16GPUs including retained comparison; incremental afternoon analysis, then continue.'}
    atomic(root / 'phase.json', manifest)
    print(json.dumps({'phase': str(root), 'remaining': len(members), 'batches': len(batches), 'workers': len(workers)}, indent=2))


def verify(root):
    manifest = load(root / 'phase.json')
    for path, expected in load(manifest['source_inventory']).items():
        require(sha(path) == expected, f'Frozen artifact changed: {path}')
    return manifest


def journal(root):
    path = root / 'control/submission.json'
    return load(path) if path.exists() else {'jobs': []}


def available(manifest):
    text = command(['squeue', '-h', '-r', '-j', ','.join(manifest['previous_array_ids']), '-o', '%i|%T|%r'])
    running = 0
    for line in text.splitlines():
        _, state, reason = line.split('|', 2)
        if state in ('RUNNING', 'COMPLETING', 'CONFIGURING'):
            running += 1
        elif state == 'PENDING':
            require(reason == 'JobHeldUser', f'Old queued task no longer held: {line}')
    require(running < 16, f'No GPU capacity under combined cap: {running}')
    return 16 - running, running


def launch(root, smoke):
    manifest = verify(root)
    record = journal(root)
    require(not any(j['smoke'] == smoke for j in record['jobs']), 'Already submitted this phase')
    capacity, old_active = available(manifest)
    if smoke:
        require(capacity >= 2, 'Need two free GPU slots for target canaries')
    else:
        proof = load(root / 'reports/canary-validation.json')
        require(proof['complete'] and proof['manifest_sha256'] == sha(root / 'phase.json'), 'Missing/stale canary proof')
        canary_rows = scheduler([j for j in record['jobs'] if j['smoke']])
        require(len(canary_rows) == 2 and all(r['state'] == 'COMPLETED' and r['exit_code'] == '0:0'
                                           for r in canary_rows), 'Canaries must release GPUs before production')
    claim = root / 'control' / ('canary-submit.claim' if smoke else 'production-submit.claim')
    with claim.open('x') as stream:
        json.dump({'at': now(), 'old_active_gpus': old_active, 'capacity': capacity}, stream)
    for role in (ARMS if smoke else ('production',)):
        rt = manifest['runtime']
        invoke = [sys.executable, str(root / 'provenance/mosaic_gated_campaign.py'), '--phase', str(root), 'worker']
        if smoke:
            invoke += ['--arm', role, '--smoke']
        lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
                 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1',
                 f'export MOSAIC_SIF={shlex.quote(rt["image"])}',
                 f'export MOSAIC_WEIGHTS={shlex.quote(rt["weights"])}',
                 f'export MOSAIC_DEV_SRC={shlex.quote(rt["dev_source"])}',
                 f'export MOSAIC_SCRATCH={shlex.quote(str(root / "scratch" / role))}/"${{SLURM_ARRAY_TASK_ID:-0}}"',
                 'export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$MOSAIC_SCRATCH/jax_cache"',
                 'export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false',
                 'export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_TRANSFORMERS_OFFLINE=1',
                 'export SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=1',
                 'export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1',
                 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',
                 'exec ' + shlex.join(invoke)]
        script = root / 'control' / f'{role}.sbatch'
        script.write_text('\n'.join(lines) + '\n')
        args = ['sbatch', '--parsable', '--account=' + manifest['account'], '--partition=' + manifest['partition'],
                '--gpus=1', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=96G',
                '--time=' + manifest['resources']['canary_walltime' if smoke else 'walltime'],
                '--job-name=' + root.name + '-' + role, '--comment=' + root.name,
                '--output=' + str(root / 'logs' / (role + '-%A_%a.log'))]
        if not smoke:
            args += [f'--array=0-{len(manifest["workers"])-1}%{capacity}']
        args.append(str(script))
        atomic(root / 'control' / f'{role}-submission-intent.json', {'at': now(), 'command': args})
        job_id = command(args).strip().split(';')[0]
        require(job_id.isdigit(), f'Ambiguous submission response: {job_id!r}')
        record['jobs'].append({'job_id': job_id, 'smoke': smoke, 'role': role, 'at': now(), 'command': args})
        atomic(root / 'control/submission.json', record)
        print(f'SUBMITTED {role} {job_id} cap={capacity}', flush=True)


def module(root):
    directory = root / 'provenance/driver'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('frozen_gated_worker', directory / 'campaign_gated.py')
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def qualify(root):
    manifest = verify(root)
    native = module(root)
    roots = {r['design_id']: r for r in load(manifest['roots'])}
    settings = native.effective_settings(manifest, True)
    proof = []
    for arm, indices in manifest['canary']['batch_indices'].items():
        for index in indices:
            batch = manifest['batches'][index]
            directory = root / 'smoke/results' / arm / batch['batch_id']
            inputs = [roots[identity] for identity in batch['root_ids']]
            require(native.validated_complete(directory, inputs, settings, sha(root / 'phase.json')), f'Invalid canary {directory}')
            proof.append({'batch_id': batch['batch_id'], 'inputs': len(inputs), 'result_sha256': sha(directory / 'result.json')})
    atomic(root / 'reports/canary-validation.json', {'complete': True, 'at': now(),
           'manifest_sha256': sha(root / 'phase.json'), 'results': proof})
    print(json.dumps(proof, indent=2))


def collect(root, manifest, native, cache):
    roots = {r['design_id']: r for r in load(manifest['roots'])}
    fingerprint = sha(root / 'phase.json')
    previous_count = len(cache)
    for batch in manifest['batches']:
        key = batch['batch_id']
        directory = root / 'results' / batch['arm'] / key
        if key not in cache and (directory / 'result.json').exists():
            require(native.validated_complete(directory, [roots[r] for r in batch['root_ids']],
                    manifest['settings'], fingerprint), f'Invalid production result: {directory}')
            cache[key] = load(directory / 'result.json')
    if len(cache) == previous_count and (root / 'reports/candidates.json').exists():
        return sum(len(result['final_members']) for result in cache.values()), len(cache)
    rows = []
    for result in cache.values():
        for member in result['final_members']:
            ancestor = roots[member['lineage_root_id']]
            original = roots[member['slot_original_id']]
            rows.append({**member, 'arm': ancestor['arm'], 'generator': ancestor['tool'],
                         'root_design_id': ancestor['design_id'], 'original_sequence': ancestor['sequence'],
                         'slot_original_sequence': original['sequence'], 'slot_original_generator': original['tool'],
                         'native_decile': ancestor.get('sampling_decile'),
                         'backbone_lineage_id': ancestor.get('backbone_lineage_id'),
                         'trajectory_lineage_id': ancestor.get('trajectory_lineage_id')})
    atomic(root / 'reports/candidates.json', rows)
    columns = ('candidate_id', 'arm', 'generator', 'root_design_id', 'slot_original_id',
               'slot_original_generator', 'sequence', 'optimization_loss', 'ranking_loss', 'native_decile')
    target = root / 'reports/candidates.csv'
    temp = target.with_suffix('.csv.tmp')
    with temp.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, target)
    return len(rows), len(cache)


def gate_summary(root, manifest):
    summary = {arm: {'checks': 0, 'eligible_checks': 0, 'replacements': 0,
                     'member_opportunities': 0} for arm in ARMS}
    for batch in manifest['batches']:
        path = root / 'results' / batch['arm'] / batch['batch_id'] / 'gates.jsonl'
        if not path.exists():
            continue
        counts = summary[batch['arm']]
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue  # A live writer may not yet have finished its last record.
            event = json.loads(line)
            counts['checks'] += 1
            counts['eligible_checks'] += event['active_count'] == 4
            counts['member_opportunities'] += event['active_count']
            counts['replacements'] += event['replacement_count']
    for counts in summary.values():
        counts['replacement_fraction'] = (counts['replacements'] / counts['member_opportunities']
                                          if counts['member_opportunities'] else None)
    return summary


def monitor(root, once=False):
    manifest = verify(root)
    native, cache = module(root), {}
    production = [j for j in journal(root)['jobs'] if not j['smoke']]
    require(len(production) == 1, 'Expected one shared-cap production array')
    job_id = production[0]['job_id']
    last_cap = None
    print('GATED_MONITOR_READY gpu=0', flush=True)
    while True:
        try:
            capacity, old_active = available(manifest)
            rows = scheduler(production)
            if capacity != last_cap and any(r['state'] not in TERMINAL for r in rows):
                command(['scontrol', 'update', 'JobId=' + job_id, 'ArrayTaskThrottle=' + str(capacity)])
                last_cap = capacity
            failures = [r for r in rows if r['state'] in TERMINAL and
                        (r['state'] != 'COMPLETED' or r['exit_code'] != '0:0')]
            require(not failures, f'Failed production allocations: {failures}')
            active = sum(r['state'] in ('RUNNING', 'COMPLETING', 'CONFIGURING') for r in rows)
            require(active + old_active <= 16, 'Combined GPU cap exceeded')
            for row in rows:
                if row['state'] == 'RUNNING' and row['elapsed_seconds'] > 3600:
                    log = root / 'logs' / f'production-{row["job_id"]}.log'
                    require(log.exists() and time.time() - log.stat().st_mtime < 3600, f'No log progress: {row["job_id"]}')
            completed, batches = collect(root, manifest, native, cache)
            terminal = len(rows) == len(manifest['workers']) and all(r['state'] in TERMINAL for r in rows)
            expected = len(load(manifest['roots']))
            require(not terminal or completed == expected, f'Terminal array incomplete: {completed}/{expected}')
            snapshot = {'phase_id': manifest['phase_id'], 'at': now(), 'complete': terminal,
                            'completed_input_slots': completed, 'expected_inputs': expected, 'completed_batches': batches,
                            'active_new_gpus': active, 'active_retained_gpus': old_active, 'shared_cap': 16,
                            'array_throttle': capacity, 'observed_new_gpu_hours': sum(r['elapsed_seconds'] for r in rows) / 3600,
                            'gate_counts': gate_summary(root, manifest), 'scheduler': rows}
            atomic(root / 'reports/progress.json', snapshot)
            print(json.dumps({k: v for k, v in snapshot.items() if k != 'scheduler'}), flush=True)
            if terminal:
                atomic(root / 'reports/final-report.json', snapshot)
                return 0
            if once:
                return 0
            time.sleep(60)
        except Exception as error:
            cancellation = subprocess.run(['scancel', job_id], capture_output=True, text=True,
                                          timeout=90, check=False)
            atomic(root / 'reports/failure-report.json', {
                'at': now(), 'error': repr(error), 'cancelled_array': job_id,
                'cancellation_returncode': cancellation.returncode, 'cancellation_stderr': cancellation.stderr})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    create = sub.add_parser('prepare')
    create.add_argument('--previous', type=Path, required=True)
    create.add_argument('--calibration', type=Path, required=True)
    create.add_argument('--repo', type=Path, default=Path.cwd())
    sub.add_parser('submit-canaries')
    sub.add_parser('qualify')
    sub.add_parser('submit')
    watch = sub.add_parser('monitor')
    watch.add_argument('--once', action='store_true')
    worker = sub.add_parser('worker')
    worker.add_argument('--arm', choices=ARMS)
    worker.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.phase.resolve()
    if args.action == 'prepare':
        prepare(root, args.previous.resolve(), args.calibration.resolve(), args.repo.resolve())
    elif args.action in ('submit-canaries', 'submit'):
        launch(root, args.action == 'submit-canaries')
    elif args.action == 'qualify':
        qualify(root)
    elif args.action == 'monitor':
        return monitor(root, args.once)
    elif args.action == 'worker':
        manifest = load(root / 'phase.json')
        indices = (manifest['canary']['batch_indices'][args.arm] if args.smoke else
                   manifest['workers'][int(os.environ['SLURM_ARRAY_TASK_ID'])]['batch_indices'])
        rt = manifest['runtime']
        invocation = [rt['wrapper'], 'python', '-u', rt['driver'], '--manifest', str(root / 'phase.json'),
                      '--batch-indices', ','.join(map(str, indices))] + (['--smoke'] if args.smoke else [])
        os.execv(rt['wrapper'], invocation)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
