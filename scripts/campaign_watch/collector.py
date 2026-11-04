#!/usr/bin/env python3
"""Read-only Mosaic campaign monitor prototype; run only in a Slurm CPU job.

Only manifest-declared designs_seed*.json artifacts are inspected. The frozen
campaign's pure validator supplies the existing lineage and sequence contract.
No collection/export main, optimizer, model, or campaign-write code is called.

GPU telemetry is one bounded Slurm accounting query over at most eight spread-out
running tasks per refresh, not instantaneous device polling. Coverage, age and
partial failures travel with the sample; missing readings never become zero.
The eight-task cap keeps sampling within the 15-second command deadline on the
observed cluster; a 32-task canary exceeded that bound. Completed output rows
survive telemetry timeouts without relaxing strict scheduler/count queries.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
SNAPSHOT_LIMIT = 1024 * 1024
MANIFEST_LIMIT = 16 * 1024 * 1024
ARTIFACT_LIMIT = 1024 * 1024
COMMAND_LIMIT = 4 * 1024 * 1024
MAX_CAMPAIGNS = 8
MAX_TASKS = 20000
MAX_GPU_SAMPLES = 8
TERMINAL = {
    'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
    'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT',
}
ALLOCATED = {'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'RESIZING',
             'SIGNALING', 'STAGE_OUT', 'STOPPED'}
PENDING = {'PENDING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def short_error(exc):
    return f'{type(exc).__name__}: {exc}'[:300]


def fingerprint(path, limit):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError(f'not a bounded regular file: {path}')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def reject_constant(value):
    raise ValueError(f'non-finite JSON value: {value}')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load_json(path, limit=MANIFEST_LIMIT):
    before = fingerprint(path, limit)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or fingerprint(path, limit) != before:
        raise ValueError(f'file changed during read or exceeds limit: {path}')
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)


def bounded_command(args, *, allow_partial=False):
    """Bound both pipes and runtime; optionally return stdout plus diagnostics."""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ, 'LC_ALL': 'C', 'TZ': 'UTC'})
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    deadline = time.monotonic() + 15
    total = 0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'{args[0]} exceeded 15 seconds')
                for key, _ in selector.select(min(remaining, 0.5)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > COMMAND_LIMIT:
                        raise ValueError(f'{args[0]} exceeded scheduler output limit')
                    buffers[key.data].extend(chunk)
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        error = None
        detail = buffers['stderr'].decode('utf-8', errors='replace')[:300]
        if process.returncode:
            error = f'{args[0]} exit {process.returncode}: {detail}'
        elif buffers['stderr'].strip():
            error = f'{args[0]} warning: {detail}'
        if error and not allow_partial:
            raise RuntimeError(error)
        output = buffers['stdout'].decode('utf-8')
        return (output, error) if allow_partial else output
    except (TimeoutError, subprocess.TimeoutExpired, ValueError) as exc:
        if not allow_partial:
            raise
        # A killed command can leave its last row truncated: never parse that row.
        complete = buffers['stdout'].rpartition(b'\n')[0]
        output = (complete + b'\n').decode('utf-8') if complete else ''
        return output, f'{type(exc).__name__}: {exc}'[:300]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def state_name(value):
    return value.strip().split(' ', 1)[0].rstrip('+')


def elapsed_seconds(value):
    days, clock = (value.split('-', 1) if '-' in value else ('0', value))
    fields = [int(part) for part in clock.split(':')]
    if len(fields) == 2:
        hours, minutes, seconds = 0, *fields
    elif len(fields) == 3:
        hours, minutes, seconds = fields
    else:
        raise ValueError(f'unrecognized Slurm elapsed time: {value}')
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def scheduler_snapshot(job_ids):
    """Two bounded, batched queries; squeue overrides accounting's current state."""
    if not job_ids:
        return {}
    joined = ','.join(sorted(set(job_ids)))
    accounting = bounded_command([
        'sacct', '-X', '-n', '-P', '-j', joined,
        '--format=JobID%64,State%32,ExitCode,ElapsedRaw,Start,End',
    ])
    # Finished array IDs disappear from squeue; querying them with -j then fails.
    # Query this user's live queue once and retain only registered array IDs.
    queue = bounded_command(['squeue', '-h', '-r', '--me', '-o', '%i|%T|%M|%N|%A'])
    records = {}
    for line in accounting.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 6:
            raise ValueError('unexpected sacct record shape')
        job, state, exit_code, elapsed, start, end = fields
        match = re.fullmatch(r'(\d+)_(\d+)', job)
        if not match:
            # Parent array allocation, compressed pending range, and job steps
            # are not unique task records. squeue expands all pending elements.
            continue
        if match[1] not in job_ids:
            continue
        record = {'state': state_name(state), 'exit': exit_code,
                  'elapsed': int(elapsed), 'start': start, 'end': end}
        records[(match[1], int(match[2]))] = record
    for line in queue.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 5:
            raise ValueError('unexpected squeue record shape')
        job, state, elapsed, _node, raw_job_id = fields
        match = re.fullmatch(r'(\d+)_(\d+)', job)
        if not match or match[1] not in job_ids:
            continue
        records[(match[1], int(match[2]))] = {
            'state': state_name(state), 'exit': None,
            'elapsed': elapsed_seconds(elapsed), 'current': True, 'raw_job_id': raw_job_id,
        }
    # Accounting RUNNING/PENDING rows can lag squeue. Never treat those rows as
    # current allocations when no matching element appeared in the live queue.
    for record in records.values():
        if not record.get('current') and record['state'] not in TERMINAL:
            record['state'] = 'UNKNOWN'
    return records


def empty_gpu_utilization(eligible_jobs=0, error=None):
    return dict(source='Slurm sstat TRESUsageInAve', observed_at=None,
                eligible_jobs=eligible_jobs, queried_jobs=0, sampled_jobs=0,
                mean_percent=None, min_percent=None, max_percent=None, low_util_jobs=0,
                memory_sampled_jobs=0, mean_memory_mib=None, error=error)


def gpu_reading(value, *, memory=False):
    """Decode Slurm quantities without conflating absent/invalid readings with zero."""
    if value is None:
        return None
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTPE]?)' if memory
                         else r'(\d+(?:\.\d+)?)', value.strip())
    if not match:
        return None
    number = float(match[1])
    if memory:
        # Slurm's binary byte suffixes: M is MiB; an unscaled quantity is bytes.
        number *= 1024.0 ** (' KMGTPE'.index(match[2] or ' ') - 2)
    if not math.isfinite(number) or number < 0 or (not memory and number > 100):
        return None
    return number


def gpu_snapshot(campaigns, records, scheduler_error):
    """One bounded batch of <=8 current tasks; accounting samples, not live GPU probes."""
    eligible = {}
    for campaign in campaigns:
        eligible[campaign.root] = {
            (campaign.job_id, task['task_id']) for task in campaign.tasks
            if not scheduler_error
            and records.get((campaign.job_id, task['task_id']), {}).get('current')
            and records[(campaign.job_id, task['task_id'])]['state'] == 'RUNNING'
        }
    result = {root: empty_gpu_utilization(len(keys), scheduler_error)
              for root, keys in eligible.items()}
    keys = sorted(set().union(*eligible.values()), key=lambda key: (int(key[0]), key[1]))
    queryable = [key for key in keys
                 if re.fullmatch(r'\d+', records[key].get('raw_job_id') or '')]
    count = min(MAX_GPU_SAMPLES, len(queryable))
    selected = (queryable if count <= 1 else
                [queryable[i * (len(queryable) - 1) // (count - 1)] for i in range(count)])
    requested = {records[key]['raw_job_id'] + '.batch' for key in selected}
    readings, seen = {}, set()
    query_error = None
    malformed = 0
    if requested:
        try:
            output, query_error = bounded_command([
                'sstat', '-j', ','.join(sorted(requested, key=lambda job: int(job.split('.')[0]))),
                '--noheader', '--parsable2', '--format=JobID%64,TRESUsageInAve%1024',
            ], allow_partial=True)
            for line in output.splitlines():
                if not line.strip():
                    continue
                fields = line.strip().split('|')
                if len(fields) != 2 or fields[0].strip() not in requested:
                    malformed += 1
                    continue
                job, tres = fields[0].strip(), fields[1]
                if job in seen:
                    readings.pop(job, None)
                    malformed += 1
                    continue
                seen.add(job)
                values, duplicates = {}, set()
                for entry in tres.split(','):
                    name, separator, value = entry.strip().partition('=')
                    if name not in ('gres/gpuutil', 'gres/gpumem'):
                        continue
                    if name in values:
                        duplicates.add(name)
                    values[name] = value if separator else None
                for name in duplicates:
                    values[name] = None
                readings[job] = (gpu_reading(values.get('gres/gpuutil')),
                                 gpu_reading(values.get('gres/gpumem'), memory=True))
        except Exception as exc:
            query_error = short_error(exc)
    observed_at = utc_now()
    selected = set(selected)
    queryable = set(queryable)
    for root, jobs in eligible.items():
        summary = result[root]
        queried = jobs & selected
        summary['queried_jobs'] = len(queried)
        errors = [scheduler_error] if scheduler_error else []
        if jobs - queryable:
            errors.append(f'{len(jobs - queryable)} running tasks lack numeric allocation IDs')
        if queried and query_error:
            errors.append(query_error)
        if queried and malformed:
            errors.append(f'{malformed} malformed or duplicate sstat rows')
        values = [readings.get(records[key]['raw_job_id'] + '.batch', (None, None))
                  for key in queried]
        utilization = [value for value, _ in values if value is not None]
        memory = [value for _, value in values if value is not None]
        if utilization:
            summary.update(sampled_jobs=len(utilization),
                           mean_percent=statistics.mean(utilization),
                           min_percent=min(utilization), max_percent=max(utilization),
                           low_util_jobs=sum(value <= 10 for value in utilization))
        if memory:
            summary.update(memory_sampled_jobs=len(memory), mean_memory_mib=statistics.mean(memory))
        if utilization or memory:
            summary['observed_at'] = observed_at
        if len(utilization) < len(queried) or len(memory) < len(queried):
            errors.append(f'Missing/invalid readings: utilization {len(queried) - len(utilization)}, '
                          f'memory {len(queried) - len(memory)} (tasks may have exited)')
        summary['error'] = '; '.join(errors)[:600] or None
    return result


def stage(identifier, name, unit, planned, state='unknown'):
    return dict(id=identifier, name=name, state=state, unit=unit, planned=planned,
                valid=0, running_jobs=0, pending_jobs=0, failed_jobs=0,
                scheduler_warning_jobs=0, active_gpus=0, eta_seconds=None,
                throughput_per_hour=None, warnings=[])


class Campaign:
    def __init__(self, config):
        self.root = Path(config['root']).resolve()
        self.name = config.get('name', self.root.name)
        self.future = config.get('future_stages', [])
        if not isinstance(self.name, str) or len(self.name) > 200:
            raise ValueError('campaign name must be a string of at most 200 characters')
        if (not isinstance(self.future, list) or len(self.future) != 2
                or {item['id'] for item in self.future} != {'esmfold', 'protenix'}):
            raise ValueError('supported future stages are exactly esmfold and protenix')
        for item in self.future:
            if (not isinstance(item['name'], str) or len(item['name']) > 200
                    or type(item['planned']) is not int or item['planned'] < 0):
                raise ValueError('invalid future-stage configuration')
        self.module = None
        self.manifest_stamp = None
        self.json_cache = {}
        self.output_cache = {}
        self.cfg = None
        self.tasks = None
        self.job_id = None
        self.last_good = None

    def cached_load(self, path):
        path = Path(path)
        # Output JSON is not retained: only its fingerprint and validation result
        # survive a sweep. Manifest and recipe objects are reused without reads.
        if path not in self.manifest_paths:
            return load_json(path, ARTIFACT_LIMIT)
        stamp = fingerprint(path, MANIFEST_LIMIT)
        cached = self.json_cache.get(path)
        if cached is None or cached[0] != stamp:
            value = load_json(path)
            self.json_cache[path] = (stamp, value)
        return self.json_cache[path][1]

    def prepare(self):
        initial_paths = [self.root / name for name in
                         ('campaign.json', 'tasks.json', 'parents.json', 'launch.json')]
        self.manifest_paths = set(initial_paths)
        cfg = self.cached_load(self.root / 'campaign.json')
        if (not isinstance(cfg, dict) or cfg.get('schema_version') != 1
                or cfg.get('batch') != 2 or cfg.get('children_per_parent') != 2
                or type(cfg.get('optimization_tasks')) is not int
                or not 0 < cfg['optimization_tasks'] <= MAX_TASKS
                or not isinstance(cfg.get('arms'), dict) or not 0 < len(cfg['arms']) <= 16):
            raise ValueError('unsupported campaign format; expected frozen Mosaic batch-two campaign')
        script = Path(cfg['runtime_root']).resolve().parent / 'scripts' / 'collect.py'
        recipes = [Path(arm['recipe']) for arm in cfg['arms'].values()]
        self.manifest_paths.update(recipes)
        paths = initial_paths + recipes + [script]
        stamp = tuple(fingerprint(path, MANIFEST_LIMIT) for path in paths)
        if stamp == self.manifest_stamp:
            return
        # The configured frozen campaign script is trusted code, audited for this
        # Mosaic format. Import does not execute its guarded collection main.
        spec = importlib.util.spec_from_file_location('_campaign_readonly_validator', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.load = self.cached_load
        validated_cfg, tasks = module.manifest(self.root)
        launch = self.cached_load(self.root / 'launch.json')
        if Path(launch['root']).resolve() != self.root:
            raise ValueError('launch root differs from campaign root')
        job_id = launch.get('jobs', {}).get('optimization')
        if not isinstance(job_id, str) or not re.fullmatch(r'\d+', job_id):
            raise ValueError('expected submitted Mosaic optimization array job ID')
        if any(item['planned'] != validated_cfg['retained_children'] for item in self.future):
            raise ValueError('future-stage plans differ from frozen retained-child count')
        if tuple(fingerprint(path, MANIFEST_LIMIT) for path in paths) != stamp:
            raise ValueError('campaign manifest changed during validation')
        self.module, self.cfg, self.tasks, self.job_id = module, validated_cfg, tasks, job_id
        self.manifest_stamp = stamp
        self.output_cache.clear()

    def validate_output(self, task):
        path = self.module.result_path(task)
        try:
            stamp = fingerprint(path, ARTIFACT_LIMIT)
        except FileNotFoundError:
            self.output_cache.pop(task['task_id'], None)
            return 0, None
        cached = self.output_cache.get(task['task_id'])
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
        try:
            count = len(self.module._validated_task(self.cfg, task))
            error = None
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            count, error = 0, short_error(exc)
        except FileNotFoundError:
            return 0, 'artifact disappeared during validation'
        if fingerprint(path, ARTIFACT_LIMIT) != stamp:
            return 0, 'artifact changed during validation'
        self.output_cache[task['task_id']] = (stamp, count, error)
        return count, error

    def empty(self):
        return dict(id=str(self.root), name=self.name, root=str(self.root),
                    observed_at=None, stages=[], warnings=[])

    def stale(self, error):
        result = copy.deepcopy(self.last_good) if self.last_good else self.empty()
        if not result['stages']:
            result['stages'] = [stage('selected_parents', 'Selected parents', 'parents', None),
                                stage('optimization', 'Optimization', 'children', None)]
            result['stages'].extend(self.future_stages())
        for item in result['stages']:
            if item['id'] == 'optimization':
                item.update(state='unknown', eta_seconds=None, throughput_per_hour=None)
                item['gpu_utilization'] = empty_gpu_utilization(error=f'Collection unavailable: {error}')
        result['warnings'].append(f'STALE / collection unavailable: {error}')
        return result

    def future_stages(self):
        return [stage(item['id'], item['name'], 'designs', item['planned'], 'not_submitted')
                for item in self.future]

    def collect(self, records, scheduler_error, gpu_utilization=None):
        result = self.empty()
        selected = stage('selected_parents', 'Selected parents', 'parents',
                         self.cfg['parent_count'], 'complete')
        selected['valid'] = self.cfg['parent_count']
        optimization = stage('optimization', 'Optimization', 'children',
                             self.cfg['retained_children'])
        optimization['job_id'] = self.job_id
        if gpu_utilization is not None:
            optimization['gpu_utilization'] = gpu_utilization
        samples = {arm: [] for arm in self.cfg['arms']}
        remaining = []
        unknown = 0
        completing = 0
        warning_ids, failed_ids, invalid_examples = [], [], []
        for task in self.tasks:
            task_id = task['task_id']
            count, error = self.validate_output(task)
            optimization['valid'] += count
            record = records.get((self.job_id, task_id), {}) if not scheduler_error else {}
            state = record.get('state', 'UNKNOWN')
            valid = count == len(task['children'])
            if state in ALLOCATED:
                optimization['running_jobs'] += 1
                optimization['active_gpus'] += 1
                completing += state == 'COMPLETING'
            elif state in PENDING:
                optimization['pending_jobs'] += 1
            if state in TERMINAL:
                abnormal = state != 'COMPLETED' or record.get('exit') != '0:0'
                if not valid:
                    optimization['failed_jobs'] += 1
                    failed_ids.append(task_id)
                elif abnormal:
                    optimization['scheduler_warning_jobs'] += 1
                    warning_ids.append(task_id)
                elif record.get('elapsed', 0) > 0:
                    samples[task['arm']].append(record['elapsed'])
            elif state not in ALLOCATED | PENDING and not valid:
                unknown += 1
            if not valid:
                remaining.append((task['arm'], record))
                if error and len(invalid_examples) < 3:
                    invalid_examples.append(f'task {task_id}: {error}')
        warnings = optimization['warnings']
        if completing:
            warnings.append(f'{completing} COMPLETING allocation(s), included once in running jobs and active GPUs.')
        if warning_ids:
            warnings.append(f'{len(warning_ids)} scheduler failures with independently valid outputs; '
                            f'children retained. Task IDs (first 20): {warning_ids[:20]}')
        if failed_ids:
            warnings.append(f'{len(failed_ids)} terminal jobs have missing or invalid outputs. '
                            f'Task IDs (first 20): {failed_ids[:20]}')
        warnings.extend(invalid_examples)
        if unknown and not scheduler_error:
            warnings.append(f'{unknown} unfinished tasks have unknown scheduler state (accounting may lag).')
        medians = {arm: statistics.median(values) for arm, values in samples.items()
                   if len(values) >= 5}
        optimization['runtime_minutes_by_arm'] = {
            arm: round(value / 60, 2) for arm, value in medians.items()}
        if scheduler_error:
            optimization['state'] = 'unknown'
            result['observed_at'] = self.last_good['observed_at'] if self.last_good else None
            result['warnings'].append(f'STALE / scheduler unavailable: {scheduler_error}; '
                                      'job and GPU counts unknown, not measured zero.')
        else:
            result['observed_at'] = utc_now()
            if optimization['failed_jobs']:
                optimization['state'] = 'needs_attention'
            elif not remaining:
                optimization['state'] = 'complete'
                optimization['eta_seconds'] = 0
            elif unknown:
                optimization['state'] = 'unknown'
            elif optimization['running_jobs']:
                optimization['state'] = 'running'
            elif optimization['pending_jobs']:
                optimization['state'] = 'queued'
            else:
                optimization['state'] = 'unknown'
            if (remaining and not unknown and not optimization['failed_jobs']
                    and all(arm in medians for arm in self.cfg['arms'])):
                self.estimate(optimization, remaining, medians)
            elif remaining and len(medians) != len(self.cfg['arms']):
                warnings.append('ETA unavailable until each arm has at least five valid COMPLETED runtime samples.')
            self.last_good = copy.deepcopy(result | {'stages': [selected, optimization] + self.future_stages()})
        result['stages'] = [selected, optimization] + self.future_stages()
        return result

    def estimate(self, optimization, remaining, medians):
        # Count only current usable allocations, never pending requests. This is
        # execution work at today's concurrency, NOT a prediction of queue wait.
        concurrency = sum(record.get('state') in ALLOCATED - {'SUSPENDED', 'STOPPED'}
                          for _, record in remaining)
        concurrency = min(concurrency, optimization['active_gpus'],
                          self.cfg['resources']['max_concurrent_gpus'])
        if concurrency <= 0:
            return
        work = [max(0, medians[arm] - (record.get('elapsed', 0)
                    if record.get('state') in ALLOCATED else 0))
                for arm, record in remaining]
        if not sum(work):
            optimization['warnings'].append('Running tasks exceed median runtime; remaining execution time is uncalibrated.')
            return
        optimization['eta_seconds'] = round(max(sum(work) / concurrency, max(work)), 1)
        average = sum(medians[arm] for arm, _ in remaining) / len(remaining)
        optimization['throughput_per_hour'] = round(concurrency * 2 * 3600 / average, 2)
        optimization['warnings'].append(
            f'ETA uses per-arm median completed runtimes and {concurrency} current allocations; '
            'subtracts elapsed running work, assumes continued concurrency, excludes queue wait.')


def publish(path, snapshot):
    data = (json.dumps(snapshot, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if len(data) > SNAPSHOT_LIMIT:
        raise ValueError('snapshot exceeds 1 MiB; keeping previous snapshot')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_cpu_job():
    if not re.fullmatch(r'\d+', os.environ.get('SLURM_JOB_ID', '')):
        raise RuntimeError('collector must run inside a Slurm CPU allocation')
    for key in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS', 'SLURM_GPUS_ON_NODE',
                'SLURM_GPUS_PER_NODE', 'SLURM_GPUS_PER_TASK'):
        if os.environ.get(key, '') not in ('', '0', '(null)', 'N/A'):
            raise RuntimeError(f'collector refuses GPU allocation ({key})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    require_cpu_job()
    interval = max(30, args.interval)
    config = load_json(args.config, SNAPSHOT_LIMIT)
    definitions = config['campaigns']
    if not isinstance(definitions, list) or not 0 < len(definitions) <= MAX_CAMPAIGNS:
        raise ValueError('configure between one and eight campaigns')
    campaigns = [Campaign(item) for item in definitions]
    if len({campaign.root for campaign in campaigns}) != len(campaigns):
        raise ValueError('duplicate campaign roots')
    output = args.output.resolve()
    for campaign in campaigns:
        if output.is_relative_to(campaign.root):
            raise ValueError('snapshot must not be written inside a campaign root')
    stop = False

    def stopping(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    while not stop:
        started = time.monotonic()
        errors, prepared, preparation_errors = [], [], {}
        total_tasks = 0
        for campaign in campaigns:
            try:
                campaign.prepare()
                total_tasks += len(campaign.tasks)
                if total_tasks > MAX_TASKS:
                    raise ValueError('configured campaigns exceed 20000-task sweep bound')
                prepared.append(campaign)
            except Exception as exc:
                preparation_errors[campaign.root] = short_error(exc)
        scheduler_error = None
        try:
            records = scheduler_snapshot([campaign.job_id for campaign in prepared])
        except Exception as exc:
            records, scheduler_error = {}, short_error(exc)
            errors.append(f'Scheduler: {scheduler_error}')
        # Telemetry failure is independent of scheduler counts, validated artifacts, and ETA.
        try:
            gpu = gpu_snapshot(prepared, records, scheduler_error)
        except Exception as exc:
            gpu = {campaign.root: empty_gpu_utilization(error=short_error(exc))
                   for campaign in prepared}
        results = []
        for campaign in campaigns:
            try:
                if campaign.root in preparation_errors:
                    raise ValueError(preparation_errors[campaign.root])
                results.append(campaign.collect(records, scheduler_error, gpu.get(campaign.root)))
            except Exception as exc:
                error = short_error(exc)
                errors.append(f'{campaign.name}: {error}')
                results.append(campaign.stale(error))
        snapshot = dict(schema_version=1, generated_at=utc_now(), refresh_seconds=interval,
                        collector=dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname()),
                        campaigns=results, errors=errors)
        try:
            publish(output, snapshot)
        except Exception as exc:
            print(f'Publication failed; previous snapshot retained: {short_error(exc)}',
                  file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return int(bool(errors))
        while not stop and time.monotonic() - started < interval:
            time.sleep(max(0, min(1, interval - (time.monotonic() - started))))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Collector refused or stopped: {short_error(exc)}', file=sys.stderr)
        sys.exit(1)
