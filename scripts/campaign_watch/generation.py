"""Bounded, read-only adapter for registered single-generate-stage bundles.

Only frozen plans, run manifests, controller journals and task status JSON are read.
No generator imports, output parsing, database access, or controller.status() calls.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re
import stat
import statistics

from collector import (ALLOCATED, PENDING, TERMINAL, MAX_TASKS, MANIFEST_LIMIT,
                       fingerprint, load_json, stage, utc_now)

MAX_FILES = 25000
MAX_BYTES = 64 * 1024 * 1024
MAX_COMPONENTS = 256
# These native adapters request an accepted-design stopping target but collect raw candidates.
ACCEPTED_STOP_TOOLS = {'bindcraft2', 'freebindcraft'}


def inside(root, value, base=None):
    path = Path(value)
    path = (base or root) / path if not path.is_absolute() else path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'metadata path escapes registered root: {path}')
    return path


def nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def reported_stage(identifier, name, planned):
    result = stage(identifier, name, 'designs', planned, 'not_submitted')
    result.update(valid=None, count_basis='reported', reported_produced=None,
                  coordinate_ready=None, phase='planned', tasks_planned=0,
                  tasks_observed=0, counts_reported_tasks=0, partial_jobs=0,
                  unknown_jobs=0, failed_by_state={}, job_ids=[], run_ids=[],
                  eta_sample_count=0, eta_basis='Execution time to finish planned tasks, not output quota; excludes queue wait and assumes current concurrency.')
    return result


class GenerationCampaign:
    def __init__(self, config):
        self.root = Path(config['root']).resolve()
        self.name = config.get('name', self.root.name)
        self.manifest = Path(config['manifest']).resolve()
        self.cache = {}
        self.last_good = None
        self.tasks = []
        self.jobs = {}
        self.job_ids = set()
        self.definition = None

    def read(self, path, *, optional=False, limit=MANIFEST_LIMIT, collected=False):
        path = path.resolve()
        if path != self.manifest and not path.is_relative_to(self.root):
            raise ValueError(f'metadata symlink escapes registered root: {path}')
        self.files += 1
        if self.files > MAX_FILES:
            raise ValueError('generation metadata file-count bound exceeded')
        self.visited.add(path)
        try:
            stamp = fingerprint(path, limit)
        except FileNotFoundError:
            self.cache.pop(path, None)
            if optional:
                return None
            raise
        self.bytes += stamp[2]
        if self.bytes > MAX_BYTES:
            raise ValueError('generation metadata sweep-byte bound exceeded')
        cached = self.cache.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        value = load_json(path, limit)
        if collected:
            # Retain only the count summary, never the candidate/structure inventory.
            value = value['collected']['run']
        self.cache[path] = (stamp, value)
        return value

    def prepare(self):
        self.files, self.bytes, self.visited = 0, 0, set()
        definition = self.read(self.manifest, limit=1024 * 1024)
        if definition.get('schema_version') != 1 or definition.get('kind') != 'bindocracy_generation':
            raise ValueError('unsupported generation registration manifest')
        if Path(definition['root']).resolve() != self.root:
            raise ValueError('registration root differs from configured root')
        tiers = definition['tiers']
        components = definition['components']
        if not 0 < len(tiers) <= 8 or not 0 < len(components) <= MAX_COMPONENTS:
            raise ValueError('generation registration exceeds tier/component bounds')
        tier_ids = {tier['id'] for tier in tiers}
        if len(tier_ids) != len(tiers):
            raise ValueError('duplicate generation tier')
        self.cells = {}
        self.arm_states = {}
        for item in components:
            key = (item['tier'], item['arm'], item['generator'])
            if key in self.cells or key[0] not in tier_ids:
                raise ValueError('duplicate or unknown generation component')
            for name in key:
                if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', name):
                    raise ValueError('invalid component identifier')
            nonnegative(item['planned'], 'planned')
            nonnegative(item['tasks_planned'], 'tasks_planned')
            self.cells[key] = item
        self.tasks, self.jobs, self.job_ids = [], {}, set()
        self.stage_controls = {}
        seen_runs, seen_status = {}, set()
        for tier in tiers:
            bundle = inside(self.root, tier['bundle_root'])
            if tier.get('control_status'):
                status = self.read(inside(self.root, tier['control_status']), optional=True, limit=65536)
                if status is not None:
                    if status.get('stage_id') != tier['id'] or status.get('state') not in (
                            'planned', 'queued', 'running', 'complete', 'failed'):
                        raise ValueError('invalid generation stage control identity/state')
                    for field in ('job_id', 'dependency_job_id'):
                        if status.get(field) is not None and not re.fullmatch(r'[0-9]+', str(status[field])):
                            raise ValueError(f'invalid stage control {field}')
                    self.stage_controls[tier['id']] = status
                    if status.get('job_id'):
                        self.job_ids.add(str(status['job_id']))
            control = self.read(bundle / 'control' / 'status.json', optional=True) or {}
            arms = control.get('stages', {}).get('generate', {})
            for arm in tier['arms']:
                if not any(key[:2] == (tier['id'], arm) for key in self.cells):
                    raise ValueError('registered arm has no components')
                entry = arms.get(arm, {})
                self.arm_states[(tier['id'], arm)] = entry.get('state')
                plan_path = inside(bundle, entry.get('plan') or
                                   f'control/plans/{arm}/generate/plan.json')
                plan = self.read(plan_path, optional=True)
                if plan is None:
                    continue
                plan_dir = inside(bundle, plan['plan_dir'])
                journal = self.read(plan_dir / 'control' / 'journal.json', optional=True) or {}
                if journal and journal.get('digest') != plan.get('digest'):
                    raise ValueError('controller journal belongs to another plan')
                associations = {}
                for run in plan['runs']:
                    manifest_path = inside(bundle, run['manifest'])
                    manifest = self.read(manifest_path)
                    key = (tier['id'], arm, manifest['tool'])
                    if key not in self.cells:
                        raise ValueError(f'unregistered generator in plan: {key}')
                    if manifest.get('kind') != 'generate' or manifest.get('arm') not in (None, arm):
                        raise ValueError('run kind/arm differs from registered generation plan')
                    directory = inside(bundle, manifest['run_dir'])
                    run_id = manifest['run_id']
                    if not isinstance(run_id, str) or len(run_id) > 200:
                        raise ValueError('invalid run identity')
                    prior = seen_runs.get(run_id)
                    if prior is not None and prior != manifest_path:
                        raise ValueError('run identity appears in different manifests')
                    duplicate = prior is not None
                    seen_runs[run_id] = manifest_path
                    resources = run.get('resources', manifest.get('resources', {}))
                    gpus = nonnegative(resources.get('gpu', resources.get('gpus', 0)), 'worker GPUs')
                    ingested = self.read(directory / 'ingested.json', optional=True, limit=65536)
                    collected, collected_warning = None, None
                    try:
                        source_path = inside(self.root, directory / 'collected.json')
                        compact = self.read(directory / 'collected-summary.json', optional=True,
                                            limit=1024 * 1024)
                        if compact is None:
                            collected = self.read(source_path, optional=True,
                                                  limit=1024 * 1024, collected=True)
                        else:
                            info = source_path.stat()
                            expected = dict(name=source_path.name, size_bytes=info.st_size,
                                            mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                                            inode=info.st_ino)
                            if (compact.get('schema_version') != 1 or
                                    not stat.S_ISREG(info.st_mode) or compact.get('source') != expected):
                                raise ValueError('compact counts do not match the current collection file')
                            collected = compact['collected']['run']
                        if collected and collected.get('run_id') != run_id:
                            raise ValueError('collected run identity mismatch')
                    except (ValueError, KeyError, OSError) as exc:
                        collected_warning = f'Collected count summary unavailable: {str(exc)[:180]}'
                        collected = None
                    reported_tasks = (collected or {}).get('count_details', {}).get('tasks', {})
                    for task in manifest['tasks']:
                        task_id = nonnegative(task['task_id'], 'task_id')
                        identity = (run_id, task_id)
                        if (run['name'], task_id) in associations:
                            raise ValueError('duplicate run/task association in plan')
                        associations[(run['name'], task_id)] = identity
                        if duplicate:
                            continue
                        status_path = inside(directory, task['status'])
                        if status_path in seen_status:
                            raise ValueError('multiple tasks share an output status path')
                        seen_status.add(status_path)
                        status = self.read(status_path, optional=True, limit=65536)
                        if status is not None:
                            if status.get('task_id') != task_id or status.get('status') not in ('succeeded', 'partial', 'failed'):
                                raise ValueError('invalid task status identity or outcome')
                            if status.get('n_produced') is not None:
                                nonnegative(status['n_produced'], 'n_produced')
                        summary = reported_tasks.get(f'{task_id:04d}', {})
                        if (status or {}).get('n_produced') is None and summary.get('n_produced') is not None:
                            count = nonnegative(summary['n_produced'], 'collected n_produced')
                            outcome = (status or summary).get('status')
                            if outcome in ('succeeded', 'partial', 'failed'):
                                status = dict(status or {}, task_id=task_id, status=outcome,
                                              n_produced=count, count_source='collected.json')
                        self.tasks.append(dict(identity=identity, key=key, run_id=run_id,
                                               task_id=task_id, status=status,
                                               requested=nonnegative(task['n_requested'], 'n_requested'),
                                               gpus=gpus, ingested=ingested is not None,
                                               warning=collected_warning))
                        if len(self.tasks) > MAX_TASKS:
                            raise ValueError('generation task-count bound exceeded')
                for attempt in journal.get('attempts', []):
                    for job in attempt['jobs']:
                        if job.get('role') != 'worker':
                            continue
                        identity = associations.get((job.get('run'), job.get('task')))
                        if identity is None:
                            raise ValueError('worker journal references unknown run/task')
                        job_id = job.get('id')
                        if job_id is None:
                            continue
                        if not isinstance(job_id, str) or not re.fullmatch(r'\d+', job_id):
                            raise ValueError('expected individual worker allocation ID')
                        if job_id in self.jobs and self.jobs[job_id] != identity:
                            raise ValueError('one allocation maps to multiple tasks')
                        self.jobs[job_id] = identity
                        if len(self.jobs) > MAX_TASKS:
                            raise ValueError('generation worker-allocation bound exceeded')
                        self.job_ids.add(job_id)
        self.cache = {path: value for path, value in self.cache.items() if path in self.visited}
        self.definition = definition

    def gpu_keys(self):
        tasks = {task['identity']: task for task in self.tasks}
        return {(job_id, -1) for job_id, identity in self.jobs.items() if tasks[identity]['gpus'] > 0}

    def empty(self):
        return dict(id=str(self.root), name=self.name, root=str(self.root),
                    kind='bindocracy_generation', observed_at=None, stages=[], warnings=[])

    @staticmethod
    def calibrate_eta(cell, tasks, task_jobs, records, scheduler_error):
        if (scheduler_error or cell['failed_jobs'] or cell['unknown_jobs']
                or cell['partial_jobs'] or not cell['running_jobs']):
            return
        samples, running = [], []
        completed = 0
        for task in tasks:
            ids = task_jobs.get(task['identity'], [])
            status = task['status'] or {}
            if not ids:
                if status:
                    return  # No trustworthy scheduler attribution for this output.
                continue
            if len(ids) != 1:
                return  # Retry elapsed times cannot be attributed to one output tally.
            record = records.get((ids[0], -1), {})
            state, elapsed = record.get('state'), record.get('elapsed')
            if state == 'COMPLETED' and record.get('exit') == '0:0' and status.get('status') == 'succeeded':
                completed += 1
                if type(elapsed) is int and elapsed > 0:
                    samples.append((elapsed, status.get('n_produced')))
            elif state == 'RUNNING' and record.get('current') and type(elapsed) is int and elapsed >= 0:
                running.append(elapsed)
            elif state not in PENDING or not record.get('current'):
                return
        cell['eta_sample_count'] = len(samples)
        if len(samples) < 5 or not running:
            return
        duration = statistics.median(elapsed for elapsed, _ in samples)
        remaining = max(cell['tasks_planned'], len(tasks)) - completed - len(running)
        work = remaining * duration + sum(max(0, duration - elapsed) for elapsed in running)
        cell['eta_seconds'] = work / len(running)
        if all(count is not None for _, count in samples):
            cell['throughput_per_hour'] = (
                sum(count for _, count in samples) * 3600 * len(running)
                / sum(elapsed for elapsed, _ in samples))

    def control_evidence(self, tier_id, records, scheduler_error):
        control = self.stage_controls.get(tier_id)
        if not control:
            return None
        job_id = control.get('job_id')
        record = records.get((str(job_id), -1), {}) if not scheduler_error else {}
        state = control['state']
        if state == 'failed':
            phase = 'blocked'
        elif job_id and (scheduler_error or not record):
            state, phase = 'unknown', 'unknown'
        elif record.get('state') in TERMINAL and (
                record['state'] != 'COMPLETED' or record.get('exit') not in (None, '0:0')):
            state, phase = 'failed', 'blocked'
        elif record.get('current') and record['state'] in PENDING:
            state, phase = 'queued', 'waiting_dependency' if control.get('dependency_job_id') else 'queued'
        elif record.get('current') and record['state'] in ALLOCATED:
            state, phase = 'running', 'preparing'
        elif state == 'complete' and record.get('state') == 'COMPLETED':
            phase = 'ingested'
        elif state == 'planned' and not job_id:
            phase = 'planned'
        else:
            state, phase = 'unknown', 'unknown'
        message = f"CPU stage job {job_id or 'not submitted'}: {state}; excluded from worker/GPU totals."
        if control.get('dependency_job_id'):
            message += f" Requires successful job {control['dependency_job_id']}."
        if control.get('error'):
            message += f" {str(control['error'])[:300]}"
        return dict(state=state, phase=phase, warning=message)

    def collect(self, records, scheduler_error, gpu_utilization=None):
        result = self.empty()
        cells = {}
        controls = {tier['id']: self.control_evidence(tier['id'], records, scheduler_error)
                    for tier in self.definition['tiers']}
        for key, item in self.cells.items():
            cell = reported_stage(':'.join(key), key[2], item['planned'])
            cell.update(arm=key[1], generator=key[2], tasks_planned=item['tasks_planned'])
            if key[2] in ACCEPTED_STOP_TOOLS:
                cell['warnings'].append('Native request is an accepted-design stop, not the nominal raw-candidate target; these counts are not compared.')
            arm_state = self.arm_states.get(key[:2])
            if arm_state:
                cell['phase'] = 'preparing'
                if arm_state == 'failed':
                    cell.update(state='needs_attention', phase='unknown')
                    cell['warnings'].append('Arm controller reports failure; worker/output evidence shown separately.')
            cells[key] = cell
        task_jobs = {}
        for job_id, identity in self.jobs.items():
            task_jobs.setdefault(identity, []).append(job_id)
        tasks_by_cell = {}
        for task in self.tasks:
            cell = cells[task['key']]
            tasks_by_cell.setdefault(task['key'], []).append(task)
            if task['warning'] and task['warning'] not in cell['warnings']:
                cell['warnings'].append(task['warning'])
            cell['tasks_observed'] += 1
            if task['run_id'] not in cell['run_ids']:
                cell['run_ids'].append(task['run_id'])
            status = task['status'] or {}
            count = status.get('n_produced')
            if count is not None:
                cell['reported_produced'] = (cell['reported_produced'] or 0) + count
                cell['counts_reported_tasks'] += 1
            partial = status.get('status') == 'partial' or (
                task['key'][2] not in ACCEPTED_STOP_TOOLS and count is not None and 0 < count < task['requested'])
            cell['partial_jobs'] += int(partial)
            ids = task_jobs.get(task['identity'], [])
            cell['job_ids'].extend(ids)
            if not ids and status and count is None:
                cell['unknown_jobs'] += 1
            failed_before = cell['failed_jobs']
            for job_id in ids:
                record = records.get((job_id, -1), {}) if not scheduler_error else {}
                state = record.get('state', 'UNKNOWN')
                if record.get('current') and state in ALLOCATED:
                    cell['running_jobs'] += 1
                    cell['active_gpus'] += task['gpus']
                elif record.get('current') and state in PENDING:
                    cell['pending_jobs'] += 1
                elif state in TERMINAL:
                    if state != 'COMPLETED' or record.get('exit') not in (None, '0:0'):
                        cell['failed_jobs'] += 1
                        cell['failed_by_state'][state] = cell['failed_by_state'].get(state, 0) + 1
                else:
                    cell['unknown_jobs'] += 1
            if status.get('status') == 'failed' and cell['failed_jobs'] == failed_before:
                cell['failed_jobs'] += 1
                cell['failed_by_state']['TASK_FAILED'] = cell['failed_by_state'].get('TASK_FAILED', 0) + 1
        for key, cell in cells.items():
            tasks = tasks_by_cell.get(key, [])
            if tasks:
                all_reported = cell['counts_reported_tasks'] == len(tasks)
                done = (all_reported and len(tasks) >= cell['tasks_planned']
                        and cell['reported_produced'] >= cell['planned']
                        and all(task['status']['status'] == 'succeeded' and
                                (key[2] in ACCEPTED_STOP_TOOLS or task['status']['n_produced'] >= task['requested'])
                                for task in tasks))
                if cell['running_jobs']:
                    cell.update(state='running', phase='generating')
                elif cell['pending_jobs']:
                    cell.update(state='queued', phase='queued')
                elif cell['failed_jobs']:
                    cell.update(state='needs_attention', phase='partial' if cell['reported_produced'] else 'unknown')
                elif cell['unknown_jobs']:
                    phase = ('ingested' if all(task['ingested'] for task in tasks) else 'generated') if done else (
                        'partial' if cell['reported_produced'] else 'unknown')
                    cell.update(state='unknown', phase=phase)
                elif done:
                    cell.update(state='complete', phase='ingested' if all(task['ingested'] for task in tasks) else 'generated')
                elif cell['reported_produced'] is not None:
                    cell.update(state='needs_attention', phase='partial')
                elif cell['job_ids'] or any(task['status'] for task in tasks):
                    cell.update(state='unknown', phase='unknown')
                else:
                    cell['phase'] = 'preparing'
                if not all_reported:
                    cell['warnings'].append(f"Reported counts available for {cell['counts_reported_tasks']}/{len(tasks)} tasks; missing counts are not zero.")
                if cell['failed_jobs'] and cell['reported_produced']:
                    cell['warnings'].append('Reported outputs retained despite worker failures; completeness and coordinates are not validated.')
            if scheduler_error and cell['job_ids']:
                cell['warnings'].append(f'Scheduler unavailable: {scheduler_error}')
            if self.arm_states.get(key[:2]) != 'failed':
                self.calibrate_eta(cell, tasks, task_jobs, records, scheduler_error)
            control = controls[key[0]]
            if control:
                cell['warnings'].append(control['warning'])
                if not cell['tasks_observed'] or (
                        not cell['job_ids'] and cell['reported_produced'] is None):
                    cell.update(state={'failed': 'needs_attention', 'running': 'running',
                                       'queued': 'queued', 'unknown': 'unknown'}.get(control['state'], cell['state']),
                                phase=control['phase'])
                if control['state'] == 'failed':
                    cell['eta_seconds'] = cell['throughput_per_hour'] = None
        for tier in self.definition['tiers']:
            children = [cell for key, cell in cells.items() if key[0] == tier['id']]
            item = reported_stage(tier['id'], tier['name'], sum(cell['planned'] for cell in children))
            item['components'] = children
            for field in ('running_jobs', 'pending_jobs', 'failed_jobs', 'active_gpus', 'tasks_planned',
                          'tasks_observed', 'counts_reported_tasks', 'partial_jobs', 'unknown_jobs'):
                item[field] = sum(cell[field] for cell in children)
            known = [cell['reported_produced'] for cell in children if cell['reported_produced'] is not None]
            item['reported_produced'] = sum(known) if known else None
            for cell in children:
                for state, count in cell['failed_by_state'].items():
                    item['failed_by_state'][state] = item['failed_by_state'].get(state, 0) + count
            for state in ('running', 'queued', 'needs_attention', 'unknown'):
                if any(cell['state'] == state for cell in children):
                    item['state'] = state
                    break
            else:
                if all(cell['state'] == 'complete' for cell in children):
                    item['state'] = 'complete'
            item['phase'] = {'running': 'generating', 'queued': 'queued', 'needs_attention': 'partial',
                             'unknown': 'unknown', 'complete': 'generated'}.get(item['state'],
                             'preparing' if any(cell['phase'] == 'preparing' for cell in children) else 'planned')
            if all(cell['phase'] in ('generated', 'ingested') for cell in children):
                item['phase'] = 'ingested' if all(cell['phase'] == 'ingested' for cell in children) else 'generated'
            if item['counts_reported_tasks'] < item['tasks_observed']:
                item['warnings'].append('Reported production is a partial tally; some task counts are unavailable.')
            control = controls[tier['id']]
            if control:
                item['warnings'].append(control['warning'])
                if control['state'] == 'failed':
                    item.update(state='needs_attention', phase='blocked')
                elif control['phase'] == 'waiting_dependency' and not item['tasks_observed']:
                    item['phase'] = 'waiting_dependency'
            result['stages'].append(item)
        for future in self.definition.get('future_stages', []):
            item = reported_stage(future['id'], future['name'], future.get('planned'))
            item['unit'] = future.get('unit', 'designs')
            item['warnings'].append('Planned downstream stage; awaiting approval, no live execution adapter.')
            result['stages'].append(item)
        result['gpu_utilization'] = gpu_utilization
        result['warnings'].append('Nominal requests, not guaranteed outputs. Production is tool-reported, not unique/coordinate-validated. ETA requires five successful same-arm/tool elapsed samples; estimates planned-task execution, not output quota, excluding queue wait.')
        result['observed_at'] = utc_now()
        self.last_good = copy.deepcopy(result)
        return result

    def stale(self, error):
        result = copy.deepcopy(self.last_good) if self.last_good else self.empty()
        for item in result['stages']:
            for row in [item, *item.get('components', [])]:
                if row['phase'] != 'planned':
                    row.update(state='unknown', phase='unknown', eta_seconds=None, throughput_per_hour=None)
        result['warnings'].append(f'STALE / collection unavailable: {error}')
        return result
