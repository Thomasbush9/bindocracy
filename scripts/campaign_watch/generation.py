"""Bounded, read-only adapter for registered single-generate-stage bundles.

Only frozen plans, run manifests, controller journals and bounded count metadata are read.
No generator imports, design/metric parsing, database access, or controller.status() calls.
"""
from __future__ import annotations

import copy
import json
import os
import re
import stat
import statistics
from datetime import datetime, timezone
from pathlib import Path

from collector import (
    ALLOCATED,
    MANIFEST_LIMIT,
    MAX_TASKS,
    PENDING,
    TERMINAL,
    fingerprint,
    load_json,
    reject_constant,
    stage,
    unique_object,
    utc_now,
)

MAX_FILES = 25000
MAX_BYTES = 64 * 1024 * 1024
MAX_COMPONENTS = 256
COLLECTED_LIMIT = 1024 * 1024
RECOVERY_MAX_AGE = 15 * 60
RECOVERY_JOB_LIMIT = 1024
RUN_PREFIX = re.compile(
    rb'\A\s*\{\s*"bundle_version"\s*:\s*1\s*,\s*"collected"\s*:\s*\{\s*"run"\s*:\s*(?=\{)'
)
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


def collection_stamp(info):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('collection is not a regular file')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def count_metadata(run):
    """Keep only counts and outcomes, not unrelated run configuration."""
    if not isinstance(run, dict):
        raise TypeError('collected run must be an object')
    count = run.get('n_produced')
    if count is not None:
        nonnegative(count, 'collected run n_produced')
    outcome = run.get('status')
    if outcome is not None and outcome not in ('succeeded', 'partial', 'failed'):
        raise ValueError('invalid collected run outcome')
    tasks = (run.get('count_details') or {}).get('tasks') or {}
    return {
        'run_id': run['run_id'], 'n_produced': count, 'status': outcome,
        'count_details': {'tasks': {
            key: {'n_produced': value.get('n_produced'), 'status': value.get('status')}
            for key, value in tasks.items()
        }},
    }


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

    def read(self, path, *, optional=False, limit=MANIFEST_LIMIT):
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
        self.cache[path] = (stamp, value)
        return value

    def read_collected(self, path):
        """Read only the leading canonical run object, bounded even for huge inventories."""
        path = inside(self.root, path)
        self.files += 1
        if self.files > MAX_FILES:
            raise ValueError('generation metadata file-count bound exceeded')
        self.visited.add(path)
        try:
            stamp = collection_stamp(path.stat())
        except FileNotFoundError:
            self.cache.pop(path, None)
            return None
        cached = self.cache.get(path)
        if cached and cached[0] == stamp:
            self.bytes += cached[2]
            if self.bytes > MAX_BYTES:
                raise ValueError('generation metadata sweep-byte bound exceeded')
            return cached[1]
        raw = bytearray()
        start = None
        depth, quoted, escaped = 0, False, False
        value = None
        with path.open('rb') as stream:
            if collection_stamp(os.fstat(stream.fileno())) != stamp:
                raise ValueError('collection changed before read')
            while len(raw) < COLLECTED_LIMIT:
                size = min(4096, COLLECTED_LIMIT - len(raw), MAX_BYTES - self.bytes)
                if size <= 0:
                    raise ValueError('generation metadata sweep-byte bound exceeded')
                chunk = stream.read(size)
                self.bytes += len(chunk)
                if not chunk:
                    break
                offset = len(raw)
                raw.extend(chunk)
                if start is None:
                    prefix = RUN_PREFIX.match(raw)
                    if prefix is None:
                        # Older tiny fixtures/bundles had an unversioned envelope.
                        # Never load a large inventory to accommodate another layout.
                        if stamp[2] > COLLECTED_LIMIT:
                            raise ValueError('unsupported canonical collection prefix')
                        continue
                    start = offset = prefix.end()
                for index in range(offset, len(raw)):
                    char = raw[index]
                    if quoted:
                        if escaped:
                            escaped = False
                        elif char == 92:
                            escaped = True
                        elif char == 34:
                            quoted = False
                    elif char == 34:
                        quoted = True
                    elif char in (123, 91):
                        depth += 1
                    elif char in (125, 93):
                        depth -= 1
                        if depth == 0:
                            value = count_metadata(json.loads(
                                raw[start:index + 1], parse_constant=reject_constant,
                                object_pairs_hook=unique_object))
                            break
                if value is not None:
                    break
            if value is None:
                if start is not None or len(raw) != stamp[2]:
                    raise ValueError('canonical run metadata exceeds prefix bound or is incomplete')
                legacy = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
                if 'bundle_version' in legacy:
                    raise ValueError('unsupported canonical collection prefix')
                value = count_metadata(legacy['collected']['run'])
            if (collection_stamp(os.fstat(stream.fileno())) != stamp
                    or collection_stamp(path.stat()) != stamp):
                raise ValueError('collection changed during read')
        self.cache[path] = (stamp, value, len(raw))
        return value

    def collected_counts(self, directory):
        source = inside(self.root, directory / 'collected.json')
        try:
            compact = self.read(directory / 'collected-summary.json', optional=True,
                                limit=COLLECTED_LIMIT)
            if compact is not None:
                before = collection_stamp(source.stat())
                expected = {
                    'name': source.name, 'size_bytes': before[2], 'mtime_ns': before[3],
                    'ctime_ns': before[4], 'inode': before[1],
                }
                if compact.get('schema_version') == 1 and compact.get('source') == expected:
                    value = count_metadata(compact['collected']['run'])
                    if collection_stamp(source.stat()) == before:
                        return value
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
            pass
        # Snakemake touches declared outputs after collection. A mismatched sidecar
        # is never trusted: recover counts from the current canonical file instead.
        return self.read_collected(source)

    def _plan_sources(self, definition):
        """Normalize live schemas before reading the shared native plan/task evidence."""
        sources = {tier['id']: [] for tier in definition['tiers']}
        self.component_controls = {}
        self.stage_controls = {}
        independent = self.read(self.root / 'control/independent.json', optional=True,
                                limit=1024 * 1024)
        self.independent = independent is not None
        if self.independent:
            if independent.get('schema_version') != 1 or independent.get('mode') != 'independent_generators':
                raise ValueError('unsupported independent generation index')
            units = independent['units']
            if not isinstance(units, list) or not 0 < len(units) <= MAX_COMPONENTS:
                raise ValueError('independent unit-count bound exceeded')
            seen_ids, history = set(), set()
            production = {key for key in self.cells if key[0] != 'qualification'}
            for unit in units:
                key = (unit['stage_id'], unit['arm'], unit['generator'])
                if key not in production or key in self.component_controls or unit['id'] in seen_ids:
                    raise ValueError('duplicate or unregistered independent unit')
                if unit['id'] != '--'.join(key):
                    raise ValueError('independent unit identity mismatch')
                seen_ids.add(unit['id'])
                for field in ('plan', 'status_path', 'database'):
                    if not Path(unit[field]).is_absolute():
                        raise ValueError('independent metadata paths must be absolute')
                    inside(self.root, unit[field])
                status = self.read(inside(self.root, unit['status_path']), optional=True, limit=65536)
                if status is None:
                    status = {'state': 'unknown', 'error': 'Component status unavailable; not proof of completion.'}
                else:
                    if status.get('state') not in ('planned', 'queued', 'running', 'complete', 'failed'):
                        raise ValueError('invalid independent component state')
                    for field, value in zip(('stage_id', 'arm', 'generator'), key):
                        if status.get(field, value) != value:
                            raise ValueError('independent component identity mismatch')
                    if status.get('id', unit['id']) != unit['id']:
                        raise ValueError('independent component identity mismatch')
                    for field in ('job_id', 'native_controller_job_id'):
                        job_id = status.get(field)
                        if job_id is not None:
                            if not re.fullmatch(r'[0-9]+', str(job_id)):
                                raise ValueError(f'invalid component control {field}')
                            self.job_ids.add(str(job_id))
                self.component_controls[key] = status
                sources[key[0]].append({
                    'arm': key[1], 'root': self.root, 'plan': inside(self.root, unit['plan']),
                    'generator': key[2]})
                if unit.get('original_plan'):
                    original = inside(self.root, unit['original_plan'])
                    identity = (key[0], key[1], original)
                    if identity not in history:
                        history.add(identity)
            if set(self.component_controls) != production:
                raise ValueError('independent units do not cover registered production components')
            # Old journals retain GPU allocations that were not resubmitted. Their
            # controls are historical, but run/task/allocation identities are shared.
            for tier, arm, path in sorted(history):
                sources[tier].append({'arm': arm, 'root': self.root, 'plan': path})
        for tier in definition['tiers']:
            if self.independent and tier['id'] != 'qualification':
                continue
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
            control = self.read(bundle / 'control/status.json', optional=True) or {}
            arms = control.get('stages', {}).get('generate', {})
            for arm in tier['arms']:
                if not any(key[:2] == (tier['id'], arm) for key in self.cells):
                    raise ValueError('registered arm has no components')
                entry = arms.get(arm, {})
                self.arm_states[(tier['id'], arm)] = entry.get('state')
                sources[tier['id']].append({
                    'arm': arm, 'root': bundle, 'plan': inside(bundle, entry.get('plan') or
                        f'control/plans/{arm}/generate/plan.json')})
        return sources

    def _recovery_sources(self):
        """Read only explicitly indexed summaries; never inspect native recovery output."""
        self.recovery_controls = {}
        self.recovery_job_ids = set()
        try:
            index = self.read(self.root / 'control/recovery/index.json', limit=1024 * 1024)
        except FileNotFoundError:
            return
        if (not isinstance(index, dict) or index.get('schema_version') != 1
                or Path(index['campaign_root']).resolve() != self.root):
            raise ValueError('invalid recovery index schema/campaign root')
        units = index['units']
        if not isinstance(units, list) or len(units) > MAX_COMPONENTS:
            raise ValueError('recovery unit-count bound exceeded')
        seen = set()
        task_keys = {task['identity']: task['key'] for task in self.tasks}
        for unit in units:
            key = (unit['stage_id'], unit['arm'], unit['generator'])
            if (key not in self.component_controls or key in seen
                    or unit['id'] != '--'.join(key)):
                raise ValueError('duplicate or unregistered recovery unit identity')
            seen.add(key)
            if nonnegative(unit['target'], 'recovery target') != self.cells[key]['planned']:
                raise ValueError('recovery target differs from original nominal target')
            expected = self.root / 'control/recovery/components' / f"{unit['id']}.json"
            # Require the declared location, then separately check resolved symlinks.
            if Path(unit['status_path']) != expected:
                raise ValueError('recovery status path differs from registered component location')
            try:
                status = self.read(inside(self.root, expected), limit=65536)
                if status.get('schema_version') != 1:
                    raise ValueError('invalid recovery status schema')
                for field in ('id', 'stage_id', 'arm', 'generator', 'target'):
                    if status.get(field) != unit[field]:
                        raise ValueError(f'recovery status {field} mismatch')
                nonnegative(status['target'], 'recovery status target')
                if status['state'] not in ('planned', 'qualifying', 'running', 'complete', 'blocked', 'failed'):
                    raise ValueError('invalid recovery state')
                nonnegative(status['retained_raw_candidates'], 'retained raw candidates')
                nonnegative(status['completed_batches'], 'completed recovery batches')
                if (not isinstance(status['count_basis'], str)
                        or not status['count_basis'].strip() or len(status['count_basis']) > 1000):
                    raise ValueError('invalid recovery count basis')
                if status['error'] is not None and not isinstance(status['error'], str):
                    raise ValueError('invalid recovery error')
                for field in ('accounting_warning', 'controller_warning'):
                    if status.get(field) is not None and not isinstance(status[field], str):
                        raise ValueError(f'invalid recovery {field}')
                controller_id = status.get('controller_job_id')
                if controller_id is not None and (
                        not isinstance(controller_id, str)
                        or not re.fullmatch(r'[0-9]{1,20}', controller_id)):
                    raise ValueError('invalid recovery controller job ID')
                ids = status['active_job_ids']
                if (not isinstance(ids, list) or len(ids) > RECOVERY_JOB_LIMIT
                        or any(not isinstance(job_id, str)
                               or not re.fullmatch(r'[0-9]{1,20}', job_id) for job_id in ids)
                        or len(set(ids)) != len(ids)):
                    raise ValueError('invalid recovery GPU allocation IDs')
                if any(job_id in self.recovery_job_ids or (
                        job_id in self.jobs and task_keys[self.jobs[job_id]] != key) for job_id in ids):
                    raise ValueError('recovery allocation belongs to multiple units')
                updated = datetime.fromisoformat(status['updated_at'].replace('Z', '+00:00'))
                if updated.tzinfo is None:
                    raise ValueError('recovery timestamp requires timezone')
                age = (datetime.now(timezone.utc) - updated).total_seconds()
                if age < -60:
                    raise ValueError('recovery timestamp is in the future')
                if status.get('reconciled_at') is not None:
                    reconciled = datetime.fromisoformat(status['reconciled_at'].replace('Z', '+00:00'))
                    if reconciled.tzinfo is None or (datetime.now(timezone.utc) - reconciled).total_seconds() < -60:
                        raise ValueError('invalid recovery reconciliation timestamp')
                if status['state'] == 'complete' and (
                        status['retained_raw_candidates'] < status['target'] or ids or status['error']):
                    raise ValueError('recovery completion contradicts count, active jobs, or error')
                # Terminal output counts remain valid after the supervisor exits.
                # Nonterminal summaries need a heartbeat, even without active GPUs.
                status = dict(status)
                if status['state'] != 'complete' and age > RECOVERY_MAX_AGE:
                    previous = status.get('error')
                    status.update(state='unknown', error='Recovery status stale (older than 15 minutes).'
                                  + (f' {previous}' if previous else ''))
                self.recovery_job_ids.update(ids)
                if len(self.recovery_job_ids) > MAX_TASKS:
                    raise ValueError('recovery allocation-count bound exceeded')
                self.job_ids.update(ids)
                if controller_id is not None:
                    self.job_ids.add(controller_id)
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
                status = {'state': 'unknown', 'error': f'Recovery status unavailable: {str(exc)[:300]}'}
            self.recovery_controls[key] = status

    def _apply_recovery(self, key, cell, records, scheduler_error):
        status = self.recovery_controls[key]
        state = status['state']
        ids = status.get('active_job_ids', [])
        cell.update(reported_produced=status.get('retained_raw_candidates'),
                    phase=f'recovery_{state}', job_ids=ids,
                    state={'planned': 'not_submitted', 'qualifying': 'running', 'running': 'running',
                           'complete': 'complete', 'blocked': 'needs_attention',
                           'failed': 'needs_attention'}.get(state, 'unknown'),
                    eta_seconds=None, throughput_per_hour=None)
        # Old task errors/counts belong to history, not the current recovery attempt.
        cell['warnings'] = []
        health = [f"{label}: {status[field]}" for field, label in (
            ('accounting_warning', 'Recovery accounting warning'),
            ('controller_warning', 'Recovery controller warning')) if status.get(field)]
        controller_id = status.get('controller_job_id')
        if scheduler_error and state != 'complete' and any(
                records.get((job_id, -1), {}).get('current') for job_id in [*ids, controller_id]):
            health.append(f'Recovery scheduler partially unavailable: {scheduler_error}')
        if controller_id is not None:
            controller = records.get((controller_id, -1), {})
            current = controller.get('current') and (
                controller.get('state') in ALLOCATED or controller.get('state') in PENDING)
            cell['warnings'].append(
                f"Recovery CPU controller {controller_id}: {controller.get('state', 'UNKNOWN')}; "
                'excluded from worker/GPU totals.')
            if state != 'complete' and not current:
                health.append('Recovery controller is not confirmed active by the scheduler; '
                              'completion is not inferred from queue absence.')
        if state != 'complete' and status.get('reconciled_at'):
            reconciled = datetime.fromisoformat(status['reconciled_at'].replace('Z', '+00:00'))
            if (datetime.now(timezone.utc) - reconciled).total_seconds() > RECOVERY_MAX_AGE:
                health.append('Recovery reconciliation stale (older than 15 minutes), even if the monitor heartbeat is fresh.')
        for job_id in ids:
            record = records.get((job_id, -1), {})
            if scheduler_error and not record.get('current'):
                record = {}
            scheduler_state = record.get('state', 'UNKNOWN')
            if record.get('current') and scheduler_state in ALLOCATED:
                cell['running_jobs'] += 1
                cell['active_gpus'] += 1  # Recovery contract: individual single-GPU allocations.
            elif record.get('current') and scheduler_state in PENDING:
                cell['pending_jobs'] += 1
            elif scheduler_state in TERMINAL:
                health.append(f'Recovery GPU job {job_id} is terminal ({scheduler_state}); awaiting reconciliation.')
                if scheduler_state != 'COMPLETED' or record.get('exit') not in (None, '0:0'):
                    cell['failed_jobs'] += 1
                    cell['failed_by_state'][scheduler_state] = cell['failed_by_state'].get(scheduler_state, 0) + 1
            else:
                cell['unknown_jobs'] += 1
        if state == 'planned' and ids:
            cell['phase'] = 'recovery_waiting_original'
        if state in {'planned', 'qualifying', 'running'}:
            if cell['failed_jobs']:
                cell['state'] = 'needs_attention'
            elif cell['running_jobs']:
                cell['state'] = 'running'
            elif cell['pending_jobs']:
                cell['state'] = 'queued'
            elif ids:
                cell['state'] = 'unknown'
            elif state != 'planned':
                cell.update(state='needs_attention', phase='recovery_idle')
                cell['warnings'].append(
                    'Recovery controller idle: no confirmed running or queued GPU workers; '
                    'published state is not evidence of generation progress.')
        if status.get('error') and state not in {'unknown', 'blocked', 'failed'}:
            health.append(f"Recovery error: {status['error'][:1000]}")
        if health:
            cell.update(state='needs_attention', phase='recovery_degraded')
            cell['warnings'].extend(health)
        count = cell['reported_produced']
        cell['warnings'].append(
            f"Recovery {'--'.join(key)}: {state}; generated {count if count is not None else '?'}/{cell['planned']} "
            f"(last-reconciled: {status.get('reconciled_at', status.get('updated_at')) or 'never'}); "
            f"completed batches {status.get('completed_batches', '?')}; "
            f"published active GPU job IDs (not proof of current activity): {', '.join(ids) or 'none'}. "
            f"Count basis: {status.get('count_basis', 'unavailable')}; includes original and recovery outputs, not added twice.")
        if status.get('error'):
            cell['warnings'].append(f"Recovery error: {status['error'][:1000]}")
        if scheduler_error and ids:
            cell['warnings'].append(f'Scheduler unavailable: {scheduler_error}')

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
        self.runs = {}
        sources = self._plan_sources(definition)
        seen_runs, seen_status = {}, set()
        for tier in tiers:
            for source in sources[tier['id']]:
                bundle, arm = source['root'], source['arm']
                plan = self.read(source['plan'], optional=True)
                if plan is None:
                    if source.get('generator'):
                        key = (tier['id'], arm, source['generator'])
                        self.component_controls[key] = dict(
                            self.component_controls[key], plan_missing=True)
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
                    if source.get('generator') and manifest['tool'] != source['generator']:
                        raise ValueError('independent plan contains another generator')
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
                        collected = self.collected_counts(directory)
                        if collected and collected.get('run_id') != run_id:
                            raise ValueError('collected run identity mismatch')
                    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
                        collected_warning = f'Collected count summary unavailable: {str(exc)[:180]}'
                        collected = None
                    self.runs[run_id] = collected or {}
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
                        self.tasks.append({'identity': identity, 'key': key, 'run_id': run_id,
                                           'task_id': task_id, 'status': status,
                                           'requested': nonnegative(task['n_requested'], 'n_requested'),
                                           'gpus': gpus, 'ingested': ingested is not None,
                                           'warning': collected_warning})
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
        self._recovery_sources()
        self.cache = {path: value for path, value in self.cache.items() if path in self.visited}
        self.definition = definition

    def gpu_keys(self):
        tasks = {task['identity']: task for task in self.tasks}
        return ({(job_id, -1) for job_id, identity in self.jobs.items() if tasks[identity]['gpus'] > 0}
                | {(job_id, -1) for job_id in self.recovery_job_ids})

    def empty(self):
        return {'id': str(self.root), 'name': self.name, 'root': str(self.root),
                'kind': 'bindocracy_generation', 'observed_at': None, 'stages': [], 'warnings': []}

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

    def _component_evidence(self, key, records, scheduler_error):
        control = self.component_controls[key]
        state = control['state']
        supervisor = control.get('job_id')
        native = control.get('native_controller_job_id')
        evidence = []
        for label, job_id in (('supervisor', supervisor), ('native controller', native)):
            if job_id is not None:
                record = records.get((str(job_id), -1), {}) if not scheduler_error else {}
                evidence.append(f"{label} {job_id}: {record.get('state', 'UNKNOWN')}")
                if (state != 'failed' and (state != 'complete' or label == 'native controller')
                        and record.get('state') in TERMINAL and (
                            record['state'] != 'COMPLETED' or record.get('exit') not in (None, '0:0'))):
                    state = 'failed'
        if state in ('running', 'queued'):
            job_id = native or supervisor
            record = records.get((str(job_id), -1), {}) if not scheduler_error else {}
            if record.get('current') and record.get('state') in ALLOCATED:
                state = 'queued' if state == 'queued' and native is None else 'running'
            elif record.get('current') and record.get('state') in PENDING:
                state = 'queued'
            else:
                state = 'unknown'
        if control.get('plan_missing') and state == 'complete':
            state = 'unknown'
        message = f"Independent unit {'--'.join(key)}: {state}."
        if evidence:
            message += ' CPU ' + '; '.join(evidence) + '; excluded from worker/GPU totals.'
        if control.get('error'):
            message += f" {str(control['error'])[:300]}"
        if control.get('plan_missing'):
            message += ' Native plan unavailable; not proof of completion.'
        return {'state': state, 'warning': message,
                'phase': {'failed': 'blocked', 'running': 'preparing', 'queued': 'queued',
                          'planned': 'planned', 'complete': 'generated'}.get(state, 'unknown')}

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
        return {'state': state, 'phase': phase, 'warning': message}

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
        counted_runs = set()
        for task in self.tasks:
            if task['key'] in self.recovery_controls:
                continue
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
                cell['counts_reported_tasks'] += 1
            run_count = self.runs[task['run_id']].get('n_produced')
            if run_count is not None:
                if task['run_id'] not in counted_runs:
                    cell['reported_produced'] = (cell['reported_produced'] or 0) + run_count
                    counted_runs.add(task['run_id'])
            elif count is not None:
                cell['reported_produced'] = (cell['reported_produced'] or 0) + count
            partial = status.get('status') == 'partial' or (
                task['key'][2] not in ACCEPTED_STOP_TOOLS and count is not None and 0 < count < task['requested'])
            cell['partial_jobs'] += int(partial)
            ids = task_jobs.get(task['identity'], [])
            cell['job_ids'].extend(ids)
            if not ids and status and count is None and run_count is None:
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
            if key in self.recovery_controls:
                self._apply_recovery(key, cell, records, scheduler_error)
                continue
            tasks = tasks_by_cell.get(key, [])
            collection_incomplete = any(
                self.runs[run_id].get('status') in ('partial', 'failed')
                for run_id in cell['run_ids'])
            if tasks:
                all_reported = cell['counts_reported_tasks'] == len(tasks)
                run_tasks = {}
                for task in tasks:
                    run_tasks.setdefault(task['run_id'], []).append(task)
                complete_runs = []
                for run_id, members in run_tasks.items():
                    summary = self.runs[run_id]
                    total = summary.get('n_produced')
                    shortfall = (total is not None and key[2] not in ACCEPTED_STOP_TOOLS
                                 and total < sum(task['requested'] for task in members))
                    collection_incomplete = collection_incomplete or shortfall
                    if total is not None and summary.get('status') is not None:
                        complete_runs.append(
                            summary['status'] == 'succeeded'
                            and not shortfall)
                    else:
                        complete_runs.append(all(
                            (task['status'] or {}).get('n_produced') is not None
                            and (key[2] in ACCEPTED_STOP_TOOLS
                                 or task['status']['n_produced'] >= task['requested'])
                            for task in members))
                done = (len(tasks) >= cell['tasks_planned'] and all(complete_runs)
                        and cell['reported_produced'] is not None
                        and cell['reported_produced'] >= cell['planned']
                        and not collection_incomplete
                        and all((task['status'] or {}).get('status') == 'succeeded' for task in tasks))
                if cell['running_jobs']:
                    cell.update(state='running', phase='generating')
                elif cell['pending_jobs']:
                    cell.update(state='queued', phase='queued')
                elif cell['failed_jobs'] or collection_incomplete:
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
                if collection_incomplete:
                    cell['warnings'].append('Canonical collection is partial, failed, or below the requested total; counts are retained, not success.')
                if cell['failed_jobs'] and cell['reported_produced']:
                    cell['warnings'].append('Reported outputs retained despite worker failures; completeness and coordinates are not validated.')
            if scheduler_error and cell['job_ids']:
                cell['warnings'].append(f'Scheduler unavailable: {scheduler_error}')
            if self.arm_states.get(key[:2]) != 'failed' and not collection_incomplete:
                self.calibrate_eta(cell, tasks, task_jobs, records, scheduler_error)
            control = (self._component_evidence(key, records, scheduler_error)
                       if key in self.component_controls else controls[key[0]])
            if control:
                cell['warnings'].append(control['warning'])
                if not cell['tasks_observed'] or (
                        not cell['job_ids'] and cell['reported_produced'] is None):
                    cell.update(state={'failed': 'needs_attention', 'running': 'running',
                                       'queued': 'queued', 'unknown': 'unknown'}.get(control['state'], cell['state']),
                                phase=control['phase'])
                if control['state'] == 'failed':
                    cell['eta_seconds'] = cell['throughput_per_hour'] = None
                if key in self.component_controls:
                    if control['state'] == 'failed':
                        cell.update(state='needs_attention',
                                    phase='partial' if cell['reported_produced'] else 'blocked')
                    elif control['state'] == 'unknown' and not cell['running_jobs'] and not cell['pending_jobs']:
                        cell.update(state='unknown', phase='unknown')
                    elif control['state'] in ('running', 'queued') and not cell['running_jobs'] and not cell['pending_jobs']:
                        cell.update(state=control['state'], phase=control['phase'])
                    elif control['state'] == 'complete' and cell['state'] == 'not_submitted':
                        cell.update(state='unknown', phase='unknown')
                    if control['state'] != 'complete' and cell['state'] == 'complete':
                        cell.update(state='unknown', phase='unknown')
                    if control['state'] in ('failed', 'unknown'):
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
            priority = ('running', 'queued', 'needs_attention', 'unknown')
            if any(cell['phase'].startswith('recovery_') and cell['state'] == 'needs_attention'
                   for cell in children):
                priority = ('needs_attention', 'running', 'queued', 'unknown')
            for state in priority:
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
            if any(cell['phase'].startswith('recovery_') for cell in children):
                item['phase'] = f"recovery_{item['phase']}"
            if item['counts_reported_tasks'] < item['tasks_observed']:
                item['warnings'].append('Some task counts are unavailable; canonical run totals are used where available, without inferring task distribution.')
            if self.independent and tier['id'] != 'qualification':
                for cell in children:
                    if cell['state'] == 'needs_attention':
                        item['warnings'].append(
                            f"{cell['arm']}/{cell['generator']}: {cell['phase']}; "
                            'other units proceed independently.')
                        if cell['phase'].startswith('recovery_'):
                            item['warnings'].extend(f"{cell['arm']}/{cell['generator']}: {warning}"
                                                    for warning in cell['warnings'])
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
