"""Metadata-only regressions: no Slurm, generators, or external package imports."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'campaign_watch'
sys.path.insert(0, str(SCRIPTS))
import collector
from generation import GenerationCampaign
from register_generation import registration


class GenerationWatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.plan_dir = self.bundle / 'control/plans/arm1/generate'
        self.run_dir = self.bundle / 'output/runs/arm1/run-one'
        self.manifest_path = self.root / 'watch.json'
        self.dump(self.manifest_path, {'schema_version': 1, 'kind': 'bindocracy_generation', 'root': str(self.root),
            'tiers': [{'id': 'tier1', 'name': 'Tier one', 'bundle_root': str(self.bundle), 'arms': ['arm1']}],
            'components': [{'tier': 'tier1', 'arm': 'arm1', 'generator': 'boltzgen', 'planned': 10, 'tasks_planned': 1}],
            'future_stages': [{'id': 'geometry', 'name': 'Geometry', 'planned': None}]})
        self.config = {'root': str(self.root), 'name': 'Test generation', 'manifest': str(self.manifest_path)}

    def dump(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def metadata(self, status=None, duplicate=False):
        self.dump(self.bundle / 'control/status.json', {'stages': {'generate': {'arm1': {
            'state': 'running', 'plan': str(self.plan_dir / 'plan.json'), 'controller': '900'}}}})
        self.dump(self.plan_dir / 'plan.json', {'schema_version': 2, 'digest': 'frozen', 'plan_dir': str(self.plan_dir),
            'runs': [{'name': 'arm1-boltzgen', 'manifest': str(self.run_dir / 'run.json'), 'resources': {'gpu': 1}}]})
        self.dump(self.run_dir / 'run.json', {'schema_version': 1, 'run_id': 'run-uuid', 'tool': 'boltzgen',
            'kind': 'generate', 'arm': 'arm1', 'run_dir': str(self.run_dir), 'resources': {'gpus': 1},
            'tasks': [{'task_id': 0, 'status': 'tasks/0000/status.json', 'n_requested': 10}]})
        worker = {'role': 'worker', 'id': '901', 'run': 'arm1-boltzgen', 'task': 0, 'state': 'RUNNING'}
        jobs = [{'role': 'controller', 'id': '900', 'state': 'RUNNING'}, worker]
        attempts = [{'id': 'first', 'jobs': jobs}]
        if duplicate:
            attempts.append({'id': 'second', 'jobs': [worker, dict(worker, id='902')]})
        self.dump(self.plan_dir / 'control/journal.json', {'digest': 'frozen', 'attempts': attempts})
        if status is not None:
            self.dump(self.run_dir / 'tasks/0000/status.json', {'task_id': 0, **status})

    def collect(self, records=None, error=None):
        campaign = GenerationCampaign(self.config)
        campaign.prepare()
        result = campaign.collect(records or {}, error)
        return campaign, result, result['stages'][0]['components'][0]

    @contextmanager
    def changing_json_read(self, source, change):
        original_open = open
        reads = []

        class ChangingStream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, size):
                reads.append(size)
                raw = self.stream.read(size)
                change()
                return raw

        def changing_open(path, mode='r', *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return ChangingStream(stream) if Path(path) == source and mode == 'rb' else stream

        with patch.object(Path, 'open', changing_open), \
             patch('collector.open', changing_open, create=True):
            yield reads

    def test_atomic_journal_replacement_keeps_published_campaign_and_refreshes(self):
        self.metadata({'status': 'partial', 'n_produced': 4})
        journal = self.plan_dir / 'control/journal.json'
        replacement = journal.with_suffix('.next')
        updated = json.loads(journal.read_text())
        updated['attempts'][0]['jobs'][1]['id'] = '902'
        self.dump(replacement, updated)
        config = self.root / 'config.json'
        output = self.root.parent / (self.root.name + '.snapshot.json')
        self.addCleanup(lambda: output.unlink(missing_ok=True))
        self.dump(config, {'campaigns': [dict(self.config, kind='bindocracy_generation')]})
        with self.changing_json_read(journal, lambda: os.replace(replacement, journal)) as reads, \
             patch.object(sys, 'argv', ['collector', '--config', str(config), '--output', str(output), '--offline']), \
             patch.object(collector, 'require_cpu_job', side_effect=AssertionError('CPU allocation requested')), \
             patch.object(collector, 'bounded_command', side_effect=AssertionError('scheduler queried')):
            self.assertEqual(collector.main(), 0)
        snapshot = json.loads(output.read_text())
        self.assertEqual(snapshot['errors'], [])
        cell = snapshot['campaigns'][0]['stages'][0]['components'][0]
        self.assertEqual((cell['reported_produced'], cell['job_ids']), (4, ['901']))
        self.assertEqual(reads, [collector.MANIFEST_LIMIT + 1])
        self.assertFalse(replacement.exists())
        _, _, refreshed = self.collect()
        self.assertEqual(refreshed['job_ids'], ['902'])

    def test_json_reads_preserve_bounds_and_strict_parsing(self):
        source = self.root / 'bounded.json'
        raw = b'{"value":1}'
        source.write_bytes(raw)
        self.assertEqual(collector.load_json(source, len(raw)), {'value': 1})
        with self.changing_json_read(source, lambda: None) as reads, \
             self.assertRaisesRegex(ValueError, 'bounded regular file'):
            collector.load_json(source, len(raw) - 1)
        self.assertEqual(reads, [])
        for invalid in (b'{"value":', b'{"value":NaN}', b'{"value":1,"value":2}', b'"\xff"'):
            with self.subTest(raw=invalid):
                source.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    collector.load_json(source)
        fifo = self.root / 'not-regular.json'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'bounded regular file'):
            collector.load_json(fifo)

    def test_json_reads_reject_in_place_changes(self):
        source = self.root / 'changing.json'
        original = b'{"value":1}'
        for raw in (b'{"value":2}', b'{}', original + b' '):
            with self.subTest(raw=raw):
                source.write_bytes(original)
                before = source.stat()

                def change(raw=raw, before=before):
                    source.write_bytes(raw)
                    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 40_000_000))

                with self.changing_json_read(source, change) as reads, \
                     self.assertRaises(ValueError):
                    collector.load_json(source, len(original))
                self.assertEqual(reads, [len(original) + 1])

    def test_json_reads_reject_ctime_change_when_mtime_is_restored(self):
        source = self.root / 'changing.json'
        source.write_bytes(b'{"value":1}')
        before = source.stat()
        # Model the metadata transition explicitly: fast writes can share a clock tick.
        after = SimpleNamespace(st_size=before.st_size, st_mtime_ns=before.st_mtime_ns,
                                st_ctime_ns=before.st_ctime_ns + 1, st_nlink=before.st_nlink)
        with self.changing_json_read(source, lambda: source.write_bytes(b'{"value":2}')), \
             patch('collector.os.fstat', side_effect=[before, after]), \
             self.assertRaises(ValueError):
            collector.load_json(source)

    def test_actual_plan_journal_mapping_and_duplicate_attempts(self):
        self.metadata({'status': 'partial', 'n_produced': 4}, duplicate=True)
        campaign, result, cell = self.collect({
            ('901', -1): {'state': 'FAILED', 'exit': '1:0'},
            ('902', -1): {'state': 'RUNNING', 'current': True, 'raw_job_id': '902'}})
        self.assertEqual(campaign.job_ids, {'901', '902'})
        self.assertEqual(campaign.gpu_keys(), {('901', -1), ('902', -1)})
        self.assertEqual(cell['reported_produced'], 4)
        self.assertEqual(cell['tasks_observed'], 1)
        self.assertEqual(cell['partial_jobs'], 1)
        self.assertEqual(cell['failed_jobs'], 1)
        self.assertEqual(cell['failed_by_state'], {'FAILED': 1})
        self.assertEqual(cell['running_jobs'], 1)
        self.assertEqual(cell['active_gpus'], 1)
        self.assertEqual(cell['job_ids'], ['901', '902'])
        self.assertIsNone(cell['valid'])
        self.assertIsNone(cell['coordinate_ready'])
        self.assertIsNone(result['stages'][1]['reported_produced'])

    def test_exit_zero_does_not_create_counts(self):
        self.metadata({'status': 'succeeded', 'exit_code': 0})
        _, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertIsNone(cell['reported_produced'])
        self.assertEqual(cell['state'], 'unknown')
        self.assertEqual(cell['counts_reported_tasks'], 0)

    def test_failure_with_reported_target_is_not_validated_success(self):
        self.metadata({'status': 'succeeded', 'n_produced': 14})
        _, _, cell = self.collect({('901', -1): {'state': 'FAILED', 'exit': '1:0'}})
        self.assertEqual(cell['reported_produced'], 14)
        self.assertEqual(cell['failed_jobs'], 1)
        self.assertEqual(cell['scheduler_warning_jobs'], 0)
        self.assertEqual(cell['state'], 'needs_attention')

    def test_collected_task_summary_fills_only_missing_reported_count(self):
        self.metadata({'status': 'succeeded', 'exit_code': 0})
        self.dump(self.run_dir / 'collected.json', {'collected': {'run': {
            'run_id': 'run-uuid', 'n_produced': 14,
            'count_details': {'tasks': {'0000': {'status': 'succeeded', 'n_produced': 14}}}}, 'designs': []}})
        campaign, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual(cell['reported_produced'], 14)
        self.assertEqual(cell['phase'], 'generated')
        self.assertNotIn('designs', campaign.cache[self.run_dir / 'collected.json'][1])

    def test_large_collection_counts_require_current_compact_summary(self):
        self.metadata({'status': 'succeeded', 'exit_code': 0})
        source = self.run_dir / 'collected.json'
        self.dump(source, {'native_inventory': 'x' * (2 * 1024 * 1024)})
        info = source.stat()
        self.dump(self.run_dir / 'collected-summary.json', {
            'schema_version': 1,
            'source': {'name': source.name, 'size_bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                       'ctime_ns': info.st_ctime_ns, 'inode': info.st_ino},
            'collected': {'run': {'run_id': 'run-uuid', 'n_produced': 14, 'status': 'succeeded',
                'count_details': {'tasks': {'0000': {'status': 'succeeded', 'n_produced': 14}}}}}})
        records = {('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}}
        _, _, cell = self.collect(records)
        self.assertEqual((cell['reported_produced'], cell['phase']), (14, 'generated'))
        with source.open('a') as handle:
            handle.write('\n')
        _, _, cell = self.collect(records)
        self.assertIsNone(cell['reported_produced'])
        self.assertEqual(cell['state'], 'unknown')

    def canonical_collection(self, count=14, outcome='succeeded', compact=False):
        source = self.run_dir / 'collected.json'
        run = {
            'run_id': 'run-uuid', 'n_produced': count, 'status': outcome,
            'count_details': {'tasks': {'0000': {'status': outcome, 'n_produced': None}}},
        }
        self.dump(source, {
            'bundle_version': 1, 'collected': {
                'run': run, 'designs': [{'unread_inventory': 'x' * (2 * 1024 * 1024)}],
            },
        })
        if compact:
            info = source.stat()
            self.dump(self.run_dir / 'collected-summary.json', {
                'schema_version': 1,
                'source': {
                    'name': source.name, 'size_bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                    'ctime_ns': info.st_ctime_ns, 'inode': info.st_ino,
                },
                'collected': {'run': run},
            })
        return source

    def test_timestamp_only_touch_falls_back_without_losing_run_total(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.canonical_collection(compact=True)
        records = {('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}}
        campaign, _, cell = self.collect(records)
        self.assertEqual((cell['reported_produced'], cell['state']), (14, 'complete'))
        before = source.stat()
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 40_000_000))
        campaign.prepare()
        cell = campaign.collect(records, None)['stages'][0]['components'][0]
        self.assertEqual((cell['reported_produced'], cell['state']), (14, 'complete'))
        self.assertEqual(cell['counts_reported_tasks'], 0)
        self.assertIsNone(campaign.tasks[0]['status']['n_produced'])
        self.assertEqual(source.stat().st_size, before.st_size)
        self.assertNotIn('designs', campaign.cache[source][1])

    def test_stale_summary_and_cached_prefix_never_mask_changed_actual_count(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.canonical_collection(compact=True)
        info = source.stat()
        os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 40_000_000))
        records = {('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}}
        campaign, _, cell = self.collect(records)
        self.assertEqual(cell['reported_produced'], 14)
        size = source.stat().st_size
        self.canonical_collection(count=19)
        self.assertEqual(source.stat().st_size, size)
        campaign.prepare()
        cell = campaign.collect(records, None)['stages'][0]['components'][0]
        self.assertEqual(cell['reported_produced'], 19)
        self.canonical_collection(count=3, outcome='partial')
        campaign.prepare()
        cell = campaign.collect(records, None)['stages'][0]['components'][0]
        self.assertEqual((cell['reported_produced'], cell['state']), (3, 'needs_attention'))
        self.assertIsNone(cell['eta_seconds'])

    def test_large_inventory_uses_bounded_prefix_and_caches_only_count_metadata(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.canonical_collection()
        opened, reads = [], []
        original_open = Path.open

        class TrackedStream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, size):
                reads.append(size)
                if size <= 0 or size > 4096:
                    raise AssertionError('unbounded collection read')
                return self.stream.read(size)

        def tracked_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path == source:
                opened.append(path)
                return TrackedStream(stream)
            return stream

        with patch.object(Path, 'open', tracked_open):
            campaign, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
            campaign.prepare()
        self.assertEqual(cell['reported_produced'], 14)
        self.assertEqual(len(opened), 1)
        self.assertEqual(sum(reads), 4096)
        self.assertLess(campaign.bytes, 20000)
        self.assertEqual(set(campaign.cache[source][1]), {'run_id', 'n_produced', 'status', 'count_details'})

    def test_run_total_is_not_added_to_known_task_counts_or_invented_for_missing_tasks(self):
        self.metadata({'status': 'succeeded', 'n_produced': 10})
        manifest = json.loads((self.run_dir / 'run.json').read_text())
        manifest['tasks'].append({'task_id': 1, 'status': 'tasks/0001/status.json', 'n_requested': 10})
        self.dump(self.run_dir / 'run.json', manifest)
        self.dump(self.run_dir / 'tasks/0001/status.json',
                  {'task_id': 1, 'status': 'succeeded', 'n_produced': None})
        self.canonical_collection(count=24, compact=True)
        campaign, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual((cell['reported_produced'], cell['counts_reported_tasks']), (24, 1))
        self.assertEqual(cell['state'], 'complete')
        self.assertIsNone(campaign.tasks[1]['status']['n_produced'])
        self.assertTrue(any('missing counts are not zero' in warning for warning in cell['warnings']))

    def test_canonical_failed_or_partial_outcome_never_becomes_success(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        for outcome in ('failed', 'partial'):
            with self.subTest(outcome=outcome):
                self.canonical_collection(count=14, outcome=outcome)
                _, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
                self.assertEqual((cell['reported_produced'], cell['state']), (14, 'needs_attention'))
                self.assertIsNone(cell['eta_seconds'])
                self.assertIsNone(cell['throughput_per_hour'])
                self.assertIsNone(cell['valid'])

    def test_canonical_total_does_not_override_missing_or_failed_scheduler_evidence(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        self.canonical_collection()
        for records, expected in (
                ({}, 'unknown'),
                ({('901', -1): {'state': 'FAILED', 'exit': '1:0'}}, 'needs_attention')):
            with self.subTest(expected=expected):
                _, _, cell = self.collect(records)
                self.assertEqual((cell['reported_produced'], cell['state']), (14, expected))
                self.assertIsNone(cell['eta_seconds'])
        (self.run_dir / 'tasks/0000/status.json').unlink()
        _, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertNotEqual(cell['state'], 'complete')
        self.assertEqual(cell['reported_produced'], 14)

    def test_prefix_bounds_and_invalid_metadata_remain_unknown(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.run_dir / 'collected.json'
        for raw in (
                '{"bundle_version":1,"collected":{"run":{"padding":"' + 'x' * (1024 * 1024),
                '{"bundle_version":2,"collected":{"run":{"run_id":"run-uuid","n_produced":14}}}',
                '{"bundle_version":1,"collected":{"run":{"run_id":"run-uuid","n_produced":14,"n_produced":99}}}',
                '{"bundle_version":1,"collected":{"run":{"run_id":"other-run","n_produced":14}}}'):
            with self.subTest(prefix=raw[:80]):
                source.write_text(raw)
                _, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
                self.assertIsNone(cell['reported_produced'])
                self.assertEqual(cell['state'], 'unknown')

    def test_collection_changed_during_prefix_read_is_not_cached(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.canonical_collection()
        original_fstat = os.fstat
        identity = source.stat()
        calls = 0

        def changing_fstat(fd):
            nonlocal calls
            info = original_fstat(fd)
            if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
                calls += 1
                if calls == 2:
                    os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 40_000_000))
                    info = original_fstat(fd)
            return info

        with patch('generation.os.fstat', side_effect=changing_fstat):
            campaign, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertIsNone(cell['reported_produced'])
        self.assertNotIn(source, campaign.cache)

    def test_prefix_reader_enforces_sweep_byte_budget(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.canonical_collection()
        campaign = GenerationCampaign(self.config)
        campaign.files, campaign.bytes, campaign.visited = 0, 0, set()
        with patch('generation.MAX_BYTES', 64), self.assertRaisesRegex(ValueError, 'sweep-byte'):
            campaign.read_collected(source)
        self.assertEqual(campaign.bytes, 64)
        self.assertNotIn(source, campaign.cache)

    def test_incremental_run_object_skips_inventory_and_rejects_invalid_sidecar(self):
        self.metadata({'status': 'succeeded', 'n_produced': None})
        source = self.run_dir / 'collected.json'
        # Delimiters and escaped quotes inside strings must not end the run object.
        self.dump(source, {'bundle_version': 1, 'collected': {'run': {
            'run_id': 'run-uuid', 'configuration': ['escaped \\" } ] {' * 700],
            'n_produced': 17, 'status': 'succeeded',
            'count_details': {'tasks': {'0000': {'n_produced': None, 'status': 'succeeded'}}},
        }, 'designs': ['x' * (2 * 1024 * 1024)]}})
        (self.run_dir / 'collected-summary.json').write_text('{invalid')
        campaign, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual((cell['reported_produced'], cell['state']), (17, 'complete'))
        self.assertGreater(campaign.cache[source][2], 4096)
        self.assertLess(campaign.cache[source][2], 32768)
        self.assertNotIn('configuration', campaign.cache[source][1])
        stale = campaign.stale('metadata disconnected')['stages'][0]['components'][0]
        self.assertEqual((stale['reported_produced'], stale['state']), (17, 'unknown'))
        self.assertIsNone(stale['eta_seconds'])

    def test_canonical_shortfall_suppresses_eta_even_with_successful_task_samples(self):
        records = self.eta_metadata()
        self.canonical_collection(count=14, outcome='succeeded')
        _, _, cell = self.collect(records)
        self.assertEqual(cell['reported_produced'], 14)
        self.assertEqual(cell['state'], 'running')
        self.assertIsNone(cell['eta_seconds'])
        self.assertIsNone(cell['throughput_per_hour'])

    def test_offline_ignores_stale_journal_running_state(self):
        self.metadata({'status': 'partial', 'n_produced': 3})
        _, _, cell = self.collect(error='Offline metadata-only snapshot')
        self.assertEqual(cell['active_gpus'], 0)
        self.assertEqual(cell['running_jobs'], 0)
        self.assertEqual(cell['unknown_jobs'], 1)
        self.assertEqual(cell['reported_produced'], 3)

    def test_changed_status_refreshes_and_symlink_escape_is_rejected(self):
        self.metadata({'status': 'partial', 'n_produced': 3})
        campaign, _, _ = self.collect()
        self.dump(self.run_dir / 'tasks/0000/status.json', {'task_id': 0, 'status': 'partial', 'n_produced': 7})
        campaign.prepare()
        self.assertEqual(campaign.collect({}, None)['stages'][0]['reported_produced'], 7)
        path = self.run_dir / 'tasks/0000/status.json'
        path.unlink()
        path.symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            campaign.prepare()

    def test_planned_18_arms_126_cells_without_bundles(self):
        arms = [f'arm{i}' for i in range(18)]
        tools = [f'tool{i}' for i in range(7)]
        groups = [arms[:1], arms[1:3], arms[3:10], arms[10:]]
        waves = [{'id': f'wave{i}', 'arms': [{'arm': arm} for arm in group]} for i, group in enumerate(groups)]
        allocation = {'arms': [{'id': arm} for arm in arms], 'generators': [{'tool': tool} for tool in tools],
            'execution': {'waves': waves}, 'total_nominal_candidate_target': 1260,
            'cells': [{'arm': arm, 'generator': tool, 'planned_candidates': 10, 'nominal_tasks': 1}
                      for arm in arms for tool in tools]}
        definition = registration(allocation, self.root,
                                  {wave['id']: self.root / wave['id'] for wave in waves}, 'Planned')
        self.dump(self.manifest_path, definition)
        _, result, _ = self.collect()
        cells = [cell for tier in result['stages'] for cell in tier.get('components', [])]
        self.assertEqual(len(cells), 126)
        self.assertEqual(len({cell['arm'] for cell in cells}), 18)
        self.assertTrue(all(cell['phase'] == 'planned' and cell['reported_produced'] is None for cell in cells))
        self.assertEqual(len(result['stages']), 10)
        self.assertTrue(all(item['planned'] is None for item in result['stages'][4:]))
        self.assertLess(len(json.dumps(result).encode()), collector.SNAPSHOT_LIMIT)

    def test_accepted_stop_is_not_compared_with_raw_candidate_count(self):
        self.metadata({'status': 'succeeded', 'n_produced': 10})
        manifest = json.loads((self.run_dir / 'run.json').read_text())
        manifest['tasks'][0]['n_requested'] = 100
        self.dump(self.run_dir / 'run.json', manifest)
        records = {('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}}
        _, _, cell = self.collect(records)
        self.assertEqual(cell['partial_jobs'], 1)
        definition = json.loads(self.manifest_path.read_text())
        definition['components'][0]['generator'] = 'freebindcraft'
        manifest['tool'] = 'freebindcraft'
        self.dump(self.manifest_path, definition)
        self.dump(self.run_dir / 'run.json', manifest)
        _, _, cell = self.collect(records)
        self.assertEqual((cell['partial_jobs'], cell['state']), (0, 'complete'))

    def independent(self, state='running', second=False):
        self.metadata({'status': 'succeeded', 'n_produced': 10})
        self.stage_control()
        self.dump(self.root / 'control/stages/tier1.json',
                  {'stage_id': 'tier1', 'state': 'failed', 'job_id': '100',
                   'error': 'Historical stage failed'})
        units = []
        generators = ['boltzgen', 'protein_hunter'] if second else ['boltzgen']
        definition = json.loads(self.manifest_path.read_text())
        for generator in generators:
            identifier = f'tier1--arm1--{generator}'
            directory = self.root / 'independent' / identifier
            status_path = self.root / 'control/components' / f'{identifier}.json'
            plan = json.loads((self.plan_dir / 'plan.json').read_text())
            plan.update(plan_dir=str(directory), digest=identifier)
            if generator != 'boltzgen':
                run_dir = self.bundle / 'output/runs/arm1/run-two'
                manifest = json.loads((self.run_dir / 'run.json').read_text())
                manifest.update(tool=generator, run_id='run-two', run_dir=str(run_dir))
                self.dump(run_dir / 'run.json', manifest)
                plan['runs'] = [{'name': generator, 'manifest': str(run_dir / 'run.json'),
                                 'resources': {'gpu': 1}}]
                definition['components'].append({
                    'tier': 'tier1', 'arm': 'arm1', 'generator': generator,
                    'planned': 10, 'tasks_planned': 1})
            else:
                journal = json.loads((self.plan_dir / 'control/journal.json').read_text())
                journal['digest'] = identifier
                self.dump(directory / 'control/journal.json', journal)
            self.dump(directory / 'plan.json', plan)
            self.dump(status_path, {
                'state': state if generator == 'boltzgen' else 'failed', 'job_id': '200',
                'native_controller_job_id': '201',
                'error': None if generator == 'boltzgen' else 'MissingOutputException: task0146'})
            units.append({
                'id': identifier, 'stage_id': 'tier1', 'arm': 'arm1', 'generator': generator,
                'plan': str(directory / 'plan.json'), 'status_path': str(status_path),
                'database': str(directory / 'campaign.duckdb'),
                'original_plan': str(self.plan_dir / 'plan.json')})
        self.dump(self.manifest_path, definition)
        self.dump(self.root / 'control/independent.json', {
            'schema_version': 1, 'mode': 'independent_generators', 'units': units,
            'supervisors': {'boltzgen': {'job_id': '200', 'state': 'running'}}})
        return units

    def recovery(self, **changes):
        unit = self.independent(state='failed')[0]
        path = self.root / 'control/recovery/components' / f"{unit['id']}.json"
        entry = {key: unit[key] for key in ('id', 'stage_id', 'arm', 'generator')}
        entry.update(target=10, status_path=str(path))
        self.dump(self.root / 'control/recovery/index.json', {
            'schema_version': 1, 'campaign_root': str(self.root), 'units': [entry]})
        status = {key: entry[key] for key in ('id', 'stage_id', 'arm', 'generator', 'target')}
        status.update(schema_version=1, state='running', retained_raw_candidates=12,
                      completed_batches=1, active_job_ids=['902'], error=None,
                      updated_at=datetime.now(timezone.utc).isoformat(),
                      count_basis='original plus terminal recovery batches')
        status.update(changes)
        self.dump(path, status)
        return path, status

    def test_recovery_replaces_old_count_and_failure_and_joins_gpu_accounting(self):
        self.recovery()
        campaign, result, cell = self.collect({
            ('901', -1): {'state': 'FAILED', 'exit': '1:0'},
            ('902', -1): {'state': 'RUNNING', 'current': True}})
        self.assertEqual((cell['state'], cell['phase']), ('running', 'recovery_running'))
        self.assertEqual((cell['reported_produced'], cell['planned']), (12, 10))
        self.assertEqual(result['stages'][0]['reported_produced'], 12)
        self.assertEqual((cell['running_jobs'], cell['active_gpus'], cell['failed_jobs']), (1, 1, 0))
        self.assertEqual(cell['job_ids'], ['902'])
        self.assertIn('902', campaign.job_ids)
        self.assertIn(('902', -1), campaign.gpu_keys())
        self.assertIsNone(cell['eta_seconds'])
        self.assertFalse(any('Independent unit' in warning for warning in cell['warnings']))
        self.assertTrue(any('12/10' in warning and '902' in warning for warning in cell['warnings']))

    def test_live_original_jobs_remain_visible_while_recovery_waits(self):
        self.recovery(state='planned')
        _, result, cell = self.collect({
            ('902', -1): {'state': 'RUNNING', 'current': True}})
        self.assertEqual(cell['state'], 'running')
        self.assertEqual(result['stages'][0]['state'], 'running')
        self.assertEqual((cell['running_jobs'], cell['active_gpus']), (1, 1))

    def test_pending_canary_is_queued_not_running(self):
        self.recovery(state='qualifying')
        _, _, cell = self.collect({
            ('902', -1): {'state': 'PENDING', 'current': True}})
        self.assertEqual(cell['state'], 'queued')
        self.assertEqual((cell['pending_jobs'], cell['active_gpus']), (1, 0))

    def test_recovery_complete_is_durable_with_overshoot(self):
        self.recovery(state='complete', active_job_ids=[],
                      updated_at='2020-01-01T00:00:00+00:00')
        _, result, cell = self.collect()
        self.assertEqual((cell['state'], cell['reported_produced']), ('complete', 12))
        self.assertEqual(result['stages'][0]['state'], 'complete')
        self.assertEqual(cell['phase'], 'recovery_complete')

    def test_recovery_stale_or_contradictory_status_never_completes(self):
        for changes in (
            {'updated_at': '2020-01-01T00:00:00+00:00'},
            {'updated_at': '2026-01-01T00:00:00'},
            {'updated_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
            {'state': 'complete', 'active_job_ids': ['902']},
            {'state': 'complete', 'active_job_ids': [], 'retained_raw_candidates': 9},
            {'state': 'complete', 'active_job_ids': [], 'error': 'not collected'},
            {'target': 11}, {'id': 'another-unit'}, {'retained_raw_candidates': -1},
            {'active_job_ids': ['902_1']}, {'active_job_ids': ['902', '902']},
            {'schema_version': 2}, {'count_basis': ''}, {'completed_batches': -1},
        ):
            with self.subTest(changes=changes):
                self.recovery(**changes)
                _, result, cell = self.collect()
                self.assertEqual(cell['state'], 'unknown')
                self.assertEqual(cell['phase'], 'recovery_unknown')
                self.assertNotEqual(result['stages'][0]['state'], 'complete')
                self.assertTrue(any('Recovery' in warning for warning in cell['warnings']))

    def test_recovery_missing_malformed_or_escaped_status_is_unknown(self):
        for mode in ('missing', 'malformed', 'escape'):
            with self.subTest(mode=mode):
                path, _ = self.recovery(state='complete', active_job_ids=[])
                if mode == 'missing':
                    path.unlink()
                elif mode == 'malformed':
                    path.write_text('{bad json')
                else:
                    path.unlink()
                    path.symlink_to('/etc/passwd')
                _, _, cell = self.collect()
                self.assertEqual(cell['state'], 'unknown')
                path.unlink(missing_ok=True)

    def test_recovery_index_validates_root_identity_bounds_and_status_location(self):
        self.recovery()
        path = self.root / 'control/recovery/index.json'
        valid = json.loads(path.read_text())
        variants = [
            dict(valid, campaign_root='/elsewhere'), dict(valid, schema_version=2),
            dict(valid, units=valid['units'] * 2),
            dict(valid, units=valid['units'] * 257),
        ]
        for field, value in (('id', 'other'), ('generator', 'other'), ('target', 11),
                             ('status_path', '/etc/passwd'),
                             ('status_path', str(self.root / 'unrelated.json'))):
            variants.append(dict(valid, units=[dict(valid['units'][0], **{field: value})]))
        for invalid in variants:
            with self.subTest(invalid=invalid):
                self.dump(path, invalid)
                with self.assertRaises((ValueError, OSError)):
                    self.collect()

    def test_recovery_metadata_refresh_and_removal_restore_original_view(self):
        path, status = self.recovery()
        campaign, _, _ = self.collect()
        self.dump(path, dict(status, state='complete', active_job_ids=[], retained_raw_candidates=15))
        campaign.prepare()
        cell = campaign.collect({}, None)['stages'][0]['components'][0]
        self.assertEqual((cell['state'], cell['reported_produced']), ('complete', 15))
        (self.root / 'control/recovery/index.json').unlink()
        campaign.prepare()
        cell = campaign.collect({}, None)['stages'][0]['components'][0]
        self.assertEqual((cell['state'], cell['reported_produced']), ('needs_attention', 10))
        self.assertNotIn('902', campaign.job_ids)

    def test_recovery_qualifying_blocked_and_scheduler_unknown_are_explicit(self):
        for state, expected in (('planned', 'not_submitted'), ('qualifying', 'running'),
                                ('blocked', 'needs_attention'), ('failed', 'needs_attention')):
            with self.subTest(state=state):
                self.recovery(state=state, active_job_ids=[], error='canary error' if state == 'blocked' else None)
                _, _, cell = self.collect()
                self.assertEqual((cell['state'], cell['phase']), (expected, f'recovery_{state}'))
                if state == 'blocked':
                    self.assertTrue(any('canary error' in warning for warning in cell['warnings']))
        self.recovery()
        _, _, cell = self.collect(error='scheduler offline')
        self.assertEqual((cell['state'], cell['running_jobs'], cell['active_gpus']), ('unknown', 0, 0))
        self.assertEqual(cell['reported_produced'], 12)

    def test_recovery_bounds_and_malformed_index_fail_closed(self):
        path, status = self.recovery(state='complete', active_job_ids=[])
        self.dump(path, dict(status, padding='x' * 65536))
        _, _, cell = self.collect()
        self.assertEqual(cell['state'], 'unknown')
        self.dump(path, dict(status, active_job_ids=[str(number) for number in range(1025)]))
        _, _, cell = self.collect()
        self.assertEqual(cell['state'], 'unknown')
        index = self.root / 'control/recovery/index.json'
        for invalid in (None, [], {'schema_version': 1, 'padding': 'x' * (1024 * 1024)}):
            self.dump(index, invalid)
            with self.assertRaises(ValueError):
                self.collect()

    def test_recovery_does_not_scan_and_leaves_other_units_unchanged(self):
        path, _ = self.recovery(state='complete', active_job_ids=[])
        self.independent(state='failed', second=True)
        with patch.object(Path, 'glob', side_effect=AssertionError('directory glob')), \
             patch.object(Path, 'rglob', side_effect=AssertionError('recursive scan')), \
             patch.object(Path, 'iterdir', side_effect=AssertionError('directory scan')):
            campaign, result, cell = self.collect()
        self.assertEqual(cell['state'], 'complete')
        other = result['stages'][0]['components'][1]
        self.assertEqual((other['state'], other['phase']), ('needs_attention', 'blocked'))
        self.assertIn(path, campaign.visited)
        self.assertLess(campaign.files, 40)

    def test_running_shared_supervisor_does_not_start_queued_unit(self):
        units = self.independent(state='queued')
        unit = units[0]
        unit.pop('original_plan')
        self.dump(self.root / 'control/independent.json', {
            'schema_version': 1, 'mode': 'independent_generators', 'units': units})
        self.dump(Path(unit['status_path']), {'state': 'queued', 'job_id': '200'})
        self.dump(Path(unit['plan']).parent / 'control/journal.json',
                  {'digest': unit['id'], 'attempts': []})
        (self.run_dir / 'tasks/0000/status.json').unlink()
        _, result, cell = self.collect({
            ('200', -1): {'state': 'RUNNING', 'current': True}})
        self.assertEqual((cell['state'], cell['phase']), ('queued', 'queued'))
        self.assertEqual((cell['active_gpus'], cell['running_jobs']), (0, 0))
        self.assertEqual(result['stages'][0]['phase'], 'queued')

    def test_independent_failure_isolated_with_original_history_deduplicated(self):
        self.independent(second=True)
        campaign, result, cell = self.collect({
            ('901', -1): {'state': 'RUNNING', 'current': True},
            ('200', -1): {'state': 'RUNNING', 'current': True},
            ('201', -1): {'state': 'RUNNING', 'current': True},
            ('100', -1): {'state': 'FAILED', 'exit': '1:0'}})
        tier = result['stages'][0]
        failed = tier['components'][1]
        self.assertEqual((tier['state'], tier['phase']), ('running', 'generating'))
        self.assertEqual((cell['state'], cell['active_gpus'], cell['tasks_observed']), ('running', 1, 1))
        self.assertEqual((failed['state'], failed['phase']), ('needs_attention', 'blocked'))
        self.assertEqual(failed['failed_jobs'], 0)  # A CPU failure is not a failed GPU worker.
        self.assertTrue(any('MissingOutputException' in warning for warning in failed['warnings']))
        self.assertTrue(any('protein_hunter' in warning for warning in tier['warnings']))
        self.assertFalse(any('Historical stage failed' in warning for warning in cell['warnings']))
        self.assertEqual(cell['reported_produced'], 10)
        self.assertEqual(cell['job_ids'], ['901'])
        self.assertEqual(campaign.gpu_keys(), {('901', -1)})
        self.assertEqual(campaign.job_ids, {'200', '201', '901'})
        self.assertEqual((tier['tasks_observed'], tier['active_gpus']), (2, 1))

    def test_independent_failure_retains_counts_even_when_workers_succeed(self):
        self.independent(state='failed')
        _, result, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual((cell['state'], cell['phase'], cell['reported_produced']),
                         ('needs_attention', 'partial', 10))
        self.assertEqual(result['stages'][0]['state'], 'needs_attention')

    def test_independent_complete_without_task_markers_is_not_success(self):
        self.independent(state='complete')
        (self.run_dir / 'tasks/0000/status.json').unlink()
        _, _, cell = self.collect({
            ('901', -1): {'state': 'COMPLETED', 'exit': '0:0'},
            ('200', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertNotEqual(cell['state'], 'complete')
        self.assertIsNone(cell['reported_produced'])

    def test_missing_independent_status_does_not_fall_back_to_failed_stage(self):
        units = self.independent()
        Path(units[0]['status_path']).unlink()
        _, _, cell = self.collect({('901', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual(cell['state'], 'unknown')
        self.assertTrue(any('status unavailable' in warning for warning in cell['warnings']))
        self.assertFalse(any('Historical stage failed' in warning for warning in cell['warnings']))

    def test_independent_sources_are_bounded_and_root_confined(self):
        units = self.independent()
        index = self.root / 'control/independent.json'
        self.dump(index, {'schema_version': 1, 'mode': 'independent_generators', 'units': units * 2})
        with patch('generation.MAX_COMPONENTS', 1), self.assertRaisesRegex(ValueError, 'unit-count bound'):
            self.collect()
        units[0]['plan'] = '/etc/passwd'
        self.dump(index, {'schema_version': 1, 'mode': 'independent_generators', 'units': units})
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.collect()
        units[0]['plan'] = str(self.plan_dir / 'plan.json')
        self.dump(index, {'schema_version': 1, 'mode': 'independent_generators',
                          'units': units, 'padding': 'x' * (1024 * 1024)})
        with self.assertRaises(ValueError):
            self.collect()

    def test_independent_duplicate_units_and_missing_coverage_rejected(self):
        units = self.independent()
        index = self.root / 'control/independent.json'
        for invalid in (units * 2, []):
            self.dump(index, {'schema_version': 1, 'mode': 'independent_generators', 'units': invalid})
            with self.assertRaises(ValueError):
                self.collect()

    def test_independent_history_only_workers_and_new_attempts_share_counts(self):
        units = self.independent(state='complete')
        plan_dir = Path(units[0]['plan']).parent
        self.dump(plan_dir / 'control/journal.json', {
            'digest': units[0]['id'], 'attempts': [{'jobs': [
                {'role': 'worker', 'id': '902', 'run': 'arm1-boltzgen', 'task': 0},
                {'role': 'controller', 'id': '201'}]}]})
        campaign, _, cell = self.collect({
            ('901', -1): {'state': 'COMPLETED', 'exit': '0:0'},
            ('902', -1): {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual(set(cell['job_ids']), {'901', '902'})
        self.assertEqual((cell['reported_produced'], cell['tasks_observed']), (10, 1))
        self.assertEqual(campaign.gpu_keys(), {('901', -1), ('902', -1)})
        self.assertEqual(cell['run_ids'], ['run-uuid'])
        self.assertEqual(cell['state'], 'complete')

    def test_completed_unit_not_poisoned_by_later_supervisor_failure(self):
        self.independent(state='complete')
        records = {('901', -1): {'state': 'COMPLETED', 'exit': '0:0'},
                   ('200', -1): {'state': 'FAILED', 'exit': '1:0'},
                   ('201', -1): {'state': 'COMPLETED', 'exit': '0:0'}}
        _, _, cell = self.collect(records)
        self.assertEqual(cell['state'], 'complete')
        records[('201', -1)] = {'state': 'FAILED', 'exit': '1:0'}
        _, _, cell = self.collect(records)
        self.assertEqual(cell['state'], 'needs_attention')

    def test_independent_component_status_size_and_sweep_bounds(self):
        units = self.independent()
        campaign = GenerationCampaign(self.config)
        with patch('generation.MAX_FILES', 2), self.assertRaisesRegex(ValueError, 'file-count'):
            campaign.prepare()
        with patch('generation.MAX_BYTES', 100), self.assertRaisesRegex(ValueError, 'sweep-byte'):
            campaign.prepare()
        self.dump(Path(units[0]['status_path']), {'state': 'failed', 'error': 'x' * 65536})
        with self.assertRaises(ValueError):
            campaign.prepare()

    def test_independent_production_retains_fourteen_legacy_qualification_cells(self):
        units = self.independent(second=True)
        definition = json.loads(self.manifest_path.read_text())
        bundle = self.root / 'qualification'
        arms = ['site_16', 'site_18']
        control_path = self.root / 'control/stages/qualification.json'
        definition['tiers'].insert(0, {
            'id': 'qualification', 'name': 'Qualification', 'bundle_root': str(bundle),
            'arms': arms, 'control_status': str(control_path)})
        self.dump(control_path, {'stage_id': 'qualification', 'state': 'complete', 'job_id': '500'})
        records = {
            ('500', -1): {'state': 'COMPLETED', 'exit': '0:0'},
            ('200', -1): {'state': 'RUNNING', 'current': True},
            ('201', -1): {'state': 'RUNNING', 'current': True},
            ('901', -1): {'state': 'RUNNING', 'current': True}}
        arm_controls = {}
        for arm_index, arm in enumerate(arms):
            plan_dir = bundle / 'control/plans' / arm / 'generate'
            runs, jobs = [], []
            for tool_index in range(7):
                tool = f'tool{tool_index}'
                run_id = f'qualified-{arm}-{tool}'
                run_dir = bundle / 'output/runs' / arm / run_id
                job_id = str(510 + 7 * arm_index + tool_index)
                definition['components'].append({
                    'tier': 'qualification', 'arm': arm, 'generator': tool,
                    'planned': 1, 'tasks_planned': 1})
                self.dump(run_dir / 'run.json', {
                    'run_id': run_id, 'tool': tool, 'kind': 'generate', 'arm': arm,
                    'run_dir': str(run_dir), 'resources': {'gpu': 1},
                    'tasks': [{'task_id': 0, 'status': 'tasks/0000/status.json', 'n_requested': 1}]})
                self.dump(run_dir / 'tasks/0000/status.json',
                          {'task_id': 0, 'status': 'succeeded', 'n_produced': 1})
                runs.append({'name': run_id, 'manifest': str(run_dir / 'run.json')})
                jobs.append({'role': 'worker', 'id': job_id, 'run': run_id, 'task': 0})
                records[(job_id, -1)] = {'state': 'COMPLETED', 'exit': '0:0'}
            self.dump(plan_dir / 'plan.json', {
                'plan_dir': str(plan_dir), 'digest': arm, 'runs': runs})
            self.dump(plan_dir / 'control/journal.json', {
                'digest': arm, 'attempts': [{'jobs': jobs}]})
            arm_controls[arm] = {'state': 'complete', 'plan': str(plan_dir / 'plan.json')}
        self.dump(bundle / 'control/status.json', {'stages': {'generate': arm_controls}})
        self.dump(self.manifest_path, definition)
        campaign, result, _ = self.collect(records)
        qualification, production = result['stages'][:2]
        self.assertEqual((qualification['id'], qualification['state'], qualification['reported_produced']),
                         ('qualification', 'complete', 14))
        self.assertEqual(len(qualification['components']), 14)
        self.assertEqual(len(campaign.component_controls), 2)
        self.assertEqual((production['state'], production['active_gpus']), ('running', 1))
        self.assertTrue(all(not any('Independent unit' in warning for warning in cell['warnings'])
                            for cell in qualification['components']))
        # A fake qualification unit must not replace immutable legacy evidence.
        forged = dict(units[0], id='qualification--site_16--tool0', stage_id='qualification',
                      arm='site_16', generator='tool0')
        self.dump(self.root / 'control/independent.json', {
            'schema_version': 1, 'mode': 'independent_generators', 'units': [*units, forged]})
        with self.assertRaisesRegex(ValueError, 'unregistered independent unit'):
            self.collect(records)

    def stage_control(self):
        definition = json.loads(self.manifest_path.read_text())
        path = self.root / 'control/stages/tier1.json'
        definition['tiers'][0]['control_status'] = str(path)
        self.dump(self.manifest_path, definition)
        self.dump(path, {'stage_id': 'tier1', 'state': 'queued', 'job_id': '100',
                         'dependency_job_id': '99'})

    def test_dependency_queue_and_cpu_activity_are_not_worker_allocations(self):
        self.stage_control()
        campaign, result, cell = self.collect({
            ('100', -1): {'state': 'PENDING', 'current': True}})
        self.assertEqual(campaign.job_ids, {'100'})
        self.assertEqual(campaign.gpu_keys(), set())
        self.assertEqual((cell['state'], cell['phase']), ('queued', 'waiting_dependency'))
        self.assertEqual(result['stages'][0]['phase'], 'waiting_dependency')
        self.assertEqual((cell['pending_jobs'], cell['active_gpus']), (0, 0))
        _, _, cell = self.collect({('100', -1): {'state': 'RUNNING', 'current': True}})
        self.assertEqual((cell['state'], cell['phase']), ('running', 'preparing'))
        self.assertEqual((cell['running_jobs'], cell['active_gpus']), (0, 0))
        _, _, cell = self.collect(error='Scheduler unavailable')
        self.assertEqual((cell['state'], cell['phase']), ('unknown', 'unknown'))

    def test_failed_stage_preserves_live_workers_and_partial_output(self):
        self.metadata({'status': 'partial', 'n_produced': 3})
        self.stage_control()
        _, result, cell = self.collect({
            ('100', -1): {'state': 'TIMEOUT', 'exit': '0:0'},
            ('901', -1): {'state': 'RUNNING', 'current': True}})
        self.assertEqual((result['stages'][0]['state'], result['stages'][0]['phase']),
                         ('needs_attention', 'blocked'))
        self.assertEqual((cell['running_jobs'], cell['active_gpus'], cell['reported_produced']), (1, 1, 3))
        self.assertEqual(cell['state'], 'running')
        self.assertIsNone(cell['eta_seconds'])

    def test_control_identity_mismatch_is_not_attributed_to_registered_stage(self):
        self.stage_control()
        self.dump(self.root / 'control/stages/tier1.json',
                  {'stage_id': 'different-stage', 'state': 'running', 'job_id': '100'})
        with self.assertRaisesRegex(ValueError, 'identity/state'):
            self.collect()

    def test_scheduler_batches_array_and_individual_allocations(self):
        outputs = ['901|FAILED|1:0|12|start|end\n800_2|RUNNING|0:0|8|start|end\n901.batch|FAILED|1:0|12|start|end\n',
                   '800_2|RUNNING|00:08|node|802\n902|RUNNING|00:03|node|902\n']
        with patch.object(collector, 'bounded_command', side_effect=outputs) as command:
            records = collector.scheduler_snapshot({'800', '901', '902'})
        self.assertEqual(command.call_count, 2)
        self.assertEqual(set(records), {('800', 2), ('901', -1), ('902', -1)})
        self.assertEqual(records[('902', -1)]['raw_job_id'], '902')

    def test_gpu_sampling_deduplicates_physical_allocation(self):
        class Campaign:
            root = Path('/test')
            def gpu_keys(self):
                return {('800', 2), ('802', -1)}
        records = {key: {'state': 'RUNNING', 'current': True, 'raw_job_id': '802'}
                   for key in Campaign().gpu_keys()}
        with patch.object(collector, 'bounded_command', return_value=('802.batch|gres/gpuutil=50,gres/gpumem=1024M\n', None)) as command:
            result = collector.gpu_snapshot([Campaign()], records, None)
        self.assertEqual(command.call_count, 1)
        self.assertEqual(result[Path('/test')]['eligible_jobs'], 1)
        self.assertEqual(result[Path('/test')]['sampled_jobs'], 1)

    def eta_metadata(self, samples=5, count=10):
        self.metadata()
        definition = json.loads(self.manifest_path.read_text())
        definition['components'][0].update(planned=100, tasks_planned=10)
        self.dump(self.manifest_path, definition)
        manifest_path = self.run_dir / 'run.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['tasks'] = [{'task_id': i, 'status': f'tasks/{i:04d}/status.json', 'n_requested': 10}
                             for i in range(samples + 2)]
        self.dump(manifest_path, manifest)
        jobs, records = [], {}
        for i in range(samples + 2):
            job_id = str(901 + i)
            jobs.append({'role': 'worker', 'id': job_id, 'run': 'arm1-boltzgen', 'task': i})
            if i < samples:
                self.dump(self.run_dir / f'tasks/{i:04d}/status.json',
                          {'task_id': i, 'status': 'succeeded', 'n_produced': count})
                records[(job_id, -1)] = {'state': 'COMPLETED', 'exit': '0:0', 'elapsed': 100}
            else:
                records[(job_id, -1)] = {'state': 'RUNNING', 'current': True,
                                        'elapsed': 40 if i == samples else 20, 'raw_job_id': job_id}
        self.dump(self.plan_dir / 'control/journal.json', {'digest': 'frozen', 'attempts': [{'jobs': jobs}]})
        return records

    def test_execution_eta_requires_five_samples_and_subtracts_elapsed(self):
        records = self.eta_metadata()
        _, _, cell = self.collect(records)
        self.assertEqual(cell['eta_sample_count'], 5)
        self.assertEqual(cell['eta_seconds'], 220)
        self.assertEqual(cell['throughput_per_hour'], 720)
        self.assertIn('not output quota', cell['eta_basis'])
        self.assertIn('excludes queue wait', cell['eta_basis'])
        # A fourth same-cell success cannot be supplemented by another component.
        records[('905', -1)] = {'state': 'PENDING', 'current': True}
        (self.run_dir / 'tasks/0004/status.json').unlink()
        _, _, cell = self.collect(records)
        self.assertEqual(cell['eta_sample_count'], 4)
        self.assertIsNone(cell['eta_seconds'])

    def test_execution_eta_missing_counts_keeps_throughput_unknown(self):
        records = self.eta_metadata(count=None)
        _, _, cell = self.collect(records)
        self.assertEqual(cell['eta_seconds'], 220)
        self.assertIsNone(cell['reported_produced'])
        self.assertIsNone(cell['throughput_per_hour'])

    def test_execution_eta_blocked_by_failures_unknown_or_shortfall(self):
        for blocker in ('failure', 'unknown', 'scheduler', 'shortfall', 'no_running'):
            with self.subTest(blocker=blocker):
                records = self.eta_metadata(count=5 if blocker == 'shortfall' else 10)
                if blocker == 'failure':
                    records[('901', -1)].update(state='FAILED', exit='1:0')
                elif blocker == 'unknown':
                    del records[('901', -1)]
                elif blocker == 'no_running':
                    for job_id in ('906', '907'):
                        records[(job_id, -1)].update(state='PENDING')
                _, _, cell = self.collect(records, 'scheduler unavailable' if blocker == 'scheduler' else None)
                self.assertIsNone(cell['eta_seconds'])
                self.assertIsNone(cell['throughput_per_hour'])

    def test_offline_cli_never_requires_or_queries_slurm(self):
        config = self.root / 'config.json'
        output = self.root.parent / (self.root.name + '.snapshot.json')
        self.addCleanup(lambda: output.unlink(missing_ok=True))
        self.dump(config, {'campaigns': [dict(self.config, kind='bindocracy_generation')]})
        with patch.object(sys, 'argv', ['collector', '--config', str(config), '--output', str(output), '--offline']), \
             patch.object(collector, 'require_cpu_job', side_effect=AssertionError('CPU allocation requested')), \
             patch.object(collector, 'bounded_command', side_effect=AssertionError('scheduler queried')):
            self.assertEqual(collector.main(), 0)
        snapshot = json.loads(output.read_text())
        self.assertIsNone(snapshot['collector']['job_id'])
        self.assertEqual(snapshot['collector']['mode'], 'metadata_only')


if __name__ == '__main__':
    unittest.main()
