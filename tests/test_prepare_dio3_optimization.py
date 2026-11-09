"""Model-free contract tests; invoke on an allocated CPU node with unittest."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_dio3_optimization as prep


class ManifestAncestry(unittest.TestCase):
    def test_approved_revision_keeps_original_anchor_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'frozen_manifest.json'
            prep.save(manifest, {'files': []})
            campaign = {'provenance': {'frozen_manifest_sha256': prep.digest(manifest)}}
            prep.save(root / 'campaign.json', campaign)
            prep._manifest_inputs(root, campaign)
            record = root / 'provenance/revisions/source-repair/revision.json'
            original = record.parent / 'frozen_manifest.before.json'
            prep.copy_checked(manifest, original)
            prep.save(manifest, {'files': [], 'approved_revisions': [{'revision_record': str(record)}]})
            revision = {'schema_version': 1, 'campaign_sha256': prep.digest(root / 'campaign.json'),
                        'original_frozen_manifest': str(original),
                        'original_frozen_manifest_sha256': prep.digest(original),
                        'frozen_manifest_sha256': prep.digest(manifest)}
            prep.save(record, revision)
            prep._manifest_inputs(root, campaign)
            for field, message in (
                    ('campaign_sha256', 'campaign changed'),
                    ('original_frozen_manifest_sha256', 'Original generation'),
                    ('frozen_manifest_sha256', 'differs from approved revision')):
                with self.subTest(field=field):
                    prep.save(record, {**revision, field: 'unverified'})
                    with self.assertRaisesRegex(ValueError, message):
                        prep._manifest_inputs(root, campaign)
            prep.save(record, revision)
            prep.save(original, {'files': ['changed']})
            with self.assertRaisesRegex(ValueError, 'Original generation'):
                prep._manifest_inputs(root, campaign)
            prep.save(manifest, {'files': ['unapproved']})
            with self.assertRaisesRegex(ValueError, 'without an approved revision'):
                prep._manifest_inputs(root, campaign)


class SamplingContract(unittest.TestCase):
    def test_family_first_reproducible_and_sequence_distinct(self):
        rows = [dict(candidate_id=f'{family}/{variant}', family_id=str(family),
                     sequence=f'{family}-{variant}')
                for family in range(100) for variant in range(4)]
        first, report = prep.select_families(rows, 20261109)
        second, _ = prep.select_families(list(reversed(rows)), 20261109)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 80)
        self.assertEqual(len({row['family_id'] for row in first}), 80)
        self.assertEqual(report['shortage'], 0)

    def test_shortage_not_filled_by_variants(self):
        rows = [dict(candidate_id=str(i), family_id='one', sequence=str(i)) for i in range(90)]
        selected, report = prep.select_families(rows, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(report['shortage'], 79)
        self.assertEqual(report['known_family_shortage'], 79)

    def test_unknown_lineage_is_sequence_proxy(self):
        rows = [dict(candidate_id=str(i), family_id=None, sequence='C' * 75) for i in range(9)]
        selected, report = prep.select_families(rows, 1)
        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0]['family_known'])
        self.assertTrue(selected[0]['sampling_family_id'].startswith('sequence-proxy:'))
        self.assertEqual(report['unknown_sequence_units'], 1)
        self.assertEqual(selected[0]['sequence'], 'C' * 75)

    def test_variant_matching_avoids_false_shortage(self):
        rows = [dict(candidate_id='a/1', family_id='a', sequence='same'),
                dict(candidate_id='a/2', family_id='a', sequence='other'),
                dict(candidate_id='b/1', family_id='b', sequence='same')]
        for seed in range(10):
            selected, _ = prep.select_families(rows, seed, quota=2)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len({row['sequence'] for row in selected}), 2)

    def test_two_trajectories_start_from_one_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = [dict(candidate_id=f'parent{i}', family_id=str(i), arm='tile_01', generator='g',
                            sequence=aa * 75) for i, aa in enumerate('ACD')]
            template = {'target': {}, 'binder': {},
                        'losses': [{'name': 'BinderTargetContact', 'weight': 1, 'params': {}}],
                        'run': {'sharp_noise': 0.1},
                        'optimizer': {'n_cycles': 3, 'soft': {'n_steps': 90}, 'sharp': {'n_steps': 30}}}
            arms = {'tile_01': {'indexing': 'target_sequence_1based', 'chain': 'A',
                               'residues': [1, 201], 'optimizer_epitope_idx_0based': [0, 200],
                               'residue_identity': ['A', 'A']}}
            original = copy.deepcopy(template)
            prep.freeze_mode(root, root, parents, arms, template, 'production')
            self.assertEqual(template, original)
            frozen = prep.load(root / 'parents.json')
            tasks = prep.load(root / 'tasks.json')
            recipe = prep.load(root / 'recipes/tile_01.json')
            sequences = prep.fasta(recipe['binder']['init_fasta'])
            by_id = {parent['candidate_id']: parent for parent in frozen}
            identifiers = set()
            for task in tasks:
                expected = by_id[task['parent_candidate_id']]['sequence']
                for trajectory in (0, 1):
                    self.assertEqual(sequences[(task['seed'] * 2 + trajectory) % len(sequences)], expected)
                self.assertEqual(len({child['parent_design_id'] for child in task['children']}), 1)
                identifiers.update(child['child_id'] for child in task['children'])
                command = prep.command_for(root, task)
                self.assertIn('--hard-sharp', command)
                self.assertEqual(command[command.index('--sharp-replicates') + 1], '2')
            self.assertEqual(len(identifiers), 6)
            self.assertEqual(recipe['losses'][0]['params']['epitope_idx'], [0, 200])
            self.assertFalse(recipe['binder']['no_cys'])
            prep.freeze_mode(root / 'qualification', root, parents[:1], arms, template, 'qualification')
            qualification = prep.load(root / 'qualification/tasks.json')
            self.assertEqual(len(qualification), 1)
            planned = qualification[0]
            original_task = tasks[planned['production_task_id']]
            for field in ('seed', 'recipe', 'children', 'output_dir', 'parent_candidate_id'):
                self.assertEqual(planned[field], original_task[field])
            self.assertEqual(prep.load(root / 'qualification/campaign.json')['resources']['max_concurrent_gpus'], 8)

    def test_all_parents_admitted_in_balanced_arrays(self):
        selected = list(range(10080))
        batches = prep.launch_batches(selected, 96)
        self.assertEqual([len(batch['task_ids']) for batch in batches], [5040, 5040])
        self.assertEqual([batch['gpu_cap'] for batch in batches], [48, 48])
        self.assertTrue(all(batch['after_batch'] is None for batch in batches))
        self.assertEqual(sorted(task for batch in batches for task in batch['task_ids']), selected)
        self.assertTrue(all(max(range(len(batch['task_ids']))) < 10000 for batch in batches))

    def test_lower_gpu_caps_and_array_boundaries(self):
        for count in (0, 18, 9999, 10000, 10001, 10062, 10080):
            for cap in (1, 2, 3, 8, 95, 96):
                batches = prep.launch_batches(list(range(count)), cap)
                self.assertEqual(sum(len(batch['task_ids']) for batch in batches), count)
                self.assertTrue(all(0 < len(batch['task_ids']) <= 10000 for batch in batches))
                self.assertLessEqual(sum(batch['gpu_cap'] for batch in batches if batch['after_batch'] is None), cap)
                if cap >= len(batches):
                    self.assertTrue(all(batch['after_batch'] is None for batch in batches))
                    self.assertLessEqual(sum(batch['gpu_cap'] for batch in batches), cap)
                elif batches:
                    self.assertEqual(batches[1]['after_batch'], 0)
                    self.assertEqual([batch['gpu_cap'] for batch in batches], [1, 1])
        with self.assertRaises(ValueError):
            prep.launch_batches([1, 1], 96)
        with self.assertRaises(ValueError):
            prep.launch_batches([1], 97)

    def test_retry_requires_terminal_unambiguous_scheduler_evidence(self):
        ledger = [{'job_id': '123', 'status': 'submitted'}]
        prep.admit_ledger(ledger, set(), {'123': ['COMPLETED', 'FAILED', 'CANCELLED']})
        prep.admit_ledger([{'job_id': None, 'status': 'not_submitted'}], set(), {})
        for active, accounting in (({'123'}, {'123': ['COMPLETED']}),
                                   (set(), {}),
                                   (set(), {'123': ['RUNNING']}),
                                   (set(), {'123': ['COMPLETED', 'PENDING']})):
            with self.assertRaises(ValueError):
                prep.admit_ledger(ledger, active, accounting)
        with self.assertRaises(ValueError):
            prep.admit_ledger([{'job_id': None, 'status': 'submitting'}], set(), {})


if __name__ == '__main__':
    prep.cpu_allocation()
    unittest.main()
