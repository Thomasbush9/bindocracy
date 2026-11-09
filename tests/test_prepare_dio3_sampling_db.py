"""CPU-only native-parser/sequence contracts; no models, jobs or production DB reads.

Parent verification on a CPU allocation with the generation environment:
  G/bindocracy/.venv/bin/python B/tests/test_prepare_dio3_sampling_db.py
Also run prepare_dio3_sampling_db.py --smoke --report <separate-path> against
stopped real production records; fixtures do not replace that parity evidence.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_dio3_sampling_db as prep


TARGET = 'AAAA'
BINDER = 'C' * 75


def put(root, path, content):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    return destination


def pdb(target=TARGET, binder=BINDER):
    names = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP'}
    lines = []
    serial = 0
    for chain, sequence in [('A', target), ('B', binder)]:
        for index, residue in enumerate(sequence, 1):
            serial += 1
            lines.append(f'ATOM  {serial:5d}  CA  {names[residue]} {chain}{index:4d}    '
                         f'{float(index):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}           C\n')
    return ''.join(lines) + 'END\n'


def expected_id(kind, *parts):
    # Independent UUID construction, not a call back into the implementation.
    return str(uuid5(NAMESPACE_URL, ':'.join(('bindocracy', kind, *parts))))


class NativeRecordParity(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.task = Path(self.directory.name) / 'tasks/task_0001'
        self.task.mkdir(parents=True)
        self.manifest = SimpleNamespace(run_id='run-parity', name='native-fixture',
                                        created_at=datetime(2026, 11, 9, tzinfo=timezone.utc))

    def record(self, generator):
        parsed, audit = prep.parse_task(self.task, generator, None)
        self.assertEqual(audit['discrepancy'], 0)
        self.assertEqual(len(parsed.candidates), 1)
        key, item = next(iter(parsed.candidates.items()))
        fast = prep.sequence_candidate(item, TARGET)
        full = prep.archive.normalize_candidate(item, TARGET)
        self.assertEqual(fast[0], full[0])
        record = prep.design_record(self.manifest, {}, self.task, key, item, fast)
        self.assertEqual(record.sequence, BINDER)
        self.assertEqual(record.design_id, expected_id('dio3-native-candidate', 'run-parity',
                                                       'task_0001', key))
        return record, item

    def test_freebindcraft_merges_models_and_preserves_rejected_metrics(self):
        put(self.task, 'bindcraft/rejected_mpnn_full_stats.csv',
            f'Design,Sequence,iptm\ntraj_mpnn1_model2_relaxed,{BINDER},0.21\n')
        put(self.task, 'bindcraft/MPNN/Sequences/traj_mpnn1.fasta', f'>variant\n{BINDER}\n')
        put(self.task, 'bindcraft/Rejected/traj_mpnn1_model3_unrelaxed.pdb', pdb())
        record, item = self.record('freebindcraft')
        self.assertEqual(record.native_id, 'task_0001:"traj_mpnn1"')
        self.assertEqual(record.metadata['family_id'], expected_id('dio3-native-family',
                                                                  'run-parity', 'task_0001', 'traj'))
        self.assertEqual(item['native_identity'], 'traj_mpnn1')
        self.assertEqual(record.metadata['intrinsic_metrics']['native_rows'][0]['row']['iptm'], '0.21')
        self.assertIsNone(record.metadata['structure_path'])
        provenance = record.metadata['source_provenance']
        self.assertFalse(provenance['artifacts_preserved'])
        self.assertEqual(provenance['coordinate_validation'], 'not_performed')
        self.assertEqual(provenance['unvalidated_structure_paths'], item['structures'])

    def test_pxdesign_scope_and_sequence_index_are_identity(self):
        put(self.task, 'global_run_1/designs/seed_2/predictions/sample_level_output.csv',
            f'name,seq_idx,sequence\nbackbone_seq3,3,{BINDER}\n')
        record, item = self.record('pxdesign')
        scope = 'global_run_1/designs/seed_2/predictions'
        self.assertEqual(item['native_identity'], [scope, 'backbone', 3])
        self.assertEqual(record.native_id, 'task_0001:' + json.dumps([scope, 'backbone', 3], separators=(',', ':')))
        self.assertEqual(record.metadata['family_id'], expected_id('dio3-native-family',
                                                                  'run-parity', 'task_0001', scope + '/backbone'))

    def test_protein_hunter_cycle_retains_run_family_and_native_metrics(self):
        put(self.task, 'summary_all_runs.csv',
            'run_id,cycle_1_seq,cycle_2_seq,cycle_3_seq,cycle_4_seq,cycle_5_seq,cycle_2_iptm\n'
            f'17,,{BINDER},,,,0.4\n')
        record, item = self.record('protein_hunter')
        self.assertEqual(item['native_identity'], [17, 2])
        self.assertEqual(record.metadata['family_id'], expected_id('dio3-native-family',
                                                                  'run-parity', 'task_0001', 'run_17'))
        self.assertEqual(record.metadata['intrinsic_metrics']['native_rows'][0]['row']['cycle_2_iptm'], '0.4')

    def test_complexa_structure_only_uses_original_coordinate_sequence(self):
        put(self.task, 'inference/pipeline/job_4/sample.pdb', pdb())
        record, item = self.record('proteina_complexa')
        self.assertEqual(item['native_identity'], ['pipeline', 'job_4', 'sample.pdb'])
        self.assertIsNone(record.metadata['family_id'])
        self.assertEqual(record.metadata['source_provenance']['coordinate_validation'], 'sequence_derivation_only')
        self.assertEqual(record.metadata['source_provenance']['chains'], {'binder': 'B', 'target': 'A'})

    def test_boltz_native_sequence_does_not_validate_coordinates(self):
        put(self.task, 'final_ranked_designs/all_designs_metrics.csv',
            f'id,designed_sequence,pass_filters\nproposal,{BINDER},False\n')
        put(self.task, 'intermediate_designs_inverse_folded/proposal.cif', 'not valid CIF\n')
        record, _ = self.record('boltzgen')
        self.assertIsNone(record.metadata['structure_path'])
        self.assertEqual(record.metadata['source_provenance']['structure_observations'], [])
        self.assertIsNone(record.metadata['family_id'])

    def test_normalization_whitespace_and_case_match_original(self):
        parsed = prep.archive.NativeTask(self.task, 'bindcraft2')
        parsed.add('parent_candidate1', sequence=' c c\n' + 'c' * 73, family='parent')
        parsed.add('parent_candidate1', sequence=BINDER, family='parent')
        item = next(iter(parsed.candidates.values()))
        self.assertEqual(prep.sequence_candidate(item, TARGET)[0], BINDER)
        self.assertEqual(prep.sequence_candidate(item, TARGET)[0],
                         prep.archive.normalize_candidate(item, TARGET)[0])

    def test_conflicting_native_sequences_never_fall_back_to_structure(self):
        coordinate = put(self.task, 'candidate.pdb', pdb())
        parsed = prep.archive.NativeTask(self.task, 'freebindcraft')
        parsed.add('variant', sequence=BINDER, structure=coordinate)
        parsed.add('variant', sequence='D' * 75)
        item = next(iter(parsed.candidates.values()))
        self.assertIsNone(prep.archive.normalize_candidate(item, TARGET)[0])
        result = prep.sequence_candidate(item, TARGET)
        record = prep.design_record(self.manifest, {}, self.task, 'variant', item, result)
        self.assertIsNone(record.sequence)
        self.assertEqual(record.status, 'invalid')

    def test_invalid_native_sequence_never_falls_back_to_structure(self):
        coordinate = put(self.task, 'candidate.pdb', pdb())
        parsed = prep.archive.NativeTask(self.task, 'freebindcraft')
        parsed.add('variant', sequence='C-CC', structure=coordinate)
        item = next(iter(parsed.candidates.values()))
        self.assertIsNone(prep.archive.normalize_candidate(item, TARGET)[0])
        result = prep.sequence_candidate(item, TARGET)
        record = prep.design_record(self.manifest, {}, self.task, 'variant', item, result)
        self.assertIsNone(record.sequence)
        self.assertEqual(record.status, 'invalid')

    def test_structure_order_matches_full_archive_not_lexicographic_order(self):
        first = put(self.task, 'z_first.pdb', pdb(binder='D' * 75))
        second = put(self.task, 'a_second.pdb', pdb())
        parsed = prep.archive.NativeTask(self.task, 'proteina_complexa')
        parsed.add('variant', structure=first)
        parsed.add('variant', structure=second)
        item = next(iter(parsed.candidates.values()))
        self.assertEqual(prep.sequence_candidate(item, TARGET), prep.archive.normalize_candidate(item, TARGET))
        self.assertEqual(prep.sequence_candidate(item, TARGET)[0], 'D' * 75)

    def test_raw_native_census_discrepancy_fails_closed(self):
        # The archive retains a header-only FASTA identity; the strict census
        # has neither a physical structure nor a sequence to count.
        put(self.task, 'bindcraft/MPNN/Sequences/empty_mpnn1.fasta', '>empty\n')
        with self.assertRaisesRegex(ValueError, 'Raw census discrepancy'):
            prep.parse_task(self.task, 'freebindcraft', None)

    def test_malformed_native_rows_are_not_silently_salvaged(self):
        put(self.task, 'bindcraft/mpnn_design_stats.csv',
            f'Design,Sequence\nvariant,{BINDER},unexpected_extra_column\n')
        with self.assertRaisesRegex(ValueError, 'Native parsing errors'):
            prep.parse_task(self.task, 'freebindcraft', None)


class FinalCensusParity(unittest.TestCase):
    def final(self):
        return {'units': [{'id': 'unit', 'stage_id': 'production', 'arm': 'site_16',
                           'generator': 'freebindcraft', 'retained_raw_candidates': 3}]}

    def test_canaries_count_in_raw_but_are_explicitly_excluded_from_production(self):
        rows = prep.reconcile(self.final(), {('production', 'site_16', 'freebindcraft'): (3, 1)}, expected_total=3)
        self.assertEqual(rows[0]['production'], 2)
        self.assertEqual(rows[0]['recovery_canary_qualification'], 1)

    def test_final_census_discrepancy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Raw census discrepancy'):
            prep.reconcile(self.final(), {('production', 'site_16', 'freebindcraft'): (2, 0)}, expected_total=3)
        with self.assertRaisesRegex(ValueError, 'Raw census discrepancy'):
            prep.reconcile(self.final(), {}, expected_total=3)

    def test_duplicate_and_unexpected_strata_fail_closed(self):
        final = self.final()
        final['units'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate final census stratum'):
            prep.reconcile(final, {('production', 'site_16', 'freebindcraft'): (3, 0)}, expected_total=6)
        with self.assertRaisesRegex(ValueError, 'absent from final census'):
            prep.reconcile(self.final(), {('production', 'site_16', 'freebindcraft'): (3, 0),
                                          ('production', 'site_17', 'freebindcraft'): (1, 0)}, expected_total=3)


if __name__ == '__main__':
    unittest.main()
