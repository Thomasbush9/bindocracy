#!/usr/bin/env python3
"""Build a complete, sequence-only DIO3 CampaignStore without preserving artifacts.

Run with G/bindocracy/.venv/bin/python in a Slurm CPU allocation (4 CPUs/16 GB):
  prepare_dio3_sampling_db.py --smoke --report /path/to/parity.json
  prepare_dio3_sampling_db.py --workers 4
The default output is R/archive/optimization_candidates.duckdb. Pass that path
unchanged to prepare_dio3_optimization.py --database. No jobs are submitted.

Uses the running archive collector's native parsers, normalizer, stable IDs and
VIEW, but never opens its database or modifies its scripts/reports. Every native
row/label and structure reference is retained in standard design metadata. No
artifacts are copied, hashed or registered as preserved. Native sequence evidence
bypasses coordinate validation; structure-only records use the original normalizer
(including its first-valid-structure sequence choice). The resulting sequences,
IDs and families match full native ingestion; coordinate status intentionally does
not. --smoke compares representative real records against full normalization.

A private, new database is published only after strict task counts and every final
production census stratum reconcile, including recovery canaries. Qualification
and canaries remain explicitly labelled/excluded by the original VIEW. Missing or
conflicting sequences fail closed, rather than silently reducing the population.
Memory is bounded by one native task plus insertion/normalization batches. Native
file discovery, CSV parsing and sequence extraction from structure-only generators
are the remaining bottlenecks; no millions-of-file preservation pass is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import islice, repeat
import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import tempfile

G = Path('/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/test_bindocracy/binder-design')
R = Path('/n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-generators-20261105')
sys.path.insert(0, str(G / 'bindocracy/src'))
sys.path.insert(0, str(G / 'scripts'))
import archive_dio3_generation as archive
from bindocracy.runs.manifest import RunManifest
from bindocracy.store.records import DesignRecord, stable_id
from bindocracy.store.store import CampaignStore

EXPECTED_RAW = 751080
BATCH_SIZE = 256
CATALOG = 'dio3-sequence-only-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sequence_candidate(item, target):
    """Delegate sequence choice/conflict handling, suppressing only unused geometry."""
    if item['sequences']:
        result = archive.normalize_candidate({**item, 'structures': []}, target)
    else:
        result = archive.normalize_candidate(item, target)
    sequence, structure, chains, observations, errors = result
    require(not item['errors'], f"Ambiguous native identity: {item['errors']}")
    return sequence, structure, chains, observations, errors


def save_report(path, value):
    """Use catalog-specific temporary names, distinct from the live archiver."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        archive.sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def design_record(manifest, source, task, key, item, result):
    """Same record identity/provenance assembly as archive.ingest, no artifact claims."""
    sequence, structure, chains, observations, errors = result
    family = (stable_id('dio3-native-family', manifest.run_id, task.name, item['family'])
              if item['family'] else None)
    provenance = {
        **source, 'task_id': task.name, 'task_path': str(task),
        'native_identity': item['native_identity'], 'native_labels': item['native_labels'],
        'raw_sequences': item['sequences'], 'native_rows': item['rows'],
        'structure_observations': observations, 'structure_paths': item['structures'],
        'unvalidated_structure_paths': [p for p in item['structures']
                                        if p not in {s['path'] for s in observations}],
        'chains': chains, 'errors': errors,
        'sampling_catalog': CATALOG, 'artifacts_preserved': False,
        'coordinate_validation': 'sequence_derivation_only' if observations else 'not_performed',
    }
    return DesignRecord(
        design_id=stable_id('dio3-native-candidate', manifest.run_id, task.name, key),
        run_id=manifest.run_id, native_id=f'{task.name}:{key}',
        candidate_type='complex' if structure else 'sequence' if sequence else 'backbone', sequence=sequence,
        status='produced' if sequence else 'invalid', created_at=manifest.created_at,
        metadata={'family_id': family, 'structure_path': structure,
                  'intrinsic_metrics': {'native_rows': item['rows'], 'native_labels': item['native_labels']},
                  'source_provenance': provenance},
    )


def source_details(path, manifest):
    return {'manifest_path': str(path), 'manifest_sha256': archive.sha256(path),
            'source_run_id': manifest.run_id, 'source_run_name': manifest.name,
            'source_run_dir': str(path.parent)}


def generation_manifests(root, arms):
    paths, _, _ = archive.discover(root)
    seen = set()
    for path in paths:
        manifest = RunManifest.read(path)
        if manifest.kind != 'generate' or manifest.tool not in archive.census.COUNT_BASIS:
            continue
        require(manifest.run_id not in seen, f'Duplicate run ID: {manifest.run_id}')
        seen.add(manifest.run_id)
        require(Path(manifest.run_dir).resolve() == path.parent.resolve(),
                f'Manifest run_dir differs from discovered path: {path}')
        yield path, manifest, archive.run_context(path, manifest, arms)


def native_tasks(path, manifest):
    planned = {task.directory: task for task in manifest.tasks}
    for task, info in sorted(archive.census._matches(path.parent, 'tasks/*')):
        if stat.S_ISDIR(info.st_mode):
            yield task, planned.get(str(task.relative_to(path.parent)))


def parse_task(task, generator, plan):
    parsed = archive.NativeTask(task, generator).parse().enrich(plan)
    require(not parsed.errors, f'Native parsing errors in {task}: {parsed.errors}')
    expected, physical, sequences = archive.census._count_task(task, generator)
    require(expected == len(parsed.candidates),
            f'Raw census discrepancy in {task}: native={len(parsed.candidates)}, census={expected}')
    return parsed, {'task': str(task), 'raw_candidates': len(parsed.candidates),
                    'generation_counts': expected, 'physical_candidate_ids': physical,
                    'sequence_bearing_candidates': sequences, 'discrepancy': 0}


def reconcile(final, observed, *, expected_total=EXPECTED_RAW):
    """Reconcile all units, even zero-count strata; canaries belong in raw census."""
    remaining = dict(observed)
    seen, rows = set(), []
    total = sum(unit['retained_raw_candidates'] for unit in final['units'])
    require(total == expected_total, f'Final raw census total {total} != {expected_total}')
    for unit in final['units']:
        key = (unit['stage_id'], unit['arm'], unit['generator'])
        require(key not in seen, f'Duplicate final census stratum: {key}')
        seen.add(key)
        actual, canaries = remaining.pop(key, (0, 0))
        expected = unit['retained_raw_candidates']
        require(actual == expected, f'Raw census discrepancy for {key}: catalog={actual}, final={expected}')
        rows.append({'unit_id': unit['id'], 'source_stage': key[0], 'arm': key[1],
                     'generator': key[2], 'expected': expected, 'catalogued': actual,
                     'production': actual - canaries, 'recovery_canary_qualification': canaries,
                     'discrepancy': 0})
    require(not remaining, f'Production strata absent from final census: {remaining}')
    return rows


def census_contract(root, campaign, arms):
    path = root / 'control/generation-final-census.json'
    final = archive.load(path)
    require(final.get('decision') == 'intentionally_curtailed_for_optimization',
            'Require the stopped-generation final census')
    expected = {(stage, arm, tool) for (stage, arm), value in arms.items()
                if not value['qualification'] for tool in campaign['tools']}
    actual = {(u['stage_id'], u['arm'], u['generator']) for u in final['units']}
    require(len(expected) == 126 and actual == expected,
            f'Incomplete final census strata: missing={expected - actual}, extra={actual - expected}')
    require(sum(u['retained_raw_candidates'] for u in final['units']) == EXPECTED_RAW,
            f'Final census must contain {EXPECTED_RAW} raw candidates including canaries')
    return path, final, archive.sha256(path)


def build(root, database, *, workers=4, report_path=None):
    archive.require_cpu()
    require(1 <= workers <= 4, 'workers must be between 1 and 4')
    root, database = Path(root).resolve(), Path(database).resolve()
    require(not database.exists(), f'Refusing to overwrite {database}')
    require(database != root / 'archive/campaign.duckdb', 'Never use the full archival database')
    campaign, target, arms = archive.context(root)
    census_path, final, census_hash = census_contract(root, campaign, arms)
    report_path = Path(report_path or str(database) + '.report.json')
    temporary = database.with_name(database.name + f'.building-{os.getpid()}')
    require(not temporary.exists(), f'Partial build already exists: {temporary}')
    report = {'started_at': archive.now(), 'complete': False, 'catalog_kind': CATALOG,
              'database': str(database), 'building_database': str(temporary),
              'artifacts_preserved': False, 'fully_archived': False,
              'coordinate_validation': 'only_when_required_for_sequence_derivation',
              'final_census': str(census_path), 'final_census_sha256': census_hash,
              'collector_path': archive.__file__, 'collector_sha256': archive.sha256(archive.__file__),
              'census_parser_sha256': archive.sha256(archive.census.__file__),
              'exclusions': 'Original qualification and recovery canaries are labelled qualification; '
                            'canaries included only in production raw census reconciliation.',
              'runs': [], 'errors': []}
    # Three child workers plus this parsing/DB process never exceed four CPUs.
    pool = (ProcessPoolExecutor(max_workers=workers - 1, mp_context=multiprocessing.get_context('spawn'))
            if workers > 1 else nullcontext(None))
    manifests = {}
    try:
        with pool as executor, CampaignStore.create(temporary) as store:
            con = store.connection
            con.execute('SET threads=1')
            con.execute("SET memory_limit='8GB'")
            for path, manifest, details in generation_manifests(root, arms):
                source = source_details(path, manifest)
                manifests[str(path)] = source['manifest_sha256']
                store.add_configs([manifest.config], target=('DIO3', 'sha256:' + campaign['target_sequence_sha256']))
                run = manifest.to_run_record().model_copy(update={
                    'stage': details['stage'], 'arm': details['arm'], 'status': 'partial',
                    'workflow_metadata': {'archive_collector': 'dio3-native-v1', 'campaign_id': root.name,
                                          'target_sequence': target, **details, 'source': source,
                                          'sampling_catalog': CATALOG, 'artifacts_preserved': False,
                                          'coordinate_validation': 'not_required_for_sampling'}})
                archive.insert_missing(store, 'runs', [run])
                audit = {'manifest': str(path), 'run_id': manifest.run_id, 'generator': manifest.tool,
                         **details, 'raw_candidates': 0, 'tasks': [], 'native_sequence_candidates': 0,
                         'structure_derived_candidates': 0, 'unknown_families': 0}
                for task, plan in native_tasks(path, manifest):
                    parsed, task_audit = parse_task(task, manifest.tool, plan)
                    entries = iter(sorted(parsed.candidates.items()))
                    while batch := list(islice(entries, BATCH_SIZE)):
                        values = [item for _, item in batch]
                        normalized = (executor.map(sequence_candidate, values, repeat(target), chunksize=8)
                                      if executor else map(sequence_candidate, values, repeat(target)))
                        records = []
                        for (key, item), result in zip(batch, normalized, strict=True):
                            records.append(design_record(manifest, source, task, key, item, result))
                            audit['native_sequence_candidates' if item['sequences'] else
                                  'structure_derived_candidates'] += 1
                            audit['unknown_families'] += item['family'] is None
                        inserted = archive.insert_missing(store, 'designs', records)
                        require(inserted == len(records), f'Duplicate candidate identity in {task}')
                    audit['tasks'].append(task_audit)
                    audit['raw_candidates'] += len(parsed.candidates)
                    print(archive.encoded({'event': 'sampling_task_catalogued', 'run_id': manifest.run_id,
                                           **task_audit}), flush=True)
                con.execute('UPDATE runs SET n_produced=?, count_details=? WHERE run_id=?',
                            [audit['raw_candidates'], archive.encoded(audit), manifest.run_id])
                report['runs'].append(audit)
                save_report(report_path, report)
            require(manifests, 'No native generation manifests discovered')
            con.execute(archive.VIEW)
            observed = {(s, a, g): (n, canaries) for s, a, g, n, canaries in con.execute(
                'SELECT source_stage,arm,generator,count(*),count(*) FILTER (WHERE recovery_canary) '
                'FROM generation_candidates WHERE campaign_id=? AND (NOT qualification OR recovery_canary) '
                'GROUP BY source_stage,arm,generator', [root.name]).fetchall()}
            report['final_census_reconciliation'] = reconcile(final, observed)
            report['raw_production_including_canaries'] = sum(n for n, _ in observed.values())
            report['counts'] = con.execute(
                'SELECT stage,qualification,recovery_canary,count(*) FROM generation_candidates '
                'GROUP BY stage,qualification,recovery_canary ORDER BY stage,qualification,recovery_canary').fetchall()
            report['database_candidates'] = con.execute('SELECT count(*) FROM generation_candidates').fetchone()[0]
            require(con.execute('SELECT count(*)=count(DISTINCT candidate_id) FROM generation_candidates').fetchone()[0],
                    'Duplicate candidate IDs')
            current = {str(path): archive.sha256(path) for path, _, _ in generation_manifests(root, arms)}
            require(current == manifests, 'Generation manifests changed during catalog build')
            require(archive.sha256(census_path) == census_hash, 'Final census changed during catalog build')
            require(archive.sha256(archive.__file__) == report['collector_sha256'] and
                    archive.sha256(archive.census.__file__) == report['census_parser_sha256'],
                    'Native parser changed during catalog build')
            con.execute('CHECKPOINT')
        require(not Path(str(temporary) + '.wal').exists(), 'Catalog WAL remains after close')
        # link is atomic and, unlike replace, cannot overwrite a concurrent publisher.
        os.link(temporary, database)
        temporary.unlink()
        report.update(complete=True, finished_at=archive.now())
        save_report(report_path, report)
        return report
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
        report['finished_at'] = archive.now()
        save_report(report_path, report)
        raise


def smoke(root, report_path, *, samples=3):
    """Parent-run, real-source parity check; does not touch either campaign DB."""
    archive.require_cpu()
    require(samples >= 1, 'smoke samples must be positive')
    root = Path(root).resolve()
    campaign, target, arms = archive.context(root)
    counts, rows = Counter(), []
    for path, manifest, details in generation_manifests(root, arms):
        if details['qualification'] or counts[manifest.tool] >= samples:
            continue
        source = source_details(path, manifest)
        for task, plan in native_tasks(path, manifest):
            parsed, _ = parse_task(task, manifest.tool, plan)
            # Spread observations through the task rather than examining one native ID.
            ordered = sorted(parsed.candidates.items())
            needed = samples - counts[manifest.tool]
            indices = sorted({i * (len(ordered) - 1) // max(1, needed - 1)
                              for i in range(needed)}) if ordered else []
            for index in indices:
                key, item = ordered[index]
                fast = sequence_candidate(item, target)
                full = archive.normalize_candidate(item, target)
                candidate = design_record(manifest, source, task, key, item, fast)
                require(candidate.sequence == full[0], f'Sequence parity failed: {candidate.design_id}')
                require(candidate.design_id == stable_id('dio3-native-candidate', manifest.run_id, task.name, key),
                        'Candidate ID parity failed')
                expected_family = (stable_id('dio3-native-family', manifest.run_id, task.name, item['family'])
                                   if item['family'] else None)
                require(candidate.metadata['family_id'] == expected_family, 'Family parity failed')
                rows.append({'generator': manifest.tool, 'candidate_id': candidate.design_id,
                             'family_id': expected_family, 'sequence': candidate.sequence,
                             'manifest': str(path), 'task': str(task), 'native_identity': item['native_identity'],
                             'sequence_source': 'native' if item['sequences'] else 'coordinates',
                             'full_archive_structure': full[1], 'full_archive_errors': full[4],
                             'sequence_parity': True})
                counts[manifest.tool] += 1
            if counts[manifest.tool] >= samples:
                break
        if all(counts[tool] >= samples for tool in campaign['tools']):
            break
    require(all(counts[tool] >= samples for tool in campaign['tools']),
            f'Insufficient real smoke records: {dict(counts)}')
    result = {'complete': True, 'mode': 'real_native_record_parity', 'created_at': archive.now(),
              'collector_sha256': archive.sha256(archive.__file__), 'counts': dict(counts), 'records': rows,
              'scope': 'Representative records, not a full archive DB comparison; full DB is never opened.'}
    save_report(report_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=R)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-samples', type=int, default=3)
    args = parser.parse_args(argv)
    if args.smoke:
        require(args.report is not None, '--smoke requires a separate --report path')
        result = smoke(args.root, args.report, samples=args.smoke_samples)
    else:
        result = build(args.root, args.database or args.root / 'archive/optimization_candidates.duckdb',
                       workers=args.workers, report_path=args.report)
    print(json.dumps({key: value for key, value in result.items() if key not in ('runs', 'records')}, indent=2))


if __name__ == '__main__':
    main()
