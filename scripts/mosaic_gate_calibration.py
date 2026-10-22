#!/usr/bin/env python3
"""Freeze CPU-only, original-quartet DIO3 gate thresholds from R32/R35 logs.

Example:
    python scripts/mosaic_gate_calibration.py --out calibration.json \
        --report calibration-report.json

The frozen threshold is the median relative gap on EVEN-seed shards; ODD-seed
shards are an untouched audit. A gate fires only for gap > threshold, never by
random fallback. Each gate is a separate counterfactual on the unchanged original
quartet, NOT a replay of replacement dynamics. Historical Boltz2+OF3+ESMC losses
are only a proxy for the current Boltz2-only objective. No GPU libraries load.

Checkpoint k uses the kth printed optimization evaluation (index k-1), whose
loss is evaluated at the extrapolated pre-update point and printed after update.
The local iteration counter resets at each soft/sharp stage. Missing or malformed
360-step schedules are excluded explicitly, never interpolated or padded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from mosaic_pipeline_traces import SEED_RE, STEP_RE

BASE = Path('/n/holylfs06/LABS/bsabatini_lab/Everyone/tomasz/mosaic_test')
RUNS = {'pocket': 'dio3_cut_r32_catalytic_1k', 'y84': 'dio3_cut_r35_y84_1k'}
CHECKPOINTS = list(range(25, 351, 25))
EXPECTED_ITERATIONS = (list(range(90)) + list(range(30))) * 3
WARNINGS = [
    'Historical total objective sums Boltz2 and OF3 at weight 1 each plus clipped ESMC at weight 0.5; current cycle_binders.build is Boltz2-only and has no ESMC. Relative gaps are not invariant to additive terms or model disagreement.',
    'Current build loads soluble MPNN with backbone noise 0.05; historical config specifies 0.0. Historical no_cys maps original C to A; production preserves original C along ancestry.',
    'Historical logs round total loss to two decimal places. Threshold precision describes arithmetic on rounded observations, not unrounded model precision.',
    'Checkpoint k uses logged evaluation k-1, at the extrapolated pre-update point of the kth update, not a separate post-update evaluation.',
    'Original-quartet, no-replacement counterfactual rates do not establish future rates after resampling, ancestry changes or objective changes. No random fallback enforces a quota.',
    'Final critic outcomes describe the best-over-cycles decoded output. A late hypothetical cull need not erase an already saved good earlier sequence; cumulative risk is not a causal replacement simulation.',
    'Tier-1 is both critics ipTM >= 0.7 and both min_dist_epitope <= 8 A, not evidence of binding; handoff reports failure of an ipTM scramble control.',
    'Source files under mosaic_test are current on-disk evidence, not an immutable snapshot of the historical executable. Job logs and per-design configs are primary run evidence.',
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source(path: Path, manifest: dict) -> bytes:
    data = path.read_bytes()
    manifest[str(path.resolve())] = {'sha256': digest(data), 'bytes': len(data)}
    return data


def load_json(path: Path, manifest: dict):
    return json.loads(source(path, manifest))


def critics(path: Path, manifest: dict) -> tuple[dict, list[str]]:
    rows = load_json(path, manifest)['results']
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence'].strip().upper()].append(row)
    valid, ambiguous = {}, []
    fields = ('iptm', 'min_dist_epitope', 'binder_plddt')
    for sequence, entries in grouped.items():
        scores = [tuple(row.get(field) for field in fields) for row in entries]
        if any(score != scores[0] for score in scores[1:]):
            ambiguous.append(sequence)
        else:
            valid[sequence] = dict(zip(fields, scores[0]))
    return valid, ambiguous


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def outcome(sequence: str, screens: list[dict]) -> dict:
    rows = [screen.get(sequence) for screen in screens]
    if any(row is None for row in rows):
        return {'min_iptm': None, 'tier1': None}
    iptms = [row['iptm'] for row in rows]
    distances = [row['min_dist_epitope'] for row in rows]
    valid_iptm = all(numeric(value) for value in iptms)
    valid_dist = all(numeric(value) for value in distances)
    return {
        'min_iptm': min(iptms) if valid_iptm else None,
        'tier1': (min(iptms) >= 0.7 and max(distances) <= 8.0)
        if valid_iptm and valid_dist else None,
        'critics': rows,
    }


def load_arm(run: Path, manifest: dict) -> tuple[list[dict], dict]:
    optimize = run / 'optimize'
    config = load_json(optimize / 'config.json', manifest)
    node = load_json(optimize / 'node.json', manifest)
    if (config['optimizer']['soft']['n_steps'],
            config['optimizer']['sharp']['n_steps'], config['run']['batch']) != (90, 30, 4):
        raise ValueError(f'{run}: historical schedule/batch mismatch; no implicit mapping')
    seed_path = Path(config['binder']['init_fasta'])
    seeds = []
    for record in source(seed_path, manifest).decode().split('>')[1:]:
        lines = record.splitlines()
        seeds.append(''.join(lines[1:]).strip().upper())
    screens, ambiguities = [], {}
    for name in ('protenix', 'esmfold2'):
        screen, ambiguous = critics(run / f'screen_{name}' / 'screen.json', manifest)
        screens.append(screen)
        ambiguities[name] = ambiguous
    exclusions, shards, seen = [], [], set()
    schedule_lengths = Counter()
    jobs = sorted(optimize.glob('job-*.out'))
    for job in jobs:
        text = source(job, manifest).decode(errors='replace')
        match = SEED_RE.search(text)
        if match is None:
            exclusions.append({'path': str(job), 'reason': 'no seed'})
            continue
        seed = int(match.group(1))
        traces, iterations = defaultdict(list), defaultdict(list)
        for line in text.splitlines():
            step = STEP_RE.match(line)
            if step:
                iteration, member, value = int(step[1]), int(step[2]), float(step[3])
                traces[member].append(value)
                iterations[member].append(iteration)
        schedule_lengths.update(len(trace) for trace in traces.values())
        if set(traces) != set(range(4)) or any(
            iterations[member] != EXPECTED_ITERATIONS or
            not all(math.isfinite(value) for value in traces[member])
            for member in range(4)
        ):
            exclusions.append({'path': str(job), 'seed': seed,
                               'reason': 'missing members/nonfinite loss/non-360-step schedule',
                               'lengths': {str(m): len(v) for m, v in traces.items()}})
            continue
        if not re.search(r'--soft-sharp-cycles\s+3(?:\s|$)', text):
            raise ValueError(f'{job}: missing explicit three-cycle command evidence')
        if seed in seen:
            raise ValueError(f'{job}: duplicate complete seed {seed}; cannot silently deduplicate')
        design_path = optimize / f'designs_seed{seed}.json'
        if not design_path.is_file():
            exclusions.append({'path': str(job), 'seed': seed, 'reason': 'missing final designs'})
            continue
        payload = load_json(design_path, manifest)
        # Results are sorted by LOSS: list position is not the trajectory ID.
        results = {row['trajectory']: row for row in payload['results']}
        if len(payload['results']) != 4 or set(results) != set(range(4)):
            raise ValueError(f'{design_path}: expected four explicit trajectory IDs')
        expected_config = {**config, 'run': {**config['run'], 'seed': seed}}
        if payload['config'] != expected_config:
            raise ValueError(f'{design_path}: config differs from optimization config')
        roots = [(seed * 4 + member) % len(seeds) for member in range(4)]
        if len(set(roots)) != 4:
            raise ValueError(f'{job}: fewer than four distinct original seed records')
        members = []
        for member in range(4):
            row = results[member]
            if row['seed'] != seed:
                raise ValueError(f'{design_path}: inconsistent member seed')
            sequence = row['sequence'].strip().upper()
            members.append({'trajectory': member, 'seed_record_index': roots[member],
                            'original_sequence': seeds[roots[member]], 'sequence': sequence,
                            'final_optimization_loss': row['loss'],
                            **outcome(sequence, screens)})
        gates = {}
        for k in CHECKPOINTS:
            values = [traces[m][k - 1] for m in range(4)]
            worst = max(range(4), key=lambda m: values[m])
            survivor_median = median(values[m] for m in range(4) if m != worst)
            gap = (values[worst] - survivor_median) / max(abs(survivor_median), 1e-6)
            gates[str(k)] = {'losses': values, 'worst_member': worst,
                             'survivor_median': survivor_median, 'gap': gap}
        shards.append({'seed': seed, 'split': 'fit' if seed % 2 == 0 else 'heldout',
                       'log': str(job), 'members': members, 'gates': gates})
        seen.add(seed)
    if not shards:
        raise ValueError(f'{run}: no complete original quartets')
    return shards, {'run': str(run), 'config': config, 'node': node,
                    'job_files': len(jobs), 'eligible_shards': len(shards),
                    'observed_member_trace_lengths': dict(schedule_lengths),
                    'exclusions': exclusions, 'ambiguous_critic_sequences': ambiguities,
                    'seed_records': len(seeds)}


def risk(shards: list[dict], candidates: list[tuple[dict, int]]) -> dict:
    members = [member for shard in shards for member in shard['members']]
    victims = [shard['members'][member] for shard, member in candidates]
    result = {'candidate_culls': len(victims)}
    for label in ('tier1', 'top_decile_iptm'):
        eligible = [m for m in members if m[label] is not None]
        known_victims = [m for m in victims if m[label] is not None]
        positives = sum(m[label] for m in eligible)
        lost = sum(m[label] for m in known_victims)
        result[label] = {'outcomes_known': len(eligible), 'high_quality_total': positives,
                         'culled_outcomes_known': len(known_victims),
                         'false_culls': lost,
                         'fraction_high_quality_culled': lost / positives if positives else None,
                         'fraction_culls_high_quality': lost / len(known_victims)
                         if known_victims else None}
    return result


def calibrate(shards: list[dict]) -> tuple[dict, dict]:
    fits = [shard for shard in shards if shard['split'] == 'fit']
    heldout = [shard for shard in shards if shard['split'] == 'heldout']
    if not fits or not heldout:
        raise ValueError('both even-seed fit and odd-seed heldout shards are required')
    scores = sorted(m['min_iptm'] for shard in shards for m in shard['members']
                    if m['min_iptm'] is not None)
    cutoff = scores[max(0, math.ceil(0.9 * len(scores)) - 1)] if scores else None
    for shard in shards:
        for member in shard['members']:
            score = member['min_iptm']
            member['top_decile_iptm'] = score >= cutoff if score is not None else None
    thresholds = {str(k): median(shard['gates'][str(k)]['gap'] for shard in fits)
                  for k in CHECKPOINTS}
    audit = {'fit_shards': len(fits), 'heldout_shards': len(heldout),
             'top_decile_iptm_cutoff': cutoff, 'by_checkpoint': {}, 'cumulative': {}}
    for k, threshold in thresholds.items():
        audit['by_checkpoint'][k] = {'threshold': threshold}
        for label, group in (('fit', fits), ('heldout', heldout), ('all', shards)):
            accepted = [(s, s['gates'][k]['worst_member']) for s in group
                        if s['gates'][k]['gap'] > threshold]
            audit['by_checkpoint'][k][label] = {
                'quartets': len(group), 'triggered': len(accepted),
                'group_trigger_rate': len(accepted) / len(group),
                'member_replacement_rate': len(accepted) / (4 * len(group)),
                'ties_at_threshold': sum(s['gates'][k]['gap'] == threshold for s in group),
                **risk(group, accepted),
            }
    for label, group in (('fit', fits), ('heldout', heldout), ('all', shards)):
        unique = {}
        for shard in group:
            for k, threshold in thresholds.items():
                gate = shard['gates'][k]
                if gate['gap'] > threshold:
                    unique[(shard['seed'], gate['worst_member'])] = (shard, gate['worst_member'])
        audit['cumulative'][label] = risk(group, list(unique.values()))
    return thresholds, audit


def encoded(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = {}
    script_dir = Path(__file__).resolve().parent
    frozen_driver = script_dir / 'driver/cycle_binders.py'
    current_driver = (frozen_driver if frozen_driver.is_file()
                      else script_dir.parent / 'drivers/mosaic/cycle_binders.py')
    for path in (Path(__file__), script_dir / 'mosaic_pipeline_traces.py',
                 current_driver,
                 args.base / 'HANDOFF_20261013_r32_r35_1k.md',
                 args.base / 'mosaic/run_design.py',
                 args.base / 'mosaic/src/mosaic/optimizers.py'):
        source(path, manifest)
    report = {'schema_version': 1, 'warnings': WARNINGS, 'by_arm': {}}
    calibration = {
        'schema_version': 1, 'metric': 'relative_gap_to_survivor_median',
        'definition': '(worst_loss - median(other3_losses)) / max(abs(median(other3_losses)), 1e-6)',
        'comparison': 'strictly_greater', 'population': 4,
        'checkpoints': CHECKPOINTS, 'schedule': {'cycles': 3, 'soft_steps': 90, 'sharp_steps': 30},
        'step_mapping': 'checkpoint k -> kth printed per-member evaluation, index k-1; no interpolation',
        'fit': 'median gap of complete original quartets with even seeds; odd seeds held out; no refit',
        'target_group_trigger_rate': 0.5, 'target_member_replacement_rate': 0.125,
        'worst_tie_break': 'smallest trajectory/member index',
        'empirical_proxy_only': True, 'warnings': WARNINGS, 'by_arm': {},
        'reproduce': [sys.executable, str(Path(__file__).resolve()), '--base', str(args.base),
                      '--out', str(args.out), '--report', str(args.report)],
    }
    for arm, name in RUNS.items():
        shards, provenance = load_arm(args.base / 'pipelines' / name, manifest)
        thresholds, audit = calibrate(shards)
        calibration['by_arm'][arm] = thresholds
        report['by_arm'][arm] = {'provenance': provenance, 'audit': audit, 'shards': shards}
        rates = [v['heldout']['member_replacement_rate'] for v in audit['by_checkpoint'].values()]
        print(f'{arm}: {len(shards)} original quartets; fit={audit["fit_shards"]}, '
              f'heldout={audit["heldout_shards"]}; heldout replacement/check '
              f'{min(rates):.3%}..{max(rates):.3%}; tier1 '
              f'{audit["cumulative"]["all"]["tier1"]}', flush=True)
    report['sources'] = manifest
    report_text = encoded(report)
    calibration['report'] = {'path': str(args.report.resolve()),
                             'sha256': digest(report_text.encode())}
    calibration['provenance'] = {
        arm: {'run': data['provenance']['run'],
              'eligible_shards': data['provenance']['eligible_shards'],
              'fit_shards': data['audit']['fit_shards'],
              'heldout_shards': data['audit']['heldout_shards'],
              'observed_member_trace_lengths': data['provenance']['observed_member_trace_lengths']}
        for arm, data in report['by_arm'].items()
    }
    calibration['source_manifest_sha256'] = digest(encoded(manifest).encode())
    for path, text in ((args.report, report_text), (args.out, encoded(calibration))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
