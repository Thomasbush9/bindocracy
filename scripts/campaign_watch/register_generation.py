#!/usr/bin/env python3
"""Register allocation metadata only; never create bundles, submit jobs, or execute manifests."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from collector import load_json, publish
from generation import inside, nonnegative

FUTURE = [
    {'id': 'geometry', 'name': 'Native-pose geometry', 'unit': 'designs', 'planned': None},
    {'id': 'selection', 'name': 'Freeze parents', 'unit': 'parents', 'planned': None},
    {'id': 'optimization', 'name': 'Optimization', 'unit': 'children', 'planned': None},
    {'id': 'esmfold2', 'name': 'ESMFold2 Full', 'unit': 'designs', 'planned': None},
    {'id': 'protenix', 'name': 'Protenix Base', 'unit': 'designs', 'planned': None},
    {'id': 'analysis', 'name': 'Analysis / figures', 'unit': 'designs', 'planned': None},
]


def registration(allocation, root, bundles, name, *, qualification_bundle=None, generation_only=False):
    root = Path(root).resolve()
    waves = allocation['execution']['waves']
    if len(waves) != 4 or {wave['id'] for wave in waves} != set(bundles):
        raise ValueError('explicitly bind all four allocation waves with --bundle ID=PATH')
    arms = {arm['id'] for arm in allocation['arms']}
    tools = {tool['tool'] for tool in allocation['generators']}
    if len(arms) != len(allocation['arms']) or len(tools) != len(allocation['generators']):
        raise ValueError('duplicate allocation arm or generator')
    tiers, arm_tier = [], {}
    roots = set()
    for wave in waves:
        bundle = inside(root, bundles[wave['id']])
        if bundle in roots:
            raise ValueError('generation tiers must use separate bundle roots')
        roots.add(bundle)
        names = [entry['arm'] for entry in wave['arms']]
        for arm in names:
            if arm in arm_tier or arm not in arms:
                raise ValueError('arm is duplicated or missing from allocation')
            arm_tier[arm] = wave['id']
        tiers.append({'id': wave['id'], 'name': wave['id'].replace('_', ' ').title(),
                      'bundle_root': str(bundle), 'arms': names,
                      'control_status': str(root / 'control' / 'stages' / f"{wave['id']}.json")})
    if set(arm_tier) != arms:
        raise ValueError('every allocation arm must belong to a generation tier')
    cells, seen = [], set()
    for cell in allocation['cells']:
        key = (cell['arm'], cell['generator'])
        if key in seen or key[0] not in arms or key[1] not in tools:
            raise ValueError('duplicate or unregistered allocation cell')
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value) for value in key):
            raise ValueError('invalid allocation identifier')
        seen.add(key)
        cells.append({'tier': arm_tier[key[0]], 'arm': key[0], 'generator': key[1],
                      'planned': nonnegative(cell['planned_candidates'], 'planned_candidates'),
                      'tasks_planned': nonnegative(cell['nominal_tasks'], 'nominal_tasks')})
    if seen != {(arm, tool) for arm in arms for tool in tools}:
        raise ValueError('allocation must include every arm x generator cell')
    if sum(cell['planned'] for cell in cells) != allocation['total_nominal_candidate_target']:
        raise ValueError('allocation cell counts disagree with nominal total')
    if qualification_bundle is not None:
        qualification_root = inside(root, qualification_bundle)
        qualification_arms = ['site_16', 'site_18']
        if qualification_root in roots or not set(qualification_arms) <= arms:
            raise ValueError('qualification needs a separate bundle and both site_16/site_18 arms')
        tiers.insert(0, {'id': 'qualification', 'name': 'Qualification', 'arms': qualification_arms,
                         'bundle_root': str(qualification_root),
                         'control_status': str(root / 'control' / 'stages' / 'qualification.json')})
        cells[:0] = [{'tier': 'qualification', 'arm': arm, 'generator': tool['tool'],
                      'planned': nonnegative(tool['per_task'], 'qualification nominal candidates'),
                      'tasks_planned': 1}
                     for arm in qualification_arms for tool in allocation['generators']]
    return {'schema_version': 1, 'kind': 'bindocracy_generation', 'root': str(root), 'name': name,
            'count_contract': allocation.get('count_contract'), 'tiers': tiers, 'components': cells,
            'future_stages': [] if generation_only else FUTURE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--bundle', action='append', required=True, metavar='WAVE_ID=PATH')
    parser.add_argument('--qualification-bundle', help='Separate site_16/site_18, one-task-per-tool bundle')
    parser.add_argument('--generation-only', action='store_true', help='Omit unapproved downstream stages')
    args = parser.parse_args()
    bundles = {}
    for binding in args.bundle:
        key, separator, value = binding.partition('=')
        if not separator or not value or key in bundles:
            parser.error('--bundle requires a unique WAVE_ID=PATH binding')
        bundles[key] = value
    manifest = registration(load_json(args.allocation), args.root, bundles, args.name,
                            qualification_bundle=args.qualification_bundle,
                            generation_only=args.generation_only)
    manifest_path, config_path = args.manifest.resolve(), args.config.resolve()
    if manifest_path == config_path or manifest_path == args.allocation.resolve() or config_path == args.allocation.resolve():
        raise ValueError('allocation, manifest and config paths must differ')
    config = load_json(config_path) if config_path.exists() else {'campaigns': []}
    entry = {'kind': 'bindocracy_generation', 'root': manifest['root'], 'name': args.name,
             'manifest': str(manifest_path)}
    matches = [item for item in config['campaigns'] if Path(item['root']).resolve() == args.root.resolve()]
    if matches and matches != [entry]:
        raise ValueError('campaign root already registered differently; refusing overwrite')
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError('manifest already exists with different content; refusing overwrite')
    if not matches:
        config['campaigns'].append(entry)
    if len(config['campaigns']) > 8:
        raise ValueError('maximum eight registered campaigns')
    # publish only the two explicitly requested files; no bundle directories are created.
    publish(manifest_path, manifest)
    publish(config_path, config)
    print(json.dumps({'manifest': str(manifest_path), 'config': str(config_path),
                      'tiers': len(manifest['tiers']), 'components': len(manifest['components']),
                      'mode': 'registration_only_no_submission'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
