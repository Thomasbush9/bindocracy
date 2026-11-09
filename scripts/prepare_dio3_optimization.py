#!/usr/bin/env python3
"""CPU-only DIO3 preparation and explicit, approval-gated optimization operations.

Run prepare/validate/collect inside an allocated CPU node. Preparation never calls
Slurm or imports a model. Run `--help` for commands. A shortage writes
reports/feasibility.json but creates no frozen cohort; fix the archive or obtain a
new sampling decision rather than lowering the quota. Qualification uses the FULL
3 x (90 soft + 30 sharp x 2 replicate) schedule, not a short smoke substitute.
The two retained trajectories in a task start from the SAME parent. Re-running a
failed task creates a new attempt without overwriting evidence or changing IDs.
No scoring jobs or automatic stage-wide dependencies are created.
Qualification selects 18 production parents (one per arm, covering all seven
generators), reusing their exact production seeds, child IDs, recipes and output
directories. Valid qualification children count toward the 20,160, not extra
trajectories. Its default cap is eight GPUs. Production is capped at 96 and gates each arm
independently. Launch defaults to a plan; --submit AND a campaign-hash-bound,
named approval JSON are required. One explicit launch queues every eligible
parent in balanced arrays below Slurm's 10,000-task limit, sharing the requested
GPU cap. Only a one-GPU request needs an automatic afterany queue dependency;
no success dependencies, manual chunk refills, or scoring jobs are created.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

W = Path('/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design')
R = Path('/n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-generators-20261105')
O = Path('/n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-confhard2-20261109')
PREVIOUS = Path('/n/netscratch/kempner_bsabatini_lab/Lab/tbush/chai1-confhard2-20261103')
PYTHON = W / 'bindocracy/.venv/bin/python'
ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWY')
AA3 = dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(), 'ACDEFGHIKLMNPQRSTVWY'))
QUOTA = 80


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cpu_allocation():
    require(os.environ.get('SLURM_JOB_ID'), 'CPU scans require an allocated compute node (SLURM_JOB_ID)')


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def copy_checked(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    actual = digest(source)
    require(expected is None or actual == expected, f'Frozen source changed: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    require(digest(destination) == actual == digest(source), f'Copy changed: {source}')
    return actual


def fasta(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            records.append('')
        elif line.strip():
            require(records, f'FASTA sequence before header: {path}')
            records[-1] += line.strip()
    return records


def _manifest_inputs(source, campaign):
    """Follow an authorized source-only revision without repinning its ancestry."""
    manifest = source / 'frozen_manifest.json'
    current = digest(manifest)
    expected = campaign['provenance']['frozen_manifest_sha256']
    if current == expected:
        return [manifest]
    revisions = load(manifest).get('approved_revisions', [])
    require(revisions, 'Generation frozen-manifest hash changed without an approved revision')
    record = Path(revisions[-1]['revision_record']).resolve()
    require(record.is_relative_to((source / 'provenance/revisions').resolve()), 'Revision receipt is outside the campaign')
    revision = load(record)
    original = Path(revision['original_frozen_manifest']).resolve()
    require(revision.get('schema_version') == 1 and original == record.parent / 'frozen_manifest.before.json',
            'Invalid generation revision provenance')
    require(revision['campaign_sha256'] == digest(source / 'campaign.json'), 'Generation campaign changed since approved revision')
    require(revision['original_frozen_manifest_sha256'] == expected == digest(original),
            'Original generation frozen inventory was not preserved')
    require(revision['frozen_manifest_sha256'] == current, 'Generation manifest differs from approved revision')
    return [manifest, record, original]


def authoritative(source):
    campaign = load(source / 'campaign.json')
    frozen = load(source / 'frozen_manifest.json')
    _manifest_inputs(source, campaign)
    hashes = {item['path']: item['sha256'] for item in frozen['files']}
    target_dir = source / 'assets/target'
    for name in ('target.fasta', 'target.a3m', 'target.pdb'):
        path = target_dir / name
        require(digest(path) == hashes[str(path)], f'Generation target asset changed: {path}')
    targets = fasta(target_dir / 'target.fasta')
    require(len(targets) == 1 and len(targets[0]) == 201, 'Expected exactly one 201aa target')
    sequence = targets[0]
    require(hashlib.sha256(sequence.encode()).hexdigest() == campaign['target_sequence_sha256'] == frozen['target_sequence_sha256'], 'Target sequence hash mismatch')
    require(fasta(target_dir / 'target.a3m')[0] == sequence, 'MSA query differs from full target')
    residues = {}
    for line in (target_dir / 'target.pdb').read_text().splitlines():
        if line.startswith('ATOM  ') and line[12:16].strip() == 'CA' and line[21] == 'A':
            require(line[26] == ' ', 'Insertion-coded target numbering is unsupported')
            number = int(line[22:26])
            require(number not in residues, f'Duplicate target residue {number}')
            residues[number] = AA3[line[17:20]]
    require(sorted(residues) == list(range(1, 202)), 'Target PDB chain A must number residues 1..201')
    require(''.join(residues[i] for i in range(1, 202)) == sequence, 'PDB residue identity differs from target sequence')
    arms, stages = {}, {}
    frozen_stages = {stage['id']: stage for stage in frozen['stages']}
    for stage in campaign['stages']:
        if stage['qualification']:
            continue
        frozen_arms = {arm['id']: arm for arm in frozen_stages[stage['id']]['arms']}
        for arm in stage['arms']:
            name, positions = arm['id'], arm['hotspots']
            require(name not in arms and positions == frozen_arms[name]['hotspots'], f'Ambiguous or changed hotspots: {name}')
            require(positions and len(set(positions)) == len(positions) and all(type(i) is int and 1 <= i <= 201 for i in positions), f'Invalid 1-based hotspots: {name}')
            arms[name] = {'indexing': 'target_sequence_1based', 'chain': 'A', 'residues': positions,
                          'optimizer_epitope_idx_0based': [i - 1 for i in positions],
                          'residue_identity': [sequence[i - 1] for i in positions]}
            stages[name] = stage['id']
    require(len(arms) == 18 and len(campaign['tools']) == 7, 'Expected 18 arms x 7 generators')
    return campaign, arms, stages, sequence


def select_families(rows, seed, quota=QUOTA):
    """Random family-first order, randomized variant choice, unique input sequences.

    Unknown lineage is explicitly a sequence-based sampling unit, never a claimed
    native family. Augmenting paths avoid a greedy variant causing false shortage.
    """
    families = defaultdict(list)
    for row in rows:
        key = 'family:' + row['family_id'] if row['family_id'] else 'sequence-proxy:' + row['sequence']
        families[key].append(row)
    rng = random.Random(seed)
    known = sorted(key for key in families if key.startswith('family:'))
    proxies = sorted(key for key in families if key.startswith('sequence-proxy:'))
    rng.shuffle(known)
    rng.shuffle(proxies)
    keys = known + proxies
    for key in sorted(families):
        variants = families[key]
        variants.sort(key=lambda row: row['candidate_id'])
        rng.shuffle(variants)
    owners, chosen = {}, {}

    def assign(key, visited):
        for row in families[key]:
            seq = row['sequence']
            if seq in visited:
                continue
            visited.add(seq)
            previous = owners.get(seq)
            if previous is None or assign(previous, visited):
                owners[seq] = key
                chosen[key] = row
                return True
        return False

    for key in keys:
        assign(key, set())
        if len(chosen) == quota:
            break
    result = []
    for key in keys:
        if key in chosen:
            result.append(dict(chosen[key], sampling_family_id=key,
                               family_known=bool(chosen[key]['family_id'])))
    return result, {'valid_candidates': len(rows), 'known_native_families': sum(k.startswith('family:') for k in keys),
                    'known_family_shortage': max(0, quota - sum(k.startswith('family:') for k in keys)),
                    'unknown_sequence_units': sum(k.startswith('sequence-proxy:') for k in keys),
                    'selected_sequence_proxies': sum(not row['family_known'] for row in result),
                    'distinct_sequences': len({row['sequence'] for row in rows}),
                    'sampling_units': len(keys), 'selected': len(result), 'required': quota,
                    'shortage': max(0, quota - len(result))}


def sample_database(connection, campaign_id, arms, generators, target, seed):
    parents, report = [], []
    require(connection.execute('SELECT count(*) = count(DISTINCT candidate_id) FROM generation_candidates').fetchone()[0], 'Duplicate candidate IDs in generation_candidates')
    for arm in sorted(arms):
        for generator in sorted(generators):
            cursor = connection.execute('SELECT * FROM generation_candidates WHERE campaign_id = ? AND stage = ? AND qualification = false AND arm = ? AND generator = ? ORDER BY candidate_id', [campaign_id, 'production', arm, generator])
            names = [item[0] for item in cursor.description]
            raw = [dict(zip(names, row)) for row in cursor.fetchall()]
            valid, invalid = [], Counter()
            for row in raw:
                for key in ('hotspots', 'intrinsic_metrics', 'source_provenance'):
                    if isinstance(row[key], str):
                        row[key] = json.loads(row[key])
                sequence = row['sequence']
                if not isinstance(sequence, str) or len(sequence) != 75 or not set(sequence) <= ALPHABET:
                    invalid['missing_or_noncanonical_75aa_sequence'] += 1
                    continue
                if row['target_sequence'] != target:
                    invalid['target_sequence_mismatch'] += 1
                    continue
                expected = {key: arms[arm][key] for key in ('indexing', 'chain', 'residues')}
                if row['hotspots'] != expected:
                    invalid['hotspot_contract_mismatch'] += 1
                    continue
                valid.append(row)
            sub_seed = hashlib.sha256(f'{seed}/{arm}/{generator}'.encode()).hexdigest()
            selected, counts = select_families(valid, sub_seed)
            counts.update(arm=arm, generator=generator, archived_rows=len(raw), invalid_input=dict(invalid), sampling_seed=sub_seed)
            report.append(counts)
            parents.extend(selected)
    return parents, {'strata': report, 'shortage_strata': sum(item['shortage'] > 0 for item in report),
                     'parent_count': len(parents), 'required_parents': 10080,
                     'policy': 'Uniform shuffled known-family order first; randomized variant matching without repeated sequences within generator-arm; labelled sequence proxies used only after exhausting known families. No score, native acceptance, geometry, or cysteine filters.',
                     'seed': seed}


def freeze_mode(root, owner, parents, arms, template, mode):
    for name in ('inputs', 'recipes', 'logs', 'reports', 'optimization'):
        (root / name).mkdir(parents=True, exist_ok=True)
    if mode == 'qualification':
        selected = {parent['candidate_id'] for parent in parents}
        ordered = [parent for parent in load(owner / 'parents.json') if parent['candidate_id'] in selected]
        tasks = []
        for production in load(owner / 'tasks.json'):
            if production['parent_candidate_id'] in selected:
                task = copy.deepcopy(production)
                task.update(production_task_id=production['task_id'], task_id=len(tasks))
                tasks.append(task)
        cfg = load(owner / 'campaign.json')
        cfg.update(name=owner.name + '-qualification', mode=mode, parent_count=len(ordered),
                   retained_children=2 * len(ordered), optimization_tasks=len(tasks),
                   reuses_production_outputs=True,
                   qualification_policy='18 existing production tasks, one per arm, covering seven generators; identical seeds/recipes/child IDs; successful outputs retained in the production cohort.')
        cfg['resources']['max_concurrent_gpus'] = 8
        save(root / 'parents.json', ordered)
        save(root / 'tasks.json', tasks)
        save(root / 'campaign.json', cfg)
        return
    ordered = []
    recipes = {}
    for arm in sorted(arms):
        arm_parents = [copy.deepcopy(p) for p in parents if p['arm'] == arm]
        path = root / 'inputs' / f'{arm}.fasta'
        with path.open('x') as stream:
            for position, parent in enumerate(arm_parents):
                parent.update(parent_position=position, design_id=parent['candidate_id'])
                # run_design.py indexes (seed * batch + trajectory) modulo FASTA length.
                for replicate in (1, 2):
                    stream.write(f'>{parent["candidate_id"]}/{replicate}\n{parent["sequence"]}\n')
        ordered.extend(arm_parents)
        recipe = copy.deepcopy(template)
        recipe['target'].update(fasta=str(owner / 'inputs/target/target.fasta'), msa=str(owner / 'inputs/target/target.a3m'), use_msa=True)
        recipe['binder'].update(length=75, no_cys=False, init_fasta=str(path))
        contacts = [loss for loss in recipe['losses'] if loss['name'] == 'BinderTargetContact']
        require(len(contacts) == 1, 'Expected one BinderTargetContact')
        contacts[0]['weight'] = 2.0
        contacts[0]['params']['epitope_idx'] = arms[arm]['optimizer_epitope_idx_0based']
        recipe['run'].update(batch=2, seed=0, sharp_replicates=2, hard_sharp=True)
        recipes[arm] = str(root / 'recipes' / f'{arm}.json')
        save(recipes[arm], recipe)
    tasks = []
    execution = ordered.copy()
    random.Random(20261109).shuffle(execution)
    for index, parent in enumerate(execution):
        children = [{'child_id': str(uuid.uuid5(uuid.NAMESPACE_URL, f'{owner.name}/{mode}/{parent["candidate_id"]}/{rep}')),
                     'child_index': index * 2 + rep - 1, 'parent_design_id': parent['candidate_id'],
                     'parent_position': parent['parent_position'], 'child_replicate': rep, 'trajectory': rep - 1}
                    for rep in (1, 2)]
        tasks.append(dict(task_id=index, arm=parent['arm'], generator=parent['generator'],
                          parent_candidate_id=parent['candidate_id'], seed=parent['parent_position'],
                          recipe=recipes[parent['arm']], output_dir=str(root / 'optimization' / f'task{index:04d}'), children=children))
    save(root / 'parents.json', ordered)
    save(root / 'tasks.json', tasks)
    save(root / 'campaign.json', {'schema_version': 1, 'name': owner.name + ('-qualification' if mode == 'qualification' else ''),
         'mode': mode, 'parent_count': len(parents), 'retained_children': len(parents) * 2,
         'optimization_tasks': len(tasks), 'batch': 2, 'children_per_parent': 2, 'runtime_root': str(owner / 'runtime'),
         'arms': {arm: dict(arms[arm], target_length=201, recipe=recipes[arm]) for arm in arms},
         'resources': {'account': 'kempner_bsabatini_lab', 'partition': 'kempner_h100_priority',
                       'cpu_partition': 'kempner_priority', 'cpu_qos': 'kemp_cpu_id59',
                       'max_concurrent_gpus': 8 if mode == 'qualification' else 96, 'cpus_per_task': 8, 'mem': '128G', 'job_walltime': '03:00:00'},
         'launch_disabled_until_explicit_approval': True, 'automatic_scoring': False,
         'qualification_policy': 'One representative frozen parent per arm, cycling all seven generators; full production schedule; production gates only its own arm.',
         'resume_policy': 'Successful task results reused; failed attempts retained; child identities immutable.'})


def prepare(args):
    cpu_allocation()
    root = args.root.resolve()
    require(root == O, f'New campaign must be prepared at {O}')
    require(not any((root / name).exists() for name in ('inputs', 'runtime', 'scripts', 'recipes', 'parents.json', 'provenance.json')), 'Refusing to overwrite frozen/partial preparation')
    cfg, arms, stages, target = authoritative(args.source)
    import duckdb
    with duckdb.connect(str(args.database), read_only=True) as connection:
        parents, feasibility = sample_database(connection, args.source.name, arms, cfg['tools'], target, args.seed)
        save(root / 'reports/feasibility.json', feasibility)
        require(feasibility['shortage_strata'] == 0 and len(parents) == 10080, f'Sampling shortages; inspect {root}/reports/feasibility.json; no cohort frozen')
        require(not Path(str(args.database) + '.wal').exists(), 'Checkpoint/archive DB before freezing: WAL exists')
        snapshot = root / 'inputs/generation_snapshot.duckdb'
        db_hash = copy_checked(args.database, snapshot)
    previous = load(args.previous / 'provenance.json')
    for name, expected in previous['files'].items():
        source = Path(name)
        if source.is_relative_to(args.previous / 'runtime'):
            copy_checked(source, root / 'runtime' / source.relative_to(args.previous / 'runtime'), expected)
    for name in ('collect.py',):
        source = args.previous / 'scripts' / name
        copy_checked(source, root / 'scripts' / ('previous_' + name), previous['files'][str(source)])
    template_path = args.previous / 'recipes/pocket.json'
    copy_checked(template_path, root / 'inputs/previous_recipe.json', previous['files'][str(template_path)])
    template = load(template_path)
    require(template['optimizer']['n_cycles'] == 3 and template['optimizer']['soft']['n_steps'] == 90 and template['optimizer']['sharp']['n_steps'] == 30, 'Previous schedule differs from requested full confhard')
    require(template['run']['sharp_replicates'] == 2 and template['run']['sharp_noise'] == 0.1, 'Unexpected previous sharpening policy')
    for name in ('target.fasta', 'target.a3m', 'target.pdb'):
        copy_checked(args.source / 'assets/target' / name, root / 'inputs/target' / name)
    for source in [args.source / 'campaign.json', *_manifest_inputs(args.source, cfg)]:
        copy_checked(source, root / 'inputs/generation' / source.relative_to(args.source))
    copy_checked(args.source / 'control/generation-final-census.json', root / 'inputs/generation/generation-final-census.json')
    copy_checked(args.previous / 'provenance.json', root / 'inputs/previous_provenance.json')
    copy_checked(Path(__file__).resolve(), root / 'scripts/prepare_dio3_optimization.py')
    save(root / 'inputs/hotspot_mapping.json', arms)
    save(root / 'inputs/cohort_provenance.json', {'database': str(args.database), 'database_snapshot': str(snapshot),
         'database_sha256': db_hash, 'view': 'generation_candidates', 'campaign_id': args.source.name,
         'stage': 'production', 'qualification': False, 'source_stages': stages, 'sampling_seed': args.seed,
         'candidate_ids': sorted(p['candidate_id'] for p in parents), 'feasibility': feasibility,
         'cross_stratum_repeated_sequences': len(parents) - len({p['sequence'] for p in parents}),
         'dedup_scope': 'generator-arm: distinct known family and sequence; cross-arm/generator equality reported, not discarded'})
    freeze_mode(root, root, parents, arms, template, 'production')
    representatives = []
    generators = sorted(cfg['tools'])
    for index, arm in enumerate(sorted(arms)):
        generator = generators[index % len(generators)]
        representatives.append(next(p for p in parents if p['arm'] == arm and p['generator'] == generator))
    freeze_mode(root / 'qualification', root, representatives, arms, template, 'qualification')
    script = root / 'scripts/prepare_dio3_optimization.py'
    (root / 'scripts/optimize.sbatch').write_text('#!/usr/bin/env bash\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=128G\n#SBATCH --no-requeue\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexec ' + str(PYTHON) + ' ' + str(script) + ' run --root "$1" --task-id "${SLURM_ARRAY_TASK_ID:?}" --approval "$2" --task-map "$3"\n')
    (root / 'scripts/collect.sbatch').write_text('#!/usr/bin/env bash\n#SBATCH --partition=kempner_priority\n#SBATCH --account=kempner_bsabatini_lab\n#SBATCH --qos=kemp_cpu_id59\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=16G\n#SBATCH --time=02:00:00\nset -euo pipefail\nexec ' + str(PYTHON) + ' ' + str(script) + ' collect --root "$1"\n')
    frozen_files = {}
    for directory in ('inputs', 'recipes', 'runtime', 'scripts', 'qualification/inputs', 'qualification/recipes'):
        for path in sorted((root / directory).rglob('*')):
            if path.is_file():
                frozen_files[str(path)] = digest(path)
    for base in (root, root / 'qualification'):
        for name in ('campaign.json', 'parents.json', 'tasks.json'):
            frozen_files[str(base / name)] = digest(base / name)
    save(root / 'provenance.json', {'files': frozen_files, 'external_assets': previous['external_assets'],
         'runtime_source': str(args.previous), 'database_sha256': db_hash,
         'external_asset_policy': 'Reuse previous working path/stat inventory and read-only wrapper mounts; shared weights are not content-frozen.'})
    print(json.dumps({'root': str(root), 'parents': len(parents), 'children': 2 * len(parents),
                      'qualification_parents': len(representatives), 'submitted': False}))


def owner_of(root):
    return Path(load(root / 'campaign.json')['runtime_root']).parent


def check_seal(root, full=False):
    owner = owner_of(root)
    manifest = load(owner / 'provenance.json')
    paths = manifest['files'] if full else [str(root / n) for n in ('campaign.json', 'parents.json', 'tasks.json')]
    for name in paths:
        require(digest(name) == manifest['files'][name], f'Frozen input changed: {name}')
    return owner, manifest


def command_for(root, task, output=None):
    runtime = Path(load(root / 'campaign.json')['runtime_root'])
    recipe = load(task['recipe'])
    return ['bash', str(runtime / 'mosaic-exec.sh'), 'python', str(runtime / 'design-scripts/run_design.py'),
            '--config', task['recipe'], '--target', recipe['target']['fasta'], '--target-msa', recipe['target']['msa'],
            '--seed', str(task['seed']), '--sharp-replicates', str(recipe['run']['sharp_replicates']),
            '--sharp-noise', str(recipe['run']['sharp_noise']), '--hard-sharp', '--out', str(output or task['output_dir'])]


def collector(owner):
    spec = importlib.util.spec_from_file_location('previous_collect', owner / 'scripts/previous_collect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_rows(root, task, implementation=None):
    implementation = implementation or collector(owner_of(root))
    return implementation._validated_task(load(root / 'campaign.json'), task)


def validate(root):
    cpu_allocation()
    owner, provenance = check_seal(root, full=True)
    cfg, tasks, parents = load(root / 'campaign.json'), load(root / 'tasks.json'), load(root / 'parents.json')
    require(len(parents) == cfg['parent_count'] == len(tasks), 'Parent/task count mismatch')
    require(len({p['candidate_id'] for p in parents}) == len(parents), 'Duplicate parent candidate')
    quota = 80 if cfg['mode'] == 'production' else 1
    counts = Counter((p['arm'], p['generator']) for p in parents)
    if cfg['mode'] == 'production':
        require(len(counts) == 126 and set(counts.values()) == {quota}, 'Generator-arm quota mismatch')
    else:
        require(len(parents) == 18 and len({p['arm'] for p in parents}) == 18 and len({p['generator'] for p in parents}) == 7, 'Qualification must cover 18 arms and all seven generators')
    for arm, generator in counts:
        group = [p for p in parents if (p['arm'], p['generator']) == (arm, generator)]
        require(len({p['sampling_family_id'] for p in group}) == len(group), f'Repeated input family: {arm}/{generator}')
        require(len({p['sequence'] for p in group}) == len(group), f'Repeated input sequence: {arm}/{generator}')
        require(all(p['stage'] == 'production' and p['qualification'] is False for p in group), 'Qualification generation row entered optimization cohort')
    require(cfg['retained_children'] == 2 * len(parents) and len(cfg['arms']) == 18, 'Child/arm count mismatch')
    by_id = {p['candidate_id']: p for p in parents}
    child_ids, pairs, task_parents = set(), Counter(), Counter()
    recipes = {arm: load(value['recipe']) for arm, value in cfg['arms'].items()}
    initial_sequences = {arm: fasta(recipe['binder']['init_fasta']) for arm, recipe in recipes.items()}
    targets = {arm: fasta(recipe['target']['fasta'])[0] for arm, recipe in recipes.items()}
    for arm, recipe in recipes.items():
        require(len(targets[arm]) == 201 and fasta(recipe['target']['msa'])[0] == targets[arm], 'Target/MSA mismatch')
    for task_id, task in enumerate(tasks):
        require(task['task_id'] == task_id, 'Task ID/index mismatch')
        parent = by_id[task['parent_candidate_id']]
        task_parents[parent['candidate_id']] += 1
        require((task['arm'], task['generator'], task['seed']) == (parent['arm'], parent['generator'], parent['parent_position']), 'Task parent metadata mismatch')
        recipe = recipes[task['arm']]
        require(task['recipe'] == cfg['arms'][task['arm']]['recipe'], 'Task recipe differs from its dedicated arm')
        sequences = initial_sequences[task['arm']]
        for trajectory in (0, 1):
            require(sequences[(task['seed'] * 2 + trajectory) % len(sequences)] == parent['sequence'], 'Both task trajectories must initialize from the same parent')
        require(recipe['binder']['no_cys'] is False and recipe['run']['batch'] == 2, 'Cysteine exclusion/batch mismatch')
        require(recipe['optimizer']['n_cycles'] == 3 and recipe['optimizer']['soft']['n_steps'] == 90 and recipe['optimizer']['sharp']['n_steps'] == 30, 'Full schedule mismatch')
        target = targets[task['arm']]
        contact = [loss for loss in recipe['losses'] if loss['name'] == 'BinderTargetContact']
        arm = cfg['arms'][task['arm']]
        require(len(contact) == 1 and contact[0]['weight'] == 2.0 and contact[0]['params']['epitope_idx'] == [i - 1 for i in arm['residues']], 'Contact mapping mismatch')
        require([target[i - 1] for i in arm['residues']] == arm['residue_identity'], 'Hotspot residue identity mismatch')
        require(len(task['children']) == 2, 'Expected two children')
        for trajectory, child in enumerate(task['children']):
            require(child['trajectory'] == trajectory and child['child_replicate'] == trajectory + 1 and child['child_index'] == task.get('production_task_id', task_id) * 2 + trajectory, 'Child ordinal mismatch')
            require(child['parent_design_id'] == parent['candidate_id'] and child['child_id'] not in child_ids, 'Child lineage/identity mismatch')
            child_ids.add(child['child_id'])
            pairs[(parent['candidate_id'], child['child_replicate'])] += 1
        require('--hard-sharp' in command_for(root, task), 'Command construction missing hard sharp')
    require(set(task_parents.values()) == {1} and len(task_parents) == len(parents) and set(pairs.values()) == {1}, 'Parent coverage mismatch')
    for asset in provenance['external_assets'].values():
        for path, expected in asset['inventory'].items():
            actual = Path(path).lstat()
            if expected['kind'] == 'file':
                require(actual.st_size == expected['size'] and actual.st_mtime_ns == expected['mtime_ns'], f'External working asset changed: {path}')
            elif expected['kind'] == 'symlink':
                require(os.readlink(path) == expected['link_target'], f'External link changed: {path}')
    result = {'valid': True, 'parents': len(parents), 'children': len(child_ids), 'recipes': 18,
              'full_length_qualification': cfg['mode'] == 'qualification', 'model_imports': False,
              'command_example': command_for(root, tasks[0]), 'submitted': False}
    save(root / 'reports/cpu_validation.json', result)
    return result


@contextmanager
def lock(path):
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'Lock exists: {path}; inspect active owner before manual recovery') from exc
    try:
        yield
    finally:
        path.rmdir()


def approval(root, path):
    owner = owner_of(root)
    value = load(path)
    require(value.get('approved') is True and value.get('approved_by') and value.get('approved_at'), 'Explicit named/date-stamped approval required')
    require(value.get('campaign_root') == str(owner) and value.get('provenance_sha256') == digest(owner / 'provenance.json'), 'Approval is not bound to this frozen campaign')
    require(load(root / 'campaign.json')['mode'] in value.get('modes', []), 'Approval does not cover requested mode')
    return value


def run_task(args):
    root = args.root.resolve()
    if args.task_map:
        mapping = load(args.task_map)
        expected_name = 'task-map-' + hashlib.sha256(json.dumps(mapping).encode()).hexdigest() + '.json'
        require(args.task_map.name == expected_name, 'Task map content hash mismatch')
        require(0 <= args.task_id < len(mapping), 'Array index outside task map')
        args.task_id = mapping[args.task_id]
    task = load(root / 'tasks.json')[args.task_id]
    require(args.task_id >= 0 and task['task_id'] == args.task_id, 'Task ID/index mismatch')
    if args.describe:
        print(json.dumps({'task': task, 'command': command_for(root, task), 'submitted': False}, indent=2))
        return
    require(os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_GPUS_ON_NODE')), 'Execution requires an explicit Slurm GPU allocation')
    require(args.approval, 'Execution requires --approval')
    approval(root, args.approval)
    owner, provenance = check_seal(root)
    required = [str(Path(__file__).resolve()), task['recipe']]
    recipe = load(task['recipe'])
    required.extend([recipe['binder']['init_fasta'], recipe['target']['fasta'], recipe['target']['msa']])
    required.extend(name for name in provenance['files'] if Path(name).is_relative_to(owner / 'runtime') or Path(name).is_relative_to(owner / 'scripts'))
    for name in required:
        require(digest(name) == provenance['files'][name], f'Frozen task input changed: {name}')
    implementation = collector(owner)
    if load(root / 'campaign.json')['mode'] == 'production':
        qualification_task = next(t for t in load(owner / 'qualification/tasks.json') if t['arm'] == task['arm'])
        task_rows(owner / 'qualification', qualification_task, implementation)
    output = Path(task['output_dir'])
    production_task_id = task.get('production_task_id', args.task_id)
    with lock(owner / 'optimization' / f'.task{production_task_id:04d}.lock'):
        attempts = owner / 'optimization' / 'attempts' / f'task{production_task_id:04d}'
        attempts.mkdir(parents=True, exist_ok=True)
        if output.exists():
            try:
                task_rows(root, task, implementation)
            except (OSError, ValueError, KeyError, TypeError):
                os.rename(output, attempts / ('invalid-published-' + str(uuid.uuid4())))
            else:
                print(json.dumps({'task_id': args.task_id, 'status': 'already_complete'}))
                return
        attempt = attempts / str(uuid.uuid4())
        attempt.mkdir()
        environment = os.environ.copy()
        environment.pop('SINGULARITY_BINDPATH', None)
        scratch_base = Path(os.environ.get('SLURM_TMPDIR', str(owner / 'scratch')))
        scratch_base.mkdir(parents=True, exist_ok=True)
        provenance = load(owner / 'provenance.json')
        with tempfile.TemporaryDirectory(prefix='dio3-confhard-', dir=scratch_base) as scratch:
            scratch = Path(scratch)
            for child in ('home/tmp', 'jax_cache', 'out'):
                (scratch / child).mkdir(parents=True)
            environment.update(MOSAIC_SIF=provenance['external_assets']['container']['path'],
                 MOSAIC_WEIGHTS=provenance['external_assets']['weights_root']['path'],
                 MOSAIC_DEV_SRC=str(owner / 'runtime/optimizer_src'), MOSAIC_SCRATCH=str(scratch),
                 SINGULARITYENV_HF_HUB_OFFLINE='1', SINGULARITYENV_TRANSFORMERS_OFFLINE='1',
                 SINGULARITYENV_HF_DATASETS_OFFLINE='1', SINGULARITYENV_TMPDIR=str(scratch / 'home/tmp'),
                 SINGULARITYENV_JAX_COMPILATION_CACHE_DIR='/jax_cache', SINGULARITYENV_PYTHONUNBUFFERED='1',
                 PYTHONDONTWRITEBYTECODE='1', SINGULARITYENV_PYTHONDONTWRITEBYTECODE='1')
            command = command_for(root, task, attempt)
            save(attempt / 'attempt.json', {'task': task, 'command': command, 'job_id': os.environ['SLURM_JOB_ID']})
            with (attempt / 'optimizer.log').open('x') as log:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            save(attempt / 'exit.json', {'returncode': result.returncode})
            require(result.returncode == 0, f'Optimizer failed; preserved attempt: {attempt}')
            task_rows(root, dict(task, output_dir=str(attempt)), implementation)
        os.rename(attempt, output)
    print(json.dumps({'task_id': args.task_id, 'status': 'complete', 'children': task['children']}))


def collect(root):
    cpu_allocation()
    owner, _ = check_seal(root)
    implementation = collector(owner)
    tasks = load(root / 'tasks.json')
    with lock(root / '.collect-lock'):
        planned, errors = [], []
        for task in tasks:
            try:
                rows = task_rows(root, task, implementation)
                error = None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rows, error = {}, str(exc)
                errors.append({'task_id': task['task_id'], 'arm': task['arm'], 'generator': task['generator'], 'error': error})
            for child in task['children']:
                raw = rows.get(child['trajectory'])
                planned.append(dict(child, design_id=child['child_id'], index=child['child_index'], role='child',
                     arm=task['arm'], generator=task['generator'], task_id=task['task_id'], seed=task['seed'],
                     sequence=raw['sequence'] if raw else None, loss=raw['loss'] if raw else None,
                     optimization_status='succeeded' if raw else 'missing_or_failed', optimization_error=error,
                     optimization_result=str(implementation.result_path(task))))
        available = [row for row in planned if row['sequence'] is not None]
        summary = {'complete': not errors, 'optimization_complete': not errors, 'planned_children': len(planned),
                   'available_children': len(available), 'failed_tasks': len(errors), 'errors': errors}
        save(root / 'collection.json', dict(summary, complete=False, optimization_complete=False, status='publishing'))
        save(root / 'planned_designs.json', planned)
        save(root / 'designs.json', available)
        implementation.atomic_text(root / 'designs.fasta', ''.join(f'>{row["child_id"]}\n{row["sequence"]}\n' for row in available))
        save(root / 'collection.json', summary)
    return summary


def launch_batches(selected, gpu_cap):
    """Cover every task once while respecting MaxArraySize=10000 and total GPUs."""
    require(1 <= gpu_cap <= 96, 'GPU cap must be 1..96')
    require(len(selected) == len(set(selected)), 'Duplicate launch task IDs')
    if not selected:
        return []
    count = (len(selected) + 9999) // 10000
    batches = []
    for index in range(count):
        ids = selected[index::count]
        if gpu_cap >= count:
            slots = gpu_cap // count + (index < gpu_cap % count)
            predecessor = None
        else:
            slots = 1
            predecessor = index - gpu_cap if index >= gpu_cap else None
        batches.append({'task_ids': ids, 'gpu_cap': min(slots, len(ids)), 'after_batch': predecessor})
    return batches


def admit_ledger(ledger, active, accounting):
    """Require positively observed terminal jobs, not merely absence from squeue."""
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    for entry in ledger:
        if entry.get('status') == 'not_submitted':
            continue
        identifier = entry.get('job_id')
        require(identifier, 'Unresolved submission intent; inspect scheduler and reconcile launches.json before retrying')
        require(identifier not in active, f'Campaign array still active: {identifier}; no overlapping retry or mode')
        states = accounting.get(identifier, [])
        require(states and all(state in terminal for state in states),
                f'No complete terminal scheduler evidence for {identifier}; reconcile before retrying')


def launch(args):
    cpu_allocation()
    root = args.root.resolve()
    owner, _ = check_seal(root, full=True)
    cfg = load(root / 'campaign.json')
    tasks = load(root / 'tasks.json')
    implementation = collector(owner)
    selected, blocked = [], []
    qualification_tasks = {t['arm']: t for t in load(owner / 'qualification/tasks.json')}
    requested = set(args.stratum or [])
    known = {f'{t["arm"]}/{t["generator"]}' for t in tasks}
    require(requested <= known, f'Unknown strata: {requested - known}')
    for task in tasks:
        if requested and f'{task["arm"]}/{task["generator"]}' not in requested:
            continue
        if Path(task['output_dir']).exists():
            try:
                task_rows(root, task, implementation)
            except (OSError, ValueError, KeyError, TypeError):
                pass  # The runner preserves invalid published evidence before retrying.
            else:
                continue
        if cfg['mode'] == 'production':
            try:
                task_rows(owner / 'qualification', qualification_tasks[task['arm']], implementation)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                blocked.append({'task_id': task['task_id'], 'reason': str(exc)})
                continue
        selected.append(task['task_id'])
    resources = cfg['resources']
    gpu_cap = args.gpu_cap or resources['max_concurrent_gpus']
    batches = launch_batches(selected, gpu_cap)
    commands, maps = [], []
    for batch in batches:
        ids = batch['task_ids']
        mapping = root / 'reports' / ('task-map-' + hashlib.sha256(json.dumps(ids).encode()).hexdigest() + '.json')
        maps.append((mapping, ids))
        command = ['sbatch', '--parsable', f'--account={resources["account"]}',
             f'--partition={resources["partition"]}', f'--time={resources["job_walltime"]}',
             f'--array=0-{len(ids) - 1}%{batch["gpu_cap"]}',
             f'--job-name=dio3-confhard-{cfg["mode"]}', f'--output={root}/logs/optimize-%A_%a.out']
        if batch['after_batch'] is not None:
            command.append(f'--dependency=afterany:<ARRAY_{batch["after_batch"]}_JOB_ID>')
        command.extend([str(owner / 'scripts/optimize.sbatch'), str(root),
                        str(args.approval or '<EXPLICIT_APPROVAL_JSON>'), str(mapping)])
        commands.append(command)
    plan = {'selected_tasks': len(selected), 'blocked_tasks': blocked, 'commands': commands,
            'submit_batch_policy': 'Queue ALL eligible parents now in balanced arrays, caps sharing the requested GPU ceiling. No success dependencies; collect any time. Explicit retries require terminal accounting evidence for all earlier arrays.',
            'gpu_cap': gpu_cap, 'arrays': [{'tasks': len(b['task_ids']), 'gpu_cap': b['gpu_cap'], 'after_batch': b['after_batch']} for b in batches],
            'collection_command': ['sbatch', f'--output={root}/logs/collect-%j.out', str(owner / 'scripts/collect.sbatch'), str(root)],
            'submitted': False, 'approval_required': True, 'automatic_scoring': False}
    if not args.submit:
        print(json.dumps(plan, indent=2))
        return
    require(args.approval, '--submit requires --approval FILE')
    approval(root, args.approval)
    require(commands, 'No eligible unfinished tasks (qualification or explicit strata may be required)')
    # The lock covers admission and every submission. Intent is committed before
    # each sbatch, so even an ambiguous scheduler response cannot trigger duplicates.
    with lock(owner / '.launch-lock'):
        ledger_path = owner / 'launches.json'
        ledger = load(ledger_path) if ledger_path.exists() else []
        state = subprocess.run(['squeue', '--noheader', '--me', '--format=%i'], check=True, capture_output=True, text=True)
        active = {line.strip().split('_')[0] for line in state.stdout.splitlines()}
        known_jobs = [entry['job_id'] for entry in ledger if entry.get('job_id')]
        accounting = defaultdict(list)
        if known_jobs:
            evidence = subprocess.run(['sacct', '--array', '--noheader', '--parsable2',
                 '--jobs', ','.join(known_jobs), '--format=JobID%40,State%40'], check=True, capture_output=True, text=True)
            for line in evidence.stdout.splitlines():
                identifier, state, *_ = line.split('|')
                identifier = identifier.split('_')[0].split('.')[0]
                accounting[identifier].append(state.split()[0].rstrip('+'))
        admit_ledger(ledger, active, accounting)
        pending = [{'root': str(root), 'command': command, 'at': datetime.now(timezone.utc).isoformat(),
                    'job_id': None, 'status': 'not_submitted'} for command in commands]
        ledger.extend(pending)
        save(ledger_path, ledger)
        for index, entry in enumerate(pending):
            save(maps[index][0], maps[index][1])
            predecessor = batches[index]['after_batch']
            if predecessor is not None:
                token = f'<ARRAY_{predecessor}_JOB_ID>'
                entry['command'] = [part.replace(token, pending[predecessor]['job_id']) for part in entry['command']]
            entry['status'] = 'submitting'
            save(ledger_path, ledger)
            response = subprocess.run(entry['command'], check=True, capture_output=True, text=True)
            identifier = response.stdout.strip().split(';')[0]
            require(re.fullmatch(r'\d+', identifier), f'Unparseable scheduler response; reconcile ledger: {response.stdout!r}')
            entry.update(job_id=identifier, status='submitted')
            save(ledger_path, ledger)
        plan.update(submitted=True, job_ids=[entry['job_id'] for entry in pending],
                    submitted_commands=[entry['command'] for entry in pending])
    print(json.dumps(plan, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'run', 'collect', 'launch'))
    parser.add_argument('--root', type=Path, default=O)
    parser.add_argument('--source', type=Path, default=R)
    parser.add_argument('--database', type=Path, default=R / 'archive/campaign.duckdb')
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--task-map', type=Path, help='Content-addressed local-array to frozen task-ID mapping')
    parser.add_argument('--describe', action='store_true', help='Print exact runtime command without importing models or executing')
    parser.add_argument('--submit', action='store_true', help='Launch only with explicit --approval JSON; default prints plan')
    parser.add_argument('--approval', type=Path, help='User-authored JSON: approved:true, approved_by, approved_at, campaign_root, provenance_sha256, modes:[qualification,production]')
    parser.add_argument('--stratum', action='append', help='Select arm/generator; production gates only its own arm qualification')
    parser.add_argument('--gpu-cap', type=int, help='Default: qualification8, production96; maximum96')
    args = parser.parse_args()
    require(args.gpu_cap is None or 1 <= args.gpu_cap <= 96, 'GPU cap must be 1..96')
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'validate':
        print(json.dumps(validate(args.root.resolve()), indent=2))
    elif args.command == 'run':
        run_task(args)
    elif args.command == 'collect':
        print(json.dumps(collect(args.root.resolve()), indent=2))
    else:
        launch(args)


if __name__ == '__main__':
    main()
