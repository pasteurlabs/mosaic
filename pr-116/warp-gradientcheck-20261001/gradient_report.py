import hashlib
import json
import tarfile
from pathlib import Path

root = Path('/data/personal/andrinr/runner/results/mosaic')
probe = root / 'pr116-warp-gradientcheck-20261001'
target = root / 'pr116-warp-repaired-20261001'

def initial_key(payload):
    normalized = json.loads(json.dumps(payload))
    training = normalized['run']['training']
    if training.get('pretrain_updates', 0) or training.get('curriculum'):
        return None
    training.pop('max_updates')
    training.pop('fd_epsilon')
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()

checks = []
for path in sorted((probe / 'configs').glob('*.json')):
    payload = json.loads(path.read_text())
    archive = probe / 'results' / path.stem / 'results.tar'
    if not archive.exists():
        raise RuntimeError(f'Missing diagnostic: {path.stem}')
    with tarfile.open(archive) as packed:
        metrics = json.load(packed.extractfile('./outcome.json'))
    training = payload['run']['training']
    assert training['max_updates'] == 1 and metrics['completed']
    checks.append({
        'cell': path.stem,
        'initial_problem_sha256': initial_key(payload),
        'model_seed': training['model_seeds'][0],
        'unroll': training['unroll'],
        'epsilon': training['fd_epsilon'],
        'fd_relative_error': metrics['end_to_end_fd_rel_error'],
        'dataset_hash': metrics['dataset_hash'],
        'source_sha256': payload['source_sha256'],
        'image': payload['image'],
    })

matches = []
for path in sorted((target / 'configs').glob('*.json')):
    payload = json.loads(path.read_text())
    if payload['run']['training']['max_updates'] != 3000:
        continue
    key = initial_key(payload)
    matching = [c for c in checks if c['initial_problem_sha256'] == key]
    if matching:
        matches.append({'training_cell': path.stem,
                        'initial_problem_sha256': key,
                        'diagnostic_cells': [c['cell'] for c in matching]})
result = {
    'description': 'Initial-model directional finite-difference refinement; not a training comparison.',
    'matching_rule': 'Exact solver, image, source and run configuration equality after removing only max_updates and fd_epsilon; no pretraining or curriculum.',
    'original_validation_unchanged': True,
    'checks': sorted(checks, key=lambda c: (c['unroll'], c['model_seed'], c['epsilon'])),
    'matching_training_cells': matches,
}
(probe / 'gradient_checks.json').write_text(json.dumps(result, indent=2) + '\n')
for c in result['checks']:
    print(c['unroll'], c['model_seed'], c['epsilon'], c['fd_relative_error'])
print('Matched training cells:', len(matches))
