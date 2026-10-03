import contextlib
import dataclasses
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from tesseract_core import Tesseract
import mosaic.benchmarks.core.runner as runner
from mosaic.benchmarks.problems import get_config

root = Path(os.environ['MOSAIC_RESULTS_DIR'])
root.mkdir(parents=True, exist_ok=True)
urls = json.loads(os.environ['MATCHED_URLS'])

@contextlib.contextmanager
def remote(tag, gpus, docker_args):
    with Tesseract.from_url(tag, timeout=(30, 1200)) as t:
        t.health()
        yield t

runner._tracked_tesseract = remote
base = get_config('ns-3d-grid')
cfg = dataclasses.replace(base, solvers=[s for s in base.solvers if s.name in urls])
keys = [key for key in cfg.experiments if '/recovery_' in key]
manifest = {
    'transport': 'HTTP/base64 Tesseract services on one allocated GPU',
    'job_id': os.environ['SLURM_JOB_ID'],
    'host': platform.node(),
    'source_base': '6ce99c0',
    'client_packages': {p: importlib.metadata.version(p) for p in ('jax', 'tesseract-core', 'tesseract-jax', 'optax')},
    'xlb_precision': 'default float64 differentiation/forward path',
    'surrogate_precision': 'float32',
    'completed': [],
}
# Run forward before gradients/cost/recovery to publish teacher comparisons early.
keys.sort(key=lambda key: ({'forward': 0, 'gradient': 1, 'cost': 2, 'optimization': 3}[key.split('/')[0]], key))
for key in keys:
    print('RUN', key, flush=True)
    result = cfg.experiments[key].fn(cfg, urls)
    rows = result.get('results', [])
    assert rows, (key, result)
    for row in rows:
        m = row.get('metrics') or {}
        assert m and m.get('valid', True) and m.get('status') != 'failed', (key, row)
    manifest['completed'].append(key)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('DONE', key, flush=True)
print('ALL MATCHED BENCHMARKS COMPLETE', flush=True)
