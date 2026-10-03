import json
from pathlib import Path
root=Path('/data/personal/andrinr/runner/surrogate')
old=json.loads((root/'gradient-pilot-20261003/results/labels.json').read_text())
normal=json.loads((root/'mosaic-3d/recovery_3d_autoregressive_weights_16k.metrics.json').read_text())
assert old['dataset_sha256']==normal['dataset_sha256']
print(json.dumps({'dataset_sha256':normal['dataset_sha256'],'output_scale':normal['output_scale'],'weights_sha256':normal['weights_sha256']}))
