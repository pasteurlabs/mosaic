import hashlib
import json
from pathlib import Path
root=Path('/data/personal/andrinr/runner/surrogate/replay-pilot-20261003')
paths=list((root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate').glob('*.py'))
paths += [root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/weights.npz',root/'source/mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py']
paths += list(root.glob('*.py'))+list(root.glob('*.sh'))
manifest={'code_commit':'d1d06bc','queue':'nice','jobs':{'training':2872650,'weak_weights':2872666,'validation':2872655,'registered_benchmarks':2872661},'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(root/'provenance.json').write_text(json.dumps(manifest,indent=2))
