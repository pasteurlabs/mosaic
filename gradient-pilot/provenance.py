import hashlib
import json
from pathlib import Path
root=Path('/data/personal/andrinr/runner/surrogate/gradient-pilot-20261003')
paths=list((root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate').glob('*.py'))
paths += [root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/weights.npz',root/'source/mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py']
paths += list(root.glob('*.py'))+list(root.glob('*.sh'))
manifest={'base_commit':'e2ac67761030766bb966ae20d07fe01e5c03412f','queue':'nice','jobs':{'pilot':2871629,'recovery_path':2871754,'validation':2871803,'validation_failed_dtype':2871648,'registered_benchmarks':2871758,'conditioning':2871881},'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(root/'provenance.json').write_text(json.dumps(manifest,indent=2))
