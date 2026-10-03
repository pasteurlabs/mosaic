import json
from pathlib import Path
import numpy as np
root=Path(__file__).parent
report=json.loads((root/'results/validation.json').read_text())
rows=[]
for name,cases in report['models'].items():
    rows.append({'model':name,
      'forward_relative_l2':float(np.mean([c['forward']['relative_l2'] for c in cases])),
      'vjp_relative_l2':float(np.mean([c['vjp']['relative_l2'] for c in cases])),
      'vjp_cosine':float(np.mean([c['vjp']['cosine'] for c in cases])),
      'self_recovery_relative_l2':float(np.mean([c['self_recovery']['ic_relative_l2'] for c in cases])),
      'xlb_recovery_relative_l2':float(np.mean([c['xlb_recovery']['ic_relative_l2'] for c in cases])),
      'fourier_xlb_recovery_relative_l2':float(np.mean([c['fourier_xlb_recovery']['ic_relative_l2'] for c in cases]))})
print(json.dumps(rows,indent=2))
(root/'pilot-summary.json').write_text(json.dumps(rows,indent=2))
