"""Retain every prior epsilon and extend toward the Float64 numerical floor."""
import hashlib
import json
from pathlib import Path
import sys

import precision_companion

precision_companion.LADDER=sorted([*precision_companion.LADDER,1e-9,3e-10,1e-10,3e-11,1e-11])
if __name__=='__main__':
    precision_companion.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])/'precision_companion.json'
    report=json.loads(output.read_text())
    report['extension_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['extension_note']='All prior13 and five additional prespecified epsilon values retained; no best-epsilon selection.'
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
