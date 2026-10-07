#!/usr/bin/env bash
set -euo pipefail
campaign=$1
env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=2 --container-image=/data/personal/andrinr/runner/artifacts/mosaic/xlb-2853707.sqsh --container-mounts="$campaign:/audit" /python-env/bin/python -c 'import pathlib,shutil,importlib.util,json,hashlib; out=pathlib.Path("/audit/source-snapshot");out.mkdir(exist_ok=True); source=pathlib.Path(importlib.util.find_spec("xlb").origin).parent; paths=[pathlib.Path("/tesseract/tesseract_api.py"),source/"operator/collision/kbc.py",source/"operator/collision/bgk.py"]; records={};[(shutil.copy2(p,out/p.name),records.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest()})) for p in paths];(out/"source-hashes.json").write_text(json.dumps(records,indent=2))'
