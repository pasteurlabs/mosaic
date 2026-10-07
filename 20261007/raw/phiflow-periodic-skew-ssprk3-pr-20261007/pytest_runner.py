import sys,importlib.util
sys.path.append('/hostsite')
import pytest
spec=importlib.util.spec_from_file_location('tesseract_api','/validation/candidate.py')
module=importlib.util.module_from_spec(spec)
sys.modules['tesseract_api']=module
spec.loader.exec_module(module)
import jax
jax.config.update('jax_enable_x64',False)
module.math.set_global_precision(32)
raise SystemExit(pytest.main(['-q','/validation/test_periodic.py']))
