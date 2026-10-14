"""External native JAX-CFD API shim. Only training state-input VJPs are perturbed."""
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from gradient_noise import controller

source = Path(os.environ.get("NOISE_SOURCE_REPO", ".")).resolve() / "mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py"
spec = importlib.util.spec_from_file_location("_noise_native_jax_cfd", source)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
InputSchema = native.InputSchema
OutputSchema = native.OutputSchema


def apply(inputs: InputSchema) -> dict[str, Any]:
    result = native.apply(inputs)
    controller.forward(inputs.state0, result["result"])
    return result


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector: dict[str, Any]) -> dict[str, Any]:
    result = native.vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector)
    if "state0" in result:
        result = dict(result)
        result["state0"] = controller.backward(inputs.state0, result["state0"])
    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    return native.abstract_eval(abstract_inputs)
