"""Shared integrity helpers; no dependency on the intervention implementation."""

import hashlib
import json
import re
from pathlib import Path
import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def array_hash(array):
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_kwargs(kwargs):
    """Remove seed and process addresses, retaining callable names and IC metadata."""
    result = json.loads(json.dumps(kwargs))
    # IcSpec repr contains process-specific callable addresses. Normalize only
    # this known field/pattern; names, descriptions, plot parameters and every
    # substantive experiment setting remain compared. Source hashes and exact
    # generated-data hashes independently bind the implementation and inputs.
    for name, spec in result.get("make_ic", {}).items():
        require(isinstance(spec, str), f"Unexpected make_ic representation: {name}")
        normalized, count = re.subn(
            r"(<function [A-Za-z_][A-Za-z0-9_.<>]* at )0x[0-9a-fA-F]+(>)",
            r"\1<process-address>\2",
            spec,
        )
        require(count == 1, f"Unknown make_ic callable representation: {name}")
        result["make_ic"][name] = normalized
    for run in result["runs"]:
        del run["training"]["model_seed"]
    return result
