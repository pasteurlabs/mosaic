# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution- and physics-conditioned periodic 3D XLB surrogate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from operator_api import (
    InputSchema,
    OutputSchema,
    abstract_eval,
    apply,
    vector_jacobian_product,
)

__all__ = [
    "InputSchema",
    "OutputSchema",
    "abstract_eval",
    "apply",
    "vector_jacobian_product",
]
