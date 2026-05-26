#!/usr/bin/env python3
"""Merge benchmark results from multiple CI artifact directories.

In CI, the GPU and CPU jobs for the same (suite, problem) write to
separate artifacts.  ``actions/download-artifact`` with
``merge-multiple: true`` does a naive file-level merge where
last-write-wins — so when both jobs produce the same ``result.json``,
one silently overwrites the other.

This script replaces that naive merge:

1. Walk every ``results-*/`` artifact directory.
2. Group files by their relative path (e.g.
   ``structural-mesh/forward/baseline/result.json``).
3. Deep-merge ``result.json`` files using the same by_solver / by_param /
   by_sweep / by_N / by_steps logic as ``save_experiment``.
4. Merge ``fields.npz`` files (combine per-solver arrays).
5. Recompute ``valid`` and ``error`` in forward ``by_param`` entries where
   the per-job run had too few solvers for consensus but the merged
   fields now have enough.
6. Write merged results to the final output directory.

Usage (in CI):
    python .github/scripts/merge-results.py staging-dir/ mosaic-results/
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2)


# ---------------------------------------------------------------------------
# result.json merge — mirrors save_experiment logic in core/utils.py
# ---------------------------------------------------------------------------


def _merge_result_pair(existing: dict, new: dict) -> dict:
    """Deep-merge *new* result into *existing*, returning merged dict.

    Uses the same merge strategy as ``save_experiment``: by_solver,
    by_param, by_sweep, by_N, by_steps.
    """
    # by_solver merge
    if isinstance(new.get("by_solver"), dict) and isinstance(
        existing.get("by_solver"), dict
    ):
        merged = {**existing["by_solver"], **new["by_solver"]}
        result = {**existing, **new, "by_solver": merged}
    # by_param merge (agreement / baseline / physical_laws)
    elif isinstance(new.get("by_param"), dict) and isinstance(
        existing.get("by_param"), dict
    ):
        merged_bp: dict = {str(k): v for k, v in existing["by_param"].items()}
        for pval, solver_map in new["by_param"].items():
            spval = str(pval)
            if (
                spval in merged_bp
                and isinstance(merged_bp[spval], dict)
                and isinstance(solver_map, dict)
            ):
                merged_bp[spval] = {**merged_bp[spval], **solver_map}
            else:
                merged_bp[spval] = solver_map
        merged_spread: dict = {str(k): v for k, v in existing.get("spread", {}).items()}
        merged_spread.update({str(k): v for k, v in new.get("spread", {}).items()})
        result = {**existing, **new, "by_param": merged_bp, "spread": merged_spread}
    # by_sweep merge (recovery / topopt)
    elif isinstance(new.get("by_sweep"), dict) and isinstance(
        existing.get("by_sweep"), dict
    ):
        merged = {**existing["by_sweep"], **new["by_sweep"]}
        result = {**existing, **new, "by_sweep": merged}
    else:
        # No recognised merge key — last-write-wins.
        result = {**existing, **new}

    # by_N / by_steps merge (cost suite) — these can coexist with other keys.
    for key in ("by_N", "by_steps"):
        if isinstance(new.get(key), dict) and isinstance(existing.get(key), dict):
            merged_k: dict = {**existing[key]}
            for sname, svals in new[key].items():
                if isinstance(svals, dict) and svals:
                    merged_k[sname] = svals
                elif sname not in merged_k:
                    merged_k[sname] = svals
            result[key] = merged_k

    # tesseract_hashes merge
    if isinstance(new.get("tesseract_hashes"), dict) or isinstance(
        existing.get("tesseract_hashes"), dict
    ):
        th = {
            **(existing.get("tesseract_hashes") or {}),
            **(new.get("tesseract_hashes") or {}),
        }
        result["tesseract_hashes"] = th

    # wall_time_s merge
    if isinstance(new.get("wall_time_s"), dict) or isinstance(
        existing.get("wall_time_s"), dict
    ):
        wt = {**(existing.get("wall_time_s") or {}), **(new.get("wall_time_s") or {})}
        result["wall_time_s"] = wt

    return result


def _merge_results(results: list[dict]) -> dict:
    """Merge a list of result dicts (from multiple artifact copies)."""
    merged = results[0]
    for r in results[1:]:
        merged = _merge_result_pair(merged, r)
    return merged


# ---------------------------------------------------------------------------
# fields.npz merge
# ---------------------------------------------------------------------------


def _merge_npz(paths: list[Path], out_path: Path) -> None:
    """Merge multiple .npz files, combining per-solver arrays."""
    merged: dict[str, np.ndarray] = {}
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as data:
                for key in data.files:
                    # Later artifacts override earlier ones for the same key.
                    # Metadata keys (sweep_values, solver_names, etc.) are
                    # overwritten; per-solver keys accumulate.
                    merged[key] = np.asarray(data[key])
        except Exception:
            continue
    if merged:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **merged)


# ---------------------------------------------------------------------------
# Recompute forward agreement valid/error from merged fields
# ---------------------------------------------------------------------------


def _l2_error_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((pred - ref) ** 2)) / (np.sqrt(np.mean(ref**2)) + 1e-30)
    )


def _trimmed_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    if len(arrays) <= 2:
        return stacked.mean(axis=0)
    lo = np.quantile(stacked, 0.05, axis=0)
    hi = np.quantile(stacked, 0.95, axis=0)
    mask = (stacked >= lo) & (stacked <= hi)
    count = np.maximum(mask.sum(axis=0), 1)
    return (stacked * mask).sum(axis=0) / count


def _recompute_agreement(result: dict, fields_path: Path) -> dict:
    """Recompute valid/error for by_param entries after fields merge.

    Handles two cases that arise when CPU and GPU jobs run separately:

    1. ALL solvers at a sweep point have ``valid=False`` — neither job had
       enough solvers for consensus alone, but the merged fields do.
    2. SOME solvers have ``valid=True`` (from a multi-solver job) while
       others have ``valid=False`` (from a single-solver job). The
       invalid solvers need their error recomputed against the consensus
       formed from all available fields.
    """
    by_param = result.get("by_param")
    if not isinstance(by_param, dict):
        return result

    if not fields_path.exists():
        return result

    try:
        fields = np.load(str(fields_path), allow_pickle=False)
    except Exception:
        return result

    # Recover sweep_values from the npz.
    try:
        sweep_values = list(fields["sweep_values"])
    except KeyError:
        return result

    # Build a mapping: sweep_index -> {solver_name: array}
    # Keys in the npz are "{solver}_{i}" where i is the sweep index.
    _snap_meta_prefixes = ("sweep_values", "solver_names", "ic", "consensus", "x_axis")
    solver_arrays: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for key in fields.files:
        if any(key.startswith(p) for p in _snap_meta_prefixes):
            continue
        # Parse "{solver_name}_{index}"
        last_us = key.rfind("_")
        if last_us < 0:
            continue
        suffix = key[last_us + 1 :]
        if not suffix.isdigit():
            continue
        solver_name = key[:last_us]
        idx = int(suffix)
        solver_arrays[idx][solver_name] = np.asarray(fields[key])

    # Build a lookup from by_param keys.  JSON round-trips may store
    # sweep values as "4" (int→str), "4.0" (float→str), or 4 (raw int).
    # The npz stores float64 sweep_values, so str(val) gives "4.0".
    # We need to match against all plausible string representations.
    bp_keys = set(by_param.keys())

    def _find_bp_key(val):
        """Find the by_param key matching a numeric sweep value."""
        for candidate in (
            str(val),
            str(int(val)) if float(val) == int(float(val)) else None,
        ):
            if candidate is not None and candidate in bp_keys:
                return candidate
        # Also try the raw value (for non-string keys, though rare after JSON).
        if val in bp_keys:
            return val
        return None

    changed = False
    new_consensus: dict[str, np.ndarray] = {}
    for i, val in enumerate(sweep_values):
        spval = _find_bp_key(val)
        if spval is None:
            continue

        solver_map = by_param[spval]
        if not isinstance(solver_map, dict):
            continue

        # Find solvers that are still marked invalid at this sweep point.
        invalid_solvers = {
            s
            for s, v in solver_map.items()
            if isinstance(v, dict) and v.get("valid") is False
        }
        if not invalid_solvers:
            continue

        # Collect all solver field arrays available after the npz merge.
        available = solver_arrays.get(i, {})
        comparable = {
            s: arr
            for s, arr in available.items()
            if s in solver_map and np.all(np.isfinite(arr))
        }
        if len(comparable) < 2:
            continue

        # Recompute consensus from ALL available fields (not just the
        # previously-valid ones) and update errors for invalid solvers.
        reference = _trimmed_mean(list(comparable.values()))
        new_consensus[f"consensus_{i}"] = reference

        for solver_name in invalid_solvers:
            entry = solver_map[solver_name]
            if not isinstance(entry, dict):
                continue
            if solver_name in comparable:
                entry["error"] = _l2_error_rel(comparable[solver_name], reference)
                entry["valid"] = True
                changed = True

    # Update consensus arrays in fields.npz if we recomputed any.
    if new_consensus and fields_path.exists():
        try:
            merged_fields: dict[str, np.ndarray] = {}
            with np.load(str(fields_path), allow_pickle=False) as data:
                for key in data.files:
                    merged_fields[key] = np.asarray(data[key])
            merged_fields.update(new_consensus)
            all_solvers = set()
            for idx_arrays in solver_arrays.values():
                all_solvers.update(idx_arrays.keys())
            if all_solvers:
                merged_fields["solver_names"] = np.array(sorted(all_solvers))
            np.savez(fields_path, **merged_fields)
        except Exception:
            pass

    if changed:
        print(f"  recomputed agreement: {fields_path.parent.name}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <staging-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    staging = Path(sys.argv[1])
    output = Path(sys.argv[2])

    if not staging.is_dir():
        print(f"Staging directory not found: {staging}", file=sys.stderr)
        sys.exit(1)

    # Collect all files grouped by relative path.
    # Staging layout: staging/<artifact-name>/<relative-path>
    # Each artifact-name dir like "results-forward-structural-mesh-gpu"
    # contains the same tree structure as mosaic-results/.
    artifact_dirs = sorted(
        d for d in staging.iterdir() if d.is_dir() and d.name.startswith("results-")
    )
    if not artifact_dirs:
        print(
            "No results-* artifact directories found in staging dir.", file=sys.stderr
        )
        sys.exit(0)

    # Group all files by their relative path (relative to the artifact dir).
    files_by_relpath: dict[str, list[Path]] = defaultdict(list)
    for adir in artifact_dirs:
        for fpath in adir.rglob("*"):
            if fpath.is_file():
                relpath = str(fpath.relative_to(adir))
                files_by_relpath[relpath].append(fpath)

    n_merged = 0
    # Track which result.json paths are forward/agreement-like (have by_param)
    # so we can recompute after fields.npz merge.
    recompute_candidates: list[Path] = []

    # First pass: merge all files.
    for relpath, sources in sorted(files_by_relpath.items()):
        out_path = output / relpath

        if relpath.endswith("result.json") and len(sources) > 1:
            # Deep-merge result.json files.
            results = []
            for src in sources:
                try:
                    results.append(_load_json(src))
                except Exception as e:
                    print(f"  warning: failed to load {src}: {e}", file=sys.stderr)
            if not results:
                continue
            if len(results) == 1:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sources[0], out_path)
            else:
                merged = _merge_results(results)
                _save_json(merged, out_path)
                n_merged += 1
                print(f"  merged {len(results)} copies: {relpath}")
                if isinstance(merged.get("by_param"), dict):
                    recompute_candidates.append(out_path)

        elif relpath.endswith(".npz") and len(sources) > 1:
            # Merge npz files.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _merge_npz(sources, out_path)
            print(f"  merged {len(sources)} copies: {relpath}")

        else:
            # Single source or non-mergeable file — copy (last source wins
            # for any remaining conflicts, matching prior merge-multiple
            # behaviour).
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources[-1], out_path)

    # Second pass: recompute forward agreement valid/error from merged fields.
    for result_path in recompute_candidates:
        fields_path = result_path.parent / "fields.npz"
        try:
            result = _load_json(result_path)
            result = _recompute_agreement(result, fields_path)
            _save_json(result, result_path)
        except Exception as e:
            print(
                f"  warning: recompute failed for {result_path}: {e}", file=sys.stderr
            )

    print(f"Merge complete: {n_merged} result.json file(s) deep-merged.")


if __name__ == "__main__":
    main()
