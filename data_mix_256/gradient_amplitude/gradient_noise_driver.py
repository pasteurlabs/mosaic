"""External controlled state-VJP noise pilot; execute on Slurm only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from gradient_noise import BANDS, controller, fields, perturb


def write(path, value):
    def finite_json(item):
        if isinstance(item, dict):
            return {key:finite_json(val) for key,val in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False, default=str))


def geometry_checks(strength):
    rng = np.random.default_rng(3201)
    clean = rng.normal(size=(64, 64, 3)).astype(np.float32)
    results = {}
    for band in BANDS:
        noisy, receipt = perturb(clean, band, strength, 17000, 0, 0)
        repeated, _ = perturb(clean, band, strength, 17000, 0, 0)
        assert np.array_equal(noisy, repeated)
        assert abs(receipt['velocity_relative_l2']-strength) < 1e-6
        assert abs(receipt['scalar_relative_l2']-strength) < 1e-6
        assert receipt['noise_divergence_l2_relative'] < 2e-5
        assert receipt['noise_normal_wall_max'] == 0
        assert receipt['velocity_mode_outside_energy'] == 0
        assert receipt['scalar_mode_outside_energy'] == 0
        assert receipt['scalar_dc_coefficient'] == 0
        v, s, vc, sc = fields(64, 17000, 0, 0, band)
        # Orthogonal unit-curl modes preserve coefficient energy before rescaling.
        assert np.isclose(np.sum(v*v), np.sum(vc*vc), rtol=1e-10)
        assert np.isclose(np.sum(s*s), np.sum(sc*sc), rtol=1e-10)
        assert np.linalg.norm(v) > 0 and np.linalg.norm(s) > 0
        zero, _ = perturb(np.zeros_like(clean), band, strength, 17000, 0, 0)
        assert not np.any(zero)
        results[band] = receipt
    unchanged, _ = perturb(clean, 'clean', 0, 17000, 0, 0)
    assert np.array_equal(clean, unchanged)
    return results


def flatten(tree):
    return np.concatenate([np.asarray(x).ravel().astype(np.float64) for x in jax.tree.leaves(tree)])


def initial_probe(loss, params, arm, seed, strength):
    controller.disable()
    controller.capture = True
    controller.trace = []
    clean_value, clean_gradient = jax.value_and_grad(loss)(params, 0)
    clean_value = np.asarray(clean_value)
    clean = flatten(clean_gradient)
    clean_trace = list(controller.trace)
    controller.trace = []
    controller.configure(arm, seed, strength)
    controller.begin(0)
    value, gradient = jax.value_and_grad(loss)(params, 0)
    value = np.asarray(value)
    noisy = flatten(gradient)
    records = list(controller.records)
    noisy_trace = list(controller.trace)
    controller.capture = False
    assert clean_trace == noisy_trace, "forward states/results changed"
    assert len({r["step"] for r in records}) == len(records)
    assert len(records) in (len(noisy_trace)-1, len(noisy_trace))
    pending = [step for steps in controller.pending.values() for step in steps]
    assert pending in ([], [0]), pending
    controller.disable()
    controller.configure(arm, seed, strength)
    controller.begin(0)
    repeat_value, repeat_gradient = jax.value_and_grad(loss)(params, 0)
    repeat_value = np.asarray(repeat_value)
    repeated = flatten(repeat_gradient)
    controller.disable()
    assert np.array_equal(clean_value, value), 'forward objective changed'
    assert np.array_equal(value, repeat_value) and np.array_equal(noisy, repeated)
    assert np.isfinite(noisy).all() and len(records) > 0
    norm = float(np.linalg.norm(clean))
    difference = float(np.linalg.norm(noisy-clean))
    assert norm > 0
    if arm == 'clean':
        assert np.array_equal(clean, noisy)
    else:
        assert difference/norm > 1e-9, 'noise did not reach model gradient'
    receipt = {
        'window': 'actual first scheduled training window, original case/scales/initial parameters',
        'forward_loss_bitwise_equal': True, 'forward_states_bitwise_equal': True,
        'forward_trace': clean_trace, 'reproducible_bitwise': True,
        'clean_loss': float(clean_value), 'corrupt_loss': float(value),
        'clean_parameter_gradient_l2': norm,
        'corrupt_parameter_gradient_l2': float(np.linalg.norm(noisy)),
        'parameter_gradient_relative_l2': difference/norm,
        'parameter_gradient_cosine': float(np.dot(clean,noisy)/(norm*np.linalg.norm(noisy))),
        'state_vjp_records': records,
    }
    controller.configure(arm, seed, strength)
    return receipt


class ChecksComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['checks', 'run'])
    parser.add_argument('--arm', choices=['clean', *BANDS], required=True)
    parser.add_argument('--model-seed', type=int, choices=[0,1,2,3,4], required=True)
    parser.add_argument('--strength', type=float, choices=[0,.03,.1,.3,1,3], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.arm == "clean") != (args.strength == 0):
        parser.error("clean requires strength 0; noisy arms require positive strength")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['MOSAIC_RESULTS_DIR'] = str(args.output.resolve())
    os.environ['NOISE_SOURCE_REPO'] = str(Path.cwd())
    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.buoyancy_grid import training
    cfg = get_config('buoyancy-grid')
    cfg.solvers = [s for s in cfg.solvers if s.key == 'jax_cfd']
    assert len(cfg.solvers) == 1
    cfg.experiments = {k:v for k,v in cfg.experiments.items() if k == 'optimization/solver_in_loop'}
    shim = Path(__file__).with_name('tesseract_api_noisy.py').resolve()
    tags = {name:f'inprocess:{shim}' for name in cfg.solver_names}
    files = [Path(__file__), shim, Path(__file__).with_name('gradient_noise.py'),
             Path(training.__file__), Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py')]
    receipt = {
        'complete': False, 'arm': args.arm, 'model_seed': args.model_seed,
        'noise_seed': 17000+args.model_seed, 'relative_strength': args.strength,
        'intervention': 'native solver INPUT-state VJP, training only; not output cotangent sensitivity',
        'velocity_basis': 'unit-L2 discrete curl of sine streamfunction modes, zero normal walls',
        'scalar_basis': 'orthonormal centered DCT modes, no DC',
        'bands': {'large':'0<q<=4','medium':'4<q<=16','small':'q>16'},
        'denominator': 'full clean field L2, separate joint-velocity and scalar budgets',
        'clean_admission': 'All benchmark finite-difference checks run without corruption; they do not validate the intentionally altered backward rule',
        'source_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'outcome': 'running', 'phase': 'setup',
    }
    original_fit = training.fit
    def fit(params, loss, validate, updates, lr, validation_interval):
        closure = inspect.getclosurevars(loss).nonlocals
        val_closure = inspect.getclosurevars(validate).nonlocals
        def array_hash(value):
            return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        receipt['data_hashes'] = {key: array_hash(value) for key, value in closure.items()
                                  if key in ('training_jax', 'scales')}
        receipt['data_hashes'].update({key:array_hash(value) for key,value in val_closure.items()
                                      if key in ('validation', 'scales')})
        receipt['initial_model_sha256'] = array_hash(flatten(params))
        receipt['phase'] = 'initial_gradient_probe'
        receipt['initial_gradient_probe'] = initial_probe(loss, params, args.arm, args.model_seed, args.strength)
        write(args.output/'noise_receipt.json', receipt)
        if args.command == 'checks':
            raise ChecksComplete()
        receipt['phase'] = 'training'
        receipt['completed_updates_lower_bound'] = 0
        receipt['observed_validation_checkpoints'] = []
        expected_count = len(receipt['initial_gradient_probe']['state_vjp_records'])
        def audit_completed_gradient():
            if not controller.active:
                return
            current = [r for r in controller.records if r['update'] == controller.update]
            assert len(current) == expected_count, (controller.update, len(current), expected_count)
            assert len({r['step'] for r in current}) == expected_count
            pending = [step for steps in controller.pending.values() for step in steps]
            assert pending in ([], [0]), pending
        def controlled_loss(p, iteration):
            audit_completed_gradient()
            if controller.active:
                receipt['completed_updates_lower_bound'] = int(iteration)
            receipt['attempted_update'] = int(iteration) + 1
            controller.begin(iteration)
            return loss(p, iteration)
        def clean_validate(p):
            audit_completed_gradient()
            if controller.active:
                receipt['completed_updates_lower_bound'] = controller.update + 1
            controller.disable()
            score = float(validate(p))
            receipt['observed_validation_checkpoints'].append({
                'update':receipt['completed_updates_lower_bound'], 'loss':score})
            write(args.output/'noise_receipt.json', receipt)
            return score
        try:
            result = original_fit(params, controlled_loss, clean_validate, updates, lr, validation_interval)
            receipt['completed_updates_lower_bound'] = updates
            receipt['phase'] = 'evaluation'
            return result
        except Exception as exc:
            receipt['training_failure'] = {'type':type(exc).__name__, 'message':str(exc)}
            raise
        finally:
            controller.disable()
            receipt['training_vjp_records'] = list(controller.records)
            write(args.output/'noise_receipt.json', receipt)
    training.fit = fit
    results = {}
    started = time.perf_counter()
    try:
        receipt['phase'] = 'geometry_checks'
        receipt['geometry_checks'] = {str(strength):geometry_checks(strength) for strength in sorted({.03,.1,3.,args.strength}-{0.})}
        receipt['phase'] = 'reference_and_clean_admission'
        write(args.output/'noise_receipt.json', receipt)
        for name, experiment in cfg.experiments.items():
            kwargs = copy.deepcopy(inspect.signature(experiment.fn).parameters['_kw'].default)
            assert len(kwargs['runs']) == 1
            kwargs['runs'][0]['training']['model_seed'] = args.model_seed
            receipt['resolved_experiment_kwargs'] = kwargs
            result = experiment.fn(cfg, tags, _kw=kwargs)
            results[name] = result
            write(args.output/'framework-summary.json', results)
        # Framework may catch kernel exceptions; checks completion is identified by receipt.
        if args.command == 'checks':
            assert 'initial_gradient_probe' in receipt
        else:
            from mosaic.benchmarks.core.status import collect_status, status_to_dict
            status = status_to_dict(collect_status(cfg))
            write(args.output/'status.json', status)
            assert status['rows']
            for row in status['rows']:
                for name, cell in row['cells'].items():
                    assert cell['status'] == 'ok', (row['experiment'], name, cell)
            assert receipt.get('training_vjp_records'), 'missing training interventions'
    except ChecksComplete:
        assert args.command == 'checks'
    except Exception as exc:
        receipt['outcome'] = 'failed'
        failure = receipt.get('training_failure', {})
        receipt['failure_kind'] = ('numerical_training_failure' if failure.get('type') == 'FloatingPointError'
                                   else 'check_or_execution_failure')
        receipt['failure'] = {'type':type(exc).__name__, 'message':str(exc), 'phase':receipt.get('phase')}
        raise
    finally:
        controller.disable()
        receipt['driver_seconds'] = time.perf_counter()-started
        write(args.output/'noise_receipt.json', receipt)
    receipt['complete'] = True
    receipt['outcome'] = 'passed'
    receipt['phase'] = 'complete'
    receipt['checks_only'] = args.command == 'checks'
    write(args.output/'noise_receipt.json', receipt)
    print(json.dumps({'complete':True,'arm':args.arm,'seed':args.model_seed}), flush=True)


if __name__ == '__main__':
    main()
