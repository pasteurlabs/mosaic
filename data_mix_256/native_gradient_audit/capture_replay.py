"""One authorized exact clean replay, capturing existing validation checkpoints."""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--saved-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ['MOSAIC_RESULTS_DIR']=str(args.output.resolve())
    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.buoyancy_grid import training
    cfg=get_config('buoyancy-grid');cfg.solvers=[s for s in cfg.solvers if s.key=='jax_cfd']
    cfg.experiments={k:v for k,v in cfg.experiments.items() if k=='optimization/solver_in_loop'}
    api=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py').resolve()
    tags={name:f'inprocess:{api}' for name in cfg.solver_names}
    original=training.fit
    def fit(params,loss,validate,updates,lr,validation_interval):
        closure=inspect.getclosurevars(loss).nonlocals
        np.savez(args.output/'training_data.npz',**{k:np.asarray(closure[k]) for k in ['training_jax','scales','case_ids','starts']})
        schedule=training.checkpoint_updates(updates,validation_interval)
        counter=0
        def capture(current):
            nonlocal counter
            update=schedule[counter];counter+=1
            np.savez(args.output/f'params_{update}.npz',**{f'{i}_{k}':np.asarray(v) for i,layer in enumerate(current) for k,v in layer.items()})
            return validate(current)
        return original(params,loss,capture,updates,lr,validation_interval)
    training.fit=fit
    result=next(iter(cfg.experiments.values())).fn(cfg,tags)
    (args.output/'framework-summary.json').write_text(json.dumps(result,indent=2,default=str))
    metrics=result['results'][0]['metrics']
    prior=json.loads((args.saved_result/'result.json').read_text())['results'][0]['metrics']
    comparisons={}
    for field in ['loss','velocity_mse','scalar_mse','selected_update','selected_candidate']:
        a=np.asarray(metrics['solver_in_loop'][field]);b=np.asarray(prior['solver_in_loop'][field])
        comparisons[field]=bool(np.array_equal(a,b))
    assert len(metrics['solver_in_loop']['candidates'])==len(prior['solver_in_loop']['candidates'])==1
    for field in ['training_loss','validation_checkpoints','selected_update','completed_updates']:
        a=metrics['solver_in_loop']['candidates'][0][field];b=prior['solver_in_loop']['candidates'][0][field]
        comparisons[f'candidate_{field}']=a==b
    from mosaic.benchmarks.core.status import collect_status,status_to_dict
    status=status_to_dict(collect_status(cfg))
    (args.output/'status.json').write_text(json.dumps(status,indent=2))
    comparisons['status_pass']=bool(status['rows']) and all(cell['status']=='ok' for row in status['rows'] for cell in row['cells'].values())
    comparisons['selected192']=metrics['solver_in_loop']['selected_update']==192
    receipt={'exact_prior_metrics_match':all(comparisons.values()),'comparisons':comparisons,
             'checkpoints':[0,64,128,192,256],'source_commit':'a2339b9587b2eccb845c324325cb5288aae6aa04',
             'windows_predeclared':'actual first scheduled (case,start), plus start24 for all four cases'}
    (args.output/'replay_admission.json').write_text(json.dumps(receipt,indent=2))
    assert receipt['exact_prior_metrics_match'],receipt


if __name__=='__main__':main()
