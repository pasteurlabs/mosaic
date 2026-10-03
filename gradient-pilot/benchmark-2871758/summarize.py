"""Summarize the standard Mosaic envelopes; run from this artifact directory."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'ns-3d-grid'
NAMES=('XLB','XLB 3D surrogate')
summary={'seeds':[0,1,2], 'solvers':{n:{} for n in NAMES}}

def rows(path):
    return {r['solver']:r['metrics'] for r in json.loads(path.read_text())['results']}

for n in NAMES:
    s=summary['solvers'][n]
    s['forward_relative_l2']=[rows(DATA/f'forward/recovery_teacher/seed_{i}/result.json')[n]['error'] for i in range(3)]
    s['fd_best_median_relative_error']=[]
    s['fd_cosine_at_best_epsilon']=[]
    for i in range(3):
        m=rows(DATA/f'gradient/recovery_fd_check/seed_{i}/result.json')[n]
        best=min(m['eps_sweep'].values(),key=lambda x:np.median(x['rel_error']))
        s['fd_best_median_relative_error'].append(float(np.median(best['rel_error'])))
        s['fd_cosine_at_best_epsilon'].append(best['cosine'])
    for operation in ('forward','vjp'):
        trials=[t for i in range(3) for t in rows(DATA/f'cost/recovery_{operation}/seed_{i}/result.json')[n]['trials_s']]
        s[f'{operation}_mean_ms']=float(np.mean(trials)*1000)
        s[f'{operation}_median_ms']=float(np.median(trials)*1000)
        s[f'{operation}_trial_count']=len(trials)
    for opt in ('bfgs','bfgs_proj'):
        m=rows(DATA/f'optimization/recovery_constant_ic_{opt}/result.json')[n]
        s[f'{opt}_ic_relative_l2']=m['final_ic_error_trials']
        s[f'{opt}_mean_ic_relative_l2']=m['final_ic_error']

gradients=[]
for i in range(3):
    with np.load(DATA/f'gradient/recovery_fd_check/seed_{i}/gradient_fields.npz') as d:
        names=d['solver_names'].tolist()
        a=d[f'grad_{names.index(NAMES[0])}'].astype(float).ravel()
        b=d[f'grad_{names.index(NAMES[1])}'].astype(float).ravel()
        gradients.append({'seed':i,'relative_l2':float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                          'cosine':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))})
summary['energy_objective_gradient_fidelity']=gradients
summary['limitations']=[
    'Three held-out seeds; this is not a complete held-out dataset evaluation.',
    'Energy-objective gradients differentiate each solver\'s own sum(u_T**2), not a full Jacobian or a common-output-cotangent VJP.',
    'Timing includes HTTP/base64 transport and a CPU JAX client; VJP timing includes the forward and reverse calls.',
    'XLB uses its default float64 path; the surrogate uses float32. One RTX 5090 allocation, sequential fixed solver order; 60 trials per operation and solver, not independent jobs.',
    'Existing runtime images were used with source/API/model/weights mounts. This is not a fresh Docker image build validation.',
]
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

colors=['#0072B2','#CC3311']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
x=np.arange(3)
axs[0,0].bar(x,100*np.array(summary['solvers'][NAMES[1]]['forward_relative_l2']),color=colors[1])
axs[0,0].set(title='Forward fidelity to XLB',ylabel='Relative field error (%)',xticks=x,xticklabels=['Seed 0','Seed 1','Seed 2'])
for j,n in enumerate(NAMES):
    s=summary['solvers'][n]
    axs[0,1].bar(np.arange(2)+(j-.5)*.34,[s['forward_mean_ms'],s['vjp_mean_ms']],width=.34,label=n,color=colors[j])
    axs[1,0].bar(np.arange(2)+(j-.5)*.34,100*np.array([s['bfgs_mean_ic_relative_l2'],s['bfgs_proj_mean_ic_relative_l2']]),width=.34,label=n,color=colors[j])
    axs[1,1].semilogy(x,100*np.array(s['fd_best_median_relative_error']),marker='o',label=n,color=colors[j])
axs[0,1].set(title='Warm API timings (60 calls each)',ylabel='Mean milliseconds',xticks=[0,1],xticklabels=['Forward','Forward + VJP'])
axs[0,1].legend()
axs[1,0].set(title='Self-target recovery (100 updates)',ylabel='Mean IC error (%)',xticks=[0,1],xticklabels=['L-BFGS','Projected L-BFGS'])
axs[1,1].set(title='Internal derivative consistency',ylabel='Best median directional FD error (%)',xticks=x,xticklabels=['Seed 0','Seed 1','Seed 2'])
fig.suptitle('XLB surrogate — matched Mosaic benchmark\nN=16 · ν=0.01 · dt=0.02 · 100 steps · RTX 5090 · HTTP/base64',fontsize=14)
fig.savefig(ROOT/'comparison.png',dpi=170)
plt.close(fig)
print(json.dumps(summary,indent=2))
