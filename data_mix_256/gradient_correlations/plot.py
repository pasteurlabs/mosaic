from pathlib import Path
import json,hashlib,itertools,statistics,os,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path('/data/personal/andrinr/runner/results/mosaic')
P=ROOT/'buoyancy-gradient-scales-20261013/plots/artifacts/gradient_spectral_comparison.json'
B=ROOT/'buoyancy-data-mix-20261013/admission_summary.json'
OUT=Path(__file__).parent/'artifacts';OUT.mkdir(exist_ok=True)
d=json.loads(P.read_text())['solvers'];base=json.loads(B.read_text())['methods']
keys={'PhiFlow':'phiflow','WaterLily':'waterlily','Oceananigans':'oceananigans','JAX-CFD':'jax-cfd','INS.jl':'ins-jl'}
colors=dict(zip(keys,['#0072B2','#E69F00','#009E73','#CC79A7','#6B4423']))
def ranks(a):return [1+sum(y<x for y in a)+(sum(y==x for y in a)-1)/2 for x in a]
def stat(x,y):
    xr,yr=ranks(x),ranks(y);rho=statistics.correlation(xr,yr)
    p=sum(abs(statistics.correlation(xr,z))>=abs(rho)-1e-12 for z in itertools.permutations(yr))/120
    return {'n':5,'spearman_rho':rho,'exact_two_sided_permutation_p_unadjusted':p}
rows=[]
for name,v in d.items():
    loss=base[keys[name]]['mean_predicted_loss'];benefit=100*(1-loss['solver_in_loop']/loss['uncorrected'])
    fd=max(r['fd_max_relative_error'] for f in v['summary'].values() for r in f.values())
    rows.append({'solver':name,'overall_benefit_percent':benefit,'worst_fd_relative_discrepancy':fd})
report={'source_hashes':{str(P):hashlib.sha256(P.read_bytes()).hexdigest(),str(B):hashlib.sha256(B.read_bytes()).hexdigest()},'overall_rows':rows,'overall_fd':stat([r['worst_fd_relative_discrepancy'] for r in rows],[r['overall_benefit_percent'] for r in rows]),'within_band':{},'limitations':['Exploratory five-backend comparison; backends are not randomized experimental units.','All correlations are shown; permutation p-values are unadjusted for multiple comparisons and not confirmatory.','Repeated fields/bands share the same five solvers; do not treat 30 points as independent replicates.','FD discrepancy depends on finite step size, nonsmoothness and floating-point effects as well as implementation accuracy.','Sampled conditioning is not the full Jacobian or neural parameter conditioning.']}
fig,ax=plt.subplots(figsize=(10,6.3));fig.subplots_adjust(left=.1,right=.97,top=.8,bottom=.24)
for row in rows:
    name=row['solver'];x=100*row['worst_fd_relative_discrepancy'];y=row['overall_benefit_percent']
    ax.scatter(x,y,color=colors[name],s=85,zorder=3)
    ax.annotate(name,(x,y),xytext=(8,6 if name!='Oceananigans' else -14),textcoords='offset points',fontsize=10)
ax.set_xscale('log');ax.set_xlabel('Worst fixed-step FD relative discrepancy (%) — smaller means closer agreement');ax.set_ylabel('Reduction in combined normalized test loss (%)');ax.set_ylim(15,65);ax.set_xlim(0.00025,15);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Do solvers with closer FD agreement benefit more from correction?',fontsize=15,y=.96)
s=report['overall_fd'];fig.text(.5,.87,f"Five solvers · Spearman ρ = {s['spearman_rho']:.2f} · exact two-sided permutation p = {s['exact_two_sided_permutation_p_unadjusted']:.3f}",ha='center',fontsize=11)
fig.text(.5,.105,'FD statistic: maximum across four states, six field/scale pairs and both fixed perturbation sizes.',ha='center',fontsize=9)
fig.text(.5,.065,'Exploratory association, not established evidence: n=5; solver differences confound the comparison.',ha='center',fontsize=10)
fig.text(.5,.03,'FD discrepancy is not a pure gradient-quality score. No fitted trend line or causal interpretation.',ha='center',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'fd_vs_overall_benefit.{ext}',dpi=150,facecolor='white')
plt.close(fig)
for metric,label,filename in [('condition_relative_H0','Sampled condition ratio κ(H8) / κ(H0)','conditioning_within_scales'),('fd_max_relative_error','Worst fixed-step FD discrepancy (%)','fd_within_scales')]:
    fig,axes=plt.subplots(2,3,figsize=(15,9));fig.subplots_adjust(left=.065,right=.985,top=.83,bottom=.22,hspace=.48,wspace=.26)
    for fi,field in enumerate(['scalar','velocity']):
        for bi,band in enumerate(['large','medium','small']):
            ax=axes[fi,bi];xs=[];ys=[]
            for name,v in d.items():
                r=v['summary'][field][band];x=r[metric]['median'] if metric=='condition_relative_H0' else 100*r[metric];y=r['error_reduction_percent'];xs.append(x);ys.append(y)
                ax.scatter(x,y,color=colors[name],s=48,zorder=3)
            s=stat(xs,ys);report['within_band'][f'{metric}/{field}/{band}']=s
            ax.set_title(f"{field.title()} · {band}\nρ={s['spearman_rho']:+.2f}, p={s['exact_two_sided_permutation_p_unadjusted']:.3f}, n=5",fontsize=10)
            if metric=='fd_max_relative_error':ax.set_xscale('log')
            ax.margins(x=.3,y=.25);ax.set_ylim(-5,88);ax.set_xlabel(label,fontsize=8)
            if bi==0:ax.set_ylabel('Error power removed (%)')
            ax.tick_params(labelsize=8);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Conditioning versus correction benefit, comparing solvers within each scale' if metric=='condition_relative_H0' else 'FD agreement versus correction benefit, comparing solvers within each scale',fontsize=15,y=.97)
    fig.text(.5,.91,'Large: 0<q≤4   ·   Medium: 4<q≤16   ·   Small: q>16   |   Same five solvers in every panel',ha='center',fontsize=11)
    fig.legend([Line2D([0],[0],marker='o',ls='',color=c) for c in colors.values()],list(colors),loc='lower center',bbox_to_anchor=(.5,.115),ncol=5,frameon=False)
    fig.text(.5,.075,'All panels reported. Exact permutation p-values are exploratory and unadjusted for multiple comparisons.',ha='center',fontsize=10)
    fig.text(.5,.035,'Fields and bands are repeated measurements, not independent replicates. No causal inference or full-Jacobian conditioning claim.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(OUT/f'{filename}.{ext}',dpi=150,facecolor='white')
    plt.close(fig)
(OUT/'correlations.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'correlations.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['comparison','n','spearman_rho','exact_two_sided_permutation_p_unadjusted'])
    for key,s in [('overall_fd',report['overall_fd']),*report['within_band'].items()]:w.writerow([key,s['n'],s['spearman_rho'],s['exact_two_sided_permutation_p_unadjusted']])
r={'job':os.environ.get('SLURM_JOB_ID'),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_new_training':True,'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name!='receipt.json'}}
(OUT/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('CORRELATION_PLOTS_COMPLETE',flush=True)
