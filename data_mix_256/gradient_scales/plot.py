from pathlib import Path
import os,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT=Path(os.environ.get('GRADIENT_CAMPAIGN','/data/personal/andrinr/runner/results/mosaic/buoyancy-gradient-scales-20261013'))
SPECTRA=Path('/data/personal/andrinr/runner/results/mosaic/buoyancy-data-mix-20261013/spectral-errors/artifacts/spectral_errors.json')
OUT=Path(__file__).parent/'artifacts';OUT.mkdir(exist_ok=True)
SOLVERS={'phiflow':'PhiFlow','waterlily':'WaterLily','oceananigans':'Oceananigans','jax-cfd':'JAX-CFD','ins-jl':'INS.jl'}
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#6B4423'];FIELDS=['scalar','velocity'];BANDS=['large','medium','small'];BANDKEYS=['Large: 0<q<=4','Medium: 4<q<=16','Small: q>16']
spectral=json.loads(SPECTRA.read_text());combined={'spectral_source_sha256':hashlib.sha256(SPECTRA.read_bytes()).hexdigest(),'summary_convention':'Median over four predetermined reference-state samples. Ranges are min/max over states, not confidence intervals. Same eight output directions for all states/backends per field/band.','solvers':{}}
csvrows=[];fdrows=[];common_basis=None
for key,name in SOLVERS.items():
    p=ROOT/key/'gradient_scales.json';d=json.loads(p.read_text());assert d['complete'] and len(d['samples'])==24
    assert d['source_hashes']==spectral['solvers'][name]['source_hashes']
    if common_basis is None:common_basis=d['basis_hashes']
    else:assert d['basis_hashes']==common_basis
    summary={}
    for field in FIELDS:
        summary[field]={}
        for band,bandkey in zip(BANDS,BANDKEYS):
            samples=[s for s in d['samples'] if s['field']==field and s['band']==band];assert len(samples)==4
            energy=spectral['solvers'][name]['fields'][field]['bands'][bandkey]
            row={'error_reduction_percent':energy['reduction_percent'],'baseline_error_fraction':energy['baseline_fraction']}
            for metric in ['rms_gain','gain_relative_H0','condition','condition_relative_H0']:
                values=[s['H8'][metric] for s in samples]
                row[metric]={'median':float(np.median(values)),'minimum':min(values),'maximum':max(values)} if all(v is not None for v in values) else {'median':None,'minimum':None,'maximum':None}
            row['H0']=samples[0]['H0'];row['fd_max_relative_error']=max(f['relative_error'] for s in samples for f in s['finite_differences'])
            row['fd_flag_over_5_percent']=row['fd_max_relative_error']>.05
            summary[field][band]=row
            csvrows.append([name,field,band,row['error_reduction_percent'],row['baseline_error_fraction'],*[row[m]['median'] for m in ['rms_gain','gain_relative_H0','condition','condition_relative_H0']],row['H0']['condition'],row['fd_max_relative_error']])
            for s in samples:
                for f in s['finite_differences']:fdrows.append([name,field,band,s['frame'],s['time'],f['epsilon'],f['autodiff'],f['finite_difference'],f['absolute_error'],f['relative_error']])
    combined['solvers'][name]={'data_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gradient_data':d,'summary':summary}
(OUT/'gradient_spectral_comparison.json').write_text(json.dumps(combined,indent=2,allow_nan=False)+'\n')
for name,header,rows in [('gradient_spectral_summary.csv',['solver','field','band','error_power_reduction_percent','baseline_error_fraction','median_rms_gain','median_gain_H8_over_H0','median_sampled_condition_H8','median_condition_H8_over_H0','condition_H0','max_fixed_fd_relative_error'],csvrows),('gradient_directional_checks.csv',['solver','field','band','frame','time','epsilon','autodiff','finite_difference','absolute_error','relative_error'],fdrows)]:
    with (OUT/name).open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)

fig,axes=plt.subplots(3,2,figsize=(15,13));fig.subplots_adjust(left=.11,right=.96,bottom=.12,top=.89,hspace=.48,wspace=.38)
for fi,field in enumerate(FIELDS):
    for mi,metric in enumerate(['error_reduction_percent','gain_relative_H0','condition']):
        ax=axes[mi,fi]
        rows=[[combined['solvers'][name]['summary'][field][band] for band in BANDS] for name in SOLVERS.values()]
        v=np.asarray([[r[metric] if mi==0 else (r[metric]['median'] if r[metric]['median'] is not None else np.nan) for r in row] for row in rows])
        if mi==0:
            norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100);cmap='RdBu';title='Corrector: error power removed (%)'
        elif mi==1:
            norm=LogNorm(vmin=.01,vmax=10);cmap='viridis';title='Solver: adjoint RMS gain, H8 / H0'
        else:
            allv=[r['summary'][f][b]['condition']['median'] for r in combined['solvers'].values() for f in FIELDS for b in BANDS]
            norm=LogNorm(vmin=1,vmax=max(2.,max(a for a in allv if a is not None)));cmap='magma_r';title='Solver: sampled condition κ at H8'
        im=ax.imshow(v,norm=norm,cmap=cmap,aspect='auto')
        ax.set_xticks(range(3));ax.set_xticklabels(['Large\n0<q≤4','Medium\n4<q≤16','Small\nq>16'],fontsize=9)
        ax.set_yticks(range(5));ax.set_yticklabels(list(SOLVERS.values()),fontsize=10);ax.set_title(title,fontsize=11)
        for i,row in enumerate(rows):
            for j,r in enumerate(row):
                if mi==0:text=f"{r[metric]:.0f}% less\n({100*r['baseline_error_fraction']:.1f}% of base)";color='white' if abs(v[i,j])>55 else '#111'
                elif mi==1:
                    m=r[metric];text=f"{m['median']:.2g}×"+(' †' if r['fd_flag_over_5_percent'] else '')+f"\n[{m['minimum']:.2g}, {m['maximum']:.2g}]";color='white' if norm(v[i,j])<.65 else '#111'
                else:
                    m=r[metric];text=('unresolved' if m['median'] is None else f"{m['median']:.2g}")+(' †' if r['fd_flag_over_5_percent'] else '')+f"\n(H0: {r['H0']['condition']:.2g})";color='white' if np.isfinite(v[i,j]) and norm(v[i,j])>.6 else '#111'
                ax.text(j,i,text,ha='center',va='center',fontsize=9,color=color)
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
fig.suptitle('Correction by spatial scale, alongside local solver-gradient behavior',fontsize=17,y=.985)
fig.text(.315,.94,'Scalar output',ha='center',fontsize=14);fig.text(.785,.94,'Cell-centered velocity output',ha='center',fontsize=14)
fig.text(.5,.91,'Held-out flow · same selected correctors · H8 = 0.32 physical time · 8 shared orthonormal probes per band',ha='center',fontsize=10)
fig.text(.5,.072,'Gradient cells: medians over t=0, 0.96, 1.92, 2.88; brackets show the state range. H0 accounts for observation/projection geometry.',ha='center',fontsize=9)
fig.text(.5,.05,'κ is conditioning of an eight-direction sampled adjoint restriction, not the full Jacobian or neural training. Input velocity tangents are divergence-free; scalar is free.',ha='center',fontsize=8.5)
fig.text(.5,.028,'All fixed-step directional checks <5%; maximum 4.95% for JAX-CFD. Error reductions span the full rollout; association is not causation.',ha='center',fontsize=8.5)
for ext in ['png','svg']:fig.savefig(OUT/f'gradient_scale_comparison.{ext}',dpi=150,facecolor='white')
plt.close(fig)

for metric,label,filename in [('gain_relative_H0','Adjoint RMS gain H8 / H0','gradient_gain_vs_correction'),('condition_relative_H0','Sampled condition ratio κ(H8) / κ(H0)','gradient_condition_vs_correction')]:
    fig,axes=plt.subplots(1,2,figsize=(14,6));fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.79,wspace=.22)
    for ax,field,title in zip(axes,FIELDS,['Scalar output','Cell-centered velocity output']):
        for (name,solver),color in zip(combined['solvers'].items(),COLORS):
            for band,marker in zip(BANDS,['o','s','^']):
                r=solver['summary'][field][band];m=r[metric]
                if m['median'] is None:continue
                ax.errorbar(m['median'],r['error_reduction_percent'],xerr=[[m['median']-m['minimum']],[m['maximum']-m['median']]],fmt=marker,color=color,markersize=7,capsize=2,alpha=.85)
        ax.set_xscale('log' if metric == 'gain_relative_H0' else 'linear');ax.axvline(1,color='gray',ls=':',lw=1);ax.axhline(0,color='gray',ls=':',lw=1);ax.set_xlabel(label);ax.set_ylabel('Error power removed by corrector (%)');ax.set_title(title);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Scale-specific gradient behavior versus correction benefit',fontsize=16,y=.97)
    fig.legend([Line2D([0],[0],color=c,marker='o',ls='') for c in COLORS],list(SOLVERS.values()),loc='upper center',bbox_to_anchor=(.5,.92),ncol=5,frameon=False)
    fig.legend([Line2D([0],[0],color='#333',marker=m,ls='') for m in ['o','s','^']],['Large scales','Medium scales','Small scales'],loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False)
    fig.text(.5,.065,'Points: median over four reference states; bars: state range, not uncertainty intervals. No causal inference from this comparison.',ha='center',fontsize=9)
    fig.text(.5,.025,'Eight-direction sampled adjoint restriction; fixed-epsilon consistency checks and raw H0/H8 singular values accompany the data.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(OUT/f'{filename}.{ext}',dpi=150,facecolor='white')
    plt.close(fig)
receipt={'method':'Plot only: joins saved gradient results with prior error spectra after matching original rollout hashes and cotangent-basis hashes across all five backends. No new training.','sampled_condition':'sigma_max/sigma_min of the 8×D projected-gradient matrix; ratio to H0 is ratio of condition numbers, not a preconditioned condition number.','reference_states':[0,24,48,72],'horizon':8,'physical_duration':.32,'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.name!='plot_receipt.json'}}
(OUT/'plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PLOTS_COMPLETE',flush=True)
for name,r in combined['solvers'].items():
    print(name,'FD max',max(s['fd_max_relative_error'] for f in r['summary'].values() for s in f.values()),flush=True)
