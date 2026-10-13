from pathlib import Path
import os, json, csv, hashlib
import numpy as np
from scipy.fft import dctn
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(os.environ.get('DATA_CAMPAIGN','/data/personal/andrinr/runner/results/mosaic/buoyancy-data-mix-20261013'))
OUT=Path(__file__).parent/'artifacts'; OUT.mkdir(exist_ok=True)
SOLVERS={'phiflow':'PhiFlow','waterlily':'WaterLily','oceananigans':'Oceananigans','jax-cfd':'JAX-CFD','ins-jl':'INS.jl'}
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#6B4423']
METHODS=['uncorrected','solver_in_loop']
N=64
kx,ky=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
q=np.hypot(kx,ky);shell=np.floor(q).astype(int);ns=shell.max()+1
counts=np.bincount(shell.ravel(),minlength=ns)
qplot=np.bincount(shell.ravel(),weights=q.ravel(),minlength=ns)/counts
bands={'DC':q==0,'Large: 0<q<=4':(q>0)&(q<=4),'Medium: 4<q<=16':(q>4)&(q<=16),'Small: q>16':q>16}

def center(s):
    ux=np.pad(s[:,:-1,:,0],((0,0),(1,1),(0,0)))
    uy=np.pad(s[:,:,:-1,1],((0,0),(0,0),(1,1)))
    return np.stack(((ux[:,:-1]+ux[:,1:])/2,(uy[:,:,:-1]+uy[:,:,1:])/2),axis=-1)

def modes(error):
    transformed=dctn(error,type=2,norm='ortho',axes=(1,2),workers=1)
    power=transformed**2/N**2
    if error.ndim==4:power=power.sum(axis=-1)
    physical=(error**2).sum(axis=-1).mean(axis=(1,2)) if error.ndim==4 else (error**2).mean(axis=(1,2))
    np.testing.assert_allclose(power.sum(axis=(1,2)),physical,rtol=1e-12,atol=1e-18)
    return power,physical

# Analytic mode placement and Parseval checks before analyzing solver results.
a=np.cos(np.pi*3*(np.arange(N)+.5)/N)[:,None]*np.cos(np.pi*4*(np.arange(N)+.5)/N)[None,:]
p,_=modes(a[None]);np.testing.assert_allclose(p[0,3,4],.25,rtol=1e-12)
assert np.sum(p)-p[0,3,4]<1e-14
z,_=modes(np.zeros((1,N,N,2)));assert np.count_nonzero(z)==0
report={'source_commit':'a2339b9587b2eccb845c324325cb5288aae6aa04','frame_selection':'Predicted frames 1..100, t=0.04..4, equal time weights; initial frame excluded','transform':'scipy.fft.dctn(type=2,norm=ortho), separately along two spatial axes, no mean subtraction or window','definition':'E(kx,ky) = mean_t |DCT(error)|^2 / N^2; velocity sums both cell-centered component powers. Transform error fields, not differences of field power spectra.','shell_definition':'Sum modal powers for floor(sqrt(kx^2+ky^2))=j; plotted x is mean radial index of modes in that shell; shell 0 is DC','physical_scale':'q=sqrt(kx^2+ky^2); cosine wavenumber pi*q/L, nominal wavelength 2L/q for q>0; L=1','band_definitions':list(bands),'ratio_definition':'Mean SOL error power / mean uncorrected error power; below1 improves. Shell ratios omitted only if baseline shell power <1e-8 times that solver/field total baseline MSE. Raw values retained in JSON/CSV.','limitations':['Cosine basis diagnostic for closed walls, not a periodic Fourier energy spectrum or a solver eigenmode decomposition.','Cell-centering attenuates highest staggered velocity modes. Vector MSE here is not the original normalized training loss.','64x64 resolved scales only; fine reference first restricted128to64. Different solvers use different fine references.','One held-out trajectory and one selected model per backend; no uncertainty estimates.','Band boundaries predefined descriptive labels, not inferred physical inertial/dissipation regimes.'],'checks':{'analytic_single_mode_3_4':True,'zero_error':True,'parseval_all_frames':True},'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},'q_shell_mean':qplot.tolist(),'shell_mode_counts':counts.tolist(),'solvers':{}}
csvrows=[];bandrows=[]
for key,name in SOLVERS.items():
    folder=ROOT/key/'results/buoyancy-grid/optimization/solver_in_loop'
    result=json.loads((folder/'result.json').read_text());m=result['results'][0]['metrics']
    with np.load(folder/'corrector_fields.npz',allow_pickle=False) as data:
        idx=list(map(str,data['solver_names'])).index(name)
        states={method:np.asarray(data[f'{method}_{idx}'],dtype=np.float64) for method in ['reference',*METHODS]}
    assert all(s.shape==(101,N,N,3) and np.isfinite(s).all() for s in states.values())
    row={'selected_update':m['solver_in_loop']['selected_update'],'source_hashes':{f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in ['corrector_fields.npz','result.json']},'fields':{}}
    for field in ['scalar','velocity']:
        errors={method:(states[method][1:,...,2]-states['reference'][1:,...,2] if field=='scalar' else center(states[method])[1:]-center(states['reference'])[1:]) for method in METHODS}
        powers={};fieldrow={}
        for method in METHODS:
            power,physical=modes(errors[method]);meanpower=power.mean(axis=0);powers[method]=meanpower
            spectrum=np.bincount(shell.ravel(),weights=meanpower.ravel(),minlength=ns)
            np.testing.assert_allclose(spectrum.sum(),physical.mean(),rtol=1e-12,atol=1e-18)
            fieldrow[method]={'mean_error_power':spectrum.tolist(),'physical_mse':float(physical.mean()),'bands':{label:float(meanpower[mask].sum()) for label,mask in bands.items()}}
        b=np.asarray(fieldrow['uncorrected']['mean_error_power']);s=np.asarray(fieldrow['solver_in_loop']['mean_error_power'])
        keep=b>=1e-8*b.sum();ratio=np.divide(s,b,out=np.full_like(s,np.nan),where=b>0)
        fieldrow['ratio_retained']=[float(v) if k and np.isfinite(v) else None for v,k in zip(ratio,keep)]
        fieldrow['bands']={}
        for label,mask in bands.items():
            baseline=float(powers['uncorrected'][mask].sum());corrected=float(powers['solver_in_loop'][mask].sum())
            entry={'uncorrected_power':baseline,'sol_power':corrected,'baseline_fraction':baseline/b.sum(),'reduction_percent':100*(1-corrected/baseline) if baseline>0 else None}
            fieldrow['bands'][label]=entry
            bandrows.append([name,field,label,baseline,corrected,entry['baseline_fraction'],entry['reduction_percent']])
        for j in range(ns):csvrows.append([name,field,j,qplot[j],int(counts[j]),b[j],s[j],ratio[j],bool(keep[j])])
        row['fields'][field]=fieldrow
    report['solvers'][name]=row
(OUT/'spectral_errors.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for filename,header,rows in [('spectral_errors.csv',['solver','field','shell','mean_radial_index','mode_count','uncorrected_power','SOL_power','SOL_over_uncorrected','ratio_above_threshold'],csvrows),('spectral_bands.csv',['solver','field','band','uncorrected_power','SOL_power','baseline_energy_fraction','reduction_percent'],bandrows)]:
    with (OUT/filename).open('w') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

for variant in ['power','ratio']:
    fig,axes=plt.subplots(1,2,figsize=(14,6.5));fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.76,wspace=.23)
    for ax,field,title in zip(axes,['scalar','velocity'],['Scalar error','Cell-centered velocity-vector error']):
        for (name,row),color in zip(report['solvers'].items(),COLORS):
            data=row['fields'][field]
            if variant=='power':
                for method in METHODS:
                    ax.plot(qplot[1:],data[method]['mean_error_power'][1:],ls='--' if method=='uncorrected' else '-',color=color,lw=1.7)
            else:
                ax.plot(qplot[1:],[np.nan if v is None else v for v in data['ratio_retained'][1:]],color=color,lw=1.8)
        for boundary in [4,16]:ax.axvline(boundary,color='gray',lw=.7,alpha=.5)
        if variant=='ratio':ax.axhline(1,color='black',lw=1.1,ls='--')
        ax.set_xscale('log');ax.set_xlim(1,90)
        if variant=='power':
            ax.set_yscale('symlog',linthresh=1e-11,linscale=.5);ax.set_ylim(bottom=0)
        else:ax.set_yscale('log')
        ax.set_xticks([1,2,4,8,16,32,64]);ax.set_xticklabels(['1','2','4','8','16','32','64'])
        ax.set_xlabel('Radial cosine mode index q (larger q = finer scale)')
        ax.set_ylabel('Mean squared error per radial shell' if variant=='power' else 'SOL error power / uncorrected error power')
        ax.set_title(title);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Where does the corrector reduce spatial error?' if variant=='ratio' else 'Spatial error spectra: uncorrected solver and learned corrector',fontsize=16,y=.97)
    fig.text(.5,.91,'Held-out seed 100 · mean over t=0.04–4 · 64² fields vs each solver’s restricted 128² reference',ha='center',fontsize=10)
    fig.legend([Line2D([0],[0],color=c,lw=2) for c in COLORS],[f'{name} (update {row["selected_update"]})' for name,row in report['solvers'].items()],loc='upper center',bbox_to_anchor=(.5,.88),ncol=5,frameon=False)
    if variant=='power':
        fig.legend([Line2D([0],[0],color='#333',lw=2,ls='--'),Line2D([0],[0],color='#333',lw=2)],['Uncorrected','SOL'],loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False)
        foot='Nonzero-mode power; mean-field bias (DC) reported separately. Vertical scale is linear below 10⁻¹¹ to keep tiny tails visible.'
    else:
        fig.text(.5,.12,'Below 1: improvement     ·     Above 1: worsening     ·     Dashed horizontal line: unchanged',ha='center',fontsize=11)
        foot='Ratios omitted where baseline shell power < 10⁻⁸ of total baseline error. Consult absolute power spectra for scale importance.'
    fig.text(.5,.055,foot,ha='center',fontsize=9)
    fig.text(.5,.025,'Closed-box cosine diagnostic; centered velocity is not the staggered training metric. Vertical lines: descriptive band boundaries q=4,16.',ha='center',fontsize=8.5)
    for ext in ['png','svg']:fig.savefig(OUT/f'spectral_error_{variant}.{ext}',dpi=150,facecolor='white')
    plt.close(fig)
# Predeclared bands summarize reduction together with the original error share.
fig,axes=plt.subplots(1,2,figsize=(13,5.7));fig.subplots_adjust(left=.13,right=.96,bottom=.22,top=.78,wspace=.32)
labels=list(bands)[1:]
for ax,field,title in zip(axes,['scalar','velocity'],['Scalar error power','Cell-centered velocity-vector error power']):
    values=np.array([[row['fields'][field]['bands'][label]['reduction_percent'] for label in labels] for row in report['solvers'].values()])
    im=ax.imshow(values,cmap='RdBu',vmin=-100,vmax=100,aspect='auto')
    ax.set_xticks(range(3));ax.set_xticklabels(['Large\n0 < q ≤ 4','Medium\n4 < q ≤ 16','Small\nq > 16'])
    ax.set_yticks(range(5));ax.set_yticklabels(list(report['solvers']));ax.set_title(title)
    for i,row in enumerate(report['solvers'].values()):
        for j,label in enumerate(labels):
            band=row['fields'][field]['bands'][label]
            ax.text(j,i,f"{band['reduction_percent']:.0f}% less\n({100*band['baseline_fraction']:.1f}% of base)",ha='center',va='center',fontsize=10,color='white' if abs(values[i,j])>55 else '#111111')
fig.suptitle('Which scales improve — and how much original error is there?',fontsize=15,y=.97)
fig.text(.5,.9,'Reduction = 100 × (1 − SOL power / uncorrected power), averaged across the held-out rollout',ha='center',fontsize=10)
fig.text(.5,.08,'Parentheses show that band’s share of uncorrected error power. Bands use exact radial indices, before shell binning.',ha='center',fontsize=9)
fig.text(.5,.04,'DC is separate: negligible except PhiFlow scalar (1.08% of baseline power), where correction increases DC error by 13.4%.',ha='center',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'spectral_error_bands.{ext}',dpi=150,facecolor='white')
plt.close(fig)
receipt={'source_commit':report['source_commit'],'job_id':os.environ.get('SLURM_JOB_ID'),'no_new_training':True,'method':{k:v for k,v in report.items() if k not in ['solvers','q_shell_mean','shell_mode_counts']},'source_hashes':{name:row['source_hashes'] for name,row in report['solvers'].items()},'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.name!='spectral_receipt.json'}}
(OUT/'spectral_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('SPECTRAL_ANALYSIS_COMPLETE',flush=True)
for name,row in report['solvers'].items():
    print(name,{field:{b:round(v['reduction_percent'],2) for b,v in data['bands'].items()} for field,data in row['fields'].items()},flush=True)
