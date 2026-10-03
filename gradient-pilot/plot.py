import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).parent
report=json.loads((root/'results/validation.json').read_text())
models=list(report['models'])
labels={'XLB':'XLB','baseline':'Original','field':'Field only','vjp001':'VJP 0.01','vjp01':'VJP 0.1','vjp1':'VJP 1','secant01':'Secant 0.1','linear_vjp01':'Linear + VJP','path_field':'Path field','path_vjp01':'Path + VJP'}
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
metrics=[('forward','relative_l2','Forward error vs XLB (%)'),('vjp','relative_l2','Common-cotangent VJP error (%)'),('xlb_recovery','ic_relative_l2','XLB-target recovery error (%)'),('fourier_xlb_recovery','ic_relative_l2','Fourier-restricted recovery error (%)')]
for ax,(key,value,title) in zip(axes.flat,metrics):
    vals=np.array([[c[key][value]*100 for c in report['models'][m]] for m in models])
    x=np.arange(len(models));colors=['#555555' if m=='XLB' else '#BB5566' if m=='baseline' else '#4477AA' for m in models]
    ax.bar(x,vals.mean(axis=1),color=colors,alpha=.75)
    for seed in range(vals.shape[1]):ax.scatter(x+(seed-1)*.13,vals[:,seed],s=15,color='black',zorder=3)
    ax.set_xticks(x,[labels[m] for m in models],rotation=45,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Gradient-training pilot: three validation cases\nBars: means; dots: individual cases. Native SciPy recovery, not registered benchmark results.',fontsize=13)
fig.savefig(root/'pilot-comparison.png',dpi=170)
