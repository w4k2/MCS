import numpy as np
import matplotlib.pyplot as plt
from utils import find_real_drift

n_features = [10,20,30]
n_drifts= [5,7,9,11]

methods = ['GNB', 'MCS-GNB', 'MLP', 'MCS-MLP', 'HTC', 'HTC-MCS']
cols = ['r', 'r', 'g', 'g', 'b', 'b']
lss = [':', '-', ':', '-', ':', '-']

res_clf = np.load('results_v4/res_compare_all.npy')
print(res_clf.shape) # features, n_drifts, drift_types, reps, methods, chunks-1

mean_res = np.mean(res_clf, axis=2)
print(mean_res.shape) # features, n_drifts, drift_types, methods, chunks-1

fig, axx = plt.subplots(4,3,figsize=(10,8), sharey=True)

for n_f_id, n_f in enumerate(n_features):
    for d_id, d in enumerate(n_drifts):
        drifts = find_real_drift(500, d)
            
        ax = axx[d_id, n_f_id]
        
        if d_id==0:
            ax.set_title('%i features' % n_f, fontsize=12)
            
        
        for m_id, m in enumerate(methods):
            temp = (np.cumsum(mean_res[n_f_id,d_id,m_id])/(np.linspace(0,1,500)[1:]))/500
            ax.plot(temp, label=m, c=cols[m_id], ls=lss[m_id])
        
        ax.set_xticks(drifts, np.arange(1,12)[:len(drifts)])

        if d_id==3:
            ax.set_xlabel('index of drift', fontsize=12)
        if n_f_id==0:
            ax.set_ylabel('%i drifts \n $accuracy$' % d, fontsize=12)

        ax.grid(ls=':')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

axx.ravel()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
          frameon=False, ncol=3)
plt.subplots_adjust(left=0.07, right=0.93, wspace=-0.35, hspace=0.05)

for aa in axx.ravel():
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.set_ylim(0.65,0.9)
   
plt.tight_layout() 
plt.savefig('foo.png')
plt.savefig('figures/comare_all_accumulated.png')
plt.savefig('figures/comare_all_accumulated.eps')
