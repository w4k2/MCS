import numpy as np
import matplotlib.pyplot as plt
from utils import find_real_drift

res_clf = np.load('results_v4/res_clf.npy')
res_concepts = np.load('results_v4/res_concepts.npy')
thresholds = np.linspace(0.5, 4, 100)

print(res_clf.shape) # drifts, reps, thresholds
print(res_concepts.shape) # drifts, reps, thresholds

drifts = find_real_drift(500, 7)
print(drifts)

concepts_gt = np.zeros((499))

c=0
for i in range(499):
    if i in drifts:
        c += 1
        c = c%2
    concepts_gt[i] = c
    
print(concepts_gt)

fig, ax = plt.subplots(1,3,figsize=(10,7), sharex=True, sharey=True)

ax[0].imshow(res_concepts[0].swapaxes(0,1).reshape(-1,499), 
             cmap='binary', vmin=0, vmax=4)
ax[1].imshow(res_concepts[1].swapaxes(0,1).reshape(-1,499),
             cmap='binary', vmin=0, vmax=4)
ax[2].imshow(res_concepts[2].swapaxes(0,1).reshape(-1,499), 
             cmap='binary', vmin=0, vmax=4)

ax[0].set_title('10 features')
ax[1].set_title('20 features')
ax[2].set_title('30 features')

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    aa.set_yticks(np.arange(5,1005,10)[::10], ['%0.2f' % a for a in thresholds][::10])
    aa.set_xticks(drifts)
    
    aa.set_xlabel('chunk', fontsize=12)

ax[0].set_ylabel('threshold', fontsize=12)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/tunning2.png')
plt.savefig('figures/tunning2.eps')