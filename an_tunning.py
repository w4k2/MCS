import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score
from scipy.ndimage import gaussian_filter1d

from old.utils import find_real_drift

res_clf = np.load('results_v4/res_clf.npy')
res_concepts = np.load('results_v4/res_concepts.npy')
thresholds = np.linspace(0.5, 4, 100)


print(res_clf.shape) # drifts, reps, thresholds, clfs, chunks
print(res_concepts.shape) # drifts, reps, thresholds, clfs, chunks


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

results_err = np.zeros((3,10,len(thresholds)))

for drf_type_id in range(3):
    for rep_id in range(10):
        for th_id in range(len(thresholds)):
            err = rand_score(concepts_gt, res_concepts[drf_type_id,rep_id,th_id,:])
            results_err[drf_type_id, rep_id, th_id] = err
            
            
res_err_mean = np.mean(results_err, axis=1)
res_clf_mean = np.mean(res_clf, axis=(1,3))


fig, ax = plt.subplots(1,2,figsize=(10,5))

s=1.
ax[0].plot(thresholds, gaussian_filter1d(res_err_mean[0], s), label = '10 features', c='r')
ax[0].plot(thresholds, gaussian_filter1d(res_err_mean[1], s), label = '20 features', c='b')
ax[0].plot(thresholds, gaussian_filter1d(res_err_mean[2], s), label = '30 features', c='g')

ax[1].plot(thresholds, gaussian_filter1d(res_clf_mean[0], s), label = '10 features', c='r')
ax[1].plot(thresholds, gaussian_filter1d(res_clf_mean[1], s), label = '20 features', c='b')
ax[1].plot(thresholds, gaussian_filter1d(res_clf_mean[2], s), label = '30 features', c='g')

ax[0].set_title('Concept identification')
ax[1].set_title('Classification quality')

ax[0].set_ylabel('rand score', fontsize=12)
ax[1].legend(frameon=False)
ax[1].set_ylabel('accuracy score', fontsize=12)

ax[0].set_xlim(0.5,4)
ax[1].set_xlim(0.5,4)

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.set_xlabel('threshold', fontsize=12)


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/tunning1.png')
plt.savefig('figures/tunning1.eps')


