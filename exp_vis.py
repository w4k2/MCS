from pymfe.mfe import MFE
import numpy as np
from sklearn import clone
from sklearn.svm import OneClassSVM
import strlearn as sl
from mcs import MCS
from sklearn.naive_bayes import GaussianNB
from strlearn.evaluators import TestThenTrain
import matplotlib.pyplot as plt

np.random.seed(122)

n_chunks = 500
chunk_size = 250
recurring = True

n_features = [10,20,30]
n_drifts= [5,7,9,11]

random_states = np.random.randint(100, 1000000, 10)

measures_names = ['mean', 'median', 't_mean', 'gravity',
                  'w_lambda', 'p_trace', 'can_cor', 'lh_trace',
                   'roy_root', 'cov', 'cor']
mfe = MFE(groups="statistical", features=measures_names, summary=['mean'])
m_oc = 25
min_concept_len = 5

thresholds = [2.2, 2.0, 1.6]
base_oneclass=OneClassSVM(kernel='rbf')

config = {
    'n_drifts': n_drifts[1],
    'n_chunks': n_chunks,
    'chunk_size': chunk_size,
    'n_features': n_features[1],
    'n_informative': int(0.3*n_features[1]),
    'n_redundant': 0,
    'recurring': True,
    'concept_sigmoid_spacing': 999,
    'random_state': random_states[0]
    }
    
stream = sl.streams.StreamGenerator(**config)

mcs = MCS(mfe, base_clf=GaussianNB(), 
        base_oneclass=clone(base_oneclass), 
        threshold=thresholds[1], 
        max_oc=m_oc,
        min_concept_len=min_concept_len)
                    
evaluator = TestThenTrain(metrics=sl.metrics.balanced_accuracy_score, verbose=True)
evaluator.process(stream, [mcs])

print(mcs.meta_arr.shape)
print(mcs._past_concepts)

measures_names = ['mean', 'median', 'truncated mean', 'gravity',
                  'wilk', 'pillai\'s trace', 'cannonical cor.', 'lh trace',
                   'roy\'s root', 'covariance', 'correlation']
r = [0,3,5,9,10]
k = len(r)
_x = mcs.meta_arr[:,r]

fig, ax = plt.subplots(k,k,figsize=(10,9))

for i in range(k):
    for j in range(k):
        ax[i,j].scatter(_x[:,j], _x[:,i], c=mcs._past_concepts, cmap='coolwarm', s=3)
        # ax[i,j].set_xticks([])
        # ax[i,j].set_yticks([])
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].grid(ls=':')

        if i==0:
            ax[-1,j].set_xlabel('%s' % measures_names[r[j]],fontsize=13)
        if j==0:
            ax[i,j].set_ylabel('%s' % measures_names[r[i]], fontsize=13)

fig.align_ylabels()

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/scatter_vis.png')
plt.savefig('figures/scatter_vis.eps')