from pymfe.mfe import MFE
import numpy as np
from sklearn import clone
from sklearn.svm import OneClassSVM
import strlearn as sl
from old.mcs import MCS
from sklearn.neural_network import MLPClassifier
from strlearn.evaluators import TestThenTrain
from tqdm import tqdm
from strlearn.ensembles import SEA, ROSE, AWE, WAE

np.random.seed(122)

class MLPWrap:
    def __init__(self, base, n_epochs):
        self.base = base
        self.n_epochs = n_epochs
        
    def predict(self, X):
        return self.base.predict(X)
    
    def partial_fit(self, X, y, classes):
        [self.base.partial_fit(X,y,classes) for i in range(self.n_epochs)]
        return self

# config

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

# experiment

n_iters = 20

n_methods=6

res_clf = np.zeros((len(n_features), len(n_drifts), len(random_states), n_methods, n_chunks-1))

pbar = tqdm(total=len(n_features)*len(n_drifts)*len(random_states))

for n_f_id, n_f in enumerate(n_features):
    for d_id, d in enumerate(n_drifts):
        for rs_id, rs in enumerate(random_states):
            
            config = {
                'n_drifts': d,
                'n_chunks': n_chunks,
                'chunk_size': chunk_size,
                'n_features': n_f,
                'n_informative': int(0.3*n_f),
                'n_redundant': 0,
                'recurring': True,
                'concept_sigmoid_spacing': 999,
                'random_state': rs
                }
                
            stream = sl.streams.StreamGenerator(**config)
            
            methods = [
                MLPWrap(MLPClassifier(random_state=1233), n_iters),
                MCS(mfe, base_clf=MLPClassifier(random_state=1233), 
                    base_oneclass=clone(base_oneclass), 
                    threshold=thresholds[n_f_id], 
                    max_oc=m_oc,
                    min_concept_len=min_concept_len,
                    n_epochs=n_iters),
                SEA(base_estimator=MLPClassifier(random_state=1233, max_iter=n_iters)),
                ROSE(base_estimator=MLPClassifier(random_state=1233, max_iter=n_iters)),
                AWE(base_estimator=MLPClassifier(random_state=1233, max_iter=n_iters)),
                WAE(base_estimator=MLPClassifier(random_state=1233, max_iter=n_iters))
            ]
                                
            evaluator = TestThenTrain(metrics=sl.metrics.balanced_accuracy_score, verbose=True)
            evaluator.process(stream, methods)
            
            pbar.update(1)

            print(evaluator.scores.shape)
            res_clf[n_f_id, d_id, rs_id] = evaluator.scores[:,:,0]                                        
        
        np.save('results_v4/res_compare_mlp_n_iter.npy', res_clf)         
