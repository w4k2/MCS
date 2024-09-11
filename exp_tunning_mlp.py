from pymfe.mfe import MFE
import numpy as np
from sklearn import clone
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
import strlearn as sl
from old.mcs import MCS
from sklearn.naive_bayes import GaussianNB
from strlearn.evaluators import TestThenTrain
from tqdm import tqdm

np.random.seed(122)

# config

n_chunks=500
chunk_size=250
recurring = True 
n_drifts= 7 

random_states = np.random.randint(100, 1000000, 10)

measures_names = ['mean', 'median', 't_mean', 'gravity',
                  'w_lambda', 'p_trace', 'can_cor', 'lh_trace',
                   'roy_root', 'cov', 'cor']
mfes = MFE(groups="statistical", features=measures_names, summary=['mean'])

max_ocs = 25
min_concept_lens = 5
thresholds = np.linspace(0.5, 4, 100)

base_clfs = MLPClassifier()
base_oneclass = OneClassSVM(kernel='rbf')

n_features= [10,20,30]


# experiment

res_clf = np.zeros((len(n_features), len(random_states), len(thresholds), n_chunks-1))
res_concepts = np.zeros((len(n_features), len(random_states), len(thresholds), n_chunks-1))

pbar = tqdm(total=len(n_features)*len(random_states))

for nf_id, nf in enumerate(n_features):
    for rs_id, rs in enumerate(random_states):
        
        config = {
            'n_drifts': n_drifts,
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'n_features': nf,
            'n_informative': int(0.3*nf),
            'n_redundant': 0,
            'recurring': recurring,
            'concept_sigmoid_spacing': 999, # sudden
            'random_state': rs
            }
            
        stream = sl.streams.StreamGenerator(**config)
        
        methods = []             
        
        for t_id, t in enumerate(thresholds):
            method = MCS(mfes, 
                            base_clf=clone(base_clfs), 
                            base_oneclass=clone(base_oneclass), 
                            threshold=t, 
                            max_oc=max_ocs, 
                            min_concept_len=min_concept_lens)
            
            methods.append(method)
        print(len(methods))
                            
        evaluator = TestThenTrain(metrics=sl.metrics.balanced_accuracy_score)
        evaluator.process(stream, methods)
        
        pbar.update(1)

        c=0                 
        for t_id, t in enumerate(thresholds):
            res_clf[nf_id, rs_id, t_id] = evaluator.scores[c,:,0]
            res_concepts[nf_id, rs_id, t_id] = methods[c]._past_concepts[1:]
            c+=1
                        

        np.save('results_v4/res_clf_mlp.npy', res_clf)              
        np.save('results_v4/res_concepts_mlp.npy', res_concepts)
        # exit()
        
        
