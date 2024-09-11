import numpy as np
from scipy import stats

n_features = [10,20,30]
n_drifts= [5,7,9,11]

methods = ['GNB', 'MCS-GNB', 'MLP', 'MCS-MLP', 'HTC', 'HTC-MCS']

res_clf = np.load('results_v4/res_compare_all.npy')
print(res_clf.shape) # features, n_drifts, reps, methods, chunks-1
res_clf = np.mean(res_clf, axis=-1) # średnia w chunkach

mean_res = np.mean(res_clf, axis=2)
std_res = np.std(res_clf, axis=2)

rows = []
for f_id, f in enumerate(n_features):
    for d_id, d in enumerate(n_drifts):
        print(f, d)
        print(mean_res[f_id, d_id])
        print(std_res[f_id, d_id])
        
        r_temp = res_clf[f_id, d_id]
        
        # stat
        alpha = 0.05

        t_stat = np.zeros((len(methods), len(methods)))
        p_val = np.zeros((len(methods), len(methods)))
        better = np.zeros((len(methods), len(methods))).astype(bool)

        for i in range(len(methods)):
            for j in range(len(methods)):
                t_stat[i,j], p_val[i,j] = stats.ttest_rel(r_temp[:,i], r_temp[:,j])
                better[i,j] = np.mean(r_temp[:,i]) > np.mean(r_temp[:,j])
                
        significant = p_val<alpha
        significantly_better = significant*better

        print(significantly_better)
        
        # rows.append(['Features: %i, Drifts: %i' % (f,d)])
        
        r = []
        r.append('F: %i, D: %i' % (f,d))
        for m_id, m in enumerate(methods):
            r.append(np.round(mean_res[f_id, d_id, m_id],3))
        rows.append(r)   
        
        r = []
        r.append('')
        for m_id, m in enumerate(methods):
            better = np.argwhere(significantly_better[m_id]==True).flatten()
            if len(better)==5:
                r.append('all')           

            else:
                r.append(' '.join(map(str,better)))           
        rows.append(r)

from tabulate import tabulate
print(tabulate(rows, methods, tablefmt="latex"))
# print(tabulate(rows, methods, tablefmt="simple"))
        
        
