"""
Method.
"""
import numpy as np
from sklearn.base import clone

class MCS:
    def __init__(self, mfe, base_clf, base_oneclass, threshold, max_oc=50, min_concept_len=5, n_epochs=None):
        self.mfe = mfe
        self.base_clf = base_clf
        self.base_oneclass = base_oneclass
        self.threshold = threshold
        self.max_oc = max_oc
        self.min_concept_len = min_concept_len

        self.one_classes = []
        self.clfs = []
        
        self.current_concept = 0

        self.meta = []
        self.counter = 0

        self._past_concepts = []
        self._past_support = []
        self.last_switch=0
        self.n_epochs = n_epochs
    
    def partial_fit(self, X, y, classes=[0,1]):
        out = self.mfe.fit(X, y).extract()
        self.meta.append(out[1])
        
        self.meta_arr = np.array(self.meta)
        
        if self.counter == 0:
            self.one_classes.append(clone(self.base_oneclass))
            self.clfs.append(clone(self.base_clf))
            self._past_support.append([])
            
        else:
            # SUPPORT
            supports = []
            for c_id in range(len(self.one_classes)):
                r = np.copy(self.meta_arr[-1])
                r[np.isnan(r)]=1
                
                s = self.one_classes[c_id].decision_function([r])
                supports.append(s)
                self._past_support[c_id].append(s)
                        
            # CHECK SHIFT
            if supports[self.current_concept] < -self.threshold and (self.counter-self.last_switch)>self.min_concept_len:
                self.last_switch = self.counter
                
                # CHECK RECURRING
                max_support_idx = np.argmax(supports)
                if supports[max_support_idx] < -self.threshold:
                    # NEW
                    self.current_concept = len(self.one_classes)
                    self.one_classes.append(clone(self.base_oneclass))
                    self.clfs.append(clone(self.base_clf))
                    print('new', np.round(self.threshold,3), self.current_concept)
                    
                    self._past_support.append([])
                else:
                    #RECURRING
                    self.current_concept = max_support_idx
                    print('rec', np.round(self.threshold,3), self.current_concept)

            
        # FIT
        if self.n_epochs is not None:
            [self.clfs[self.current_concept].partial_fit(X, y, classes) for i in range(self.n_epochs)]
        else:
            self.clfs[self.current_concept].partial_fit(X, y, classes)
        self._past_concepts.append(self.current_concept)
        # print(self._past_concepts)

        # FIT ONECLASS
        _x = self.meta_arr[self.last_switch:, :]
        _x[np.isnan(_x)]=1
        if len(_x)>self.max_oc:
            idx = np.random.choice(np.arange(len(_x)), self.max_oc)
            _x = _x[idx]
        self.one_classes[self.current_concept].fit(_x)
        
        self.counter+=1
        
        
    def predict(self, X):
        return self.clfs[self.current_concept].predict(X)
