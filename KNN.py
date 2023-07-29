import numpy as np
from collections import Counter
def eucl_dist(x1,x2):
    dist=np.sqrt(sum((x1-x2)**2))
    return dist

class KNN:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,Y):
        self.X_train=X
        self.y_train=Y

    def predict(self,X):
        prediction=[self._pred(x) for x in X]
        return prediction
    
    def _pred(self,x):
        distances=[eucl_dist(x,x_train) for x_train in self.X_train]

        indices=np.argsort(distances[:self.k])
        label=[self.y_train[i] for i in indices]
        print(label)
        #most common
        most_comm=Counter(label).most_common()
        print(most_comm)
     
    
