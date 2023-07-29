import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['green','black','red'])
from KNN import KNN
iris=datasets.load_iris()
X,y=iris.data,iris.target 
X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=.2,random_state=123)
plt.figure()
plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolors='k',s=30)
plt.show()



clf=KNN(k=5)
clf.fit(X=X_train,Y=y_train)
preds=clf.predict(X_test)
print(preds)