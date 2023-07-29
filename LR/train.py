import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['green','black','red'])
from sklearn.metrics import mean_squared_error
from LR import LR
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print("x_shape",X_train.shape)
print("y_shape",y_train.shape)
lr=LR()
lr.fit(X_train,y_train)
res=lr.pred(X_test)

print(res)
print("[INFO].........RESULT ANALYSIS:")
lr.analyze_linear_regression(y_test=y_test,y_pred=res)

print("[INFO].......OPTIMIZING:")
# Define a list of hyperparameter values to search
lr_values = [0.001, 0.01, 0.1, 0.2]
n_iter_values = [500, 1000, 1500]

best_lr = None
best_n_iter = None
best_rmse = float('inf')

# Perform grid search
for lr in lr_values:
    for n_iter in n_iter_values:
        # Create and train the model with current hyperparameters
        model = LR(lr=lr, n_iters=n_iter)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.pred(X_test)

        # Calculate root mean squared error (RMSE)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Check if current hyperparameters give better RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_lr = lr
            best_n_iter = n_iter

# Print the best hyperparameters and RMSE
print("Best Learning Rate:", best_lr)
print("Best Number of Iterations:", best_n_iter)
print("Best RMSE:", best_rmse)
print("[INFO]....TRAINING ON BEST HYPERT PARAMETERS:")
lr=LR(lr=best_lr,n_iters=best_n_iter)
lr.fit(X_train,y_train)
res=lr.pred(X_test)

print(res)

lr.analyze_linear_regression(y_test=y_test,y_pred=res)
# Define a list of hyperparameter values to search




