import numpy as np
import matplotlib.pyplot as plt

class LR:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def pred(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    def analyze_linear_regression(self, y_test, y_pred):

        error = y_pred - y_test
        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)

        # R-squared (coefficient of determination)
        total_variance = np.sum((y_test - np.mean(y_test)) ** 2)
        explained_variance = np.sum((y_pred - y_test) ** 2)
        r2 = 1.0 - (explained_variance / total_variance)

        # Visualization for MAE
        plt.scatter(y_test, error, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Actual')
        plt.ylabel('Error')
        plt.title('MAE - Mean Absolute Error')
        plt.show()

        # Visualization for MSE
        plt.scatter(y_test, error ** 2, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Actual')
        plt.ylabel('Squared Error')
        plt.title('MSE - Mean Squared Error')
        plt.show()

        # Visualization for RMSE
        plt.scatter(y_test, np.sqrt(error ** 2), color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Actual')
        plt.ylabel('Root Mean Squared Error')
        plt.title('RMSE - Root Mean Squared Error')
        plt.show()

        # Visualization for R-squared (Actual vs. Predicted)
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot(y_test, y_test, color='red', label='Perfect Fit')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('R-squared - Actual vs. Predicted')
        plt.legend()
        plt.show()

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R-squared (R2):", r2)
