import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import numpy as np
from src.multiple_linear_regression import MultipleLinearRegression

class RegressionPlotter():
    def __init__(self, model:MultipleLinearRegression) -> None:
        self._model = model
    
    def _plot_single_feature(self, X:np.ndarray, y:np.ndarray, axis:plt.Axes) -> Axes:
        axis.scatter(X, y)
        axis.plot(X, self._model.predict(X))
        axis.set_xlabel("Feature")
        axis.set_ylabel("Target")
        axis.set_title("Linear Regression")
        return axis

    
    def _plot_multiple_features(self, X:np.ndarray, y:np.ndarray) -> None:
        fig, ax = plt.subplots(X.shape[1], 1)
        for i in range(X.shape[1]):
            self._plot_single_feature(X[:,i], y, ax[i])
        fig.show()
    
    def _plot_two_features(self, X:np.ndarray, y:np.ndarray) -> None:
        # make 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:,0], X[:,1], y)
        # add plane
        x1 = np.linspace(min(X[:,0]), max(X[:,0]), 10)
        x2 = np.linspace(min(X[:,1]), max(X[:,1]), 10)
        x1, x2 = np.meshgrid(x1, x2)
        y = self._model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
        ax.plot_surface(x1, x2, y, alpha=0.5)
        fig.show()

    def _check_dimensions(self, X:np.ndarray, y:np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows. Found {X.shape[0]} rows in X and {y.shape[0]} rows in y.")
        if (data_features:=X.shape[1]) != (model_features:=len(self._model.get_parameters()["weights"])):
            raise ValueError(f"Features mismatch between the data passed and the model. Found {data_features} and {model_features}")

    def plot(self, X:np.ndarray, y:np.ndarray, plot_3d_if_2_covariates:bool=True) -> None:
        self._check_dimensions(X, y)
        features_dimensions = X.shape[1]
        if features_dimensions == 1:
            fig, ax = plt.subplots(figsize=(7,7))
            self._plot_single_feature(X, y, ax)
            plt.show()
        elif features_dimensions == 2 and plot_3d_if_2_covariates:
            self._plot_two_features(X, y)
        else:
            self._plot_multiple_features(X, y)