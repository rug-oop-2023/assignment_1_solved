import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import numpy as np
from src.multiple_linear_regression import MultipleLinearRegression
from typing import List

class RegressionPlotter():
    def __init__(self, model:MultipleLinearRegression) -> None:
        if not isinstance(model, MultipleLinearRegression):
            raise TypeError(f"model: expected MultipleLinearRegression, got {type(model)}")
        self._model = model
    
    def _plot_single_feature(self, X:np.ndarray, y:np.ndarray, axis:plt.Axes, feature_name:str="Feature", target_name:str="Target") -> Axes:
        axis.scatter(X, y)
        points_for_line = np.array([[X.min()],[X.max()]])
        pred_for_line = self._model.predict(points_for_line)
        axis.plot(points_for_line, pred_for_line, color="red")
        axis.set_xlabel(feature_name)
        axis.set_ylabel(target_name)
        return axis
    
    def _plot_multiple_features(self, X:np.ndarray, y:np.ndarray, feature_names:List[str]=None, target_name:str="Target") -> None:
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
        fig, ax = plt.subplots(X.shape[1], 1)
        for i in range(X.shape[1]):
            model = MultipleLinearRegression()
            model.train(X[:,[i]], y)
            sub_plotter = RegressionPlotter(model)
            sub_plotter._plot_single_feature(X[:,i], y, ax[i], feature_names[i], target_name)
        plt.show()
    
    def _plot_two_features(self, X:np.ndarray, y:np.ndarray, feature_names:List[str]=("Feature 1", "Feature 2"), target_name:str="Target") -> None:
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
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        
        plt.show()

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
