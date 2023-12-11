import numpy as np
from .utils import numpy_to_list


class MultipleLinearRegression():
    def __init__(self) -> None:
        self._weights = None
        self._intercept = None
    
    def train(self, X:np.ndarray, y:np.ndarray) -> None:
        X = np.c_[np.ones(X.shape[0]), X]
        params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self._intercept = np.array(params[0])
        self._weights = params[1:]

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self._intercept + X.dot(self._weights)

    def get_parameters(self) -> dict:
        return {
            "weights": numpy_to_list(self._weights),
            "intercept": numpy_to_list(self._intercept)
        }
    
    def set_parameters(self, params_dict:dict) -> None:
        self._weights = np.array(params_dict["weights"])
        self._intercept = np.array(params_dict["intercept"])
    
    
