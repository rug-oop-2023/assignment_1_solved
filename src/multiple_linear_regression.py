import numpy as np
from .utils import numpy_to_list


class MultipleLinearRegression():
    def __init__(self) -> None:
        self._weights = None
        self._intercept = None
    
    def _check_ndarray(self, array:np.ndarray, name:str=None) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name}: expected numpy.ndarray, got {type(array)}")
        if array.dtype not in [np.float64, np.float32, np.int64, np.int32]:
            raise TypeError(f"{name}: expected numpy.ndarray of numeric types, got {array.dtype}")
    
    def train(self, X:np.ndarray, y:np.ndarray) -> None:
        self._check_ndarray(X, "X")
        self._check_ndarray(y, "y")
        try:
            X = np.c_[np.ones(X.shape[0]), X]
            params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y).flatten()
            self._intercept = np.array(params[0])
            self._weights = params[1:]
        except np.linalg.LinAlgError:
            raise ValueError("The matrix X^T X is not invertible. Please check that the features are not linearly dependent.")

    def predict(self, X:np.ndarray) -> np.ndarray:
        self._check_ndarray(X, "X")
        return self._intercept + X.dot(self._weights)

    def get_parameters(self) -> dict:
        return {
            "weights": numpy_to_list(self._weights),
            "intercept": numpy_to_list(self._intercept)
        }
    
    def set_parameters(self, params_dict:dict) -> None:
        for param_name, param in params_dict.items():
            self._check_ndarray(param, param_name)
        self._weights = np.array(params_dict["weights"])
        self._intercept = np.array(params_dict["intercept"])
    
    
