import numpy as np
from .utils import numpy_to_list, dict_to_numpy


class MultipleLinearRegression():
    def __init__(self) -> None:
        self._parameters = None
        
    @property
    def parameters(self) -> dict:
        return self._parameters

    @parameters.setter
    def parameters(self, params_dict:dict) -> None:
        params_dict = dict_to_numpy(params_dict)
        for param_name, param in params_dict.items():
            self._check_ndarray_type(param, param_name)
        self._parameters = np.concatenate(
            [params_dict["intercept"], params_dict["weights"]]
            )
        print(f"Incoming parameters:\n{params_dict}\nNew params\n{self._parameters}")
        
    @parameters.getter
    def parameters(self) -> dict:
        if self._parameters is None:
            return None
        return {
            "weights": numpy_to_list(self._parameters[1:]),
            "intercept": numpy_to_list(self._parameters[:1])
        }
    
    def _check_input_data(self, array:np.ndarray, name:str=None) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name}: expected numpy.ndarray, got {type(array)}")
        self._check_ndarray_type(array, name)
    
    def _check_ndarray_type(self, array:np.ndarray, name:str=None) -> None:
        if array.dtype not in [np.float64, np.float32, np.int64, np.int32]:
            raise TypeError(f"{name}: expected numpy.ndarray of numeric types, got {array.dtype}")
    
    def train(self, X:np.ndarray, y:np.ndarray, verbose:bool=False) -> None:
        self._check_input_data(X, "X")
        self._check_input_data(y, "y")
        print(f"Incoming X:\n{X}\nIncoming y:\n{y}")
        print(f"X shape: {X.shape}\ny shape: {y.shape}")

        try:
            X_new = np.concatenate([np.ones(X.shape[0]), X])
            if verbose:
                print(f"X after adding intercept:\n{X_new}")
                print(f"X shape after adding intercept: {X_new.shape}")
            params = np.linalg.inv(X.T.dot(X))
            print(f"X^T X:\n{params}")
            print(f"X^T X shape: {params.shape}")
            params = params.dot(X.T)
            print(f"(X^T X)^-1 X^T:\n{params}")
            print(f"(X^T X)^-1 X^T shape: {params.shape}")
            params = params.dot(y)
            print(f"(X^T X)^-1 X^T y:\n{params}")
            print(f"(X^T X)^-1 X^T y shape: {params.shape}")
            params = params.flatten()
            print(f"Flattened params:\n{params}")
            print(f"Flattened params shape: {params.shape}")
            self._parameters = params
        except np.linalg.LinAlgError:
            raise ValueError("The matrix X^T X is not invertible. Please check that the features are not linearly dependent.")

    def predict(self, X:np.ndarray) -> np.ndarray:
        self._check_input_data(X, "X")
        return self._parameters[0] + X.dot(self._parameters[1:])

    
    
