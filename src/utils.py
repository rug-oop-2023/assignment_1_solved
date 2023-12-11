import numpy as np

def numpy_to_list(array:np.ndarray) -> list:
    if isinstance(array, np.ndarray):
        return array.tolist()
    elif isinstance(array, (list, set)):
        return array
    elif isinstance(array, (int, float)):
        return [array]
    else:
        raise TypeError(f"Expected numpy.ndarray or list, found {type(array)}")