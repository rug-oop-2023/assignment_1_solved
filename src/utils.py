import numpy as np
from typing import Dict, List, Any

def dict_to_numpy(dictionary:dict) -> Dict[Any, np.ndarray]:
    for key in dictionary:
        dictionary[key] = np.array(dictionary[key])
        if len(dictionary[key].shape) == 0:
            dictionary[key] = dictionary[key].reshape(1)
    return dictionary

def numpy_to_list(array:np.ndarray) -> List:
    if isinstance(array, np.ndarray):
        return array.tolist()
    elif isinstance(array, (list, set)):
        return array
    elif isinstance(array, (int, float)):
        return [array]
    else:
        raise TypeError(f"Expected numpy.ndarray or list, found {type(array)}")