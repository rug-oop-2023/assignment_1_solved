from typing import Any
from src.multiple_linear_regression import MultipleLinearRegression
import json
import pickle

class ModelSaver():
    def _save_json(self, model:Any, path:str) -> None:
        with open(path, "w") as f:
            json.dump(model.get_parameters(), f)
    
    def _load_json(self, model:Any, path:str) -> None:
        with open(path, "r") as f:
            params = json.load(f)
        model.set_parameters(params)
    
    def _save_pkl(self, model:Any, path:str) -> None:
        with open(path, "wb") as f:
            pickle.dump(model.get_parameters(), f)
    
    def _load_pkl(self, model:Any, path:str) -> None:
        with open(path, "rb") as f:
            params = pickle.load(f)
        model.set_parameters(params)
    
    def save(self, model:Any, path:str, format:str="json") -> None:
        if format == "json":    
            self._save_json(model, path)
        elif format in ("pkl", "pickle"):
            self._save_pkl(model, path)
        else:
            raise ValueError(f"Invalid format {format}. Valid formats are 'json' and 'pkl'")

    def load(self, model:Any, path:str, format:str="json") -> None:
        if format == "json":    
            self._load_json(model, path)
        elif format in ("pkl", "pickle"):
            self._load_pkl(model, path)
        else:
            raise ValueError(f"Invalid format {format}. Valid formats are 'json' and 'pkl'")