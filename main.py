from src.multiple_linear_regression_property import MultipleLinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 1, 2]
    ])
    y = np.array([1, 2, 3, 0])

    model = MultipleLinearRegression()
    model.train(X, y)
    pred = model.predict(X)

    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)
    pred_sklearn = model_sklearn.predict(X)

    print(X)
    print(y)
    
    print("---MLR---")
    print(pred)

    print("---SKLEARN---")
    print(pred_sklearn)
