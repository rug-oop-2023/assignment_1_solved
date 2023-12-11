from src import model_saver, multiple_linear_regression, regression_plotter
from sklearn.datasets import load_diabetes

if __name__ == "__main__":
    model = multiple_linear_regression.MultipleLinearRegression()
    data = load_diabetes()
    model.train(data.data, data.target)

    print("Model parameters after training")
    print(model.get_parameters())

    saver = model_saver.ModelSaver()
    saver.save(model, "model.json", format="json")
    saver.save(model, "model.pkl", format="pkl")

    mock_parameters = {
        "weights": [1,2,3],
        "intercept": [0.5]
    }
    model.set_parameters(mock_parameters)

    print("Model parameters after setting mock parameters")
    print(model.get_parameters())

    saver.load(model, "model.json", format="json")

    print("Model parameters after loading from json")
    print(model.get_parameters())

    model.set_parameters(mock_parameters)

    print("Model parameters after setting mock parameters")
    print(model.get_parameters())

    saver.load(model, "model.pkl", format="pkl")

    print("Model parameters after loading from pickle")
    print(model.get_parameters())

    plotter = regression_plotter.RegressionPlotter(model)
    
    # one-d plot
    # replace params
    old_params = model.get_parameters()
    new_params = old_params.copy()
    new_params["weights"] = old_params["weights"][:1]
    model.set_parameters(new_params)
    plotter.plot(data.data[:,:1], data.target)

    # two-d plot
    # replace params
    new_params["weights"] = old_params["weights"][:2]
    model.set_parameters(new_params)
    # 3d plot
    plotter.plot(data.data[:, :2], data.target, plot_3d_if_2_covariates=True)
    # 2 x 2d plot
    plotter.plot(data.data[:, :2], data.target, plot_3d_if_2_covariates=False)

    # full 2d plots
    model.set_parameters(old_params)
    plotter.plot(data.data, data.target)







