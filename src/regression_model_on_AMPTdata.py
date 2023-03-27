""" ML regression models to predict the dependent variable """

import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Add project path using sys.
# import sys; sys.path.insert(0, "..")

# Define project path using os
project_dir = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


def load_dataset(data_file):
    data_folder = "data"
    inputfile = os.path.join(project_dir, data_folder, data_file)
    # Following will also work with ''' import sys; sys.path.insert(0, "..")'''
    # inputfile = "data/amptsm.csv"
    try:
        dataset = pd.read_csv(inputfile)
    except:
        print(
            f"Your data file {data_file} does not exist "
            f"in {project_dir}/{data_folder}.\n"
            f"Kindly, place your data file correctly."
        )
        exit()
    return dataset


def pred_score(estimator, ps_train_x, ps_test_x, ps_train_y):
    """predict dependent variable"""
    estimator.fit(ps_train_x, ps_train_y)
    prediction = estimator.predict(ps_test_x)
    return prediction


def main():
    """implement ml models"""
    dataset = load_dataset("amptsm.csv")
    X = dataset.iloc[:, :15].values
    y = dataset.iloc[:, -29].values

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.8, random_state=40
    )

    sc_x = StandardScaler()
    # sc_y = StandardScaler()
    train_X = sc_x.fit_transform(train_X)
    test_X = sc_x.transform(test_X)

    model_predictions = {}
    estimators = {
        "linear": LinearRegression(),
        "Random_forest": RandomForestRegressor(n_estimators=10),
        "Decision_tree": DecisionTreeRegressor(),
    }

    for estimator_name, estimator in estimators.items():
        predicted_output = pred_score(estimator, train_X, test_X, train_y)
        score = cross_val_score(estimator, X=train_X, y=train_y, cv=10)

        model_predictions[f"{estimator_name}"] = {
            "prediction": predicted_output,
            "score": score,
        }

    linear_t = LinearRegression()
    linear_t.fit(train_X, train_y)
    y_pred = linear_t.predict(test_X)

    random_f = RandomForestRegressor(n_estimators=10)
    random_f.fit(train_X, train_y)
    y_pred_rf = random_f.predict(test_X)

    decision_t = DecisionTreeRegressor()
    decision_t.fit(train_X, train_y)
    y_pred_dt = decision_t.predict(test_X)

    # score = cross_val_score(estimator=rf, X=train_X, y=train_y, cv=10)

    breakpoint()

    # save the trained model using pickle
    with open("model_pickle.pkl", "wb") as f:
        pickle.dump(random_f, f)

    # load the trained model and predict the values
    # with open('model_pickle', 'rb') as f:
    #     mp = pickle.load(f)
    #
    # y_pred_mp = mp.predict(X_test)
    # print(y_pred_mp, y_test)

    # plotting true vs predicted values
    x_var = test_y
    y_var = y_pred_rf
    xlim = x_var.min(), x_var.max()
    ylim = y_var.min(), y_var.max()

    fig, (bx0, bx1) = plt.subplots(
        ncols=1, nrows=2, sharey="all", figsize=(8, 8)
    )
    # hb1 = bx1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
    # bx1.set(xlim=(300,2000),ylim=(250,2000))
    # bx1.set(xlim=xlim,ylim=ylim)
    # bx1.set_title("log scale")
    # cb1 = fig.colorbar(hb1, ax = bx1, label= 'log')

    hb1 = bx0.hexbin(x, y, gridsize=50, cmap="inferno")
    # bx0.set(xlim=(500,1500),ylim=(500,1500))
    bx0.set(xlim=xlim, ylim=ylim)
    bx0.set_title("hexa bining")
    cb0 = fig.colorbar(hb1, ax=bx0, label="counts")
    plt.scatter(
        test_y, y_pred_rf, color="red", alpha=0.4
    )  # , linewidth=0.5, edgecolor="white")
    plt.show()
    plt.savefig("randomforest_chMult.png")


if __name__ == "__main__":
    main()
