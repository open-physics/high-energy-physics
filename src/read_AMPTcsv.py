import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import inputfile # csv file
dataset = pd.read_csv(inputfile)
X = dataset.iloc[:, :15].values
y = dataset.iloc[:, -29].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8
)  # , random_state=0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def train_model(model, xtrain, ytrain, xtest):
    inst = model()
    inst.fit(xtrain, ytrain)
    pred = inst.predict(xtest)
    return pred


print("Linear reg: ", train_model(LinearRegression, X_train, y_train, X_test))
print(
    "DecisionTree reg: ", train_model(DecisionTreeRegressor, X_train, y_train, X_test)
)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("RandomForest reg: ", y_pred_rf)

score = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10)
print(score)
print(np.mean(score))

# save the trained model using pickle
with open("model_pickle.pkl", "wb") as f:
    pickle.dump(rf, f)

# plotting true vs predicted values
x = y_test
y = y_pred_rf
xlim = x.min(), x.max()
ylim = y.min(), y.max()

fig, (bx0, bx1) = plt.subplots(ncols=1, nrows=2, sharey="all", figsize=(8, 8))
hb1 = bx0.hexbin(x, y, gridsize=50, cmap="inferno")
bx0.set(xlim=xlim, ylim=ylim)
bx0.set_title("hexa bining")
cb0 = fig.colorbar(hb1, ax=bx0, label="counts")
plt.scatter(
    y_test, y_pred_rf, color="red", alpha=0.4
)  # , linewidth=0.5, edgecolor="white")
plt.show()
plt.savefig("randomforest_chMult.png")
