
''' ###################################################################### '''
''' Apply different ML regression models to predict the dependent variable '''
''' ###################################################################### '''

import time
import uproot
from pprint import pprint
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pickle 

#############################
##############################################
# file_root = uproot.open("~/Desktop/myThesisWork/smOnAMPT/AnaRes1020.root")
# # pprint(file_root.keys())
# # pprint(file_root.values())
# dfs = []
# tree = file_root["fDBEvtTree"]
# t0 = time.time()
# for key in tree.keys():
#     # print(key, tree[key].array())
#     # print(key, len(tree[key].array()))
#     print(key)
#
#     value = tree[key].array()
#     df = pd.DataFrame({key:value})
#     dfs.append(df)
#
# dataframe = pd.concat(dfs, axis=1)
# t1 = time.time()
# print(t1-t0)
# print(dataframe)
# t2 = time.time()

#dataframe.to_csv("amptsm.csv")

#############################################
#############################


# load csv file
from config import inputfile
dataset = pd.read_csv(inputfile)

X = dataset.iloc[:, :15].values
y = dataset.iloc[:, -29].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)#, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("linear reg: ", y_pred)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print("random forest: ", y_pred_rf)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("decision tree:", y_pred_dt)

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator=rf, X = X_train, y= y_train, cv = 10)
print(score)
print(np.mean(score))

# save the trained model using pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(rf, f)

# load the trained model and predict the values
# with open('model_pickle', 'rb') as f:
#     mp = pickle.load(f)
#
# y_pred_mp = mp.predict(X_test)
# print(y_pred_mp, y_test)


# plotting true vs predicted values
x = y_test
y = y_pred_rf
xlim = x.min(), x.max()
ylim = y.min(), y.max()

fig, (bx0, bx1) = plt.subplots(ncols=1, nrows=2, sharey='all', figsize=(8, 8))
# hb1 = bx1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
# bx1.set(xlim=(300,2000),ylim=(250,2000))
# bx1.set(xlim=xlim,ylim=ylim)
# bx1.set_title("log scale")
# cb1 = fig.colorbar(hb1, ax = bx1, label= 'log')

hb1 = bx0.hexbin(x, y, gridsize=50, cmap='inferno')
#bx0.set(xlim=(500,1500),ylim=(500,1500))
bx0.set(xlim=xlim,ylim=ylim)
bx0.set_title("hexa bining")
cb0 = fig.colorbar(hb1, ax = bx0, label='counts')
plt.scatter(y_test, y_pred_rf, color= "red", alpha=0.4) #, linewidth=0.5, edgecolor="white")
plt.show()
plt.savefig("randomforest_chMult.png")


