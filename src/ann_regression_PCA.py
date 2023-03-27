#######################################################################
######## PCA (dimensionality reduction) and DNN layers ########
#######################################################################

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import inputfile

dataset = pd.read_csv(inputfile)

X = dataset.iloc[:, :15].values
y = dataset.iloc[:, -29].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply PCA
pca = PCA(n_components=7)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
# print(X_train.shape)

# build DNN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=7, activation="relu", input_shape=[7]))
ann.add(tf.keras.layers.Dense(units=128, activation="relu"))
ann.add(tf.keras.layers.Dense(units=128, activation="relu"))
ann.add(tf.keras.layers.Dense(units=128, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer="adam", loss="mean_squared_error")
history = ann.fit(
    X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10
)

history_df = pd.DataFrame(history.history)

y_pred = ann.predict(X_test)


""" plot the loss function"""
plt.plot(history_df["loss"])
plt.plot(history_df["val_loss"])
plt.title("Loss function (Train and Test set)")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
