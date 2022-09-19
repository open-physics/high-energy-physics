#######################################################################
######## ANN regression with dense neural network layers (DNN) ########
#######################################################################

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from configfile import input_csv

dataset = pd.read_csv("input_csv")
X = dataset.iloc[:, :15].values
y = dataset.iloc[:, -29].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

''' build DNN model with 3 dense layers '''
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 15, activation = "relu", input_shape = [15]))
ann.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1))

''' train the model '''
ann.compile(optimizer = "adam", loss = "mean_squared_error")
history = ann.fit(
            X_train, y_train,
            validation_data = (X_test, y_test),
            batch_size = 32,
            epochs = 50)


history_df = pd.DataFrame(history.history)
#print(history_df['loss'], history_df['val_loss'])

# prediction on the test data set 
y_pred = ann.predict(X_test)
#np.set_printoptions(precision = 2)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# data visualization (plotting loss functions)
plt.plot(history_df['loss'])
y_pred = ann.predict(X_test)
plt.plot(history_df['val_loss'])
plt.title("Loss function (Adam)")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

