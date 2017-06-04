# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold

#Importing data from csv file
file = 'Flagdata.csv'
df = pd.read_csv(file)

#Splitting the data into test and train
train, test = train_test_split(df, test_size = 0.2)

#Separating output variable("religion") from the input variables
x_test = test.drop(["religion"], axis = 1)
y_test = test.filter(items = ["religion"])
x_train = train.drop(["religion"], axis = 1)
y_train = train.filter(items = ["religion"])

#Scaling the training input variables
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

#Learning the model with training data
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation = "logistic", learning_rate_init = 0.3, momentum = 0.2, max_iter = 500)
mlp.fit(x_train, y_train)

#Scaling the test input variables
scaler.fit(x_test)
x_test = scaler.transform(x_test)

#Generating predictions for test and train data
train_predictions = mlp.predict(x_train)
test_predictions = mlp.predict(x_test)

#Accuracy for train and test data
accuracy_train = mlp.score(x_train, y_train)
accuracy_test = mlp.score(x_test, y_test)
print("training accuracy = ", accuracy_train)
print("test accuracy = ", accuracy_test)

#Confusion matrix for test data
con_mat = confusion_matrix(y_test, test_predictions)
print(con_mat)

#Classification report for test data
cla_rep = classification_report(y_test, test_predictions)
print(cla_rep)







