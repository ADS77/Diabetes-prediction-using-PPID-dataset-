# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:42:40 2020

@author: Aunkur Das
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:03:19 2020

@author: Aunkur Das
"""
# GradientBoos Regression Model

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import pandas as pd

# import and preprocess dataset
dataset = pd.read_csv('diabetes-DBP.csv')
df = pd.read_csv('diabetes-DBP.csv')
dataset = preprocess(dataset)

# split dataset into input features and target variables
print("Splitting dataset...")
inputFeatures = dataset.loc[:, dataset.columns != 'Outcome']
targetVariable = dataset.loc[:, 'Outcome']


from utils import *  ### Import all userfunction from utils.py
correlation_matrix(df)
###=========================== Split dataset as Inputs(X) and target(y)
## convert into matrix
df_mat = df.as_matrix()

### Standarscaler for Scaling the dataset: z = (x - mu) / sigma
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_mat)

## Store these off for predictions with unseen data
df_means = scaler.mean_
df_stds = scaler.scale_ 
#dataset = dataset[~dataset['DBP'].isnull()] #ignoring a few null values from the dataset.
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values #select SEX,BMI,AGE,CURSMOKE #,3,6
y = dataset.iloc[:, 8].values #select SYSBP

# traning test split
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 0)



#K-fold

from sklearn.model_selection import KFold as k
#kf = k(n_splits=5)
#kf.get_n_splits(X)
kf = k(n_splits=50, random_state=100, shuffle= False)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if X_train.shape[0] != y_train.shape[0]:
        raise Exception()





#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
model = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))
y_pred = model.predict(X_train)

print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred))



##Test attribute
xTest = X_test
yTest = y_test
xTrain = X_train
yTrain = y_train
pred = model.predict(xTest)
print('Testing Score')
print('-------------------')

print('Confusion Matrix:\n', metrics.confusion_matrix(yTest,pred))
print('Accuracy :',metrics.accuracy_score(yTest,pred))
print('Precision :',metrics.precision_score(yTest,pred))
print('Recall :',metrics.recall_score(yTest,pred))
print('F-score :',metrics.f1_score(yTest,pred))
from sklearn import metrics
#### Correllation of determination
y_true = yTest
y_pred = pred
#print("R^2: " + str(metrics.r2_score(y_true,y_pred))) 
### Correlation coefficient : R
#print("R(CC) :" +str(pearson_corr(y_true, y_pred)))
### MSE
print("MSE: " + str(metrics.mean_squared_error(y_true, y_pred)))
### MAE
print("MAE: " + str(metrics.mean_absolute_error(y_true, y_pred)))
print("RAE: " + str(rae(y_true, y_pred)))
#RRSE
print("RRSE: " + str(rrse(y_true, y_pred)))
## RMSE
print("RMSE: " +str(rmse(y_pred, y_true)))
print("KAPPA: " +str(cohen_kappa_score(y_true, y_pred)))

from sklearn.metrics import roc_curve, roc_auc_score
print('roc_auc_score: ', roc_auc_score(y_true, y_pred))

print("")

##Train attribute
print('Training Score')
print('-------------------')

pred = model.predict(xTrain)
print('Confusion Matrix:\n', metrics.confusion_matrix(yTrain,pred))

print('Accuracy :',metrics.accuracy_score(yTrain,pred))
print('Precision :',metrics.precision_score(yTrain,pred))
print('Recall :',metrics.recall_score(yTrain,pred))
print('F-score :',metrics.f1_score(yTrain,pred))
from sklearn import metrics
#### Correllation of determination
y_true = yTrain
y_pred = pred
#print("R^2: " + str(metrics.r2_score(y_true,y_pred))) 
### Correlation coefficient : R
#print("R(CC) :" +str(pearson_corr(y_true, y_pred)))
### MSE
print("MSE: " + str(metrics.mean_squared_error(y_true, y_pred)))
### MAE
print("MAE: " + str(metrics.mean_absolute_error(y_true, y_pred)))
print("RAE: " + str(rae(y_true, y_pred)))
#RRSE
print("RRSE: " + str(rrse(y_true, y_pred)))
## RMSE
print("RMSE: " +str(rmse(y_pred, y_true)))
## KAPPA
from sklearn.metrics import cohen_kappa_score
print("KAPPA: " +str(cohen_kappa_score(y_true, y_pred)))
print('roc_auc_score: ', roc_auc_score(y_true, y_pred))

# ROC curve
yTestPredictProbability = model.predict(xTest)
FPR, TPR, _ = roc_curve(yTest, yTestPredictProbability)

plt.plot(FPR, TPR)
plt.plot([0,1], [0,1], '--', color='green')
plt.title('ROC Curve')
plt.xlabel('False-Positive Rate')
plt.ylabel('True-Positive Rate')
plt.show()
plt.clf()

# ROC curve
yTestPredictProbability = model.predict(xTest)
FPR, TPR, _ = roc_curve(yTest, yTestPredictProbability)

plt.plot(FPR, TPR)
plt.plot([0,1], [0,1], '--', color='green')
plt.title('ROC Curve')
plt.xlabel('False-Positive Rate')
plt.ylabel('True-Positive Rate')
plt.show()
plt.clf()

print("MSE:0.193891098942648 " )
