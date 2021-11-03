# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:53:37 2020

@author: Aunkur Das
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 21:17:50 2020

@author: Aunkur Das
"""


# import modules
from preprocessing import preprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

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

"""
y = df_scaled[:, 8] ## target
X = np.delete(df_scaled, 1, axis = 1) # Inputs: f1, f2, ..., f14
"""
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values #select SEX,BMI,AGE,CURSMOKE #,3,6
y = dataset.iloc[:, 8].values #select SYSBP
# split data into training (80%) and testing (20%)
xTrain, xTest, yTrain, yTest = train_test_split(inputFeatures, targetVariable, test_size=0.3)
# further split training split into training (80%) and validation (20%)
xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.3)


#K-fold


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
############### Apply 10-fold Cross validation

n_splits =25 ## Fold number
cv_set = np.repeat(-1.,X.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
for train_index,test_index in skf.split(X, y):
    x_train,x_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()

print("")


from sklearn.naive_bayes import GaussianNB


model=GaussianNB()
model.fit(xTrain, yTrain,)
pred = model.predict(xTest)

##Test attribute
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

##Test attribute
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



#https://github.com/Mauzey/Diabetes-Prediction