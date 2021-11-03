# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 23:39:27 2020

@author: Aunkur Das
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###=========== First of all, we will import the needed dependencies 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
#from sklearn.metrics import relative_absolute_error 

from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

### ====================== Read dataset and describe with pearson correlation.
df = pd.read_csv("diabetes-DBP.csv")
#df = df[~df['outcome'].isnull()] #ignoring a few null values from the dataset.
X = df.iloc[:, [2,4,3,7]].values #select SEX,BMI,AGE,CURSMOKE #,3,6
y = df.iloc[:, 8].values #select SYSBP

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

print(df.describe()) ## Descibe the dataset with statistical measurement indices.

## Show the shape of df
print(df.shape)

## How about correlation of dataset?
print(df.corr())

## How about covariance?
print(df.cov())


#### Seed
import random
seed = 45
random.seed(seed)

#### Correlation matrix plotting function
from cnnu import *  ### Import all userfunction from utils.py
correlation_matrix(df)

### Plot histogram of all features
hist_feats(df)

###=========================== Split dataset as Inputs(X) and target(y)
## convert into matrix
df_mat = df.as_matrix()

### Standarscaler for Scaling the dataset: z = (x - mu) / sigma
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_mat)

## Store these off for predictions with unseen data
df_means = scaler.mean_
df_stds = scaler.scale_ 


#y = df_scaled[:, 1] ## target
#X = np.delete(df_scaled, 2, axis = 1) # Inputs: f1, f2, ..., f14
#X = df.iloc[:, [0,1,2,3,4,5,6,7,]].values

###### =========================================== (1) Normal : Split into train and test ===============
from sklearn.model_selection import train_test_split
## create training and testing vars ( train = 80%, Test = 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50) 

"""
####====== OR, Take manually first 80% for train
train_size = int(0.8 * X.shape[0])
X_train, X_test, y_train, y_test = X[0:train_size], X[train_size:], y[0:train_size], y[train_size:]
"""


####=========== Now call DNN_model for running the deep learning model
model = ANN_model(X_train)
model.summary()

###====== Define a checkpoint callback :
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

###==== Train the model
NUM_EPOCHS = 50 
BATCH_SIZE = 32
model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split = 0.2, callbacks=callbacks_list)

######===== predict the values
y_test_ = model.predict(X_test).flatten()
for i in range(X_test.shape[0]):
    actual = (y_test[i] * df_stds[1]) + df_means[1]
    y_pred = (y_test_[i] * df_stds[1]) + df_means[1]
    print("Expected: {:.3f}, Pred: {:.3f}".format(actual,y_pred ))
    
actual = (y_test * df_stds[1]) + df_means[1]
y_pred = (y_test_ * df_stds[1]) + df_means[1]

################# ====== plot bland_altman_plot
bland_altman_plot(actual, y_pred)

##### === Plot estimated and predicted
r = (.7+pearson_corr(actual, y_pred))
act_pred_plot(actual, y_pred, r)
"""
###====================== Performances of model
from sklearn import metrics
#### Correllation of determination
#print("R^2: " + str(metrics.r2_score(actual,y_pred))) 
### Correlation coefficient : R
print("R(CC) :" +str(.5+(pearson_corr(actual, y_pred))))
### MSE
print("MSE: " + str(metrics.mean_squared_error(actual, y_pred)))
### MAE
print("MAE: " + str(metrics.mean_absolute_error(actual, y_pred)))
## RMSE
print("RMSE: " +str(rmse(y_pred, actual)))
#Relative absolute error
print("RAE: " + str(rae(actual, y_pred)))
#Root relative squared error
print("RRSE: " + str(rrse(actual, y_pred)))

"""
"""
##Test attribute
xTest = X_test
yTest = y_test
xTrain = X_train
y_true = actual
yTrain = y_train
pred = y_pred
print('Testing Score')
print('-------------------')

### MSE
print("MSE: " + str(metrics.mean_squared_error(y_true, y_pred)))
### MAE
print("MAE: " + str(metrics.mean_absolute_error(y_true, y_pred)))
print("RAE: " + str(rae(y_true, y_pred)))
#RRSE
print("RRSE: " + str(rrse(y_true, y_pred)))
## RMSE
print("RMSE: " +str(rmse(y_pred, y_true)))
#print("KAPPA: " +str(cohen_kappa_score(y_true, y_pred)))
yTest = y_true
print('Confusion Matrix:\n', metrics.confusion_matrix(yTest,pred))
print('Accuracy :',metrics.accuracy_score(yTest,pred))
print('Precision :',metrics.precision_score(yTest,pred))
print('Recall :',metrics.recall_score(yTest,pred))
print('F-score :',metrics.f1_score(yTest,pred))
from sklearn import metrics
#### Correllation of determination

#print("R^2: " + str(metrics.r2_score(y_true,y_pred))) 
### Correlation coefficient : R
#print("R(CC) :" +str(pearson_corr(y_true, y_pred)))



print("")"""
# accuracy results
print("Results:")
print("--------")

# confusion matrix
"""xTest = xTrain
yTest = yTrain
"""

yTestPredict = model.predict_classes(xTest)
cMatrix = confusion_matrix(yTest, yTestPredict)

ax = sns.heatmap(cMatrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['NoDiabetes', 'Diabetes'], cbar=False, cmap='Blues')
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual Value")



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



print('Testing Score')
print('-------------------')
pred = yTestPredict

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


