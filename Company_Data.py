# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:35:39 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
company=pd.read_csv("Company_Data.csv")
company.head()

#Converting the categorical into numeric
company["High"]=company.Sales.map(lambda x: 1 if x>8 else 0)

company["ShelveLoc"]=company["ShelveLoc"].astype("category")
company["Urban"]=company["Urban"].astype("category")
company["US"]=company["US"].astype("category")

#label encoding to convert the categorical into numeric
company["ShelveLoc"]=company["ShelveLoc"].cat.codes
company["Urban"]=company["Urban"].cat.codes
company["US"]=company["US"].cat.codes

company.columns
#Setting x(features) and y(target)
feature_col=["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]
#x= company.drop(["Sales","High"], axis=1)

x=company[feature_col]
y=company.High
print(x)
print(y)

#train_test_split
X_train, X_test, y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#Model building using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=0)
model = RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
model.fit(X_train,y_train)
pred = model.predict(X_test)

##evaluating the model(by using crosstab or confusion matrix)
pd.crosstab(y_test,pred)
#or by using confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))

#Accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))

#Here accuracy is 77%



