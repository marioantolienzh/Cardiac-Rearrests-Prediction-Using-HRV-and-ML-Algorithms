#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:11:00 2021

@author: laros
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set(color_codes=True)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('results1.csv')
data1=pd.DataFrame(data)
data1[["bpm min", "bpm max"]] = data1[["bpm min", "bpm max"]].apply(pd.to_numeric)
data1[["bpm std", "sdnn","RMSSD","sdsd","nn50","pnn50","tinn n","tinn m","tinn"]] = data1[["bpm std", "sdnn","RMSSD","sdsd","nn50","pnn50","tinn n","tinn m","tinn"]].apply(pd.to_numeric)
data1[["tri index", "VLF peak","LF peak","HF peak","VLF power","LF power","HF power","fft ratio"]] = data1[["tri index", "VLF peak","LF peak","HF peak","VLF power","LF power","HF power","fft ratio"]].apply(pd.to_numeric)
data1[["sd1", "sd2","sd ratio","ellipse area","Sample entropy"]] = data1[["sd1", "sd2","sd ratio","ellipse area","Sample entropy"]].apply(pd.to_numeric)
#data1[["Type"]]= data1[["Type"]].apply(pd.to_numeric)

del data1['ID']
del data1['nni counter (sample size)']

#print(data1.info())
# print(data1.shape)

#Check duplicate rows in data
# duplicate_rows = data1[data1.duplicated()]
# print("Number of duplicate rows :: ", duplicate_rows.shape)

#remove null values 
data2 = data1.dropna()

# #Looking for null values
# print("Null values :: ")
# print(data2.isnull() .sum())

# plt.figure(1)
# # plt.scatter(data2["Sample entropy"],data2["Type"])
# # plt.xlabel("sdnn")
# sns.pairplot(data2, vars= ['sdnn', 'RMSSD','sdsd','pnn50'],hue=("Type"))

# plt.figure(2)
# sns.pairplot(data2, vars= ['VLF power', 'LF power','HF power'],hue=("Type"))

# plt.figure(3)
# sns.pairplot(data2, vars= ['sd1', 'sd2','ellipse area'],hue=("Type"))

#correlation
plt.figure(5)
pearcor = data2.corr(method='pearson')
spearcor = data2.corr(method='spearman')
cmap=sns.diverging_palette(20, 220, n=200)
# cmap = sns.diverging_palette(0,250,150,50,as_cmap=True)
sns.heatmap(pearcor, vmin=-1, vmax=1, cmap=cmap, linewidth=0.1)
plt.title("Pearson Correlation")


plt.figure(6)
sns.heatmap(spearcor, vmin=-1, vmax=1, cmap=cmap, linewidth=0.1)
plt.title("Spearman Correlation")

#machine learning
x = data2.drop("Type",axis=1)
y = data2["Type"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#Logistic regression
logreg = LogisticRegression(random_state=0,solver='liblinear')
logreg.fit(x_train,y_train)
y_pred_logreg = logreg.predict(x_test)
print("Accuracy of 1st log reg::" , metrics.accuracy_score(y_test,y_pred_logreg))

data3 = data2[["Type","sdnn","RMSSD","sdsd","VLF power","LF power","HF power","sd1","sd2","ellipse area"]]
# print(data3.info())

#machine learning
x1 = data3.drop("Type",axis=1)
y1 = data3["Type"]
x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=0.3)

#Logistic regression
logreg = LogisticRegression(random_state=0,solver='liblinear')
logreg.fit(x1_train,y1_train)
y1_pred_logreg = logreg.predict(x1_test)
# score = logreg.score(x1_test, y1_test)
# print("score::", score)
print("Accuracy of 2nd log reg::" , metrics.accuracy_score(y1_test,y1_pred_logreg))

#cross validation
# scores = cross_val_score(logreg, x1_train, y1_train, cv=10)
# print('Cross-Validation Accuracy Scores', scores)

# ***********************Decision Tree Classification***********************

decTree = DecisionTreeClassifier(max_depth=12, random_state=0)
decTree.fit(x_train,y_train)
y2_pred_decTree = decTree.predict(x_test)
print("Accuracy of Decision Trees :: " , metrics.accuracy_score(y_test,y2_pred_decTree))

# plt.figure()
# tree.plot_tree(decTree)

# Using Random forest classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)
print("Accuracy of Random Forest Classifier :: ", metrics.accuracy_score(y_test, y_pred_rf))



plt.figure(7)
#Find the score of each feature in model and drop the features with low scores
f_imp = rf.feature_importances_
sorted_indices = np.argsort(f_imp)[::-1]

plt.title('Feature Importance based on random forest')
plt.bar(range(x_train.shape[1]), f_imp[sorted_indices], align='center')
plt.xticks(range(x_train.shape[1]), x_train.columns[sorted_indices], rotation=90)
plt.ylabel('Feature Importance score')
# plt.xlabel('Features')
plt.tight_layout()
plt.show()



