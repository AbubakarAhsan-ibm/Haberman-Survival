# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Sat Jun 22 13:25:35 2019

@author: MuhammadAbuBakarAhsa
"""
"""
Attribute Information:

1. Age of patient at time of operation (numerical) 
2. Patient's year of operation (year - 1900, numerical) 
3. Number of positive axillary nodes detected (numerical) 
4. Survival status (class attribute) 
-- 1 = the patient survived 5 years or longer 
-- 2 = the patient died within 5 year

"""

from sklearn.neighbors import KNeighborsClassifier
#This line imports KNN classifier from sklearn neighbors library
import pandas as pd
#this line import pandas as pd in short form
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
#this line import numpy as np



print("--------------------------HABERMAN DATA----------------------------")
#this line print data
colname = ["Age","Year_of_Op","Pos_Axi","Survival_status"]
#this line defines the column name
dataframe = pd.read_csv(r"C:\Users\MuhammadAbuBakarAhsa\Documents\RElated\IDS/haberman.data",names=colname)
#In this line we define the path of data ,name of column of the data
print(dataframe)
#this line print dataframe



array = dataframe.values
# converting the dataframe to an array for calculations
X = array[:,0:3]
#In this line we define the values except class on which the model will be trained
Y = array[:,3]
#In this line we just define class label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#In this line we devide it into two parts(X,Y) X_train=80% of data, X_test=20%) 
#the test_size = the percentage of the testing set


KNN=KNeighborsClassifier(n_neighbors=3)
# We are just specifying the alogrithm that we will be using 
# n_neighbors = The value of K in KNN
KNN.fit(X_train,Y_train)
#this line we train 80% data of X_train and Y_train
predictions=KNN.predict(X_test)
#this line we test 20% data to check how are model is performing (accuracy) 
print(classification_report(predictions,Y_test))
#It gives all the accuracy measures (recall , precision, f1 measure)

print(KNN.predict(np.array([27,76,2])))
#this line is just to check an individual data point/row/person 
#the answer will be either 1 or 2 which means either he will die within 5 years or not

