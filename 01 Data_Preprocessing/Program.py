# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
#df = pd.DataFrame(X)
y = dataset.iloc[:, 3].values
#df = pd.DataFrame(y)

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])
#df = pd.DataFrame(X)

#Encoding categorical data
#we have a problem that machine learning is completely based on 
#numbers and mathematical equations so we cannot have text or words in 
#it so we encode that text(categorial data) into numbers
#------------------------
#from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#df = pd.DataFrame(X)

#------------------------------
# but now the problem is that we have assigned numbers to each country
# so some has larger numbers ans some are small so this creates a bias

# to remove that problem we use different columns for each category and create 
#dummy variable so as to ensure we donot bias the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into tarining set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# FEATURE SCALING
# the problem is that now the range of age is much smaller than range of salary
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)















































