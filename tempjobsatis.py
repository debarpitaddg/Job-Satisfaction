import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d=pd.read_csv('our.csv')
x1=d.iloc[:]
data=pd.read_csv('IBMdataset.csv')
x=data.iloc[:,[0,10,12,13,18,28,30,31]]
print(x.head())
y=data.iloc[:,16]
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(x).transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=200)
clf.fit(x_train,y_train)
y_pred2=clf.predict(x1)
print(y_pred2)

