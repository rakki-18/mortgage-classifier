#!/usr/bin/env python
# coding: utf-8

# In[145]:


#Importing csv into python as a dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv(r'C:\Users\rahul\Downloads\acm-ml-pip-master\loan_mortgage_data.csv')

#Filling empty values with mean
dataset=dataset.fillna(dataset.mean())

#Dropping unnecessary columns
dataset.drop(['applicant_sex','applicant_ethnicity','applicant_race','state_code','row_id','tract_to_msa_md_income_pct','preapproval'],axis=1,inplace=True)

#Converting into numpy array
dataset.to_numpy

#Scaling or Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(dataset)

#Slicing the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into a training:testing split of 90:10
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

#Importing random forest classifier and training it
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=90,n_jobs=4,max_depth=17, random_state=0)
clf.fit(X_train,y_train)

#predicting loan approval for test cases
y_pred = clf.predict(X_test)

#Checking accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[ ]:





# In[ ]:





# In[ ]:




