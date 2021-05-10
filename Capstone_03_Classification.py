#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[100]:


bcc = pd.read_csv(r"C:\Users\dell\Documents\Data Science\Excel\Capstone03\BankCreditCard_masterdata1.csv",header=0)


# In[101]:


bcc.head()


# In[103]:


bcc.isnull().sum()


# In[115]:


bccds = bcc.drop(['RecNum','AccountNum','AccOrgDt','CustomerID','CCIssueDt','MonthsAveBal','CCPurchase','NoofDays','Age'],axis=1)


# In[116]:


bccds.info()


# In[117]:


from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

bccds['MVB_Range'] = number.fit_transform(bccds['MVB_Range'])
bccds['CCP_Range'] = number.fit_transform(bccds['CCP_Range'])


# In[118]:


bccds.astype(int)


# In[119]:


#load data into dependent and independent variables

IndepVar = []

for col in bccds.columns:
    if col != 'CreditCard':
        IndepVar.append(col)
        
TargetVar = ['CreditCard']

X = bccds[IndepVar]
y = bccds[TargetVar]


# In[120]:


# Splitting the dataset into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.24, random_state = 15) 


# # LogisticRegression

# In[121]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# In[122]:


#Build the model

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[123]:


y_pred = logreg.predict(X_test)


# In[124]:


params = logreg.get_params()
print(params)


# In[125]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[126]:


#import scikit_learn metrics module for accuracy calculation
from sklearn import metrics

#Moddel Accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test,y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("f1-score:",metrics.f1_score(y_test,y_pred))


# # RandomForestClassifier

# In[127]:


#Importing the algorithm
from sklearn.ensemble import RandomForestClassifier


# In[128]:


# Build the model 
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[129]:


#Predictions with test data
y_pred1 = rfc.predict(X_test)


# In[130]:


#evavluation the algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
print(accuracy_score(y_test, y_pred1)*100, '%')


# In[131]:


#import scikit_learn metrics module for accuracy calculation
from sklearn import metrics

#Moddel Accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test,y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("f1-score:",metrics.f1_score(y_test,y_pred))


# # KNN

# In[132]:


# StandardScaler : It transforms the data in such a manner that it has mean as 0 and standard deviation as 1. 
# Standardization is useful for data which has negative values. It arranges the data in a standard normal distribution. 
# It is more useful in classification than regression
# StandardScaler removes the mean and scales each feature/variable to unit variance

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(X)
ss_transform = ss.transform(X)
sc_bankm= pd.DataFrame(ss_transform)

sc_bankm.head()


# In[134]:


# Split the dataset into train and test

from sklearn.model_selection import train_test_split

X = ss_transform
y = bccds[TargetVar]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[138]:


# Initialize an array that stores the Accuracy and build the KNN algorithm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = []

for a in range(1, 2, 1):
    k = a
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_preds2 = knn.predict(X_test)
    ascore = accuracy_score(y_test, y_preds2)*100
    ascore = "{:.2f}".format(ascore)
    print('Accuracy value for k=', k, 'is:', ascore)


# In[140]:


# Display the confusion matrix and classification Report
print(confusion_matrix(y_test, y_preds2))
print(classification_report(y_test, y_preds2))


# In[142]:


#import scikit_learn metrics module for accuracy calculation
from sklearn import metrics

#Moddel Accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test,y_preds2))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_preds2))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test,y_preds2))

# Model Recall: what percentage of positive tuples are labelled as such?
print("f1-score:",metrics.f1_score(y_test,y_preds2))


# In[ ]:




