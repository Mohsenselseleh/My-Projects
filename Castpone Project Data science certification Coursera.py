#!/usr/bin/env python
# coding: utf-8

# In[49]:


#K-Nearest Neighbors
import itertools
import csv
import pandas as pd
import numpy as np
import pandas as np
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import scipy.optimize as opt


# In[17]:


df = pd.read_csv("C:/Users/Mohsen/Desktop/Collision Reference No.csv")


# In[18]:


df.head(10)


# In[12]:


#Features


# In[26]:


X =df[['Day of Collision','Month of Collision','Hour of Collision (24 hour)','Carriageway Type','Speed Limit','Junction Detail','Junction Control','Pedestrian Crossing – Human Control']]


# In[25]:


Y = df['Collision Severity'].values


# In[ ]:


#Normalize data


# In[27]:


X =preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
neigh


# In[36]:


Yhat = neigh.predict(X_test)
Yhat[0:5]


# In[37]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Yhat))


# In[38]:


#Decision Trees


# In[41]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[46]:


drugTree.fit(X_train,Y_train)
predTree = drugTree.predict(X_test)


# In[47]:


from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(Y_test, predTree))


# In[48]:


#Logistic regression


# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
LR


# In[51]:


Yhat = LR.predict(X_test)
Yhat


# In[52]:


Yhat_prob = LR.predict_proba(X_test)
Yhat_prob


# In[59]:


from sklearn.metrics import log_loss
log_loss(Y_test, Yhat_prob)


# In[60]:


# Use score method to get accuracy of model
score = LR.score(X_test, Y_test)
print(score)


# In[ ]:




