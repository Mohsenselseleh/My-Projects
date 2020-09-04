#!/usr/bin/env python
# coding: utf-8

# In[83]:


import itertools
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import scipy.optimize as opt
import pylab as pl
import scipy.optimize as opt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs


# In[86]:


df = pd.read_csv("C:/Users/Mohsen/Desktop/Collision Reference No.csv")
df1=df.head(5)
print(df1)
X =df[['Day of Collision','Month of Collision','Hour of Collision (24 hour)','Carriageway Type','Speed Limit','Junction Detail','Junction Control','Pedestrian Crossing – Human Control']]


# In[77]:


Y = df['Collision Severity'].values


# In[38]:


X =preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)


# In[40]:


#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier


# In[41]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
neigh


# In[42]:


Yhat = neigh.predict(X_test)
Yhat[0:5]


# In[34]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Yhat))
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    Yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(Y_test, Yhat)

    
    std_acc[n-1]=np.std(Yhat==Y_test)/np.sqrt(Yhat.shape[0])

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()







# In[43]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[44]:


drugTree.fit(X_train,Y_train)
predTree = drugTree.predict(X_test)


# In[46]:


from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(Y_test, predTree))


# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
LR


# In[48]:


Yhat = LR.predict(X_test)
Yhat


# In[49]:


Yhat_prob = LR.predict_proba(X_test)
Yhat_prob


# In[50]:


from sklearn.metrics import log_loss
log_loss(Y_test, Yhat_prob)


# In[51]:


score = LR.score(X_test, Y_test)
print(score)


# In[56]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train) 


# In[58]:


Yhat = clf.predict(X_test)
Yhat [0:5]


# In[59]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[60]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[62]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(Y_test, Yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


# In[74]:


from sklearn.metrics import f1_score
f1_score(Y_test, Yhat, average='weighted')


# In[ ]:





# In[66]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




