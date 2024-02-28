#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[78]:


data = pd.read_csv('Social_Network_Ads.csv')
data


# In[ ]:





# In[79]:


le = LabelEncoder()
d = le.fit_transform(data['Gender'])
data['Gender'] = d
data


# In[80]:


x = data.drop('Purchased',axis = 1)


# In[81]:


y = data.drop(['User ID','Gender','Age','EstimatedSalary'], axis = 1)


# In[82]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[83]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[85]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) 
classifier.fit(x_train, y_train)


# In[87]:


y_pred=classifier.predict(x_test)


# In[ ]:





# In[ ]:





# In[88]:


accuracy = accuracy_score(y_test, y_pred)


# In[89]:


accuracy 


# In[90]:


###REGRESSION


# In[91]:


from sklearn.linear_model import LinearRegression


# In[92]:


ml = LinearRegression()


# In[93]:


ml.fit(x_train,y_train) ## model fitting


# In[94]:


y_pred = ml.predict(x_test)


# In[95]:


y_pred


# In[96]:


plt.scatter(y_test,y_pred)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)


# In[97]:


from sklearn import metrics


# In[98]:


meanAbErr = metrics.mean_absolute_error(y_test, y_pred) 


# In[99]:


meanAbErr * 100

