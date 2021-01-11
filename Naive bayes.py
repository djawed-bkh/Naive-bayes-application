#!/usr/bin/env python
# coding: utf-8

# # Théoreme de bayes en python
#     En utilisant le dataset iris
#     La régle: P(class|data) = (P(data|class) * P(class)) / P(data)
#     

# In[39]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[17]:


iris=load_iris()
iris.data[1:5]


# ### Attributs des fleurs

# In[19]:


iris.feature_names


# ### Les classes 

# In[22]:


iris.target_names


# In[23]:


iris.target


# In[41]:


Gaussian_classifier=GaussianNB()
Gaussian_classifier.fit(X_train,Y_train)


# ## Phase d'apprentissage a travers GaussianNB()

# In[44]:


X_train, X_tests, Y_train, Y_tests= train_test_split(iris.data,iris.target,test_size=0.3, random_state=108)


# ## Phase de test 

# In[46]:


Y_pred=Gaussian_classifier.predict(X_tests)
Y_pred


# ## Voir le pourcentage d'apprentissage

# In[47]:


from sklearn import metrics
print("Precision:",metrics.accuracy_score(Y_tests, Y_pred))


# In[ ]:




