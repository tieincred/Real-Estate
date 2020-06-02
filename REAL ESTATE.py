#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


housing = pd.read_csv('C:\\Users\91820\\Downloads\\original.CSV')


# In[3]:


housing


# In[4]:


housing.describe()


# In[5]:


# observation: missing values in "n_hos_beds"


# In[6]:


plt.scatter(x='n_hot_rooms',y='price',data=housing)


# In[7]:


# observation: two outliers


# In[8]:


plt.scatter(x='rainfall',y='price',data=housing)


# In[9]:


# observation: single outlier


# In[10]:


housing.head()


# In[11]:


import seaborn as sns


# In[12]:


sns.countplot(x='airport', data=housing)


# In[13]:


sns.countplot(x='waterbody', data=housing)


# In[14]:


sns.countplot(x='bus_ter', data=housing)


# In[15]:


# observation: all values are "yes", the data is not going to effect predictions.


# In[16]:


housing.info()


# In[17]:


ul = np.percentile(housing.n_hot_rooms,[99])[0]


# In[18]:


ul


# In[19]:


housing[(housing.n_hot_rooms>ul)]


# In[20]:


housing.n_hot_rooms[(housing.n_hot_rooms>3*ul)] = 3*ul


# In[21]:


uv = np.percentile(housing.rainfall,[1])[0]


# In[22]:


uv


# In[23]:


housing[(housing.rainfall<uv)]


# In[24]:


housing.rainfall[(housing.rainfall<uv)]=0.3*uv


# In[25]:


housing[(housing.rainfall<uv)]


# In[26]:


housing.n_hos_beds=housing.n_hos_beds.fillna(housing.n_hos_beds.mean())


# In[27]:


housing.info()


# In[28]:


# fixed missing values


# In[29]:


plt.scatter(x='crime_rate',y='price',data=housing)

# it apparently has outliers but it's not linear so will be treated differently


# In[30]:


housing.crime_rate = np.log(1+housing.crime_rate)


# In[31]:


plt.scatter(x='crime_rate',y='price',data=housing)


# In[32]:


housing['avg_dist']= (housing.dist1+housing.dist2+housing.dist3+housing.dist4)/4


# In[33]:


del housing['dist1']


# In[34]:


del housing['dist2']


# In[35]:


del housing['dist3']


# In[36]:


del housing['dist4']


# In[37]:


housing.describe()


# In[38]:


#all locations have bus terminal no effect on prediction so


# In[39]:


del housing['bus_ter']


# In[40]:


housing.head()


# In[41]:


housing = pd.get_dummies(housing)


# In[42]:


housing.head()


# In[43]:


del housing['airport_NO']


# In[44]:


del housing['waterbody_None']


# In[45]:


# dealing with categorical data


# In[46]:


housing.corr()


# In[47]:


del housing['parks']


# In[50]:


from sklearn.linear_model import LinearRegression


# In[51]:


y = housing['price']


# In[52]:


X = housing[['room_num']]


# In[53]:


lm = LinearRegression()


# In[54]:


lm.fit(X,y)


# In[55]:


print(lm.intercept_,lm.coef_)


# In[56]:


lm.predict(X)


# In[58]:


sns.jointplot(x= housing['room_num'], y=housing['price'], data = housing, kind='reg')


# In[ ]:




