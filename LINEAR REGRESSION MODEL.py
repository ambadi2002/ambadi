#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION

# In[1]:


#IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # loading data set

# In[2]:


data=pd.read_csv(r"C:\Users\ambad\Desktop\DATA SCIENCE\DATA SETS\Salary_Data.csv")
data.head()


# # first split of data

# In[25]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# # visualization

# In[27]:


sns.displot(data['YearsExperience'],kde=False,bins=10)


# In[28]:


sns.countplot(y='YearsExperience',data=data)


# In[29]:


sns.barplot(x="YearsExperience",y="Salary",data=data)


# In[30]:


sns.heatmap(data.corr())


# # second split of data

# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# In[32]:


#model creation


# In[33]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[34]:


model.fit(X_train,y_train)


# # prediction

# In[36]:


y_prediction=model.predict(X_test)
y_prediction


# # visualizing the training set results

# In[49]:


plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,model.predict(X_train),color="red")
plt.title("salary-yr of exp(train set)")
plt.xlabel("yr of experience")
plt.ylabel("salary")


# # visualizing the test set results

# In[44]:


plt.scatter(X_test,y_test,color="blue")
plt.plot(X_test,model.predict(X_test),color="red")
plt.title("salary-yr of exp(test set)")
plt.xlabel("yr of experience")
plt.ylabel("salary")


# # finding resuduals

# In[46]:


from sklearn import metrics


# In[48]:


print("MSE",metrics.mean_squared_error(y_prediction,y_test))
print("MAE",metrics.mean_absolute_error(y_prediction,y_test))
print("RSE",np.sqrt(metrics.mean_squared_error(y_prediction,y_test)))


# In[ ]:




