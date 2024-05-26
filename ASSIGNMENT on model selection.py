#!/usr/bin/env python
# coding: utf-8

# In[375]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # QUESTION 1

# In[376]:


data=pd.read_csv(r"C:\Users\ambad\Desktop\DATA SCIENCE\ASSIGNMENT on model selection\titanic_dataset .csv")
data.head()


# # QUESTION 2

# In[377]:


data.info()


# In[378]:


data.describe()


# In[379]:


data.shape


# In[380]:


#missing values


# In[381]:


data.isna().sum()


# In[384]:


data['Age']=data['Age'].fillna(data['Age'].median())
data['Embarked'].fillna(method='ffill',inplace=True)
data['Cabin']=data['Cabin'].fillna(data['Cabin'].mode())


# In[385]:


data.isna().sum()


# # OUTLIER

# In[280]:


data.columns


# In[281]:


num_col=['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare']
     


# In[282]:


for i in num_col:
    plt.figure()
    sns.boxplot(data[i])
    plt.title(i)


# In[283]:


#outlier correction in SibSp


# In[284]:


q1=np.percentile(data['SibSp'],25,method='midpoint')
q2=np.percentile(data['SibSp'],50,method='midpoint')
q3=np.percentile(data['SibSp'],75,method='midpoint')


# In[285]:


q1


# In[286]:


q2


# In[287]:


q3


# In[288]:


IQR=q3-q1
IQR


# In[289]:


lower_lmt=q1-1.5*IQR
lower_lmt


# In[290]:


upper_lmt=q3+1.5*IQR
upper_lmt


# In[291]:


outlier=[]
for x in data['SibSp']:
    if((x>upper_lmt)or(x<lower_lmt)):
        outlier.append(x)
        
outlier


# In[292]:


data['SibSp']=data['SibSp'].clip(lower=lower_lmt,upper=upper_lmt)


# In[293]:


outlier=[]
for x in data['SibSp']:
    if((x>upper_lmt)or(x<lower_lmt)):
        outlier.append(x)
        
outlier


# In[294]:


#oulier correction in Parch


# In[295]:


q1=np.percentile(data['Parch'],25,method='midpoint')
q2=np.percentile(data['Parch'],50,method='midpoint')
q3=np.percentile(data['Parch'],75,method='midpoint')


# In[296]:


q1


# In[297]:


q2


# In[298]:


q2


# In[299]:


IQR=q3-q1
IQR


# In[300]:


lower_lmt=q1-1.5*IQR
lower_lmt


# In[301]:


upper_lmt=q3+1.5*IQR
upper_lmt


# In[302]:


outlier=[]
for x in data['Parch']:
    if((x>upper_lmt)or(x<lower_lmt)):
        outlier.append(x)
        
outlier


# In[303]:


data['Parch']=data['Parch'].clip(lower=lower_lmt,upper=upper_lmt)


# In[304]:


outlier=[]
for x in data['Parch']:
    if((x>upper_lmt)or(x<lower_lmt)):
        outlier.append(x)
        
outlier


# In[305]:


#outlier correction in fare


# In[306]:


q1=np.percentile(data['Fare'],25,method='midpoint')
q2=np.percentile(data['Fare'],50,method='midpoint')
q3=np.percentile(data['Fare'],75,method='midpoint')


# In[307]:


q1


# q2

# In[308]:


q3


# In[309]:


IQR=q3-q1
IQR


# In[310]:


lower_lmt=q1-1.5*IQR
lower_lmt


# In[311]:


upper_lmt=q3+1.5*IQR
upper_lmt


# In[312]:


outlier=[]
for x in data['Fare']:
    if((x>upper_lmt)or(x<lower_lmt)):
        outlier.append(x)
        
outlier


# In[313]:


data['Fare']=data['Fare'].clip(lower=lower_lmt,upper=upper_lmt)


# In[314]:


outlier=[]
for x in data['Fare']:
    if((x>upper_lmt)or(x<lower_lmt)):
        outlier.append(x)
        
outlier


# In[315]:


#Encoding


# In[316]:


data1=data.copy()


# In[317]:


data1.columns


# In[318]:


from sklearn.preprocessing import LabelEncoder


# In[319]:


label_en=LabelEncoder()


# In[320]:


data1.nunique()


# In[321]:


data1['Name']=label_en.fit_transform(data1['Name'])
data1['Sex']=label_en.fit_transform(data1['Sex'])
data1['Ticket']=label_en.fit_transform(data1['Ticket'])
data1['Cabin']=label_en.fit_transform(data1['Cabin'])
data1['Embarked']=label_en.fit_transform(data1['Embarked'])


# In[322]:


data1.head()


# In[323]:


#scaling


# In[324]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data_sc=sc.fit_transform(data1)
data_sc=pd.DataFrame(data_sc)
data_sc.head()


# # QUESTION 3

# # 1)KNN

# In[325]:


#first split of data

y=data1['Survived']
x=data1.drop(['Survived','Name','Sex'],axis=1)


# In[326]:


x.shape


# In[327]:


#second split of data

from sklearn.model_selection import train_test_split


# In[328]:


X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# In[329]:


X_train.shape


# In[330]:


X_test.shape


# In[333]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


# In[334]:


from sklearn.neighbors import KNeighborsClassifier 
metrics_k=[]
neighbors=np.arange(1,15)


# In[335]:


for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    classifier.fit(X_train,y_train)
    y_prediction=classifier.predict(X_test)
    acc=accuracy_score(y_test,y_prediction)
    metrics_k.append(acc)
    
metrics_k


# In[336]:


plt.plot(neighbors,metrics_k,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()


# In[337]:


classifier=KNeighborsClassifier(n_neighbors=3,metric='euclidean')
classifier.fit(X_train,y_train)
y_prediction=classifier.predict(X_test)
acc=accuracy_score(y_test,y_prediction)
print('accuracy is',accuracy_score(y_test,y_prediction))
print('precision is',precision_score(y_test,y_prediction))
print('recal is',recall_score(y_test,y_prediction))
print('f1_score is',f1_score(y_test,y_prediction))
acc=accuracy_score(y_test,y_prediction)


# In[338]:


confusion_matrix(y_test,y_prediction)


# # 2)SVM

# In[339]:


from sklearn.svm import SVC
svm_clf=SVC(kernel='linear')
svm_clf.fit(X_train,y_train)
y_svm_prediction=svm_clf.predict(X_test)


# In[341]:


print('accuracy is',accuracy_score(y_test,y_svm_prediction))


# In[340]:


print(confusion_matrix(y_test,y_svm_prediction))


# # QUESTION 4

# In[342]:


#applying kfold cross validation


# In[346]:


from sklearn.model_selection import cross_val_score,KFold


# In[347]:


kf=KFold(n_splits=10)


# In[348]:


#printing the kfolds


# In[349]:


for train_index,test_index in kf.split(x):
  print('train_index:',train_index)
  print('test_index:',test_index)


# In[350]:


#getting the average score


# In[353]:


score=cross_val_score(classifier,x,y,cv=kf)
print('averagecross validation scores: {}'.format(score.mean()))


# In[354]:


#applying straitified kfold


# In[357]:


from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10)


# In[358]:


for train_index,test_index in skf.split(x,y):
  print('train_index:',train_index)
  print('test_index:',test_index)


# In[360]:


shcv_result=cross_val_score(classifier,x,y,cv=skf)
shcv_result


# In[361]:


#getting average score


# In[362]:


print('average cross validation score:{}'.format(shcv_result.mean()))


# In[ ]:




