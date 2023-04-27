#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("loan-train.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[8]:


pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True)


# In[9]:


dataset.boxplot(column='ApplicantIncome')


# In[10]:


dataset['ApplicantIncome'].hist(bins=20)


# In[11]:


dataset['CoapplicantIncome'].hist(bins=20)


# In[12]:


dataset.boxplot(column='ApplicantIncome', by='Education')


# In[13]:


dataset.boxplot(column='LoanAmount')


# In[14]:


dataset['LoanAmount'].hist(bins=20)


# In[15]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[16]:


dataset.isnull().sum()


# In[29]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[37]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[38]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[39]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[40]:


dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount.mean())


# In[41]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[42]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[43]:


dataset.isnull().sum()


# In[44]:


dataset['TotalIncome']= dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[45]:


dataset['TotalIncome_log'].hist(bins=20)


# In[46]:


dataset.head()


# In[48]:


x= dataset.iloc[:,np.r_[1:5,9:11,13:14]].values
y= dataset.iloc[:,12].values


# In[49]:


x


# In[50]:


y


# In[51]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[52]:


print(x_train)


# In[56]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()


# In[57]:


for i in range(0, 5):
    x_train[:,i]= labelencoder_x.fit_transform(x_train[:,i])


# In[59]:


x_train[:,6]= labelencoder_x.fit_transform(x_train[:,6])


# In[60]:


x_train


# In[64]:


labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_train


# In[65]:


for i in range(0, 5):
    x_test[:,i]= labelencoder_x.fit_transform(x_test[:,i])


# In[66]:


x_test[:,6]= labelencoder_x.fit_transform(x_test[:,6])


# In[67]:


labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)


# In[68]:


x_test


# In[69]:


y_test


# In[71]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[72]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier =  DecisionTreeClassifier(criterion = 'entropy', random_state=0)
DTClassifier.fit(x_train, y_train)


# In[73]:


y_pred = DTClassifier.predict(x_test)
y_pred


# In[74]:


from sklearn import metrics
print('The accuracy of decision tree is:', metrics.accuracy_score(y_pred, y_test))


# In[75]:


from sklearn.naive_bayes import GaussianNB
NBClassifier =  GaussianNB()
NBClassifier.fit(x_train, y_train)


# In[76]:


y_pred = NBClassifier.predict(x_test)
y_pred


# In[77]:


print('The accuracy of naive bayes is:', metrics.accuracy_score(y_pred, y_test))


# In[78]:


testdata = pd.read_csv("loan-test.csv")


# In[79]:


testdata.head()


# In[80]:


testdata.info()


# In[81]:


testdata.isnull().sum()


# In[83]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[84]:


testdata.isnull().sum()


# In[85]:


testdata.boxplot(column='LoanAmount')


# In[86]:


testdata.boxplot(column='ApplicantIncome')


# In[88]:


testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())
testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[89]:


testdata.isnull().sum()


# In[90]:


testdata['TotalIncome']= testdata['ApplicantIncome'] + testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[91]:


testdata.head()


# In[92]:


test= testdata.iloc[:,np.r_[1:5,9:11,13:14]].values


# In[93]:


for i in range(0, 5):
    test[:,i]= labelencoder_x.fit_transform(test[:,i])


# In[94]:


test[:,6]= labelencoder_x.fit_transform(test[:,6])


# In[95]:


test


# In[96]:


test= ss.fit_transform(test)


# In[97]:


pred = NBClassifier.predict(test)


# In[98]:


pred


# In[ ]:




