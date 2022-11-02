#!/usr/bin/env python
# coding: utf-8

# ## Project Overview

# In this project I will be analyzing a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. I will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# |
# * 'Daily  Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# ### Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


import scipy.stats as stats


# ### Reading the Data and Exploration

# In[2]:


df = pd.read_csv('advertising.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# ### Datatype Conversions

# In[6]:


#Lets convert Timestamp to dat,month and year columns to be able to capture it


# In[7]:


df.Timestamp = pd.to_datetime(df.Timestamp)


# In[8]:


df['Day']= df.Timestamp.dt.day
df['Month'] = df.Timestamp.dt.month
df['Year'] = df.Timestamp.dt.year


# ### Checking for Null Values

# In[9]:


df.isna().all()


# In[10]:


df.isnull().sum()


# In[11]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# There are no missing values in the dataset, I will continue my data exploration through visualizations

# ### Exploratory Data Analysis

# #### Histogram

# In[17]:


sns.set_style('whitegrid')
df['Age'].hist(bins=30, lw=1)
plt.xlabel('Age')


# #### Jointplot Area Income vs Age

# In[18]:


j = sns.jointplot(x = 'Age',y='Area Income',data = df)
j.annotate(stats.pearsonr)


# #### Kde Distributions of Daily Time Spent on Site vs Age

# In[20]:


j = sns.jointplot(x= 'Age', y= 'Daily Time Spent on Site', data = df, kind = 'kde')
j.annotate(stats.pearsonr)


# #### Jointplot of Daily TIme Spent on Site vs Daily Internet Usage

# In[21]:


j = sns.jointplot(x= 'Daily Time Spent on Site', y= 'Daily Internet Usage', data = df, edgecolors='black')
j.annotate(stats.pearsonr)


# In[58]:


p = sns.pairplot(df,hue = 'Clicked on Ad',palette='bwr')
p.map_diag(plt.hist)


# ### Logistic Regression Model

# In[79]:


X = df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male','Day','Month','Year']]
y= df['Clicked on Ad']


# In[80]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state=42)


# In[81]:


from sklearn.linear_model import LogisticRegression
logm = LogisticRegression()


# In[82]:


logm.fit(X_train,y_train)


# ### Predictions and Evaluations

# In[83]:


predictions = logm.predict(X_test)


# In[84]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




