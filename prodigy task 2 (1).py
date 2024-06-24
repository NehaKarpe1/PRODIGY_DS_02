#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic_df=pd.read_csv('titanic.csv')


# In[3]:


print(titanic_df.head())


# In[4]:


print(titanic_df.info())


# In[5]:


print(titanic_df.describe())


# In[6]:


# drop unnecesarry columns
titanic_df=titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# In[14]:


print(titanic_df.head())


# In[21]:


#ckeck for  missing values 
missing_values = titanic_df.isnull().any()
print(missing_values[missing_values == True].index)


# In[23]:


#fill the missing values of age column with median
titanic_df['Age'].fillna(titanic_df['Age'].median(),inplace=True)


# In[14]:


#filling the missing columns of embarked with mode
mode_embarked = titanic_df['Embarked'].mode()[0]
titanic_df.fillna(mode_embarked,inplace=True)


# In[15]:


# Convert categorical variables into dummy/indicator variables
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)

# Check for any remaining missing values
print(titanic_df.isnull().sum())


# In[21]:


plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[10]:


plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Sex_male', data=titanic_df)
plt.title('Survival Count by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
#plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(['Female', 'Male'])
plt.show()


# In[2]:


# Explore the survival rate by passenger class
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Pclass', data=titanic_df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Passenger Class')
plt.show()


# In[ ]:




