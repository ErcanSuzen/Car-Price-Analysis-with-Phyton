#!/usr/bin/env python
# coding: utf-8

# # Car Price Analysis
# 
# 
# DATA SET INCLUDE OVER 46K CAR DATAS FROM GERMANY , REGISTRATION FROM 2011 TO 2022
# 
# ## Importing data set
# ## Data Wrangling
# ## Exploratory Data Analyses
# ## Model Development
# ## Model Evaluation and Refinement
# 
# 

# In[136]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[137]:


path=r"C:\Users\ERCAN\Desktop\PORTFOLIO\CAR PRICE\dataset_germany-cars-zenrows.csv"


# In[138]:


df=pd.read_csv(path)


# In[139]:


df.head(10)


# In[140]:


df.info()


# ## Drop nulls

# In[141]:


cars_data=df.dropna()
cars_data.count()


# ## Check the typo of datas

# In[142]:


cars_data["fuel"].unique()


# In[143]:


cars_data["gear"].unique()


# In[144]:


cars_data["offerType"].unique()


# In[145]:


cars_data["make"].unique()


# ### Define the data dominant car brands and type
# 
# Volkswagen, Opel and Ford seems dominant brands
# 

# In[146]:


cars_data["make"].value_counts().head(10)


# In[147]:


cars_data["fuel"].value_counts()


# In[148]:


cars_data["model"].value_counts().head(10)


# In[149]:


cars_data["gear"].value_counts()


# In[150]:


cars_data["offerType"].value_counts()


# In[151]:


cars_data.describe()


# ### There are big differences between max and mean values of "price" and "hp" datas. Find and remove all prices over the 99.5% percentile.

# In[152]:


cars_data=cars_data[cars_data["price"]<cars_data["price"].quantile(0.995)].reset_index(drop=True)


# In[153]:


cars_data=cars_data[cars_data["hp"]<cars_data["hp"].quantile(0.995)].reset_index(drop=True)


# ### Analysis of Non Numerical parameters effect on prices

# In[154]:


sns.boxplot(x="gear", y="price", data=cars_data)


# In[155]:


sns.boxplot(x="fuel", y="price", data= cars_data )
plt.gcf().set_size_inches(13, 6)


# In[156]:


cars_data.corr()


# ### We should make Volkswagen car price model according to this datasets

# In[163]:


Volkswagen_data=cars_data.query("make=='Volkswagen'")
Volkswagen_data


# In[160]:


from sklearn.linear_model import LinearRegression


# In[167]:


lm=LinearRegression()


# In[174]:


X=Volkswagen_data[["hp"]]
Y=Volkswagen_data[["price"]]
Z=Volkswagen_data[["hp","mileage","year"]]


# ### X will be used as Independent variable for Linear Regression 
# ### Z will be used as Independent variable for Multiple Linear Regression

# In[175]:


lm.fit(X,Y)
lm.score(X,Y)


# In[177]:


lm2=LinearRegression()


# In[179]:


lm2.fit(Z,Y)
lm2.score(Z,Y)


# In[181]:


sns.regplot(x="price", y="hp" , data=Volkswagen_data , color="green", marker="+")
plt.gcf().set_size_inches(13, 6)


# ### Pipeline

# In[187]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[188]:


Input=[("scale",StandardScaler()),("polynomial",PolynomialFeatures(include_bias=False)),("model",LinearRegression())]


# In[192]:


pipe=Pipeline(Input)


# In[193]:


pipe.fit(X,Y)


# In[194]:


pipe.score(X,Y)


# In[195]:


pipe.fit(Z,Y)


# In[196]:


pipe.score(Z,Y)


# ### Model Evaluation and Refinement

# In[210]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[211]:


z_train, z_test, y_train, y_test = train_test_split(Z, Y, test_size=0.15, random_state=1)


print("number of test samples :", z_test.shape[0])
print("number of training samples:",z_train.shape[0])


# In[212]:


from sklearn.linear_model import Ridge


# In[213]:


pr=PolynomialFeatures(degree=2)
z_train_pr=pr.fit_transform(z_train[["hp","mileage","year"]])
z_test_pr=pr.fit_transform(z_test[["hp","mileage","year"]] )


# In[214]:


RidgeModel=Ridge(alpha=0.1)

RidgeModel.fit(z_train_pr, y_train)


# In[215]:


RidgeModel.score(z_train_pr, y_train)


# In[224]:


z_train_pr=pr.fit_transform(z_train[["hp","mileage","year"]])


# In[225]:


z_polly=pr.fit_transform(z_train[["hp","mileage","year"]])


# In[226]:


RidgeModel=Ridge(alpha=0.1)

RidgeModel.fit(z_train_pr, y_train)

RidgeModel.score(z_train_pr, y_train)


# In[227]:


z_test_pr=pr.fit_transform(z_test[["hp","mileage","year"]])


# In[228]:


z_polly=pr.fit_transform(z_test[["hp","mileage","year"]])


# In[229]:



RidgeModel=Ridge(alpha=0.1)

RidgeModel.fit(z_test_pr, y_test)

RidgeModel.score(z_test_pr, y_test)


# In[ ]:




