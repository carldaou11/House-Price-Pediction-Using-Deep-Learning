#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import json
import random
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train=pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
#train.head()


# In[3]:


last_index = train[-1:]['Id'].values[0]
last_index


# In[4]:


data = pd.concat([train,test],axis=0,sort=False)


# In[5]:


data.info()


# In[6]:


data.duplicated().sum()


# In[7]:


data.isnull().sum()


# In[8]:


data.isnull().mean()


# In[9]:


def missing (data):
    missing_number = data.isnull().sum().sort_values(ascending=False)
    missing_percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values

missing(data)


# In[10]:


top_features = train.corr()[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(30)
plt.figure(figsize=(5,10))
sns.heatmap(top_features,cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=-1)


# In[11]:


#for col in data.columns:
  #  if data[col].isnull().mean()*100>40:
       # data.drop(col,axis=1,inplace=True)

data = data.drop(columns=['PoolQC', 'MiscFeature','Alley', 'Fence','FireplaceQu'])


# In[12]:


data.shape


# In[13]:


sns.countplot(data.dtypes.map(str))
plt.show()


# In[14]:


num_features=data.select_dtypes(include=['int64','float64'])
categorical_features=data.select_dtypes(include='object')


# In[15]:


data['BsmtFinSF1'].fillna(0, inplace=True)
data['BsmtFinSF2'].fillna(0, inplace=True)
data['TotalBsmtSF'].fillna(0, inplace=True)
data['BsmtUnfSF'].fillna(0, inplace=True)
data['Electrical'].fillna('FuseA',inplace = True)
data['KitchenQual'].fillna('TA',inplace=True)
data['LotFrontage'].fillna(data.groupby('1stFlrSF')['LotFrontage'].transform('mean'),inplace=True)
data['LotFrontage'].interpolate(method='linear',inplace=True)
data['MasVnrArea'].fillna(data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace=True)
data['MasVnrArea'].interpolate(method='linear',inplace=True)


# In[16]:


missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending = False)
NAN_col = list(missing_values.to_dict().keys())
missing_values_data = pd.DataFrame(missing_values)
missing_values_data.reset_index(level=0, inplace=True)
missing_values_data.columns = ['Feature','Number of Missing Values']
missing_values_data['Percentage of Missing Values'] = (100.0*missing_values_data['Number of Missing Values'])/len(data)
missing_values_data


# In[17]:


for col in NAN_col:
    data_type = data[col].dtype
    if data_type == 'object':
        data[col].fillna('NA',inplace=True)
    else:
        data[col].fillna(data[col].mean(),inplace=True)


# In[18]:


data = pd.get_dummies(data)
data


# In[19]:


from scipy.stats import skew

numerical_features = data.dtypes[data.dtypes != 'object'].index

# checking the skewness in all the numerical features
skewed_features = data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)

# converting the features into a dataframe
skewness = pd.DataFrame({'skew':skewed_features})

# checking the head of skewness dataset
skewness


# In[20]:


train_clean = data[data['Id'] < last_index]


# In[21]:


train_clean['SalePrice']


# In[22]:


test_clean = data[data['Id'] > last_index]


# In[23]:


test_clean['SalePrice']


# In[24]:



x=train_clean.drop(['Id','SalePrice'],axis=1) ## all the features
y=train_clean['SalePrice']  ## target variable
test_clean_id = test_clean['Id'].to_list()
test_clean = test_clean.drop(['Id','SalePrice'],axis=1)


# In[25]:


from sklearn.preprocessing import MinMaxScaler
mc=MinMaxScaler()
scaled_x=mc.fit_transform(x)
test_clean_scaled = mc.transform(test_clean)


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(scaled_x,y,test_size=0.20,random_state=0)


# In[27]:


x_train.shape


# In[28]:


x_test.shape


# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


LR=LinearRegression()
LR.fit(x_train,y_train)  ## fitting the training data

x_test_pred_LR=LR.predict(x_test)  ## predicted x test


# In[31]:


x_test_pred_LR


# In[32]:


y_test


# In[33]:


x_train_pred_LR=LR.predict(x_train) ##predicted x train

x_train_pred_LR


# In[34]:


y_train


# In[35]:


print('Linear Regression trainind score is',LR.score(x_train,y_train))


# In[36]:


print('Linear Regression testing score is',LR.score(x_test,y_test))


# In[37]:


from sklearn.metrics import r2_score


# In[38]:


train_score=r2_score(y_train,x_train_pred_LR)
print('Linear Regression r2_score for training is',train_score)


# In[39]:


y_final_pred_LR=LR.predict(test_clean_scaled)


# In[40]:


y_final_pred_LR


# In[54]:


test_clean


# In[53]:


final_prediction = pd.DataFrame([])
final_prediction['Id'] = test_clean_id
final_prediction['SalePrice'] = y_final_pred_LR
final_prediction.to_csv('LR_Results.csv', index=False)


# In[54]:


from xgboost import XGBRegressor


# In[77]:


XGB=XGBRegressor()
XGB.fit(x_train,y_train) 

xtest_XGB_pred=XGB.predict(x_test) ## predicted x test


# In[78]:


xtest_XGB_pred


# In[70]:


xtrain_XGB_pred=XGB.predict(x_train)


# In[71]:


xtrain_XGB_pred


# In[79]:


y_test


# In[ ]:


xtrain_XGB_pred=LR.predict(x_train) ##predicted x train

x_train_pred_LR


# In[72]:


print('Training score for XGB is',XGB.score(x_train,y_train))


# In[73]:


print('Testing score for XGB is',XGB.score(x_test,y_test))


# In[81]:


y_final_pred_XGB=XGB.predict(test_clean_scaled)


# In[82]:


final_prediction = pd.DataFrame([])
final_prediction['Id'] = test_clean_id
final_prediction['SalePrice'] = y_final_pred_XGB
final_prediction.to_csv('XGB_Results.csv', index=False)


# In[61]:


# Multiple Liner Regression
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()  
#regressor.fit(x_train, y_train)
#evaluate the model (intercept and slope)
#print(regressor.intercept_)
#print(regressor.coef_)
#predicting the test set result
#y_pred = regressor.predict(x_test)
#put results as a DataFrame
#coeff_df = pd.DataFrame(regressor.coef_, data.drop('SalePrice',axis =1).columns, columns=['Coefficient']) 
#coeff_df


# In[62]:


# visualizing residuals
#fig = plt.figure(figsize=(10,5))
#residuals = (y_test- y_pred)
#sns.distplot(residuals)


# In[63]:


#compare actual output values with predicted values
#y_pred = regressor.predict(x_test)
#data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#df1 = data.head(10)
#df1
# evaluate the performance of the algorithm (MAE - MSE - RMSE)
#from sklearn import metrics
#print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
#print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


# In[83]:


# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# In[84]:


# having 19 neuron is based on the number of available features
model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='Adam',loss='mes')


# In[67]:


model.fit(x=x_train,y=y_train,
          validation_data=(x_test,y_test),
          batch_size=128,epochs=400)
model.summary()


# In[85]:


y_pred = model.predict(x_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
# Visualizing Our predictions
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions
plt.plot(y_test,y_test,'r')


# In[87]:


# visualizing residuals
#fig = plt.figure(figsize=(10,5))
#residuals = (y_test - y_pred)
#sns.distplot(residuals)


# In[ ]:




