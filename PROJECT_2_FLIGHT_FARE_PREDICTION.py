#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING IMPORTANT LIBRARIES 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error , mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib.inline', '')


# # IMPORTING DATASETS

# In[2]:


# importing the dataset 
train_data=pd.read_excel('flight_fare_prediction_train.xlsx')
test_data=pd.read_excel('flight_fare_prediction_test.xlsx')


# In[3]:


train_data


# In[4]:


train_data.isnull().sum()


# In[5]:


train_data=train_data.dropna()


# In[6]:


train_data


# In[7]:


train_data.describe()


# In[8]:


train_data.info()


# In[9]:


train_data.nunique()


# # FEATURE ENGINEERING

# In[10]:


train_data['journey_day']=pd.to_datetime(train_data.Date_of_Journey,format='%d/%m/%Y').dt.day
train_data['journey_month']=pd.to_datetime(train_data.Date_of_Journey,format='%d/%m/%Y').dt.month


# In[11]:


train_data.head()


# In[12]:


train_data.drop(['Date_of_Journey'],axis=1)


# In[14]:


train_data['dep_min']=pd.to_datetime(train_data.Dep_Time).dt.minute
train_data['dep_hour']=pd.to_datetime(train_data.Dep_Time).dt.hour


# In[15]:


train_data


# In[16]:


train_data.drop(['Dep_Time'],axis=1,inplace=True)


# In[17]:


train_data


# In[18]:


## Time taken to reach destination is called Duration
## It is difference between Departure Time and Arrival Time

## Assigning and converting Duration column into list
duration = list(train_data.Duration)

for i in range(len(duration)):
    if len(duration[i].split()) != 2:                       # Check if duration contains only hour or mins
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # Adding 0 mins
        else:
            duration[i] = "0h " + duration[i]


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split("h")[0]))    # Extract hours from Duration
    duration_mins.append(int(duration[i].split('m')[0].split()[-1]))  # Extract onlye minutes from Duration
    


# In[19]:


train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins


# In[20]:


train_data


# In[21]:


train_data['arr_hour']=pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data['arr_min']=pd.to_datetime(train_data.Arrival_Time).dt.minute


# In[22]:


train_data.drop(['Arrival_Time'],axis=1,inplace=True)
train_data.drop(['Duration'],axis=1,inplace=True)


# In[23]:


train_data


# In[24]:


## Additional_Info conatins almost 80% no_info
# Route and Total_Stops are related to each other

train_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)


# In[25]:


train_data.drop(['Date_of_Journey'], axis=1, inplace=True)


# In[26]:


train_data


# In[27]:


train_data.columns


# # SPLITTING DATA INTO FEATURES AND TARGET

# In[28]:


# splitting data into features and target

X=train_data[['Airline', 'Source', 'Destination', 'Total_Stops',
       'journey_day', 'journey_month', 'dep_hour', 'dep_min', 'Duration_hours',
       'Duration_mins', 'arr_hour', 'arr_min']]
y=train_data['Price']


# In[29]:


X.dtypes


# In[30]:


y


# # TRAIN TEST SPLIT

# In[31]:


# splitting data into train and test data


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42)


# In[32]:


print(X_train)
print(X_test)
print(y_train)
print(y_test)


# In[33]:


#  splitting the train and test datainto categorical and numeical data

X_train_cat=X_train.select_dtypes(include='object')
X_train_num=X_train.select_dtypes(include='int64')

X_test_cat=X_test.select_dtypes(include='object')
X_test_num=X_test.select_dtypes(include='int64')


# In[34]:


print(X_train_cat)
print(X_train_num)
print(X_test_cat)
print(X_test_num)


# # ENCODING AND PREPARING DATA FOR MODELLING

# In[35]:


# Onehotencoder on train and test categorical data
OE=OrdinalEncoder()
OE.fit(X_train_cat)
X_train_cat_enc=OE.transform(X_train_cat)


# In[36]:


print(X_train_cat_enc)


# In[37]:


OE.fit(X_test_cat)
X_test_cat_enc=OE.transform(X_test_cat)


# In[41]:


# concat the train and test data categorical and numerical data
X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)
X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)
X_train_cat_enc_df.reset_index(drop=True,inplace=True)
X_train_num.reset_index(drop=True,inplace=True)
X_test_cat_enc_df.reset_index(drop=True,inplace=True)
X_test_num.reset_index(drop=True,inplace=True)
print(X_train_cat_enc_df.shape)
print(X_test_cat_enc_df.shape)
print(X_train_num.shape)
print(X_test_num.shape)

X_train_final=pd.concat([X_train_cat_enc_df,X_train_num],axis=1,ignore_index=True)
X_test_final=pd.concat([X_test_cat_enc_df,X_test_num],axis=1,ignore_index=True)


# In[42]:


print(X_train_cat_enc_df.shape)
print(X_train_num.shape)
print(X_test_cat_enc_df.shape)
print(X_test_num.shape)


# In[43]:


print(X_train_final.isnull().sum())
print(X_train_final.info())
print(X_train_final.shape)


# In[46]:


y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
print(y_train)


# # MODEL BUILDING

# In[47]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train_final,y_train)
y_pred=model.predict(X_test_final)


# In[48]:


print(y_pred)


# In[50]:


RMSE=mean_squared_error(y_pred,y_test)


# In[51]:


print(RMSE)


# # MODEL EVALUATION

# In[52]:


from sklearn.metrics import r2_score
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[53]:


metrics.r2_score(y_test, y_pred)


# # HYPER PARAMETER TUNING

# In[56]:


from sklearn.model_selection import RandomizedSearchCV


# In[57]:


# Randomized Search CV

## Number of trees in ramdom forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
## Number of features to consider at every split
max_features = ['auto', 'sqrt']
## Maximum number of level in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
## Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
## Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[58]:


## create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)


# In[59]:


rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)


# In[60]:


rf_random.fit(X_train_final, y_train)


# In[61]:


rf_random.best_params_


# In[64]:


model_1=RandomForestRegressor(n_estimators= 700,
 min_samples_split= 15,
 min_samples_leaf= 1,
 max_features='auto',
 max_depth=20)
model_1.fit(X_train_final,y_train)


# In[65]:


y_pred_1 = model_1.predict(X_test_final)


# In[66]:


print("MAE: ", metrics.mean_absolute_error(y_test, y_pred_1))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred_1))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred_1)))


# In[67]:


metrics.r2_score(y_test, y_pred_1)


# In[ ]:




