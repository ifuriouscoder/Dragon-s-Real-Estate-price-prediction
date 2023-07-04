#!/usr/bin/env python
# coding: utf-8

# # Dragon Real Estate - Price Predictor
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plot
import numpy as np


# In[2]:


housing = pd.read_csv("housing.csv")   #housing_df


# In[3]:


housing.head()


# In[4]:


housing.shape


# In[5]:


housing.info()


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing['AGE'].value_counts()


# In[8]:


housing.describe()


# In[9]:


# housing.hist(bins=50,figsize=(20,15))   #plot.show() not required in jupyter %matplotlib inline


# # Train-Test splitting

# In[10]:


#def split_train_test(data,test_ratio):
    #np.random.seed(42)
    #shuffled=np.random.permutation(len(data))
    #test_set_size=int(len(data)*test_ratio)
    #test_indices=shuffled[:test_set_size]
    #train_indices=shuffled[test_set_size:]
    #return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


#train_set,test_set=split_train_test(housing,0.2)


# In[12]:


#print(f"Rows in train set :{len(train_set)}\nRows in test set:{len(test_set)}")


# In[13]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)

print(f"Rows in train set :{len(train_set)}\nRows in test set:{len(test_set)}")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[15]:


strat_test_set.head()


# In[16]:


strat_test_set.shape


# In[17]:


strat_test_set.info()


# In[18]:


strat_test_set.describe()


# In[19]:


strat_test_set['CHAS'].value_counts()    #number of zeroes and ones equally distributed to both train and test datasets


# In[20]:


strat_train_set['CHAS'].value_counts()


# In[21]:


376/28


# In[22]:


95/7


# In[23]:


housing=strat_train_set.copy()


# # Looking for Correlations

# In[24]:


corr_matrix=housing.corr()


# In[25]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[26]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[27]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)    #positive correlation


# # Trying out Attribute Combinations

# In[28]:


housing['TAXRM']=housing['TAX']/housing['RM']


# In[29]:


housing['TAXRM']


# In[30]:


housing.head()


# In[31]:


#corr_matrix=housing.corr()


# In[32]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[33]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)    #strong negative correlation


# In[34]:


#separate features
housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set['MEDV'].copy()


# # Missing attributes

# In[35]:


housing.dropna(subset=['RM'])   #option1

a=housing.dropna(subset=['RM'])   #option1
a.shape
#note that the original data frame will remain unchangedhousing.drop("RM",axis=1)   #option2
#Note there is no RM column now and also note that the original data frame will remain unchanged
# In[36]:


housing.describe()    #before imputing and filling the missing attributes


# In[37]:


#option 3 Median
median=housing['RM'].median()


# In[38]:


median


# In[39]:


housing["RM"].fillna(median)
#note that the original data frame will remain unchanged


# In[40]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[41]:


imputer.statistics_


# In[42]:


imputer.statistics_.shape


# In[43]:


X = imputer.transform(housing)


# In[44]:


housing_tr=pd.DataFrame(X, columns=housing.columns)


# In[45]:


housing_tr.describe()


# # Scikit - learn design
# # Creating a Pipeline

# In[46]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[47]:


my_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),  #...add as many as you want in your pipeline
    ("std_scaler",StandardScaler())
])


# In[48]:


housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr  #gives you the numPy array
housing_num_tr.shape


# # Selecting a desired model for Dragon Real Estates

# ## Linear Regression

# In[49]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_num_tr,housing_labels)


# In[50]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[51]:


list(some_labels)


# # Evaluating the Model

# In[52]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)


# In[53]:


lin_mse


# In[54]:


lin_rmse


# # Using better Evaluation Technique - Cross Validation

# In[65]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores


# In[66]:


def print_scores(scores):
    print("Scores : ",scores)
    print("Mean : ",scores.mean())
    print("Standard Deviation : ",scores.std())

print_scores(rmse_scores)


# # Decision Tree Regression

# In[55]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[56]:


mse  #overfitted model


# In[57]:


rmse


# In[60]:


#1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[61]:


rmse_scores


# In[62]:


def print_scores(scores):
    print("Scores : ",scores)
    print("Mean : ",scores.mean())
    print("Standard Deviation : ",scores.std())


# In[63]:


print_scores(rmse_scores)


# # Random Forest Regression

# In[67]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[68]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores


# In[69]:


def print_scores(scores):
    print("Scores : ",scores)
    print("Mean : ",scores.mean())
    print("Standard Deviation : ",scores.std())

print_scores(rmse_scores)


# Quiz: Convert this notebook into a python file & run the pipeline using Visual Studio Code

# ## Saving The Model

# In[70]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# ## Testing the Model

# In[77]:


X_test = strat_test_set.drop("MEDV", axis=1)

Y_test = strat_test_set["MEDV"].copy()

X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
a=list(Y_test)
for i in range(len(final_predictions)):
    print(final_predictions[i],'\t',a[i])
    
#print(final_predictions[i],list(Y_test))


# In[72]:


final_rmse


# # Using the Model

# In[78]:


from joblib import dump, load 
import numpy as np 
model = load('Dragon.joblib')
features = np.array([[-0.43942006, 3.12628155, -1.12165014, -0.27288841, -1.4221456,
-0.23979304, -1.31238772, 2.61111401, -1.0016859 , -0.5778192 , 
-0.97491834, 0.41164221, -0.86091034]]) 

model.predict(features)


# In[ ]:




