#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import dump, load 
import numpy as np 
model = load('Dragon.joblib')


# In[4]:


# number of elements as input
lst = list(map(float, input("Enter the elements").split(",")))

 
print(lst)

features = np.array(lst).reshape(1,13)    #reshaped to 1row and 13columns to convert it into 2D array
 
# displaying list
print ("List: ", lst)

print()
 
# displaying array
print ("Array: ", features)

# model.predict(lst)    #gives ValueError as it requires 2D array
model.predict(features)


# In[ ]:




