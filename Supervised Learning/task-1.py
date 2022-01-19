#!/usr/bin/env python
# coding: utf-8

# # SIMPLE LINEAR REGRESSION
# 
# In given task we have to predict the percentage of marks expected by the student based upon the number of hours they studied.In this task only two variables are involved.

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[2]:


#Import the data
url="http://bit.ly/w-data"
data=pd.read_csv(url)
data1=data
print("The data is imported successfully")
data


# In[3]:


data.describe()


# # DATA VISUALIZATION
# 
# Now let's plot a graph of our data so that it will give us clear idea about data.

# In[4]:


#Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Linear Regression Model
# 
# Now we prepare the data and split it in test data

# In[5]:


#Splitting training and testing data
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
x_train, x_test, y_train, y_test= train_test_split(x, y,train_size=0.80,test_size=0.20,random_state=0)


# # Training the model

# In[6]:


from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predict= linearRegressor.predict(x_train)


# # Training the Algorithm
# 
# Now the spliting of our data into training and testing sets is done, now it's time to train our algorithm.

# In[7]:


regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")


# In[8]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# # Checking the accuracy scores for training and test set

# In[9]:


print('Test Score')
print(regressor.score(x_test, y_test))
print('Training Score')
print(regressor.score(x_train, y_train))


# # Now we make predictions

# In[10]:


a={'Actual' : y_test,'Predicted' : y_predict }
data= pd.DataFrame.from_dict(a, orient='index')
data=data.transpose()
data


# In[11]:


#Let's predict the score for 9.25 hpurs
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))


# In[12]:


#Let's predict the score for 8.9 hpurs
print('Score of student who studied for 8.9 hours a dat', regressor.predict([[8.9]]))

