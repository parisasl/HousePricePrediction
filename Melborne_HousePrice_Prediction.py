
# In[1]:


import pandas as pd # to read and manipulate data
from sklearn.tree import DecisionTreeRegressor #DecisionTree model
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor #RandomForest model


# In[2]:


#understand the dataset
# save filepath
melbourne_file_path = '/Users/paris/OneDrive/Desktop/Prog_docum/home-data-for-ml-course/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
# print a summary 
melbourne_data.describe()


# In[3]:


melbourne_data.columns


# In[4]:
# drop missing values
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
# A giving (selected) list of features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]


# In[5]:

X.describe()


# In[6]:

X.head()

# In[7]:

# Define model. Specify a number for random_state to ensure same results will happen for each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)


# In[8]:

#EXAMPLE
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# In[9]:


#Evaluate on TRAINING set
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[10]:


# spliting the data to training and evaluating set and train the model with traing set
# split data into training and validation data.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# Evaluate on VALIDATION data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[11]:


# improve accuracy of the model
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# as th tree become deeper, the model would become overfitted...


# In[12]:


#building a randomforest model (improving the model)
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[13]:


# changing the MAX LEAF NODE in randomForest to check the MAE
def randomForestMae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    forestModel = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    forestModel.fit(train_X, train_y)
    preds_forest_value = forestModel.predict(val_X)
    forest_mae = mean_absolute_error(val_y, preds_forest_value)
    return(forest_mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000, 10000]:
    RandomForestMae = randomForestMae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, RandomForestMae))


# In[14]:


# compare MAE with differing values of MAX_DEPTH
def randomForestMae(max_depth, train_X, val_X, train_y, val_y):
    forestModel = RandomForestRegressor(max_depth=max_depth, random_state=1)
    forestModel.fit(train_X, train_y)
    preds_forest_value = forestModel.predict(val_X)
    forest_mae = mean_absolute_error(val_y, preds_forest_value)
    return(forest_mae)

for max_depth in [5, 50, 500, 5000, 10000]:
    RandomForestMae = randomForestMae(max_depth, train_X, val_X, train_y, val_y)
    print("Max depth: %d  \t\t Mean Absolute Error:  %d" %(max_depth, RandomForestMae))





