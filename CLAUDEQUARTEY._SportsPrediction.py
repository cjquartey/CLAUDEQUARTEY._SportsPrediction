#!/usr/bin/env python
# coding: utf-8

# QUESTION 1 - Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


playerData = pd.read_csv("male_players (legacy).csv")


# In[3]:


def dropColumns(data):
    #------------------------DROP COLUMNS WITH MORE THAN 30% MISSING VALUES------------------------
    threshold = 0.3
    thresh_count = int((1 - threshold) * len(data))
    data = data.dropna(axis=1, thresh=thresh_count)
    
    
    #------------------------SPLIT DATA INTO CAETEGORICAL AND NUMERICAL FOR IMPUTING------------------------
    numericalData = data.select_dtypes(include = [np.number])
    categoricalData = data.select_dtypes(exclude = [np.number])
    print()
    
    #Keep only the potentially useful categorical variables
    categoricalData = categoricalData[["preferred_foot"]]
            
    #Encode categorical variables
    categoricalData = pd.get_dummies(categoricalData, prefix = "preferred_foot_").astype(int)
    
    data = pd.concat([numericalData, categoricalData], axis = 1).reset_index(drop=True)
    
    #Impute Data
    from sklearn.impute import SimpleImputer
    
    numeric_imputer = SimpleImputer(strategy = "median")
    categorical_imputer = SimpleImputer(strategy = "most_frequent")
    
    data[numericalData.columns] = numeric_imputer.fit_transform(data[numericalData.columns])
    data[categoricalData.columns] = categorical_imputer.fit_transform(data[categoricalData.columns])
    
    return data


# In[4]:


playerData = dropColumns(playerData)


# QUESTION 2 - FEATURE ENGINEERING

# In[5]:


def featureEngineering(data):
    corrMatrix = playerData.corr()
    columnNames = corrMatrix["overall"].index.tolist()
    ##Drop useless numerical variables; i.e. those with a weak correlation to the overall rating
    for i, corr in enumerate(corrMatrix["overall"]):
        if (columnNames[i] not in ["preferred_foot__Left", "preferred_foot__Right"]):
            if abs(corr) < 0.5 or np.isnan(corr):
                playerData.drop([columnNames[i]], axis = 1, inplace = True)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    columns_to_scale = data.columns[1:]
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data


# In[6]:


playerData = featureEngineering(playerData)


# QUESTION 3 - Training Models

# In[7]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error


# In[8]:


def splitData(data):
    Y = data["overall"]
    X = data.drop("overall", axis = 1)
    
    from sklearn.model_selection import train_test_split
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    
    return Xtrain, Xtest, Ytrain, Ytest


# In[9]:


Xtrain,Xtest,Ytrain,Ytest = splitData(playerData)


# In[10]:


###MODEL 1 - LINEAR REGRESSION


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


l = LinearRegression()


# In[13]:


#Training with Cross Validation
cv = 5 
scores = cross_val_score(l, Xtrain, Ytrain, cv=cv)
print("Average cross-validation score:", scores.mean())


# In[14]:


l.fit(Xtrain, Ytrain)


# In[15]:


linear_y_pred = l.predict(Xtest)


# In[16]:


###MODEL 2 - POLYNOMIAL REGRESSION


# In[17]:


from sklearn.preprocessing import PolynomialFeatures


# In[18]:


poly = PolynomialFeatures(degree=2)


# In[19]:


X_poly_train = poly.fit_transform(Xtrain)
X_poly_test = poly.fit_transform(Xtest)


# In[20]:


model = LinearRegression()


# In[21]:


#Training with Cross Validation
cv = 5 
scores = cross_val_score(model, X_poly_train, Ytrain, cv=cv)
print("Average cross-validation score:", scores.mean())


# In[22]:


model.fit(X_poly_train, Ytrain)


# In[23]:


poly_y_pred = model.predict(X_poly_test)


# In[24]:


###MODEL 3 - RANDOM FOREST REGRESSOR


# In[25]:


from xgboost import XGBRegressor


# In[26]:


xgb_model = XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.5, n_estimators=50, random_state=42)


# In[27]:


scores = cross_val_score(xgb_model, Xtrain,Ytrain, cv=3, scoring='neg_mean_squared_error')


# In[28]:


print(f"XGBoost Regressor : {scores.mean():.3f} (+/- {scores.std():.3f})")


# In[29]:


xgb_model.fit(Xtrain, Ytrain)


# In[30]:


xgb_y_pred = xgb_model.predict(Xtest)


# QUESTION 4 - Evaluation

# In[31]:


print(f"""LINEAR REGRESSION MODEL
Mean Absolute Error: {mean_absolute_error(linear_y_pred, Ytest)},
R2 Score: {r2_score(linear_y_pred, Ytest)}""")


# In[32]:


print(f"""POLYNOMIAL REGRESSION MODEL
Mean Absolute Error: {mean_absolute_error(poly_y_pred, Ytest)},
R2 Score: {r2_score(poly_y_pred, Ytest)}""")


# In[33]:


print(f"""RANDOM FOREST REGRESSOR MODEL
Mean Absolute Error: {mean_absolute_error(xgb_y_pred, Ytest)},
R2 Score: {r2_score(xgb_y_pred, Ytest)}""")


# In[34]:


from sklearn.model_selection import GridSearchCV


# In[35]:


# Fine-tuning the XGB Model


# In[36]:


param_grid = {
    'n_estimators': [50, 100, 200, 300],  # number of boosting rounds
    'max_depth': [3, 5, 7, 10],           # maximum depth of a tree
    'learning_rate': [0.01, 0.05, 0.1, 0.2]  # step size shrinkage
}


# In[37]:


grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="neg_mean_absolute_error")
grid_search.fit(Xtrain, Ytrain)


# In[38]:


print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")


# In[39]:


xgb_model = grid_search.best_estimator_


# In[40]:


xgb_y_pred = xgb_model.predict(Xtest)


# In[41]:


print(f"""RANDOM FOREST REGRESSOR MODEL
Mean Absolute Error: {mean_absolute_error(xgb_y_pred, Ytest)},
R2 Score: {r2_score(xgb_y_pred, Ytest)}""")


# In[42]:


###FEATURE IMPORTANCE


# In[43]:


importances = xgb_model.feature_importances_
features = xgb_model.feature_names_in_
importantFeatures = {}

print("Feature Importances:")
for i, importance in enumerate(importances):
    print(f"Feature {features[i]}: {importance:.3f}")
    importantFeatures[features[i]] = importance


# In[44]:


sorted_importances = sorted(importantFeatures.items(), key=lambda x: x[1], reverse=True)

top_five_features = sorted_importances[:5]


# In[45]:


top_five_feature_names = [feature for feature, importance in top_five_features]


# In[46]:


###RETRAINING THE MODEL WITH ONLY THE FIVE IMPORTANT FEATURES


# In[47]:


best_model = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=300, random_state=42)


# In[48]:


for column in Xtrain.columns:
    if column not in top_five_feature_names:
        Xtrain.drop(column, axis = 1, inplace = True)


# In[49]:


for column in Xtest.columns:
    if column not in top_five_feature_names:
        Xtest.drop(column, axis = 1, inplace = True)


# In[50]:


best_model.fit(Xtrain, Ytrain)


# In[51]:


best_model_y_pred = best_model.predict(Xtest)


# In[52]:


print(f"""RANDOM FOREST REGRESSOR MODEL
Mean Absolute Error: {mean_absolute_error(best_model_y_pred, Ytest)},
R2 Score: {r2_score(best_model_y_pred, Ytest)}""")


# QUESTION 5 - Test with new data

# In[53]:


testData = pd.read_csv("players_22-1.csv")


# In[54]:


testData = dropColumns(testData)


# In[55]:


for column in testData.columns:
    if column not in playerData.columns:
        testData.drop(column, axis = 1, inplace = True)


# In[56]:


testData = featureEngineering(testData)


# In[57]:


Ynew = testData["overall"]
Xnew = testData.drop("overall", axis = 1)


# In[58]:


for column in Xnew.columns:
    if column not in top_five_feature_names:
        Xnew.drop(column, axis = 1, inplace = True)


# In[59]:


new_y_pred = best_model.predict(Xnew)


# In[60]:


# Evaluate the model's performance on new data
print("R2 Score: ", (r2_score(new_y_pred, Ynew)))


# QUESTION 6 - Deployment

# In[61]:


import pickle as pkl


# In[62]:


pkl.dump(best_model, open("./" + "FINAL_" + best_model.__class__.__name__ + ".pkl", "wb"))


# In[63]:


###Deployment continued in separate python file

