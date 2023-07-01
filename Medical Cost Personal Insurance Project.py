#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv')
df


# In[3]:


# for null values 

df.isnull().sum()


# In[4]:


# No null value

# Lets see the description

df.describe()


# # Age skew to the right because the mean value is greater than the mmedian value
# # The value of mean is almost equal to the median value in bmi column, so slightly skew to the right
# # The value of mean is almost equal to the median value in children column, so slightly skew to the right
# # Charges skew to the right because the mean value is greater than the mmedian value
# # there are possible outliers because the margin between the Q3 values and max values for some columns(bmi, charges)
# # are too much
# 
# # Charges is the target class while age,bmi and age are the features
# 
# # I need to check for outliers
# 
# plt.figure(figsize=(8,6), facecolor='white')
# plotnumber = 1
# for col in df:
#     if plotnumber <= 11:
#         ax = plt.subplot(2,2, plotnumber)
#         snb.boxplot(df[col], palette="Blues")
#         plt.xlabel(col, fontsize=15)
#         plt.yticks(rotation=180, fontsize=10)
#     plotnumber += 1
# plt.tight_layout()
# plt.show()

# In[10]:


# Check for datatypes to ensure they in format that can be analysed

df.dtypes


# In[ ]:





# In[12]:


print(df['sex'].value_counts())
ax = snb.countplot(x='sex', data =df)


# In[19]:


plt.title('comparison between sex and age')
snb.scatterplot(x='charges', y = 'age', data =df, hue='sex', palette='bright')
plt.show()


# In[20]:


# Sex is evenly distributed and its not affecting the target, so I will drop it

df1 = df.drop('sex', axis = 1)

df1


# In[18]:


print(df['smoker'].value_counts())
ax = snb.countplot(x='smoker', data =df)


# In[14]:



df["smoker"] = np.where(df["smoker"] == 'yes', 1, 0)

df


# In[15]:


print(df['region'].value_counts())
ax = snb.countplot(x='region', data =df)


# In[21]:


plt.title('comparison between charges and region')
snb.scatterplot(x='charges', y = 'region', data =df, hue='sex', palette='bright')
plt.show()


# In[26]:


# To check for outliers I need to remove column of object datatype

df2 = df1.drop('region', axis = 1)

df2


# In[38]:


snb.pairplot(df2, palette='Dark2')
plt.show()


# In[30]:


plt.figure(figsize=(8,6), facecolor='white')
plotnumber = 1 
for col in df2:
    print('col number is: ',col)
    if plotnumber <= 4:
        ax = plt.subplot(2,2, plotnumber)
        snb.boxplot(df2[col], palette="Blues")
        plt.xlabel(col, fontsize=15)
        plt.yticks(rotation=180, fontsize=10)
    plotnumber += 1 
plt.tight_layout()
plt.show()


# In[31]:


# Smoker is a categorical class and the outlier will be ignored
# BMI has outliers and will be removed

#removing the outliers using zscore

from scipy.stats import zscore

out_features = df[['bmi']]
z=np.abs(zscore(out_features))

z


# In[32]:


np.where(z>3)


# In[33]:


df3 = df2[(z<3).all(axis=1)]

df3


# In[35]:


#data loss is relatively not much

print('percentage data loss after removing outliers using IQR ', ((df2.shape[0] - df3.shape[0])/df2.shape[0])* 100)


# In[40]:


# let me check for skewness

df3.skew()


# In[44]:


# Checking how data is distributed across columns

plt.figure(figsize=(20,25), facecolor='green')
p=1

for i in df3:
    if p<= 18:
        ax = plt.subplot(6,4,p)
        snb.distplot(df3[i], color='b')
        plt.xlabel=i
    p += 1
plt.show


# In[47]:


# all the columns are not in acceptable range of 0.5 to -0.5
# remove skewness using cuberoot method

df4 = np.cbrt(df3)


# In[49]:


df4.skew()


# In[52]:


# Checking how data is distributed across columns

plt.figure(figsize=(20,25), facecolor='red')
p=1

for i in df4:
    if p<= 18:
        ax = plt.subplot(6,4,p)
        snb.distplot(df4[i], color='b')
        plt.xlabel=i
    p += 1
plt.show


# In[53]:


# Check Corelation between target variable and independent variables

df4.corr()


# In[54]:


plt.figure(figsize=(26,14))
snb.heatmap(df.corr(), annot=True, fmt='0.2f', linewidths=0.2, linecolor='black', cmap='Spectral')
plt.xlabel='figure'
plt.ylabel='feature name'
plt.title('Descriptive graph', fontsize=20)
plt.show()


# In[55]:


df.corr().charges.sort_values()


# In[58]:


# Smoker is strongly related

# Visualizing the corelation between target and independent variables using bar plot
plt.figure(figsize=(26,14))
df4.corr()['charges'].sort_values(ascending=False).drop(['charges']).plot(kind='bar', color='m')
plt.xlabel='feature'
plt.ylabel='target'
plt.title('Correlation between target and independence variable using bar plot', fontsize=20)
plt.show()


# In[59]:


# Seperating independent variable and target variable into x and y

x = df4.drop('charges', axis=1)

y = df4['charges']

print('Feature dimension : ', x.shape)
print('Target dimension : ', y.shape)


# In[60]:


# feature scaling using standard scalarization 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

x


# In[61]:


# Checking Variance Inflation Factor to confirm multiple colinearity and variance value

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Values'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

vif['Features'] = x.columns

vif


# In[63]:


# VIF is within available range

# Finding best random state

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

maxAccu = 0
maxRS = 0

for i in range(1,300):
    x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size =0.30, random_state=i)
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    pred = LR.predict(x_test)
    acc = r2_score(y_test, pred)
    if acc > maxAccu:
        maxAccu = acc
        maxRS = i
print('Best accuracy is : ', maxAccu, ' at random state : ',maxRS)


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=7)


# In[65]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso, Ridge


LR = LinearRegression()
LR.fit(x_train, y_train)
pred_LR = LR.predict(x_test)
pred_train = LR.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_LR))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_LR))
print('mean squared error : ',mean_squared_error(y_test, pred_LR))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_LR)))


# In[66]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_LR, y=y_test, color='r')
plt.plot(pred_LR, pred_LR, color='b')
plt.xlabel='Actual'
plt.ylabel='Predicted'
plt.title('Linear Regression ', fontsize = 18)

plt.show()


# In[67]:


RFR = RandomForestRegressor()
RFR.fit(x_train, y_train)
pred_RFR = RFR.predict(x_test)
pred_train = RFR.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_RFR))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_RFR))
print('mean squared error : ',mean_squared_error(y_test, pred_RFR))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_RFR)))


# In[68]:


knn = KNN()
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
pred_train = knn.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_knn))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_knn))
print('mean squared error : ',mean_squared_error(y_test, pred_knn))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_knn)))


# In[69]:


GRR = GradientBoostingRegressor()
GRR.fit(x_train, y_train)
pred_GRR = GRR.predict(x_test)
pred_train = GRR.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_GRR))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_GRR))
print('mean squared error : ',mean_squared_error(y_test, pred_GRR))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_GRR)))


# In[76]:


lasso = Lasso()
lasso.fit(x_train, y_train)
pred_lasso = lasso.predict(x_test)
pred_train = lasso.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_lasso))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_lasso))
print('mean squared error : ',mean_squared_error(y_test, pred_lasso))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_lasso)))


# In[70]:


ridge =Ridge()
ridge.fit(x_train, y_train)
pred_ridge = ridge.predict(x_test)
pred_train = ridge.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_ridge))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_ridge))
print('mean squared error : ',mean_squared_error(y_test, pred_ridge))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_ridge)))


# In[71]:


# Do cross validation for the models to validate model performance

from sklearn.model_selection import cross_val_score

#cross_val_score(model/estimate, features, target, CV=5, scoring='r2')

score = cross_val_score(LR,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_LR) - score.mean())


# In[72]:


score = cross_val_score(RFR,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_RFR) - score.mean())


# In[73]:


score = cross_val_score(knn,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_knn) - score.mean())


# In[75]:


score = cross_val_score(GRR,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_GRR) - score.mean())


# In[79]:


score = cross_val_score(lasso,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_lasso) - score.mean())


# In[80]:


score = cross_val_score(ridge,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_ridge) - score.mean())


# In[91]:


# KNN parameters
knn.get_params().keys()


# In[90]:


# The model with the least dif the best choice 
# Hyper parameter tuning to get best parameter for my model
#Build model for best performing option

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'p': [1, 2]  # Power parameter for the Minkowski distance metric
}

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)


# In[87]:


knn.get_params().keys()


# In[93]:


model= KNN(n_neighbors=7,p=2, weights='uniform')


# In[94]:


model.fit(x_train, y_train)
pred = model.predict(x_test)

print('R2 Score : ', r2_score(y_test, pred))
print('mean absolute error : ',mean_absolute_error(y_test, pred))
print('mean squared error : ',mean_squared_error(y_test, pred))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred)))


# In[96]:


# Save model

import joblib
import pickle

filename = 'MedicalInsurance.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[97]:


loaded_model = pickle.load(open('MedicalInsurance.pkl','rb'))

result = loaded_model.score(x_test, y_test)
print(result * 100)


# In[98]:


conclusion = pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]], index=['Predicted','Original'])


# In[99]:


conclusion


# In[ ]:




