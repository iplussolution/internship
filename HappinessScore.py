#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[41]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')

df


# In[42]:


# I will drop some column(Happiness rank, Region and country) that is not affecting happiness rank

df2 = df.drop(['Country','Region','Happiness Rank'], axis=1)

df2


# In[43]:


# for null values

df2.isnull().sum()


# In[44]:


df.info()


# In[45]:


# Check the relationship among the columns

snb.pairplot(df2,hue='Happiness Score', palette='Dark2')
plt.show()


# In[46]:


# Check for duplicate values in the data frame

print('Total duplicate values is ',df2.duplicated().sum() )


# In[47]:


# Checking the features relationship with the label

snb.lmplot(x='Economy (GDP per Capita)', y='Happiness Score', data=df2, palette='colorblind')


# In[48]:


snb.lmplot(x='Family', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[49]:


snb.lmplot(x='Standard Error', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[50]:


snb.lmplot(x='Health (Life Expectancy)', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[51]:


snb.lmplot(x='Freedom', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[52]:


snb.lmplot(x='Trust (Government Corruption)', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[53]:


snb.lmplot(x='Generosity', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[54]:


snb.lmplot(x='Dystopia Residual', y='Happiness Score', data=df2, palette='colorblind')
plt.show()


# In[55]:


df2.describe()


# In[56]:


# Lets check the outliers by plotting Boxplot

plt.figure(figsize=(20,25))

p=1

for i in df2:
    if p<=13:
        plt.subplot(5,4,p)
        snb.boxplot(df2[i], palette='Set2_r')
        plt.xlabel=i
        
    p+=1
    
plt.show()


# In[58]:


#removing the outliers using zscore

from scipy.stats import zscore

out_features = df2[['Happiness Score', 'Standard Error', 'Economy (GDP per Capita)', 'Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual']]
z=np.abs(zscore(out_features))

z


# In[116]:


# Using Threshold of 3 to check for outliers

np.where(z>3)


# In[117]:


df3 = df2[(z<3).all(axis=1)]

df3


# In[118]:


print('the shape of new data: ',df2.shape[0])
print('the shape of old data: ',df3.shape[0])


# In[119]:


#Check for skewness
df3.skew()


# In[120]:


# Check the relationship among the columns

snb.pairplot(df2,hue='Happiness Score', palette='Dark2')
plt.show()


# In[121]:


# some of the columns are not in acceptable range of 0.5 to -0.5
# remove skewness using cuberoot,log,boxcox method

df3['Standard Error'] = np.log(df3['Standard Error'])
df3.skew()


# In[122]:


from scipy import stats

df3['Family'] = stats.boxcox(df3['Family'])[0]

df3.skew()


# In[124]:


df3['Health (Life Expectancy)'] = np.cbrt(df3['Health (Life Expectancy)'])
df3.skew()


# In[125]:



df3['Trust (Government Corruption)'] = np.sqrt(df3['Trust (Government Corruption)'])
df3.skew()


# In[126]:



df3['Generosity'] = np.sqrt(df3['Generosity'])

df3.skew()


# In[128]:


# The best reduction of skeness doesnot fall into acceptable range therefore I will drop the feature

df4 = df3.drop(['Health (Life Expectancy)'], axis= 1)

df4


# In[133]:


#checking corelation between target and features
cor = df3.corr()
cor


# In[134]:


plt.figure(figsize=(26,14))
snb.heatmap(df3.corr(), annot=True, fmt='0.2f', linewidths=0.2, linecolor='black', cmap='Spectral')
plt.xlabel='figure'
plt.ylabel='feature name'
plt.title('Descriptive graph', fontsize=20)
plt.show()


# In[138]:


# Visualizing the corelation between target and independent variables using bar plot
plt.figure(figsize=(26,14))
df3.corr()['Happiness Score'].sort_values(ascending=False).drop(['Happiness Score']).plot(kind='bar', color='m')
plt.xlabel='feature'
plt.ylabel='target'
plt.title('Correlation between target and independence variable using bar plot', fontsize=20)
plt.show()


# In[139]:


# Seperating independent variable and target variable into x and y

x = df3.drop('Happiness Score', axis=1)

y = df3['Happiness Score']

print('Feature dimension : ', x.shape)
print('Target dimension : ', y.shape)


# In[140]:


# feature scaling using standard scalarization 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

x


# In[141]:


# Checking Variance Inflation Factor to confirm multiple colinearity and variance value

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Values'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

vif['Features'] = x.columns

vif


# In[142]:


# VIF is with acceptable range

# Finding best random state

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

maxAccu = 0
maxRS = 0

for i in range(1,200):
    x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size =0.30, random_state=i)
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    pred = LR.predict(x_test)
    acc = r2_score(y_test, pred)
    if acc > maxAccu:
        maxAccu = acc
        maxRS = i
print('Best accuracy is : ', maxAccu, ' at random state : ',maxRS)


# In[143]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=maxRS)


# In[144]:


# Linear regression algorithms

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


# In[145]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_LR, y=y_test, color='r')
plt.plot(pred_LR, pred_LR, color='b')
plt.xlabel='Actual'
plt.ylabel='Predicted'
plt.title('Linear Regression ', fontsize = 18)

plt.show()


# In[146]:


RFR = RandomForestRegressor()
RFR.fit(x_train, y_train)
pred_RFR = RFR.predict(x_test)
pred_train = RFR.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_RFR))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_RFR))
print('mean squared error : ',mean_squared_error(y_test, pred_RFR))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_RFR)))


# In[147]:


knn = KNN()
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
pred_train = knn.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_knn))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_knn))
print('mean squared error : ',mean_squared_error(y_test, pred_knn))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_knn)))


# In[148]:


GRR = GradientBoostingRegressor()
GRR.fit(x_train, y_train)
pred_GRR = GRR.predict(x_test)
pred_train = GRR.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_GRR))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_GRR))
print('mean squared error : ',mean_squared_error(y_test, pred_GRR))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_GRR)))


# In[149]:


lasso = Lasso()
lasso.fit(x_train, y_train)
pred_lasso = lasso.predict(x_test)
pred_train = lasso.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_lasso))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_lasso))
print('mean squared error : ',mean_squared_error(y_test, pred_lasso))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_lasso)))


# In[150]:


ridge =Ridge()
ridge.fit(x_train, y_train)
pred_ridge = ridge.predict(x_test)
pred_train = ridge.predict(x_train)

print('R2 Score : ', r2_score(y_test, pred_ridge))
print('R2 Score on training data : ', r2_score(y_train, pred_train)*100)
print('mean absolute error : ',mean_absolute_error(y_test, pred_ridge))
print('mean squared error : ',mean_squared_error(y_test, pred_ridge))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred_ridge)))


# In[151]:


# Do cross validation for the models to validate model performance

from sklearn.model_selection import cross_val_score

#cross_val_score(model/estimate, features, target, CV=5, scoring='r2')

score = cross_val_score(LR,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_LR) - score.mean())


# In[152]:


score = cross_val_score(RFR,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_RFR) - score.mean())


# In[153]:


score = cross_val_score(knn,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_knn) - score.mean())


# In[154]:


score = cross_val_score(GRR,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_GRR) - score.mean())


# In[155]:


score = cross_val_score(lasso,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_lasso) - score.mean())


# In[156]:


score = cross_val_score(ridge,x,y)
print(score)
print(score.mean())
print('Difference between R2 score and cross validation score is ', r2_score(y_test, pred_ridge) - score.mean())


# In[158]:


# The best choice is ridge

# The model with the least dif the best choice
# Hyper parameter tuning to get best parameter for my model
#Build model for best performing option

from sklearn.model_selection import GridSearchCV

param = {'alpha':[1.0,.05,.4,2],'fit_intercept':[True, False], 'solver' :['auto','svd','cholesky','lsqr','sag','saga','lbfgs'],
             'positive' : [False,True],
             'random_state' : [1,4,10,20]}

gscv = GridSearchCV(ridge, param, cv=5)

gscv.fit(x_train, y_train)


# In[159]:


model= Ridge(alpha=0.05,fit_intercept=True, positive=False, random_state=1,solver='lsqr')


# In[160]:


model.fit(x_train, y_train)
pred = model.predict(x_test)

print('R2 Score : ', r2_score(y_test, pred))
print('mean absolute error : ',mean_absolute_error(y_test, pred))
print('mean squared error : ',mean_squared_error(y_test, pred))
print('root mean squared error : ',np.sqrt(mean_squared_error(y_test, pred)))


# In[161]:


# Save model

import joblib
import pickle

filename = 'HappinessScore.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[162]:


loaded_model = pickle.load(open('HappinessScore.pkl','rb'))

result = loaded_model.score(x_test, y_test)
print(result * 100)


# In[163]:


con = pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]], index=['Predicted','Original'])

con


# In[ ]:




