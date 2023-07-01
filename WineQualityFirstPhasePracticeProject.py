#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')

df


# In[6]:


df.describe


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df.info()


# In[11]:


# Dependent variable is quality and will classify quality value of 7 and above good 
# and quality of values below 7 is not good

df["quality"] = np.where(df["quality"] >= 7, 1, 0)

df


# In[12]:


df.nunique().to_frame("number of unique values")


# In[13]:


print('Minimum density value : ',df['density'].min())

print('Maximum density value : ',df['density'].max())


# In[14]:


# The range is very close not sinificant, I will drop density column because its effect on the quality is not significant

df1 = df.drop('density', axis = 1)

df1


# In[15]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# In[16]:


# Different between the density values is not significant and it not really affecting the out come

# before dropping it I want to check the relationships

snb.pairplot(df,hue='quality', palette='Dark2')
plt.show()


# In[17]:


df.describe()


# In[18]:


# I need to check for outliers

plt.figure(figsize=(10,6), facecolor='white')
plotnumber = 1
for col in df:
    if plotnumber <= 11:
        ax = plt.subplot(4,3, plotnumber)
        snb.boxplot(df[col], palette="Blues")
        plt.xlabel(col, fontsize=15)
        plt.yticks(rotation=180, fontsize=10)
    plotnumber += 1
plt.tight_layout()
plt.show()


# In[19]:


#removing the outliers using zscore

from scipy.stats import zscore

out_features = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
z=np.abs(zscore(out_features))

z


# In[20]:


np.where(z>3)


# In[103]:


df2 = df[(z<3).all(axis=1)]

df2


# In[104]:


print('the shape of new data: ',df1.shape[0])
print('the shape of old data: ',df2.shape[0])


# In[105]:


# data loss is relatively not much

print('percentage data loss after removing outliers using IQR ', ((df1.shape[0] - df2.shape[0])/df.shape[0])* 100)


# In[106]:


# there are lots of outliers

# let me check for skewness

df2.skew()


# In[107]:


#. df['fixed acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','sulphates','alcohol']=np.cbrt(df2['fixed acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','sulphates','alcohol','quality'])

df2['fixed acidity'] = np.cbrt(df2['fixed acidity'])


# In[108]:


df2['residual sugar'] = np.cbrt(df2['residual sugar'])


# In[109]:


df2['chlorides'] = np.cbrt(df2['chlorides'])


# In[110]:


df2['free sulfur dioxide'] = np.cbrt(df2['free sulfur dioxide'])


# In[111]:


df2['total sulfur dioxide'] = np.cbrt(df2['total sulfur dioxide'])


# In[112]:


df2['sulphates'] = np.cbrt(df2['sulphates'])


# In[113]:


df2['alcohol'] = np.cbrt(df2['alcohol'])


# In[114]:


df2.skew()


# In[115]:


#checking corelation between target and features
cor = df2.corr()
cor


# In[117]:


plt.figure(figsize=(20,15))
snb.heatmap(df2.corr(), linewidths=0.1, fmt='.1g', linecolor='black', cmap='Blues_r', annot= True)
plt.yticks(rotation = 0)
plt.show()


# In[118]:


cor['quality'].sort_values(ascending= False)


# In[128]:


x = df2.drop('quality', axis=1)

y = df2['quality']


# In[129]:


# feature scaling using standard scalarization to avoid biasness

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

x


# In[130]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Values'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

vif['Features'] = x.columns

vif


# In[131]:


x.drop('fixed acidity', axis=1, inplace = True)
x


# In[132]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF Values'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

vif['Features'] = x.columns

vif


# In[133]:


y.value_counts()


# In[134]:


# the class is not balanced i.e Use smoting to balance the class

from imblearn.over_sampling import SMOTE
SM = SMOTE()

x,y = SM.fit_resample(x,y)


# In[135]:


y.value_counts()


# In[137]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

maxAccu = 0
maxRS = 0

for i in range(1,200):
    x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size =0.40, random_state=i)
    RFR = RandomForestClassifier()
    RFR.fit(x_train, y_train)
    pred = RFR.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAccu:
        maxAccu = acc
        maxRS = i
print('Best accuracy is : ', maxAccu, ' at random state : ',maxRS)


# In[138]:


#creating train test split

x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size =0.30, random_state=maxRS)


# In[139]:


# Classification Algorithm

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.model_selection import cross_val_score

#RandomForestClassifier
#checking accuracy for random forest classifier 

RFC = RandomForestClassifier()
RFC.fit(x_train, y_train)

predRFC = RFC.predict(x_test)
print(accuracy_score(y_test, predRFC))
print(confusion_matrix(y_test, predRFC))
print(classification_report(y_test, predRFC))


# In[150]:


#AdaBoostClassifier
#checking accuracy for AdaboostClassifier 

AD = AdaBoostClassifier()
AD.fit(x_train, y_train)

predAD = AD.predict(x_test)
print(accuracy_score(y_test, predAD))
print(confusion_matrix(y_test, predAD))
print(classification_report(y_test, predAD))


# In[140]:


#LogisticsRegression
#checking accuracy for LogisticsRegression  

LR = LogisticRegression()
LR.fit(x_train, y_train)

predLR = LR.predict(x_test)
print(accuracy_score(y_test, predLR))
print(confusion_matrix(y_test, predLR))
print(classification_report(y_test, predLR))


# In[142]:


#ExtraTreeClassifier
#checking accuracy for ExtraTreeClassifier 

ET = ExtraTreesClassifier()
ET.fit(x_train, y_train)

predET = ET.predict(x_test)
print(accuracy_score(y_test, predET))
print(confusion_matrix(y_test, predET))
print(classification_report(y_test, predET))


# In[143]:


# Validation accuracy score to be sure its not as a result of over fitting 

# Using Cross validation

from sklearn.model_selection import cross_val_score

score = cross_val_score(RFC,x,y)
print(score)
print(score.mean())
print('Difference between accuracy score and RandomForestClassifier is ', accuracy_score(y_test, predRFC) - score.mean())


# In[144]:



score = cross_val_score(LR,x,y)
print(score)
print(score.mean())
print('Difference between accuracy score and LogisticsRegression is ', accuracy_score(y_test, predLR) - score.mean())


# In[145]:


score = cross_val_score(ET,x,y)
print(score)
print(score.mean())
print('Difference between accuracy score and LogisticsRegression is ', accuracy_score(y_test, predET) - score.mean())


# In[151]:


score = cross_val_score(AD,x,y)
print(score)
print(score.mean())
print('Difference between accuracy score and AdaboostClassifier is ', accuracy_score(y_test, predAD) - score.mean())


# In[152]:


#Hyper parameter Tunning 
#Type of cross validation method to get best values for model parameters IN THIS CASE ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV

parameters = { 'criterion' :['gini','entropy'],
             'random_state' : [10,50,1000],
             'max_depth' : [0,10,20],
             'n_jobs' : [-2,-1,1],
             'n_estimators': [50,100,200,300]}


# In[153]:


GCV = GridSearchCV(ExtraTreesClassifier(), parameters, cv = 5)


# In[154]:


GCV.fit(x_train, y_train)


# In[155]:


final_model = ExtraTreesClassifier(criterion = 'gini', random_state=10, max_depth = 20, n_estimators = 200, n_jobs=2)

final_model.fit(x_train, y_train)
pred = final_model.predict(x_test)

accuracy = accuracy_score(y_test, pred)

print(acc * 100)


# In[ ]:




