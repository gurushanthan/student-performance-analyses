import numpy as np
import pandas as pd


data=pd.read_csv('/content/student-por.csv')

import seaborn as sns 
import matplotlib.pyplot as plt #making statistical graphics in Python
b=sns.countplot(data['G3']) #counts the number of students for a grade recieved
b.axes.set_title('Final grade of students',fontsize=30) #title of the plot
b.set_xlabel('Final Grade',fontsize=20) # x axis
b.set_ylabel('Count',fontsize=20)#y axis
plt.show

plt.figure(figsize=(20,25))
sns.heatmap(data.corr(),annot=True)  #darker shades of the chart represent higher values than the lighter shade
plt.show()

data.shape

data.isnull().sum()

male_students=len(data[data['sex']=='M'])
female_students=len(data[data['sex']=='F'])
print('No. of male students',male_students)
print('No. of female students',female_students)

data.head()

data.drop(['school','age','Pstatus'],axis=1,inplace=True)

data.columns

data.head()

data.describe()

data.corr()

encoding


#MAPPING to ensure we have binary data to work
# with our model coz we cannot use string data
# for machine learning
# yes/no values:
d = {'yes':1,'no':0}
data['schoolsup']=data['schoolsup'].map(d)
data['famsup']=data['famsup'].map(d)
data['paid']=data['paid'].map(d)
data['activities']=data['activities'].map(d)
data['nursery']=data['nursery'].map(d)
data['higher']=data['higher'].map(d)
data['internet']=data['internet'].map(d)
data['romantic']=data['romantic'].map(d)

d={'F':1,'M':0}
data['sex']=data['sex'].map(d)

d={'teacher':0,'health':1,'services':2,'at_home':3,'other':4}
data['Mjob']=data['Mjob'].map(d)
data['Fjob']=data['Fjob'].map(d)

d={'home':0,'reputation':1,'course':2,'other':3}
data['reason']=data['reason'].map(d)

d={'mother':0,'father':1,'other':2}
data['guardian']=data['guardian'].map(d)

d={'R':0,'U':1}
data['address']=data['address'].map(d)

#mapping the familysize
d={'LE3':0,'GT3':1}
data['famsize']=data['famsize'].map(d)

data.columns

data.dtypes

from sklearn.model_selection import train_test_split
x=data.drop('G3',axis=1)
y=data['G3']

data['G3']

data.isnull().any()

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
#we have four parts of our data for training and testing



LINEAR REGRESSION


from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)

y_pred11 = regressor1.predict(X_test)
from sklearn.metrics import r2_score
r2 =  r2_score(y_test,y_pred11)
print("r2 score is: ,", r2)

DECISION TREE


from sklearn.tree import DecisionTreeRegressor
regressor2 = DecisionTreeRegressor(criterion='squared_error',random_state = 0)
regressor2.fit(X_train, y_train)

y_pred22=regressor2.predict(X_test)
np.set_printoptions(precision=2)


from sklearn.metrics import r2_score,accuracy_score,confusion_matrix
r2=r2_score(y_test,y_pred22)
print(r2)

cm=confusion_matrix(y_test,y_pred22)

RIDGE


from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()

reg1.fit(X_train, y_train)

y_pred222 = reg1.predict(X_test)
r2_score(y_test, y_pred222)

from sklearn.linear_model import Ridge
ridge=Ridge(alpha=1)
ridge.fit(X_train,y_train)

y_pred1 = ridge.predict(X_test)
r2_score(y_test, y_pred1)

LASSO

from sklearn.linear_model import Lasso
lasso=Lasso(alpha=1)
lasso.fit(X_train,y_train)

y_pred2 = lasso.predict(X_test)
r2_score(y_test, y_pred1) 

from sklearn.linear_model import LinearRegression
regressor77 = LinearRegression()
regressor77.fit(X_train, y_train)

y_pred77 = regressor77.predict(X_test)
from sklearn.metrics import r2_score
r2 =  r2_score(y_test,y_pred77)
print("r2 score is: ,", r2)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
cf = classifier.fit(X_train,y_train)
y_pred0= cf.predict(X_test)
print(y_pred)

r2=r2_score(y_test,y_pred0)

print(r2)
