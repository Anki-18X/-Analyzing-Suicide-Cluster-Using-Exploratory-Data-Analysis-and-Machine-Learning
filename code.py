import pandas as pd
import numpy as np
df=pd.read_csv(r"/content/new ML.csv")

df.shape

df.info()

df.head(10)

df.tail(10)

df.columns

df.country.value_counts()

df.year.value_counts()

df.sex.value_counts()

df.age.value_counts()

country=df.country.unique()
print(country)
print(len(country))

age=df.age.unique()
print(age)
print(len(age))

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

plt.figure(figsize=(10,3))
sns.barplot(x = 'suicides_no',
            y = 'sex',
            data = df)
plt.title('Gender - Suicide Count')
plt.show()

from google.colab import drive
drive.mount('/content/drive')



plt.figure(figsize=(10,3))
sns.barplot(x = 'age',
            y = 'suicides_no',
            data = df,)
plt.title('Age - Suicide Count')
plt.show()

df.describe()

df.isnull().sum()

df.isnull().sum().sum()

df.head(10)

mean=df.mean()
df.fillna(mean,inplace=True)
df.head(10)

from sklearn.preprocessing import LabelEncoder
df=df.apply(LabelEncoder().fit_transform)

features=list(df)
features

x=df[features]
y=df.suicides_no
y.head()

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)
train_x.shape, test_y.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import time

df[list(df)].apply(lambda y: y.astype("category"))

from sklearn.preprocessing import LabelEncoder
df=df.apply(LabelEncoder().fit_transform)

gini_clf=DecisionTreeClassifier(criterion="gini",random_state=7)
start=time.time()
gini_clf=gini_clf.fit(train_x,train_y)
stop=time.time()
print(f"training time: {stop - start}s")

pred_y=gini_clf.predict(test_x)
print("accuracy of training data is : " ,metrics.accuracy_score(test_y,pred_y))

ent_clf=DecisionTreeClassifier(criterion="entropy",random_state=7)
start=time.time()
ent_clf=ent_clf.fit(train_x,train_y)
stop=time.time()
print(f"training time: {stop - start}s")

pred_y=ent_clf.predict(test_x)
print("accuracy of training data is : " ,metrics.accuracy_score(test_y,pred_y))

from sklearn.linear_model import LinearRegression
#importing required libraries
from sklearn.metrics import mean_squared_error
# instantiate the model
lr = LinearRegression()
# fit the model
lr.fit(train_x, train_y)
#predicting the target value from the model for the samples
test_y_lr = lr.predict(test_x)
train_y_lr = lr.predict(train_x)
#computing the accuracy of the model performance
acc_train_lr = lr.score(train_x, train_y)
acc_test_lr = lr.score(test_x, test_y)

#computing root mean squared error (RMSE)
rmse_train_lr = np.sqrt(mean_squared_error(train_y, train_y_lr))
rmse_test_lr = np.sqrt(mean_squared_error(test_y, test_y_lr))

print("Linear Regression: Accuracy on training Data: {:.3f}".format(acc_train_lr))
print("Linear Regression: Accuracy on test Data: {:.3f}".format(acc_test_lr))
print('\nLinear Regression: The RMSE of the training set is:', rmse_train_lr)
print('Linear Regression: The RMSE of the testing set is:', rmse_test_lr)

rfc=RandomForestClassifier(n_estimators=500,random_state=7)
start=time.time()
rfc=rfc.fit(train_x,train_y)
stop=time.time()
print(f"training time: {stop - start}s")

from sklearn.preprocessing import StandardScaler
import numpy as np


# the scaler object (model)
scaler = StandardScaler()
# fit and transform the data
scaled_data = scaler.fit_transform(train_x)
print()
print(scaled_data)
