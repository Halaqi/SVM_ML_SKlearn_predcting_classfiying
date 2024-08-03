import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.datasets import load_iris

# loading data
data = pd.read_csv('heart.csv')
data2 = load_iris()

# heart dataset x and y
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# iris dataset x and y
X2 = data2.data
y2 = data2.target

# checking the missing values
missing_values = SimpleImputer(missing_values = np.nan, strategy='mean')
data = missing_values.fit(data)

# heart dataset scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# iris dataset scaling
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(X2)

# heart dataset spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# heart dataset spliting
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.20, random_state=0)

# calling SVC class for both datasets
svc_model = SVC(kernel='rbf', C=2, random_state=0)
svc_model.fit(X_train, y_train)

svc_model2 = SVC(kernel='rbf', random_state=10)
svc_model2.fit(X_train2, y_train2)

# printing the predicted values for heart dataset
y_pred = svc_model.predict(X_test)
print("the actual y is:    ",list(y_test[:10]))
print("the predicted y is: ",list(y_pred[:10]))

# printing the predicted values for iris dataset
y_pred2 = svc_model2.predict(X_test2)
print("the iris actual y is:    ", y_test2[:10])
print("the iris predicted y is: ", y_pred2[:10])

# printiing the accuracy for heart dataset 
score_train = svc_model.score(X_train, y_train)
print("the heart training accuracy is: ", score_train*100 ,'%')

score_test = svc_model.score(X_test, y_test)
print("the heart testing accuracy is: ", score_test*100 ,'%')

# printiing the accuracy for iris dataset
score_train2 = svc_model2.score(X_train2, y_train2)
print("the iris training accuracy is: ", score_train2*100 ,'%')

score_test2 = svc_model2.score(X_test2, y_test2)
print("the irsi testing accuracy is: ", score_test2*100 ,'%')


# Making the Confusion Matrix for heart dataset
cm = confusion_matrix(y_test, y_pred)
print('the error cost between y test and y pred is:\n',cm)

# Making the Confusion Matrix for iris datasets
cm2 = confusion_matrix(y_test2, y_pred2)
print('the error cost between y test and y pred is:\n',cm2)

# ploting the confusion matrix for heart dataset
sns.heatmap(cm, center=True)
plt.show()

# ploting the confusion matrix for iris dataset
sns.heatmap(cm2, center=True)
plt.show()

