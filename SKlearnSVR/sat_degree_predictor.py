from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
import pandas as pd
import numpy as np

# loading the data
data = pd.read_csv('satf.csv')

# preparing the x and y axises 
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# scaling the data between 1 and -1
scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)

# spliting the data to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=40)

# calling the the class and make object to use the SVR model
svr_model = SVR(kernel='rbf',epsilon=0.1)
svr_model.fit(X_train,y_train)

# printing the accuracy for training and testing data
score_train = svr_model.score(X_train, y_train)
print('accuracy of train data is: ',score_train*100,'%')

score_test = svr_model.score(X_test, y_test)
print('accuracy of test data is: ',score_test*100,'%')

# showing the actual data and the predicted one
print('actual y_test data is:    ',list(y_test[:10]))

y_pred = svr_model.predict(X_test)
print('predicted y_test data is: ',list(np.round(y_pred[:10],2)))

# showing the error between the actaul and the predicted values
absolute_error = mean_absolute_error( y_test,y_pred)
print(' mean absolute error is: ',absolute_error)
squared_error = mean_squared_error(y_test, y_pred)
print(' mean squared error is: ',squared_error)
med_absolute_error =  median_absolute_error(y_test,y_pred)
print('median_absolute_error is: ',med_absolute_error)





