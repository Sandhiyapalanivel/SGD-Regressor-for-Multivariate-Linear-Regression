# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step-1.Start
Step-2.Import necessary libraries
Step-3.Load and preprocess the data (define features and target).
Step-4.Split the dataset into training and testing sets.
Step-5.Scale the features using StandardScaler.
Step-6.Train the SGDRegressor model on the training set.
Step-7.Evaluate the model on both training and testing sets using MSE or other metrics.
Step-8.End 
```
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SANDHIYA P
RegisterNumber:  212223230183
*/
import numpy as np 
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X= data.data[:, :3]
Y = np.column_stack((data.target, data.data[:, 6]))
x_train, x_test, Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state =42)
scaler_X=StandardScaler()
scaler_Y= StandardScaler()
x_train = scaler_X.fit_transform(x_train)
x_test = scaler_X.fit_transform(x_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.fit_transform(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, Y_train)
y_pred =multi_output_sgd.predict(x_test)
y_pred = scaler_Y.inverse_transform(y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
print(y_pred)
mse = mean_squared_error(Y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
```
## Output:
```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/5bbe4bf7-0584-46dc-86a7-66b78f5913df)

```
Mean squared error
```

![image](https://github.com/user-attachments/assets/9c066eb3-f5d4-464a-aa90-4ae299689bf7)

```
Predictions:
```

![image](https://github.com/user-attachments/assets/c2eaac82-ef46-4c76-9ec0-a0fcf23a3c2a)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
