# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.

2.Load the dataset into a Pandas DataFrame and preview it using head() and tail().

3.Extract the independent variable X and dependent variable Y from the dataset.

4.Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.

5.In a loop over the number of epochs:

.Compute the predicted value Y_pred using the formula . Calculate the gradients

.Update the parameters m and c using the gradients and learning rate.

.Track and store the error in each epoch.

6.Plot the error against the number of epochs to visualize the convergence.

7.Display the final values of m and c, and the error plot.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRAVEENA M
RegisterNumber:  212223040153
*/
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

```

## Output:
![image](https://github.com/user-attachments/assets/dcd8eea5-a75d-4459-a225-59a2b6f980b7)


```
dataset.info()
```

## Output:
![image](https://github.com/user-attachments/assets/1b2e1569-8d74-486b-8573-9b3f4199f88b)

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

```
## Output:

![image](https://github.com/user-attachments/assets/d6854e83-9e8a-41d3-be5a-060f6b8d7091)

```

print(X.shape)
print(Y.shape)
```

## Output:
![image](https://github.com/user-attachments/assets/99198cbd-12e2-4812-935f-57941403350b)


```
m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
```


## Output:
![image](https://github.com/user-attachments/assets/2973645f-5d8c-4b7f-a92d-8a956e2a6f26)

```
plt.plot(range(0,epochs),error)
```


## Output:
![image](https://github.com/user-attachments/assets/89c89751-0be5-4140-b4b2-a18d18f47277)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
