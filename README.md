# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset, then preprocess the data by removing unnecessary columns and converting categorical data into numerical form.


2.Split the dataset into training and testing sets and separate features (X) and target variable (y).


3.Train the Linear Regression model (and Polynomial Regression by transforming features if required) using the training data.


4.Predict car prices using the test data and evaluate the model using metrics like MSE, R², and MAE.


## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data= pd.read_csv('CarPrice_Assignment (1).csv')
data.head()

data = data.drop(['car_ID','CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
data.head()

X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print('Name: KRITHIKAA P ')
print('Reg. No: 212225040193')
print("\n== Cross-Validation ==")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R^2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R^2:{cv_scores.mean():.4f}")

y_pred =model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):>10.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()

## Output:
![WhatsApp Image 2026-03-28 at 4 24 28 PM](https://github.com/user-attachments/assets/1bb427a1-2478-4963-9a84-d60b46fa6574)


![WhatsApp Image 2026-03-28 at 4 25 03 PM](https://github.com/user-attachments/assets/a3a4c900-3533-458f-a010-ad5f860984c9)


![WhatsApp Image 2026-03-28 at 4 25 28 PM](https://github.com/user-attachments/assets/d365add3-5037-4946-a923-4e9cd0cc20fe)


![WhatsApp Image 2026-03-28 at 4 25 55 PM](https://github.com/user-attachments/assets/da82591a-02aa-4bd8-ab7e-58607513b8ec)


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
