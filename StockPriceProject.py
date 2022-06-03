import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

credit_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\MLDATA\tesla_data.csv")

#print(credit_data.describe())
#print(credit_data.corr())
#print(credit_data)

features = credit_data.iloc[:, 0:5]
target = credit_data.iloc[:, 5:6]

print(features)
print(target)

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

target_train = np.ravel(target_train)
target_test = np.ravel(target_test)

model = LinearRegression()
model.fit(feature_train, target_train)

predictions = model.predict(feature_test)

predictions=np.array(predictions).reshape(-1,1)

print("Mean Squared Error is:", mean_squared_error(predictions,target_test))
print("R squared statistic:", model.score(feature_train,target_train))

print("Weights: ", model.coef_)
print("Intercept:", model.intercept_)
# print(predictions['test_score'])
