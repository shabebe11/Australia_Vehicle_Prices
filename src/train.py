import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score, root_mean_squared_error
from getData import getData



data = getData()


X = np.array(data.drop(['price'], axis = 1))
Y = np.array(data['price'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(X_train, Y_train)
acc = linear.score(X_test, Y_test)
# print(acc)

train_pred = linear.predict(X_train)
test_pred = linear.predict(X_test)

print("\nTraining Performance:")
print(f"R²: {r2_score(Y_train, train_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_train, train_pred):.6f}")
print(f"RMSE: {root_mean_squared_error(Y_train, train_pred):.6f}")

print("\nTest Performance:")
print(f"R²: {r2_score(Y_test, test_pred):.4f}")
print(f"MAE: {mean_absolute_error(Y_test, test_pred):.6f}")
print(f"RMSE: {root_mean_squared_error(Y_test, test_pred):.6f}") 

# Uncomment to plot results
plt.figure(figsize=(14,7))
plt.plot(range(len(Y_test)), Y_test, label='Actual Returns')
plt.plot(range(len(Y_test)), test_pred, label='Predicted Returns', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Returns')
plt.show()
