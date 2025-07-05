import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def getData():
    data = pd.read_csv('data/aus_car_prices.csv', sep =',')

    data.duplicated().sum()
    data.drop_duplicates(inplace=True)

    data = data.drop(['ColourExtInt','Car/Suv','Model','Title'], axis = 1)
    data.replace(['-','POA','- / -'],np.nan,inplace=True)
    data.isnull().sum()

    data['engine'] = data['engine'].str.split(',').str[1].str.split().str[0]
    data['fuelConsumption'] = data['fuelConsumption'].str.split('/').str[0].str.split().str[0]
    data['cylinders'] = data['cylinders'].str.split().str[0]
    data['doors'] = data['doors'].str.split().str[0]
    data['seats'] = data['seats'].str.split().str[0]

    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['engine'] = pd.to_numeric(data['engine'], errors='coerce')
    data['fuelConsumption'] = pd.to_numeric(data['fuelConsumption'], errors='coerce')
    data['kilometres'] = pd.to_numeric(data['kilometres'], errors='coerce')
    data['cylinders'] = pd.to_numeric(data['cylinders'], errors='coerce')
    data['doors'] = pd.to_numeric(data['doors'], errors='coerce')
    data['seats'] = pd.to_numeric(data['seats'], errors='coerce')
    data['price'] = pd.to_numeric(data['price'], errors='coerce')


    data = pd.get_dummies(data, columns=['UsedOrNew', 'Transmission'], drop_first=True, dtype = int)

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop('price')
    categorical_columns = data.select_dtypes(include=['object']).columns

    def drop_rare_categories(data, column, threshold=10):
        value_counts = data[column].value_counts()
        rare_categories = value_counts[value_counts<threshold].index
        data_filtered = data[~data[column].isin(rare_categories)]
        return data_filtered

    for col in ['Brand', 'FuelType', 'Location', 'BodyType']:
        data = drop_rare_categories(data, column = col)

    data['carAge'] = 2025 - data['year']
    data.drop('year', axis=1, inplace=True)

    data = data.drop(['Brand', 'DriveType', 'FuelType', 'Location', 'BodyType'], axis = 1)
    data.dropna(inplace=True)

    # plt.figure(figsize=(12, 6))
    # sns.scatterplot(x=data['year'], y=data['price'])
    # plt.title('Price vs. Year')
    # plt.xlabel('Year')
    # plt.ylabel('Price')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.histplot(data['Price'], kde=True, bins=50, color='skyblue')
    # plt.title("Distribution of car prices Before Removing Outliers")
    # plt.xlabel("Price")
    # plt.ylabel("N_Cars")
    # plt.grid(True)
    # plt.show()

    # sns.boxplot(data['Price'])
    # plt.title('Box Plot of Price')
    # plt.xlabel('Price')
    # plt.show()

    Q1 = data['price'].quantile(0.25)
    Q3 = data['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data['price'] < lower_bound) | (data['price'] > upper_bound)]
    outlier_percentage = (len(outliers) / len(data)) * 100
    data = data[(data['price'] >= lower_bound) & (data['price'] <= upper_bound)]
    
    # sns.boxplot(data['price'])
    # plt.title('Box Plot of Price (Data without extreme outliers)')
    # plt.xlabel('price')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.histplot(data['Price'], kde=True, bins=50, color='skyblue')
    # plt.title("Distribution of car prices (Data without extreme outliers)")
    # plt.xlabel("Price")
    # plt.ylabel("N_Cars")
    # plt.grid(True)
    # plt.show()

    return data

