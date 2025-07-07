# 🚗 Australian Car Price Prediction with Machine Learning
A machine learning project that predicts used car prices in Australia using structured listing data. Built with Python, this project explores how features like engine size, car age, odometer reading, and transmission type impact resale value. Linear regression is used as a baseline model to estimate car prices based on these technical and categorical features.

# 🧠 Overview
This regression-based machine learning project aims to forecast the selling price of used cars in the Australian market. The goal is to understand the influence of various car attributes on price and evaluate the model's ability to generalise to unseen data. The project also serves as a platform for experimenting with preprocessing, feature engineering, and model evaluation techniques.

# 📂 Dataset
The dataset (aus_car_prices.csv) contains:

Thousands of used car listings from Australian marketplaces

Features include:
Make, Year, Kilometres, Engine, Fuel Consumption, Transmission, Doors, Seats, UsedOrNew, DriveType, FuelType, BodyType, Location, and Price

# 🛠️ Data Preprocessing
Removed duplicate and irrelevant columns

Cleaned and converted fields (e.g., extracted numeric engine size, fuel consumption, number of doors/seats/cylinders)

Converted object fields to numeric types

Encoded selected categorical features (UsedOrNew, Transmission)

Removed rare categories from fields like Brand, FuelType, Location, and BodyType

Engineered a carAge feature from the vehicle's year

Removed outliers in price using the IQR method

# 📈 Features
The model uses the following cleaned and engineered features:<br>
Engine size<br>
Fuel consumption<br>
Kilometres travelled<br>
Cylinders<br>
Doors<br>
Seats<br>
Car age<br>
Encoded values for transmission (automatic or manuel) and used or new <br>

Target variable:<br>
Car price rice (in AUD)

# 🤖 Model & Performance
Baseline model: Linear Regression

**Training Performance**

R²: ~0.6

MAE: ~7476 AUD

RMSE: ~9774 AUD

**Test Performance**

R²: ~0.62

MAE: ~7688 AUD

RMSE: ~9972 AUD

These results show moderate performance, with similar metrics across training and test sets. While the model captures some trends in pricing, high RMSE values suggest there is still significant variance in prediction accuracy—likely due to the complexity and non-linearity of real-world pricing patterns.

# ⚠️ Challenges
Over-simplification from using linear regression on complex, non-linear data

Imbalanced categories (e.g., rare brands or models)

Feature correlations and data noise

# 🔧 Future Improvements <br>
✅ Test with non-linear models like XGBoost, Random Forest, or LightGBM <br> 
✅ Apply feature scaling and use pipelines for better modularity <br>
✅ Explore time-aware validation or grouped cross-validation by brand <br>
✅ Build a web interface to allow users to input car details and get a price estimate
