import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)


# Loading data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

print(df.head())
print(df.describe())
print(df.shape)

# Exploratory data analysis 
print(df.isnull().sum())  # check for missing values
print("\nCorrelation with Price:")
print(df.corr()['Price'].sort_values(ascending=False))


# Splitting the data
X = df[['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


# Fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {list(zip(X.columns, model.coef_))}")



# Predictions and Evaluation
y_pred = model.predict(X_test)

intercept = model.intercept_
coefficient = model.coef_
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Model Interpretation
print('Intercept:', intercept)
print('Coefficient:', coefficient)
print('Mean Squared Error:', mse)
print('R² Score:', r2)


## Visualizations ##

# Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Housing Prices")
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.3, color='coral')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()