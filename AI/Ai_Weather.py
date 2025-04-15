# Install these with pip if you haven't
import pandas as pd         # For handling data
import numpy as np          # For numerical operations
from sklearn.linear_model import LinearRegression  # ML model
from sklearn.model_selection import train_test_split
import tkinter as tk        # For GUI interaction with cursor


print(pd.__version__)



# Load mock dataset
data = pd.read_csv('weather_data.csv')

# Features (X): today's temp, humidity, wind speed
X = data[['Temp', 'Humidity', 'Wind']]

# Target (y): tomorrow's temperature
y = data['TomorrowTemp']

#Train the AI Model
model = LinearRegression()

if len(X) > 1:
    # Split data if we have more than one point
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully with multiple data points")
else:
    # Use all data for training if we only have one point
    model.fit(X, y)
    print("Model trained with single data point")

# Print model coefficients
print("\nModel coefficients:")
print(f"Temperature coefficient: {model.coef_[0]:.2f}")
print(f"Humidity coefficient: {model.coef_[1]:.2f}")
print(f"Wind coefficient: {model.coef_[2]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")