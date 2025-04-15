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

# Split into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#Train the AI Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")