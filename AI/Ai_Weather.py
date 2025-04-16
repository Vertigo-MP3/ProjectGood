# These are comments - they help explain the code but don't do anything
# Install these with pip if you haven't

# Import necessary tools (like getting tools from a toolbox)
import pandas as pd         # pd is a nickname for pandas - helps with data handling
import numpy as np          # np is a nickname for numpy - helps with numbers
from sklearn.linear_model import LinearRegression  # This is our weather prediction tool
from sklearn.model_selection import train_test_split  # Helps split our data
import tkinter as tk        # Helps create the window/buttons for our program

# Print the version of pandas we're using
print(pd.__version__)

# Load our weather data from the CSV file (like opening a spreadsheet)
data = pd.read_csv('weather_data.csv')

# Prepare our data for the model
# X is what we use to make predictions (today's weather)
X = data[['Temp', 'Humidity', 'Wind']]
# y is what we want to predict (tomorrow's temperature)
y = data['TomorrowTemp']

# Create our weather prediction model
model = LinearRegression()

# Check if we have enough data to split
if len(X) > 1:
    # If we have enough data, split it into training and testing parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Teach the model using the training data
    model.fit(X_train, y_train)
    print("Model trained successfully with multiple data points")
else:
    # If we only have one data point, use all of it for training
    model.fit(X, y)
    print("Model trained with single data point")

# Show how the model makes its predictions
print("\nModel coefficients:")
print(f"Temperature coefficient: {model.coef_[0]:.2f}")  # How much temperature affects tomorrow
print(f"Humidity coefficient: {model.coef_[1]:.2f}")    # How much humidity affects tomorrow
print(f"Wind coefficient: {model.coef_[2]:.2f}")        # How much wind affects tomorrow
print(f"Intercept: {model.intercept_:.2f}")             # Base temperature prediction

# This function runs when you click the Predict button
def predict_weather():
    # Get the numbers you type into the boxes
    temp = float(entry_temp.get())      # Get temperature
    humidity = float(entry_humidity.get())  # Get humidity
    wind = float(entry_wind.get())      # Get wind speed
    
    # Use the model to predict tomorrow's temperature
    prediction = model.predict([[temp, humidity, wind]])
    
    # Show the prediction on the screen
    result_label.config(text=f"Predicted Temp Tomorrow: {prediction[0]:.2f}°C")

# Create the window for our program
root = tk.Tk()
root.title("AI Weather Predictor")  # Give the window a title

# Create labels and input boxes
tk.Label(root, text="Today's Temp (°C)").grid(row=0)  # Label for temperature
tk.Label(root, text="Humidity (%)").grid(row=1)       # Label for humidity
tk.Label(root, text="Wind Speed (km/h)").grid(row=2)  # Label for wind speed

# Create boxes where you can type numbers
entry_temp = tk.Entry(root)      # Box for temperature
entry_humidity = tk.Entry(root)  # Box for humidity
entry_wind = tk.Entry(root)      # Box for wind speed

# Put the boxes in the right places
entry_temp.grid(row=0, column=1)
entry_humidity.grid(row=1, column=1)
entry_wind.grid(row=2, column=1)

# Create the Predict button
tk.Button(root, text='Predict', command=predict_weather).grid(row=3, column=0, pady=4)

# Create a label where the prediction will appear
result_label = tk.Label(root, text="Prediction will appear here")
result_label.grid(row=4, columnspan=2)

# Start the program and keep it running
root.mainloop()