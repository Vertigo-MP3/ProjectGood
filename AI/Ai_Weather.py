import pandas as pd
import kagglehub

# Load the dataset
df = pd.read_csv('WeatherHistory.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)


# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Fill missing values using forward fill method
df.fillna(method='ffill', inplace=True)

from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
le = LabelEncoder()

# Encode the 'weather_condition' column
df['weather_condition'] = le.fit_transform(df['weather_condition'])

# Extract year, month, and day from the 'date' column
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Select relevant features for prediction
features = ['temperature', 'humidity', 'wind_speed', 'year', 'month', 'day']
target = 'weather_condition'

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = df[features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

from sklearn.model_selection import train_test_split

# Define features and target variable
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the weather condition for the test set
y_pred = model.predict(X_test)

# Convert numeric predictions back to original labels
predicted_conditions = le.inverse_transform(y_pred)

# Display the predictions
print(predicted_conditions)

from sklearn.metrics import accuracy_score, classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display classification report
print(classification_report(y_test, y_pred))

# Example features for tomorrow
tomorrow_features = [[22.5, 65, 15, 2025, 4, 15]]  # [temperature, humidity, wind_speed, year, month, day]

# Predict the weather condition
tomorrow_pred = model.predict(tomorrow_features)

# Convert numeric prediction back to original label
tomorrow_condition = le.inverse_transform(tomorrow_pred)

# Display the prediction
print(f'Tomorrow\'s weather condition: {tomorrow_condition[0]}')