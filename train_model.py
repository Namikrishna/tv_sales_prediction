import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Ensure the 'model' directory exists
os.makedirs("model", exist_ok=True)

# Load the dataset
try:
    dataset = pd.read_csv('tv_sales_dataset.csv')
except FileNotFoundError:
    print("Error: 'tv_sales_dataset.csv' not found in the current directory.")
    exit()

# Print column names for debugging
print("Columns in the dataset:", dataset.columns.tolist())

# Rename columns
expected_columns = ['TV  ($)', 'Sales ($)']
if all(col in dataset.columns for col in expected_columns):
    dataset = dataset.rename(columns={'TV  ($)': 'TV', 'Sales ($)': 'Sales'})
else:
    print(f"Error: Expected columns {expected_columns} not found in {dataset.columns.tolist()}")
    exit()

# Select columns
try:
    dataset = dataset[['TV', 'Sales']]
except KeyError as e:
    print(f"Error: {e}. Available columns: {dataset.columns.tolist()}")
    exit()

# Handle missing values and data types
dataset = dataset.dropna()
dataset['TV'] = pd.to_numeric(dataset['TV'], errors='coerce')
dataset['Sales'] = pd.to_numeric(dataset['Sales'], errors='coerce')
dataset = dataset.dropna()

# Remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Outliers in {column}: Lower bound = {lower_bound}, Upper bound = {upper_bound}")
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

dataset = remove_outliers(dataset, 'TV')
dataset = remove_outliers(dataset, 'Sales')

# Prepare data
X = dataset[['TV']].values
y = dataset['Sales'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Train model
regressor = LinearRegression()
regressor.fit(x_train_scaled, y_train)

# Save model and scaler
model_path = "model/tv_sales_model.pkl"
scaler_path = "model/scaler.pkl"

with open(model_path, "wb") as file:
    pickle.dump(regressor, file)
with open(scaler_path, "wb") as file:
    pickle.dump(scaler, file)

# Verify files were saved
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Model and scaler saved successfully to 'model/' folder:")
    print(f"- {model_path}")
    print(f"- {scaler_path}")
else:
    print("Error: Failed to save model or scaler files.")