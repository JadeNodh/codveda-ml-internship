# Import necessary libraries for data preprocessing
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load Dataset
df_file_path = './datasets/house_Prediction_Data_Set.csv'  # Update this path to your dataset
df = pd.read_csv(df_file_path)

print("Original Data:\n", df.head()) # Display the first few rows of the dataset

# Handle Missing Values
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill numerical missing values with the mean
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].mean(), inplace=True)
    
# Fill categorical missing values with the mode
for col in df.select_dtypes(include=[object]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
# Encode Categorical Variables
label_encoders = {}

for col in df.select_dtypes(include=[object]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the label encoder for future use
    
# Feature Scaling
scaler = StandardScaler()

numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split the dataset into features and target variable
# Assume last column is the target variable
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nPreprocessed Data:\n", df.head()) # Display the first few rows of the preprocessed dataset

print("\nTraining Features Shape:", X_train.shape)  
print("Testing Features Shape:", X_test.shape)


