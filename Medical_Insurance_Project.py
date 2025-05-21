import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import models from scikit-learn module:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Column names and data loading
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
try:
    df = pd.read_csv('adult.data', header=None, names=col_names)
except FileNotFoundError:
    print("Error: 'adult.data' file not found. Please ensure the file is in the correct directory.")
    exit()

# Strip whitespace in object-type columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Create feature dataframe X and dummy variables
X = pd.get_dummies(df.drop('income', axis=1))
y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Instantiate and score baseline model
rf = RandomForestClassifier(random_state=1)
rf.fit(x_train, y_train)
print("Baseline Accuracy:", rf.score(x_test, y_test))

# Tune max_depth
np.random.seed(0)
accuracy_train = []
accuracy_test = []

depths = range(1, 26)
for depth in depths:
    model = RandomForestClassifier(max_depth=depth, random_state=1)
    model.fit(x_train, y_train)
    accuracy_train.append(model.score(x_train, y_train))
    accuracy_test.append(model.score(x_test, y_test))

# Find best depth
best_depth = depths[np.argmax(accuracy_test)]
best_accuracy = max(accuracy_test)
print("Best max_depth:", best_depth)
print("Best test accuracy:", best_accuracy)

# Plot accuracy over depth
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracy_train, label='Train Accuracy', marker='o')
plt.plot(depths, accuracy_test, label='Test Accuracy', marker='o')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Random Forest Accuracy by Tree Depth')
plt.grid()
plt.show()

# Save best model and feature importances
best_rf = RandomForestClassifier(max_depth=best_depth, random_state=1)
best_rf.fit(x_train, y_train)
importances = pd.DataFrame({'feature': X.columns, 'importance': best_rf.feature_importances_})
importances = importances.sort_values(by='importance', ascending=False)
print("Top 5 Feature Importances:")
print(importances.head())

# Add education_bin and native_country_bin, then rerun
df['education_bin'] = df['education'].apply(lambda x: 'Bachelors+Higher' if x in ['Bachelors', 'Masters', 'Doctorate'] else 'Lower')
df['native_country_bin'] = df['native-country'].apply(lambda x: 'US' if x == 'United-States' else 'Non-US')

# Select new features
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education_bin', 'native_country_bin']
df_encoded = pd.get_dummies(df[feature_cols])
y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Split data with new features
x_train, x_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=1)

# Retrain with new features
accuracy_train = []
accuracy_test = []
for depth in depths:
    model = RandomForestClassifier(max_depth=depth, random_state=1)
    model.fit(x_train, y_train)
    accuracy_train.append(model.score(x_train, y_train))
    accuracy_test.append(model.score(x_test, y_test))

# Find best depth for new features
best_depth = depths[np.argmax(accuracy_test)]
print("New best max_depth:", best_depth)

# Train best model with new features
best_rf = RandomForestClassifier(max_depth=best_depth, random_state=1)
best_rf.fit(x_train, y_train)
importances = pd.DataFrame({'feature': x_train.columns, 'importance': best_rf.feature_importances_})
importances = importances.sort_values(by='importance', ascending=False)
print("Top 5 Feature Importances with New Features:")
print(importances.head())