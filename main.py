import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("restaurant_data.csv")

# Drop irrelevant columns
df.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose',
         'Longitude', 'Latitude', 'Switch to order menu', 'Rating color', 'Rating text'], axis=1, inplace=True)

# Drop rows with missing 'Cuisines'
df = df.dropna(subset=['Cuisines'])

# Label Encoding for binary categorical columns
binary_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Label Encoding for remaining categorical columns
label_enc_cols = ['City', 'Currency', 'Cuisines']
le = LabelEncoder()
for col in label_enc_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(['Aggregate rating'], axis=1)
y = df['Aggregate rating']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Preprocessing complete!")
print("🧠 Features shape:", X_train.shape)
print("🎯 Target shape:", y_train.shape)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 2️⃣ Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 📊 Evaluation Function
def evaluate_model(name, y_test, y_pred):
    print(f"\n📈 {name} Performance:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

# Evaluate both models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree Regressor", y_test, y_pred_dt)

import matplotlib.pyplot as plt
import seaborn as sns

# Feature importances from Decision Tree
importances = dt.feature_importances_
feature_names = X.columns

# Create a dataframe
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('🔍 Feature Importance for Restaurant Rating Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
