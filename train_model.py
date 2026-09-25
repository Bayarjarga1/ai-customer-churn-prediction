import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/customer_churn_business_dataset.csv")

# Drop ID column
if "customer_id" in df.columns:
    df = df.drop("customer_id", axis=1)

# Remove missing values
df = df.dropna()

# Separate features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# Convert categorical columns into numeric columns
X = pd.get_dummies(X)

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Apply SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
probs = model.predict_proba(X_test)[:, 1]
predictions = (probs > 0.25).astype(int)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and columns
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")

import matplotlib.pyplot as plt

# Get feature importance scores
importance = model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort descending
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Top 10 features
top10 = feature_importance.head(10)

print("\nTop 10 Features:")
print(top10)

# Plot
plt.figure(figsize=(10,6))
plt.barh(top10["Feature"], top10["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Influencing Customer Churn")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("static/images/feature_importance.png")

plt.show()

print("\nModel saved successfully!")