import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/customer_churn_business_dataset.csv")

# Remove ID column
if "customer_id" in df.columns:
    df = df.drop("customer_id", axis=1)

# Remove missing values
df = df.dropna()

# Features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# Convert categorical variables
X = pd.get_dummies(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SMOTE balancing
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)

# Predict probabilities
nn_probs = nn_model.predict_proba(X_test_scaled)[:, 1]

# Same threshold as Random Forest
nn_predictions = (nn_probs > 0.25).astype(int)

print("\n===== Neural Network Results =====")

print("Accuracy:", accuracy_score(y_test, nn_predictions))

print("\nClassification Report:")
print(classification_report(y_test, nn_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nn_predictions))