import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("1. Loading training and testing datasets...")
train_df = pd.read_csv("data/mitbih_train.csv", header=None)
test_df = pd.read_csv("data/mitbih_test.csv", header=None)

# Separate features (columns 0-186) and targets (column 187)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

print("\n2. Training Random Forest Classifier (this may take 1-2 minutes)...")
# Using 100 trees with multi-threading across all CPU cores (n_jobs=-1)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n3. Evaluating model on unseen test data...")
y_pred = model.predict(X_test)

# Print overall accuracy and detailed metrics per heartbeat class
acc = accuracy_score(y_test, y_pred)
print(f"\n--- Model Results ---")
print(f"Overall Accuracy: {acc * 100:.2f}%\n")

class_names = ["Normal (0)", "Supraventricular (1)", "Ventricular (2)", "Fusion (3)", "Unknown (4)"]
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model to disk so we can use it in our web application later
joblib.dump(model, "src/ecg_random_forest.pkl")
print("\n4. Model successfully saved as 'src/ecg_random_forest.pkl'!")