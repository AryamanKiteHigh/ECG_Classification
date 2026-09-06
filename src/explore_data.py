import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
print("Loading MIT-BIH dataset...")
train_df = pd.read_csv("data/mitbih_train.csv", header=None)

# 2. Check basic dimensions
print(f"Dataset Shape: {train_df.shape} (Rows: Heartbeats, Columns: Signal Points + Label)")

# 3. Separate features (signal points) and target labels
X = train_df.iloc[:, :-1].values  # First 187 columns are voltage points
y = train_df.iloc[:, -1].values   # 188th column is the heartbeat class (0 to 4)

# Map numeric labels to standard MIT-BIH arrhythmia classes
class_names = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown / Unclassifiable (Q)"
}

print("\nHeartbeat Distribution:")
label_counts = pd.Series(y).map(class_names).value_counts()
print(label_counts)

# 4. Plot one sample heartbeat from each class
plt.figure(figsize=(14, 8))
for class_id, class_label in class_names.items():
    # Find the index of the first sample belonging to this class
    sample_idx = np.where(y == class_id)[0][0]
    
    plt.subplot(2, 3, class_id + 1)
    plt.plot(X[sample_idx], color='crimson' if class_id != 0 else 'navy')
    plt.title(f"Class {class_id}: {class_label}")
    plt.xlabel("Time Step (Normalized)")
    plt.ylabel("Normalized Voltage")
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()