🌲 Random Forest – ML Notes
📘 What is Random Forest?
Random Forest is a powerful ensemble learning method used for classification and regression tasks.

It works by building multiple decision trees and combining their outputs (via majority vote for classification or averaging for regression).

🔍 Why Random Forest?
✅ Reduces Overfitting (compared to a single decision tree)

✅ Handles missing values and unbalanced data

✅ Works well on large datasets

✅ Provides Feature Importance

🧠 How It Works (Simplified):
Bootstrap Sampling: Take random samples from the dataset with replacement.

Train multiple decision trees on these samples.

Random Feature Selection: At each tree split, use a random subset of features.

Voting:

Classification → Majority vote

Regression → Average prediction

📊 Real-World Applications:
Fraud detection

Medical diagnosis

Customer churn prediction

Credit scoring

🧪 Code Example: Random Forest on Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(acc * 100, 2), "%")
print("Confusion Matrix:\n", cm)

📈 Sample Output:
Accuracy: 93.33 %
Confusion Matrix:
[[16  0  0]
 [ 0 14  2]
 [ 0  1 12]]
📌 Pros of Random Forest
Robust to outliers

Works well with categorical & numerical data

Can measure feature importance

⚠️ Limitations
Slower for very large datasets

Less interpretable than a single tree

📚 Sklearn Summary:

RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=None,       # Let trees grow fully
    random_state=42,      # Reproducibility
)
✅ Key Terms Recap:
Term	Meaning
Ensemble Learning	Combining multiple models to improve performance
Bagging	Sampling with replacement
Majority Voting	Most common prediction among models
Feature Importance	Which features the model used the most

