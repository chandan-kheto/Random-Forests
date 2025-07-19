
# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Data
iris = load_iris()
X = iris.data # Features
y = iris.target # Target

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create Random Forest Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train) # Train on training data

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Model
feature_names = iris.feature_names
class_names = iris.target_names

plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_names, color='blue')
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.show()