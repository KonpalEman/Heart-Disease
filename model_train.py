import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load CSV
df = pd.read_csv("heart.csv")

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save model
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model retrained and saved successfully.")
