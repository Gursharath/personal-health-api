import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("stroke.csv")

# Drop non-numeric or ID column
df = df.drop(["id"], axis=1)

# Convert categorical columns
df['gender'] = df['gender'].map({"Male": 0, "Female": 1, "Other": 2})
df['ever_married'] = df['ever_married'].map({"No": 0, "Yes": 1})
df['work_type'] = df['work_type'].astype('category').cat.codes
df['Residence_type'] = df['Residence_type'].map({"Rural": 0, "Urban": 1})
df['smoking_status'] = df['smoking_status'].astype('category').cat.codes

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model_stroke.pkl", "wb"))
print("✅ Stroke model saved as model_stroke.pkl")
