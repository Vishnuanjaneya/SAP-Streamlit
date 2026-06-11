import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../data/sap_transport_dataset.csv")

print("✅ Dataset loaded successfully")
print(df.head())

# ---------------- ENCODING ----------------
# Convert categorical columns to numeric

module_map = {'FI': 0, 'MM': 1, 'SD': 2, 'HR': 3}
df['module'] = df['module'].map(module_map)

stage_map = {'Development': 0, 'Quality': 1, 'Production': 2}
df['transport_stage'] = df['transport_stage'].map(stage_map)

status_map = {'Approved': 0, 'Pending': 1, 'Rejected': 2}
df['change_request_status'] = df['change_request_status'].map(status_map)

# ---------------- TARGET VARIABLE ----------------
# Convert risk label to numeric
risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
df['risk'] = df['risk_level'].map(risk_map)

# ---------------- FEATURES ----------------
X = df[[
    'module',
    'objects_changed',
    'lines_changed',
    'conflicts',
    'history_failures',
    'transport_stage',
    'change_request_status'
]]

# ---------------- TARGET ----------------
y = df['risk']

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("✅ Model training completed")

# ---------------- SAVE MODEL ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully (model.pkl)")