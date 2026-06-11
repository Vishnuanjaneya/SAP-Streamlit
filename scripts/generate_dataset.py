import pandas as pd
import random

modules = ['FI', 'MM', 'SD', 'HR']
stages = ['Development', 'Quality', 'Production']
status = ['Approved', 'Pending', 'Rejected']

data = []

for i in range(10000):

    module = random.choice(modules)

    # realistic ranges
    objects = random.randint(1, 6)
    lines = random.randint(10, 300)
    conflicts = random.randint(0, 2)
    failures = random.randint(0, 4)

    stage = random.choice(stages)
    change_status = random.choice(status)

    # ✅ REALISTIC RISK LOGIC (important)
    if conflicts > 1 or failures > 2 or (module == "FI" and lines > 120):
        risk = "HIGH"
    elif lines > 80 or conflicts > 0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    data.append([
        f"TR{i+1:05d}",
        module,
        objects,
        lines,
        conflicts,
        failures,
        stage,
        change_status,
        risk
    ])

df = pd.DataFrame(data, columns=[
    "transport_id",
    "module",
    "objects_changed",
    "lines_changed",
    "conflicts",
    "history_failures",
    "transport_stage",
    "change_request_status",
    "risk_level"
])

# ✅ Save dataset
df.to_csv("data/sap_transport_dataset.csv", index=False)

print("✅ 10,000 realistic SAP records created!")