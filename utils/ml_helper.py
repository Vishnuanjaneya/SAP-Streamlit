import pandas as pd

def preprocess(df):
    """
    NOTE: With the CatBoost model, module / transport_stage /
    change_request_status must stay as their original string values
    ('FI', 'Development', 'Approved', etc.) — NOT converted to numbers.

    CatBoost was trained with these columns passed as raw categorical
    strings (cat_features=[...]), so it learned categories like 'FI'
    directly. If you convert them to 0/1/2/3 here before predicting,
    the model will see a category it never learned and produce wrong
    or broken predictions. This is the opposite of your old
    RandomForest setup, which needed manual numeric encoding.
    """
    df = df.copy()

    # Just make sure these are clean strings — no numeric mapping.
    df['module'] = df['module'].astype(str)
    df['transport_stage'] = df['transport_stage'].astype(str)
    df['change_request_status'] = df['change_request_status'].astype(str)

    return df

def predict(model, df):
    X = df[[
        'module', 'objects_changed', 'lines_changed',
        'conflicts', 'history_failures',
        'transport_stage', 'change_request_status'
    ]]
    preds    = model.predict(X)
    # CatBoost's .predict() on a MultiClass model returns a 2D array
    # like [[0], [2], [1]] rather than a flat [0, 2, 1] — flatten it
    # so the rest of the app (which expects a simple list) still works.
    preds = [int(p[0]) if hasattr(p, '__len__') else int(p) for p in preds]
    risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return [risk_map[p] for p in preds]