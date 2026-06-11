import pandas as pd

def preprocess(df):
    module_map = {'FI': 0, 'MM': 1, 'SD': 2, 'HR': 3}
    stage_map = {'Development': 0, 'Quality': 1, 'Production': 2}
    status_map = {'Approved': 0, 'Pending': 1, 'Rejected': 2}

    df['module'] = df['module'].map(module_map)
    df['transport_stage'] = df['transport_stage'].map(stage_map)
    df['change_request_status'] = df['change_request_status'].map(status_map)

    return df

def predict(model, df):
    X = df[[
        'module','objects_changed','lines_changed',
        'conflicts','history_failures',
        'transport_stage','change_request_status'
    ]]

    preds = model.predict(X)

    risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return [risk_map[p] for p in preds]