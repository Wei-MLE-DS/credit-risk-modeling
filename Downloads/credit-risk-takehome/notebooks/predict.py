import joblib
import pandas as pd

artifacts = joblib.load("credit_default_model.joblib")
pipeline = artifacts["pipeline"]
threshold = artifacts["threshold"]

def predict(df: pd.DataFrame):
    proba = pipeline.predict_prob(df)[:, 1]
    decision = (proba >= threshold).astype(int)
    return {
        "probability" : proba.tolist(),
        "decision" : decision.tolist()
    }