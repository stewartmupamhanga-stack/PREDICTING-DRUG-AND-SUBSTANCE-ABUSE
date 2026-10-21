def predict_combined_risk(X, model_ever, model_used30, used30_threshold=None):
    pred_ever = model_ever.predict(X)[0]
    if pred_ever == 0:
        return {"risk_level": "low", "used_30_days": None}
    if used30_threshold is None:
        pred_used30 = model_used30.predict(X)[0]
    else:
        classes = list(model_used30.classes_)
        probabilities = model_used30.predict_proba(X)[0]
        pred_used30 = int(probabilities[classes.index(1)] >= used30_threshold)
    risk_level = "high" if pred_used30 == 1 else "moderate"
    return {"risk_level": risk_level, "used_30_days": pred_used30}
