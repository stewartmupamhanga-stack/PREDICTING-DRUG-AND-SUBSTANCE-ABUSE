import os
import json

import joblib
import pandas as pd
from django.apps import apps
from django.conf import settings

from .preprocessing import load_artifacts, preprocess_raw_input
from .inference import predict_combined_risk


ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "predictor", "ml", "artifacts")

# Pilot models live in their own subfolder so training a pilot never
# touches or overwrites the production model_ever.joblib / model_used30.joblib.
# The notebook-aligned pilot estimators are sklearn Pipelines and accept raw
# survey columns. Production estimators retain their existing feature matrix.
PILOT_DIR = os.path.join(ARTIFACT_DIR, "pilot")
PILOT_MODEL_EVER_PATH = os.path.join(PILOT_DIR, "model_ever.joblib")
PILOT_MODEL_USED30_PATH = os.path.join(PILOT_DIR, "model_used30.joblib")

_pilot_cache = {
    "loaded": False,
    "model_ever": None,
    "model_used30": None,
    "artifacts": None,
    "used30_threshold": None,
    "trained_at": None,
}


def _load_pilot_bundle():
    paths = [PILOT_MODEL_EVER_PATH, PILOT_MODEL_USED30_PATH]
    if not all(os.path.exists(p) for p in paths):
        raise FileNotFoundError(
            "Pilot mode is enabled but pilot models are missing. "
            "Run train-pilot first, or disable pilot mode."
        )
    _pilot_cache["model_ever"] = joblib.load(PILOT_MODEL_EVER_PATH)
    _pilot_cache["model_used30"] = joblib.load(PILOT_MODEL_USED30_PATH)
    metrics_path = os.path.join(PILOT_DIR, "pilot_metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        pilot_metrics = json.load(f)
    _pilot_cache["used30_threshold"] = pilot_metrics.get("deployment", {}).get(
        "model_used30_threshold"
    )
    _pilot_cache["trained_at"] = pilot_metrics.get("trained_at")
    # Kept for the artifact-inspection endpoint. Pilot estimator pipelines
    # do not use the production artifacts for feature engineering.
    _pilot_cache["artifacts"] = load_artifacts(ARTIFACT_DIR)
    _pilot_cache["loaded"] = True


def get_bundle(force_reload=False):
    """
    Returns {"model_ever", "model_used30", "artifacts"} — either the
    production bundle (from PredictorConfig) or the pilot bundle,
    depending on MLConfig.pilot_mode_enabled.

    force_reload=True re-checks MLConfig and, if pilot mode is on,
    re-loads the pilot model files from disk — used right after a
    toggle or a fresh train-pilot run so the change takes effect
    without a server restart. It has no effect on the production
    bundle, which only ever loads once per process via
    ensure_models_loaded().
    """
    from ..models import MLConfig  # local import avoids a circular import at module load time

    config_row = MLConfig.get_solo()

    if config_row.pilot_mode_enabled:
        config_trained_at = (
            config_row.pilot_trained_at.isoformat()
            if config_row.pilot_trained_at is not None else None
        )
        if force_reload or not _pilot_cache["loaded"] or _pilot_cache["trained_at"] != config_trained_at:
            _load_pilot_bundle()
        return {
            "model_ever": _pilot_cache["model_ever"],
            "model_used30": _pilot_cache["model_used30"],
            "artifacts": _pilot_cache["artifacts"],
            "input_mode": "raw",
            "used30_threshold": _pilot_cache["used30_threshold"],
        }

    app_config = apps.get_app_config("predictor")
    app_config.ensure_models_loaded()
    return {
        "model_ever": app_config.model_ever,
        "model_used30": app_config.model_used30,
        "artifacts": app_config.artifacts,
        "input_mode": "preprocessed",
        "used30_threshold": None,
    }


def prepare_input(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply the input contract belonging to the selected model bundle."""
    if bundle.get("input_mode") == "raw":
        return raw_df
    return preprocess_raw_input(raw_df, bundle["artifacts"])


def predict_details(raw_data: dict) -> dict:
    """Predict risk and confidence using the currently selected model pair."""
    bundle = get_bundle()
    raw_df = pd.DataFrame([raw_data])
    X = prepare_input(raw_df, bundle)
    model_ever, model_used30 = bundle["model_ever"], bundle["model_used30"]

    ever_classes = list(model_ever.classes_)
    ever_probabilities = model_ever.predict_proba(X)[0]
    p_ever = float(ever_probabilities[ever_classes.index(1)])
    pred_ever = int(model_ever.predict(X)[0])

    if pred_ever == 0:
        return {"risk_level": "low", "used_30_days": None,
                "confidence": float(ever_probabilities[ever_classes.index(0)]),
                "probability_ever_used": p_ever}

    used_classes = list(model_used30.classes_)
    used_probabilities = model_used30.predict_proba(X)[0]
    p_used = float(used_probabilities[used_classes.index(1)])
    threshold = bundle.get("used30_threshold")
    pred_used = int(p_used >= threshold) if threshold is not None else int(model_used30.predict(X)[0])
    risk_level = "high" if pred_used == 1 else "moderate"
    confidence = p_used if pred_used == 1 else float(used_probabilities[used_classes.index(0)])
    return {"risk_level": risk_level, "used_30_days": pred_used,
            "confidence": confidence, "probability_ever_used": p_ever,
            "probability_used_30_days": p_used}


def predict(raw_data: dict) -> dict:
    bundle = get_bundle()

    raw_df = pd.DataFrame([raw_data])
    X = prepare_input(raw_df, bundle)
    result = predict_combined_risk(
        X, bundle["model_ever"], bundle["model_used30"],
        used30_threshold=bundle.get("used30_threshold"),
    )

    # numpy ints/bools aren't JSON-serializable as-is
    used_30 = result["used_30_days"]
    if used_30 is not None:
        used_30 = int(used_30)

    return {"risk_level": result["risk_level"], "used_30_days": used_30}


def compare_with_common_responses(raw_data: dict, feature_names: list[str]) -> dict:
    """Compare a prediction with one study-common answer changed at a time.

    This is a local sensitivity check, not a causal or formal feature
    attribution. The reference values are the modal answers in the configured
    source survey data.
    """
    try:
        modes = pd.read_csv(settings.SURVEY_DATA_CSV, usecols=feature_names).mode(dropna=True).iloc[0]
    except (OSError, ValueError, KeyError, IndexError):
        return {}

    original_class = predict(raw_data).get("risk_level")
    comparisons = {}
    for feature in feature_names:
        if feature not in raw_data or feature not in modes.index or pd.isna(modes[feature]):
            continue
        reference = str(modes[feature])
        observed = str(raw_data[feature])
        if observed == reference:
            comparisons[feature] = {"reference": reference, "changed": False, "risk_level": original_class}
            continue
        counterfactual = dict(raw_data)
        counterfactual[feature] = reference
        changed_class = predict(counterfactual).get("risk_level")
        comparisons[feature] = {
            "reference": reference,
            "changed": changed_class != original_class,
            "risk_level": changed_class,
        }
    return comparisons
