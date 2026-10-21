"""Notebook-aligned pilot model comparison and fitting.

The production Random Forest artifacts are never written here.  Pilot
estimators are sklearn Pipelines that accept raw survey columns, matching the
preprocessing and evaluation in FIRSTSET/resultsAndFindings.ipynb.
"""

import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


RANDOM_STATE = 42
ARTIFACT_DIR = Path(settings.BASE_DIR) / "predictor" / "ml" / "artifacts"
PILOT_DIR = ARTIFACT_DIR / "pilot"
PILOT_CANDIDATE_DIR = PILOT_DIR / "candidates"

# Keep all notebook target/leakage fields out of the predictors.  The
# notebook excludes province for both prediction tasks.
LEAKAGE_COLUMNS = {
    "id", "date", "province", "everUsedIllegalDrugs",
    "howManyTimesYouUsedLast30Days", "mostUsedDrug", "drugsUsed",
}
KNOWN_FREQUENCIES = {"0", "1_2", "3_5", "6_9", "10_19", "20+"}


def make_preprocessor(feature_columns):
    """Notebook preprocessing: fold-fitted mode imputation then one-hot."""
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("categorical", categorical, list(feature_columns))],
        remainder="drop",
        sparse_threshold=0,
    )


def make_models():
    """The four classifiers used in the notebook's shared comparison."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_tuned30_specs():
    """Five estimators and grids copied from the notebook's 30-day section."""
    return {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"classifier__n_estimators": [200], "classifier__min_samples_leaf": [2, 4],
             "classifier__max_features": ["sqrt", 0.7],
             "classifier__class_weight": ["balanced", None]},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"classifier__n_estimators": [200], "classifier__min_samples_leaf": [2, 4],
             "classifier__max_features": ["sqrt", 0.7],
             "classifier__class_weight": ["balanced", None]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {"classifier__C": [0.1, 0.5, 2.0],
             "classifier__class_weight": ["balanced", None]},
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"classifier__C": [0.1, 0.5, 2.0],
             "classifier__kernel": ["linear", "rbf"],
             "classifier__class_weight": ["balanced", None]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"classifier__n_estimators": [80, 150],
             "classifier__learning_rate": [0.03, 0.08],
             "classifier__max_depth": [1, 2],
             "classifier__min_samples_leaf": [3, 5]},
        ),
    }


def _pipeline(classifier, feature_columns):
    return Pipeline([
        ("preprocess", make_preprocessor(feature_columns)),
        ("classifier", clone(classifier)),
    ])


def _positive_probabilities(model, X):
    probabilities = model.predict_proba(X)
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    classes = list(estimator.named_steps["classifier"].classes_)
    return probabilities[:, classes.index(1)]


def _binary_metrics(y_true, predictions, probabilities, target_names=("No", "Yes")):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "classification_report": classification_report(
            y_true, predictions, labels=[0, 1],
            target_names=list(target_names), output_dict=True, zero_division=0,
        ),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix_target_names": list(target_names),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def _notebook_holdout_comparison(X, y, feature_columns, target_names):
    """Reproduce the notebook's 80/20 holdout plus 5-fold train CV."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    fitted = {}

    for name, classifier in make_models().items():
        print(f"Notebook holdout/CV comparison: {name}", flush=True)
        model = _pipeline(classifier, feature_columns)
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = _positive_probabilities(model, X_test)
        item = _binary_metrics(y_test, predictions, probabilities, target_names)
        item.update({
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_accuracy_sd": float(np.std(cv_scores["test_accuracy"], ddof=1)),
            "cv_macro_f1_mean": float(np.mean(cv_scores["test_macro_f1"])),
            "cv_macro_f1_sd": float(np.std(cv_scores["test_macro_f1"], ddof=1)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        })
        results[name] = item
        fitted[name] = model

    # Pick the deployment winner by validation accuracy, never by held-out
    # test score. Macro F1 remains reported alongside accuracy.
    selected = max(results, key=lambda name: results[name]["cv_accuracy_mean"])
    deployment_models = {}
    for name, classifier in make_models().items():
        deployment_model = _pipeline(classifier, feature_columns)
        deployment_model.fit(X, y)
        deployment_models[name] = deployment_model
    return results, selected, deployment_models


def _nested_30day_comparison(X, y, feature_columns):
    """Notebook's pooled outer 5-fold evaluation with inner tuning/cutoffs."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    threshold_grid = np.arange(0.20, 0.801, 0.01)
    result = {}
    deployment_candidates = {}

    for name, (classifier, grid) in make_tuned30_specs().items():
        print(f"Nested 30-day tuning: {name}", flush=True)
        prob_oof = np.full(len(y), np.nan, dtype=float)
        pred_oof = np.full(len(y), -1, dtype=int)
        fold_thresholds, fold_params = [], []

        for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
            print(f"  outer fold {fold}/5", flush=True)
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold)
            pipeline = _pipeline(classifier, feature_columns)
            search = GridSearchCV(
                pipeline, grid, scoring="f1_macro", cv=inner,
                n_jobs=1, refit=True, error_score="raise",
            )
            search.fit(Xtr, ytr)
            fold_params.append(search.best_params_)

            inner_prob = cross_val_predict(
                clone(search.best_estimator_), Xtr, ytr, cv=inner,
                method="predict_proba", n_jobs=1,
            )[:, 1]
            cutoff = max(
                threshold_grid,
                key=lambda threshold: f1_score(
                    ytr, (inner_prob >= threshold).astype(int), average="macro", zero_division=0
                ),
            )
            outer_prob = _positive_probabilities(search, X.iloc[te])
            prob_oof[te] = outer_prob
            pred_oof[te] = (outer_prob >= cutoff).astype(int)
            fold_thresholds.append(float(cutoff))

        metrics = _binary_metrics(
            y, pred_oof, prob_oof, ("No recent use", "Any recent use")
        )
        metrics.update({
            "outer_folds": 5,
            "inner_folds": 4,
            "pooled_outer_oof": True,
            "fold_thresholds": fold_thresholds,
            "mean_selected_threshold": float(np.mean(fold_thresholds)),
            "best_params_by_outer_fold": fold_params,
        })
        result[name] = metrics

        # This refit is for deployment only. Its score is not reported as an
        # evaluation score; those come from the untouched outer predictions.
        final_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        final_search = GridSearchCV(
            _pipeline(classifier, feature_columns), grid,
            scoring="f1_macro", cv=final_cv, n_jobs=1,
            refit=True, error_score="raise",
        )
        final_search.fit(X, y)
        deployment_candidates[name] = {
            "model": final_search.best_estimator_,
            "threshold": float(np.mean(fold_thresholds)),
            "best_params_full_data_refit": final_search.best_params_,
        }

    # Compare candidates by pooled outer-fold accuracy; each score comes from
    # out-of-fold predictions, not the deployment refit.
    winner = max(result, key=lambda name: result[name]["accuracy"])
    return result, winner, deployment_candidates


def _model_slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _as_json(value):
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_pilot(data_path: str) -> dict:
    """Run notebook comparisons; save only separate pilot artifacts/metrics."""
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Pilot training CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"everUsedIllegalDrugs", "howManyTimesYouUsedLast30Days"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing notebook target columns: {missing}")

    ever_raw = df["everUsedIllegalDrugs"].astype("string").str.strip().str.lower()
    y_ever_all = ever_raw.map({"no": 0, "yes": 1})
    ever_valid = y_ever_all.notna()
    ever_feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X_ever = df.loc[ever_valid, ever_feature_cols].reset_index(drop=True)
    y_ever = y_ever_all.loc[ever_valid].astype(int).reset_index(drop=True)

    response = df["howManyTimesYouUsedLast30Days"].astype("string").str.strip().replace("", pd.NA)
    frequency_valid = response.isin(KNOWN_FREQUENCIES)
    y_recent_all = response.loc[frequency_valid].ne("0").astype(int)
    recent_feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X_recent = df.loc[frequency_valid, recent_feature_cols].reset_index(drop=True)
    y_recent = y_recent_all.reset_index(drop=True)

    for target_name, target in (("lifetime", y_ever), ("30-day use", y_recent)):
        if target.nunique() != 2 or int(target.value_counts().min()) < 5:
            raise ValueError(
                f"{target_name} does not have enough examples per class for the notebook's 5-fold CV: "
                f"{target.value_counts().to_dict()}"
            )

    # Four-method holdout/CV comparisons, matching the notebook's shared section.
    lifetime_comparison, lifetime_winner, lifetime_candidates = _notebook_holdout_comparison(
        X_ever, y_ever, ever_feature_cols, ("No", "Yes")
    )
    recent_comparison, recent_baseline_winner, _ = _notebook_holdout_comparison(
        X_recent, y_recent, recent_feature_cols,
        ("No recent use", "Any recent use")
    )

    # Five-method, nested 30-day-only evaluation. This supplies the pilot's
    # recent-use classifier and the primary pooled outer-fold metrics.
    tuned_recent, recent_winner, recent_candidates = _nested_30day_comparison(
        X_recent, y_recent, recent_feature_cols
    )
    recent_deployment = recent_candidates[recent_winner]

    trained_at = timezone.now()
    metrics = {
        "schema_version": 2,
        "evaluation_source": "FIRSTSET/resultsAndFindings.ipynb",
        "trained_at": trained_at.isoformat(),
        "dataset": {
            "path": str(data_path),
            "rows": int(len(df)),
            "lifetime_valid_rows": int(len(y_ever)),
            "recent_valid_rows": int(len(y_recent)),
            "recent_missing_or_unrecognized_excluded": int((~frequency_valid).sum()),
            "lifetime_class_counts": {str(k): int(v) for k, v in y_ever.value_counts().sort_index().items()},
            "recent_class_counts": {str(k): int(v) for k, v in y_recent.value_counts().sort_index().items()},
            "features_lifetime": ever_feature_cols,
            "features_recent": recent_feature_cols,
        },
        "model_ever": {
            "winner": lifetime_winner,
            "selection_metric": "mean five-fold training CV accuracy",
            "cv_f1_macro": lifetime_comparison[lifetime_winner]["cv_macro_f1_mean"],
            "all_candidates": {
                name: value["cv_macro_f1_mean"] for name, value in lifetime_comparison.items()
            },
            "comparison": lifetime_comparison,
        },
        "model_used30": {
            "winner": recent_winner,
            "selection_metric": "pooled outer five-fold out-of-fold accuracy",
            "cv_f1_macro": tuned_recent[recent_winner]["macro_f1"],
            "all_candidates": {
                name: value["macro_f1"] for name, value in tuned_recent.items()
            },
            "holdout_comparison_winner_by_training_cv": recent_baseline_winner,
            "holdout_comparison": recent_comparison,
            "nested_outer_cv_comparison": tuned_recent,
        },
        "deployment": {
            "model_ever_name": lifetime_winner,
            "model_ever_fit_rows": int(len(y_ever)),
            "model_ever_candidate_files": {
                name: f"candidates/ever_used/{_model_slug(name)}.joblib"
                for name in lifetime_candidates
            },
            "model_used30_name": recent_winner,
            "model_used30_fit_rows": int(len(y_recent)),
            "model_used30_threshold": recent_deployment["threshold"],
            "model_used30_best_params_full_data_refit": recent_deployment["best_params_full_data_refit"],
            "model_used30_candidate_files": {
                name: f"candidates/used30/{_model_slug(name)}.joblib"
                for name in recent_candidates
            },
            "models_accept_raw_survey_columns": True,
        },
        "target_definition": {
            "lifetime": "0=no, 1=yes; unrecognized and missing responses excluded",
            "recent_use": "0=explicit zero, 1=any recognized nonzero frequency; missing/unrecognized excluded",
            "selection_warning": "Held-out test metrics are reported but are not used to select winners.",
        },
    }

    # Separate pilot filenames; no production Random Forest files are touched.
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (PILOT_CANDIDATE_DIR / "ever_used").mkdir(parents=True, exist_ok=True)
    (PILOT_CANDIDATE_DIR / "used30").mkdir(parents=True, exist_ok=True)
    for name, candidate in lifetime_candidates.items():
        joblib.dump(candidate, PILOT_CANDIDATE_DIR / "ever_used" / f"{_model_slug(name)}.joblib")
    for name, candidate in recent_candidates.items():
        joblib.dump(candidate["model"], PILOT_CANDIDATE_DIR / "used30" / f"{_model_slug(name)}.joblib")
    joblib.dump(lifetime_candidates[lifetime_winner], PILOT_DIR / "model_ever.joblib")
    joblib.dump(recent_deployment["model"], PILOT_DIR / "model_used30.joblib")
    metrics = _as_json(metrics)
    with open(PILOT_DIR / "pilot_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)

    from predictor.models import MLConfig

    config = MLConfig.get_solo()
    config.pilot_model_ever_name = lifetime_winner
    config.pilot_model_used30_name = recent_winner
    config.pilot_trained_at = trained_at
    config.pilot_metrics = metrics
    config.pilot_mode_enabled = True
    config.save()
    return metrics
