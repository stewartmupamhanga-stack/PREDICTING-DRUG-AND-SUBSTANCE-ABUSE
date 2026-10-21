# ML model serving and pilot comparison

## Model bundles

- Default mode uses the existing Random Forest artifacts in `predictor/ml/artifacts/` and their saved feature preprocessing.
- Pilot mode uses the separately saved estimators in `predictor/ml/artifacts/pilot/`. These are notebook-aligned sklearn pipelines and take raw survey columns.
- The assessment form, `/predict/`, legacy prediction APIs, and bulk predictions all use the same mode-aware predictor service.
- Pilot training never writes the default model files or default `metrics.json`. Pilot models and all pilot evaluation metrics are written only under `artifacts/pilot/`.
- Every evaluated lifetime and 30-day pilot candidate is retained under `artifacts/pilot/candidates/`; the notebook-selected winners are copied to `model_ever.joblib` and `model_used30.joblib` and are the pair served in Pilot mode.

## Pilot methodology

`pilot_training.py` follows `FIRSTSET/resultsAndFindings.ipynb`:

- Lifetime use and broad recent-use comparisons use the notebook's four classifiers: Random Forest, Logistic Regression, SVM, and Gradient Boosting. Each uses a stratified 80/20 split and five-fold CV on training data. The winning lifetime model is selected by training CV macro F1, not its test score.
- Recent use is also evaluated by the notebook's five-method nested procedure: Random Forest, Extra Trees, Logistic Regression, SVM, and Gradient Boosting; outer five-fold evaluation; inner four-fold model tuning; training-fold-only threshold selection; pooled outer-fold predictions.
- Recent use is coded as explicit zero versus any recognized nonzero frequency. Rows with missing or unrecognized frequency answers are excluded from that task. The 30-day predictors exclude outcome, drug identity, ID, date, and province fields.
- Imputation and one-hot encoding are fit inside each model pipeline and therefore inside each CV fold.
- `pilot_metrics.json` stores the holdout and CV metrics, classification reports, confusion matrices, nested outer-fold results, tuned parameters, thresholds, selected model names, and data row counts. The notebook's ROC-AUC and PR-AUC are included.

The pilot recent-use model uses the full eligible cohort, matching the notebook's 30-day-only task. The default Model B retains its existing behavior. As a result, pilot metrics are a supplementary comparison and should not be reported as a reproduction of the Chapter 4 chained Random Forest results.

## Toggle and training endpoints

All URLs are rooted at the Django site:

```text
POST /predict/       body: raw survey-answer JSON; returns chained risk
GET  /pilot-mode/    staff login required; returns toggle state and pilot metrics
POST /pilot-mode/    staff login + CSRF required; body: {"enabled": true|false}
POST /train-pilot/   staff login + CSRF required; body: {"data_path": "<configured study CSV>"}
```

The web training endpoint only accepts the configured study CSV. For local research on another authorized CSV, run the management command from the project directory:

```bash
python manage.py train_pilot_models --data /path/to/data.csv
```

Training completes the comparisons, writes the pilot-only artifacts and metrics, records the selected methods, and enables pilot mode. Switching pilot mode off returns prediction routes to the unchanged default Random Forest artifacts.
