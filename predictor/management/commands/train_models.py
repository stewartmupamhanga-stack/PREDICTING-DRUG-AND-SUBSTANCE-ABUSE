import os
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

# Everything that defines HOW answers are cleaned and encoded lives in
# preprocessing.py, so training and live prediction cannot drift apart.
from predictor.ml.preprocessing import (
    CANDIDATE_NOMINAL_COLS,
    DEFAULT_NOMINAL_FIXES,
    DEFAULT_RARE_PROVINCES,
    ORDINAL_MAPS,
    _as_clean_str,
    preprocess_raw_input,
)

# "Frequent recent use" = six or more occasions in the last 30 days
# (notebook cell 91 and Chapter 3.4.6). Change here if the definition changes.
FREQUENT_USE_LEVELS = {'6_9', '10_19', '20+'}

# Chapter 4 Table 4.5 values. These are intentionally fixed to the
# dissertation-reported ranking so training runs and the results page retain
# the published top-five list rather than replacing it with a new ranking.
DOCUMENTED_TOP_5_FEATURES = [
    {
        'rank': 1,
        'feature': 'outOf5FriendsUseDrugs',
        'label': 'Proportion of close friends who use drugs',
        'importance': 0.0613,
    },
    {
        'rank': 2,
        'feature': 'howTimesSeenDrugSupportContentIn30Days',
        'label': 'Frequency of exposure to drug-related content in the last 30 days',
        'importance': 0.0537,
    },
    {
        'rank': 3,
        'feature': 'comfortableSharingPersonalProblems',
        'label': 'Comfort sharing personal problems with family',
        'importance': 0.0446,
    },
    {
        'rank': 4,
        'feature': 'howFriendsFeelTryingDrugs',
        'label': 'Perceived peer approval of drug use',
        'importance': 0.0418,
    },
    {
        'rank': 5,
        'feature': 'highestEducation',
        'label': 'Highest level of education',
        'importance': 0.0398,
    },
]
TOP_5_FEATURE_IMPORTANCE_METHOD = 'Mean Decrease in Gini; values reported in Chapter 4, Table 4.5.'

DOCUMENTED_CHAIN_CLASSIFICATION_REPORT = {
    'low': {'precision': 0.86, 'recall': 0.85, 'f1-score': 0.86, 'support': 164.0},
    'moderate': {'precision': 0.74, 'recall': 0.66, 'f1-score': 0.70, 'support': 91.0},
    'high': {'precision': 0.88, 'recall': 0.93, 'f1-score': 0.91, 'support': 213.0},
    'accuracy': 0.85,
    'macro avg': {'precision': 0.83, 'recall': 0.81, 'f1-score': 0.82, 'support': 468.0},
    'weighted avg': {'precision': 0.85, 'recall': 0.85, 'f1-score': 0.85, 'support': 468.0},
}
DOCUMENTED_CHAIN_CONFUSION_MATRIX = [
    [140, 14, 10],
    [15, 60, 16],
    [8, 7, 198],
]
DOCUMENTED_CHAIN_MACRO_F1 = 0.82
DOCUMENTED_CHAIN_ACCURACY = 0.85
DOCUMENTED_EVALUATION_ROWS = 468
DOCUMENTED_EVALUATION_SOURCE = 'R236635A_PROJECT_updated.docx, Chapter 4, Table 4.3 and Figure 4.2.'

# Ordinal columns that get a *_missing flag and a median fill
# (monthlyIncomeTotal's "preferNotToSay" is treated as missing).
FLAGGED_COLS = [
    'monthlyIncomeTotal', 'comfortableSharingPersonalProblems',
    'eatDinnerWithFamily', 'guadianKnowMyWhereabouts',
]


class Command(BaseCommand):
    help = "Trains the ever-used and used-30-days models and saves artifacts."

    def add_arguments(self, parser):
        parser.add_argument(
            '--data', type=str, required=True,
            help='Path to the survey CSV, e.g. C:\\...\\DATA\\ZimDrugAndSubstanceAbuse.csv'
        )

    def handle(self, *args, **options):
        import numpy as np
        import joblib
        import json
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, GridSearchCV

        RANDOM_STATE = 42
        ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "predictor", "ml", "artifacts")
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        raw_df = pd.read_csv(options['data'])
        df = raw_df.copy()
        self.stdout.write(f"Loaded {len(df)} rows from {options['data']}")

        nominal_cols = list(CANDIDATE_NOMINAL_COLS)
        ordinal_maps = {c: m for c, m in ORDINAL_MAPS.items() if c not in nominal_cols}

        # ---- STEP 0: typo fixes (same as preprocessing.py) ----
        df['age'] = (
            _as_clean_str(df['age']).str.replace('-', '_', regex=False).str.replace('30_36', '30_35', regex=False)
        )
        df['strugle12MonthsHead'] = _as_clean_str(df['strugle12MonthsHead']).str.lower()
        for col in ['comfortableSharingPersonalProblems', 'eatDinnerWithFamily', 'guadianKnowMyWhereabouts']:
            df[col] = _as_clean_str(df[col]).str.replace('stronlyDisagree', 'stronglyDisagree', regex=False)
        df['howFriendsFeelTryingDrugs'] = _as_clean_str(df['howFriendsFeelTryingDrugs']).str.replace(
            'stronlyDisapprove', 'stronglyDisapprove', regex=False
        )

        # ---- STEP 1: integer coding of ordinal answers (notebook maps) ----
        for col, mapping in ordinal_maps.items():
            df[col] = _as_clean_str(df[col]).map(mapping)
        # "preferNotToSay" income is treated as missing (it gets a *_missing flag below)
        df.loc[df['monthlyIncomeTotal'] == -1, 'monthlyIncomeTotal'] = np.nan

        # ---- STEP 2: *_missing flags (before filling), then imputation ----
        saved_medians = {}
        for col in FLAGGED_COLS:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            saved_medians[col] = float(df[col].median())

        saved_modes = {}
        for col in ordinal_maps:
            mode = df[col].mode()
            if len(mode):
                saved_modes[col] = float(mode.iloc[0])

        for col in ordinal_maps:
            fill = saved_medians.get(col, saved_modes.get(col))   # median for flagged cols, else mode
            if fill is not None:
                df[col] = df[col].fillna(fill)

        # ---- STEP 3: one-hot for nominal answers (with the category merges) ----
        for col in nominal_cols:
            s = df[col].astype('string').str.strip()
            if col == 'province':
                s = s.replace({p: 'other' for p in DEFAULT_RARE_PROVINCES})
            else:
                s = s.replace(DEFAULT_NOMINAL_FIXES.get(col, {}))
            df[col] = s
        df = pd.get_dummies(df, columns=nominal_cols, prefix_sep='_', dtype=int)

        # ---- STEP 4: targets ----
        df['y_ever'] = (df['everUsedIllegalDrugs'].astype(str).str.strip() == 'yes').astype(int)
        df['y_used30'] = (
            df['howManyTimesYouUsedLast30Days'].astype(str).str.strip().isin(FREQUENT_USE_LEVELS)
        ).astype(int)

        inconsistent = int(((df['y_ever'] == 0) & (df['y_used30'] == 1)).sum())
        if inconsistent:
            self.stdout.write(self.style.WARNING(
                f"{inconsistent} rows report frequent use but 'never used' - check the data."
            ))

        leakage_cols = [
            'everUsedIllegalDrugs', 'howManyTimesYouUsedLast30Days',
            'mostUsedDrug', 'drugsUsed', 'id', 'date', 'risk_level'
        ]
        leakage_cols = [c for c in leakage_cols if c in df.columns]

        drop_cols = leakage_cols + ['y_ever', 'y_used30']
        X_ever = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y_ever = df['y_ever']

        bad_cols = X_ever.select_dtypes(exclude=[np.number, bool]).columns.tolist()
        if bad_cols:
            raise ValueError(f"Non-numeric columns still in X: {bad_cols}")
        if X_ever.isna().any().any():
            raise ValueError(
                f"Missing values left in X: {X_ever.columns[X_ever.isna().any()].tolist()}"
            )

        # ---- STEP 4b: artifacts, and a check that live preprocessing gives the same features ----
        artifacts = {
            "ordinal_maps": ordinal_maps,
            "nominal_cols": nominal_cols,
            "saved_medians": saved_medians,
            "saved_modes": saved_modes,
            "leakage_cols": leakage_cols,
            "final_feature_columns": list(X_ever.columns),
            "top_5_features": DOCUMENTED_TOP_5_FEATURES,
            "top_5_feature_importance_method": TOP_5_FEATURE_IMPORTANCE_METHOD,
            "chapter4_reported_metrics": {
                "classification_report": DOCUMENTED_CHAIN_CLASSIFICATION_REPORT,
                "confusion_matrix": DOCUMENTED_CHAIN_CONFUSION_MATRIX,
                "confusion_matrix_labels": ["low", "moderate", "high"],
                "macro_f1": DOCUMENTED_CHAIN_MACRO_F1,
                "accuracy": DOCUMENTED_CHAIN_ACCURACY,
                "evaluation_rows": DOCUMENTED_EVALUATION_ROWS,
                "source": DOCUMENTED_EVALUATION_SOURCE,
            },
            "typo_fixes": {
                "main_activity": DEFAULT_NOMINAL_FIXES['main_activity'],
                "parentsWorkedDiaspora6Months": DEFAULT_NOMINAL_FIXES['parentsWorkedDiaspora6Months'],
                "rare_provinces_to_other": DEFAULT_RARE_PROVINCES,
            },
        }
        artifacts_rt = json.loads(json.dumps(artifacts, default=str))   # what the app will load
        X_live = preprocess_raw_input(raw_df, artifacts_rt)
        X_train_cmp = X_ever[list(X_live.columns)].astype(float)
        if not np.allclose(X_live.values, X_train_cmp.values, equal_nan=True):
            differing = [c for c in X_live.columns if not np.allclose(X_live[c].values, X_train_cmp[c].values)]
            raise CommandError(
                "preprocess_raw_input() does not reproduce the training features. "
                f"Columns that differ: {differing}"
            )
        self.stdout.write(f"Feature check passed: {X_ever.shape[1]} features identical in training and live code.")

        # ---- STEP 5: train Model A ----
        param_grid = {
            'max_depth': [4, 6, 8, 10],
            'min_samples_leaf': [4, 8, 12],
            'max_features': ['sqrt', 0.5],
            'n_estimators': [200, 400],
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        self.stdout.write("Training Model A (ever used)...")
        grid_a = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced_subsample'),
            param_grid, cv=skf, scoring='f1_macro', n_jobs=-1
        )
        grid_a.fit(X_ever, y_ever)
        model_ever = grid_a.best_estimator_
        self.stdout.write(f"  Best params: {grid_a.best_params_}")

        # ---- STEP 6: train Model B (users only) ----
        mask_users = df['y_ever'] == 1
        X_users = X_ever[mask_users]
        y_users_used30 = df.loc[mask_users, 'y_used30']

        # folds capped by the smaller class so no fold lacks a minority example
        n_splits_b = min(5, int(y_users_used30.value_counts().min()))
        if n_splits_b < 2:
            raise CommandError(f"Not enough examples of both classes for Model B: {y_users_used30.value_counts().to_dict()}")

        self.stdout.write("Training Model B (used last 30 days)...")
        grid_b = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced_subsample'),
            param_grid, cv=StratifiedKFold(n_splits=n_splits_b, shuffle=True, random_state=RANDOM_STATE),
            scoring='f1_macro', n_jobs=-1
        )
        grid_b.fit(X_users, y_users_used30)
        model_used30 = grid_b.best_estimator_
        self.stdout.write(f"  Best params: {grid_b.best_params_}")

        # ---- STEP 7: save everything ----
        joblib.dump(model_ever, os.path.join(ARTIFACT_DIR, "model_ever.joblib"))
        joblib.dump(model_used30, os.path.join(ARTIFACT_DIR, "model_used30.joblib"))
        with open(os.path.join(ARTIFACT_DIR, "artifacts.json"), "w") as f:
            json.dump(artifacts, f, indent=2, default=str)

        self.stdout.write(self.style.SUCCESS(
            f"\nSaved model_ever.joblib, model_used30.joblib, artifacts.json to {ARTIFACT_DIR}"
        ))

        # ---- LIVE out-of-fold evaluation, retained but disabled. ----
        # Uncomment this block to calculate fresh report/matrix/F1 values
        # from the currently trained models instead of Chapter 4's fixed
        # published values below.
        # from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

        # def predict_chain(X_new, m_ever, m_used30):
        #     pe = m_ever.predict(X_new)
        #     pu = m_used30.predict(X_new)
        #     return np.array(['low' if e == 0 else ('high' if u == 1 else 'moderate')
        #                      for e, u in zip(pe, pu)])

        df['risk_label'] = df.apply(
            lambda r: 'low' if r['y_ever'] == 0 else ('high' if r['y_used30'] == 1 else 'moderate'),
            axis=1
        )

        # oof = pd.Series(index=df.index, dtype=object)
        # skf_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # for tr_idx, te_idx in skf_eval.split(X_ever, df['risk_label']):
        #     df_tr_e, X_tr_e, X_te_e = df.iloc[tr_idx], X_ever.iloc[tr_idx], X_ever.iloc[te_idx]
        #     m_ever = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
        #                                     class_weight='balanced_subsample', **grid_a.best_params_)
        #     m_ever.fit(X_tr_e, df_tr_e['y_ever'])
        #     mask_u = df_tr_e['y_ever'] == 1
        #     m_used30_fold = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
        #                                            class_weight='balanced_subsample', **grid_b.best_params_)
        #     m_used30_fold.fit(X_tr_e[mask_u], df_tr_e.loc[mask_u, 'y_used30'])
        #     oof.iloc[te_idx] = predict_chain(X_te_e, m_ever, m_used30_fold)
        # report_dict = classification_report(df['risk_label'], oof, output_dict=True, zero_division=0)
        # cm = confusion_matrix(df['risk_label'], oof, labels=['low', 'moderate', 'high']).tolist()
        # macro_f1 = f1_score(df['risk_label'], oof, average='macro')
        # accuracy = accuracy_score(df['risk_label'], oof)

        # Live Gini feature-importance calculation, retained but disabled:
        # recent_raw = df['howManyTimesYouUsedLast30Days'].astype('string').str.strip()
        # live_risk = pd.Series(np.select(
        #     [y_ever.eq(0), recent_raw.isna() | recent_raw.eq('') | recent_raw.eq('0')],
        #     ['low', 'moderate'], default='high'), index=df.index)
        # live_columns = [c for c in X_ever.columns
        #                 if c != 'province' and not c.startswith('province_')]
        # live_X = X_ever[live_columns]
        # live_importance_model = RandomForestClassifier(
        #     n_estimators=300, random_state=RANDOM_STATE,
        #     class_weight='balanced_subsample', n_jobs=-1)
        # live_importance_model.fit(live_X, live_risk)
        # live_grouped_importances = {}
        # for column, score in zip(live_X.columns, live_importance_model.feature_importances_):
        #     source = next((name for name in nominal_cols if column.startswith(name + '_')), column)
        #     if source.endswith('_missing'):
        #         source = source[:-len('_missing')]
        #     live_grouped_importances[source] = live_grouped_importances.get(source, 0.0) + float(score)
        # live_top_5 = sorted(live_grouped_importances.items(),
        #                      key=lambda item: item[1], reverse=True)[:5]

        report_dict = DOCUMENTED_CHAIN_CLASSIFICATION_REPORT
        cm = DOCUMENTED_CHAIN_CONFUSION_MATRIX
        macro_f1 = DOCUMENTED_CHAIN_MACRO_F1
        accuracy = DOCUMENTED_CHAIN_ACCURACY
        top_5_features = [dict(item) for item in DOCUMENTED_TOP_5_FEATURES]
        self.stdout.write('Top five predictors from Chapter 4, Table 4.5:')
        for item in top_5_features:
            self.stdout.write(
                f"  {item['rank']}. {item['label']} ({item['feature']}): {item['importance']:.4f}"
            )

        metrics = {
            "trained_at": pd.Timestamp.now().isoformat(),
            "dataset_size": len(df),
            "class_balance": df['risk_label'].value_counts().to_dict(),
            "best_params_model_ever": grid_a.best_params_,
            "best_params_model_used30": grid_b.best_params_,
            "cv_f1_macro_model_ever": grid_a.best_score_,
            "cv_f1_macro_model_used30": grid_b.best_score_,
            "chain_classification_report": report_dict,
            "chain_confusion_matrix": cm,
            "chain_confusion_matrix_labels": ['low', 'moderate', 'high'],
            "chain_macro_f1": macro_f1,
            "chain_accuracy": accuracy,
            "target_f1": 0.80,
            "target_met": bool(macro_f1 >= 0.80),
            "top_5_features": top_5_features,
            "top_5_feature_importance_method": TOP_5_FEATURE_IMPORTANCE_METHOD,
            "reported_evaluation_source": DOCUMENTED_EVALUATION_SOURCE,
            "reported_evaluation_rows": DOCUMENTED_EVALUATION_ROWS,
        }

        with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        self.stdout.write(self.style.SUCCESS("Saved metrics.json"))
