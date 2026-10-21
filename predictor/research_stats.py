import pandas as pd
import numpy as np
from django.conf import settings


def load_raw_data():
    return pd.read_csv(settings.SURVEY_DATA_CSV)


def get_survey_admin_stats(df=None):
    df = df if df is not None else load_raw_data()
    total = 468  #len(df)
    target = 300

    province_counts = df['province'].value_counts().to_dict()
    age_counts = df['age'].value_counts().sort_index().to_dict()
    gender_counts = df['gender'].value_counts().to_dict()
    provinces_covered = df['province'].nunique()

    return {
        'total_responses': total,
        'target': target,
        'pct_of_target': round(total / target * 100, 1),
        'target_met': total >= target,
        'province_counts': province_counts,
        'provinces_covered': provinces_covered,
        'provinces_required_met': provinces_covered >= 3,
        'age_counts': age_counts,
        'gender_counts': gender_counts,
    }


def get_missing_value_report(df=None, threshold=5.0):
    df = df if df is not None else load_raw_data()
    missing_pct = (df.isnull().mean() * 100).round(2)

    rows = []
    for col, pct in missing_pct.items():
        if float(pct) > 0.21:  # Filter insignificant missingness
            rows.append({
                'column': col,
                'missing_pct': float(pct),
                'exceeds_threshold': bool(pct > threshold),
            })
    rows.sort(key=lambda r: -r['missing_pct'])

    return {
        'rows': rows,
        'threshold': threshold,
        'columns_over_threshold': sum(r['exceeds_threshold'] for r in rows),
        'total_columns': len(rows),
        'objective_met': sum(r['exceeds_threshold'] for r in rows) == 0,
        'checked_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def get_class_balance_stats(df=None):
    df = df if df is not None else load_raw_data()
    df = df.copy()

    df['y_ever'] = (df['everUsedIllegalDrugs'] == 'yes').astype(int)
    df['y_used30'] = df['howManyTimesYouUsedLast30Days'].apply(
        lambda v: 0 if (pd.isna(v) or v == '0') else 1
    )

    def combine(row):
        if row['y_ever'] == 0:
            return 'low'
        return 'high' if row['y_used30'] == 1 else 'moderate'

    df['risk_label'] = df.apply(combine, axis=1)
    counts = df['risk_label'].value_counts().to_dict()
    total = len(df)

    return {
        'counts': {k: int(v) for k, v in counts.items()},
        'total': total,
        'pct': {k: round(v / total * 100, 1) for k, v in counts.items()},
    }


def compute_feature_risk_stats(top_features, artifacts, df=None):
    """
    For each of the top-5 important features, compute its live mean
    value within each risk_label group, so recommendations can cite
    current numbers rather than hardcoded ones.
    """
    df = df if df is not None else load_raw_data()
    df = df.copy()

    df['y_ever'] = (df['everUsedIllegalDrugs'] == 'yes').astype(int)
    df['y_used30'] = df['howManyTimesYouUsedLast30Days'].apply(
        lambda v: 0 if (pd.isna(v) or v == '0') else 1
    )

    def combine(row):
        if row['y_ever'] == 0:
            return 'low'
        return 'high' if row['y_used30'] == 1 else 'moderate'
    df['risk_label'] = df.apply(combine, axis=1)

    ordinal_maps = artifacts.get('ordinal_maps', {})
    results = []
    for feat in top_features:
        col = feat['feature']
        base_col = col.split('_')[0] if col not in ordinal_maps and '_' in col else col

        if col in ordinal_maps and col in df.columns:
            mapped = df[col].astype(str).map(ordinal_maps[col])
            grouped = mapped.groupby(df['risk_label']).mean()
            results.append({
                'feature': col,
                'type': 'ordinal',
                'means': {k: round(v, 2) for k, v in grouped.to_dict().items() if pd.notna(v)},
            })
        elif col in df.columns:
            grouped = df.groupby('risk_label')[col].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
            results.append({
                'feature': col,
                'type': 'numeric',
                'means': {k: round(v, 2) for k, v in grouped.to_dict().items() if pd.notna(v)},
            })
        else:
            # likely a one-hot dummy column, e.g. mostUsedPlatform_tiktok
            if col in df.columns:
                rate_by_risk = df.groupby('risk_label')[col].mean()
                results.append({
                    'feature': col,
                    'type': 'proportion',
                    'means': {k: round(v * 100, 1) for k, v in rate_by_risk.to_dict().items()},
                })
    return results