import os
import json
import numpy as np
import pandas as pd


def load_artifacts(artifact_dir: str) -> dict:
    with open(os.path.join(artifact_dir, "artifacts.json")) as f:
        return json.load(f)


def preprocess_raw_input(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Takes a DataFrame of raw survey responses and returns a feature matrix
    X that EXACTLY matches what the models were trained on in the
    `train_risk_models` management command / the notebook.

    Kept 1:1 with your ipynb version, with one fix: the notebook's training
    script maps mostUsedDrug's "no" -> NaN before building artifacts.json,
    but that step was missing from the pasted preprocess_raw_input — added
    back below (tf["mostUsedDrug_no_to_nan"]) so live requests match training.
    """
    df = raw_df.copy()

    tf = artifacts["typo_fixes"]
    if "age" in df.columns:
        df["age"] = df["age"].replace(tf["age"])
    if "strugle12MonthsHead" in df.columns:
        df["strugle12MonthsHead"] = (
            df["strugle12MonthsHead"].astype(str).str.strip().replace(tf["strugle12MonthsHead"])
        )
    if "parentsWorkedDiaspora6Months" in df.columns:
        df["parentsWorkedDiaspora6Months"] = df["parentsWorkedDiaspora6Months"].replace(
            tf["parentsWorkedDiaspora6Months"]
        )
    for col in ["comfortableSharingPersonalProblems", "eatDinnerWithFamily", "guadianKnowMyWhereabouts"]:
        if col in df.columns:
            df[col] = df[col].replace(tf["likert_typo"])
    if "howFriendsFeelTryingDrugs" in df.columns:
        df["howFriendsFeelTryingDrugs"] = df["howFriendsFeelTryingDrugs"].replace(tf["likert_typo"])
    if "participationInOrganizedProsocialActivitiesInAWeek" in df.columns:
        df["participationInOrganizedProsocialActivitiesInAWeek"] = df[
            "participationInOrganizedProsocialActivitiesInAWeek"
        ].replace(tf["participation_zero"])
    if "main_activity" in df.columns:
        df["main_activity"] = df["main_activity"].replace(tf["main_activity"])
    if "mostUsedDrug" in df.columns and tf.get("mostUsedDrug_no_to_nan"):
        df["mostUsedDrug"] = df["mostUsedDrug"].replace({"no": np.nan})
    if "province" in df.columns:
        df["province"] = df["province"].replace({p: "other" for p in tf["rare_provinces_to_other"]})

    for col, mapping in artifacts["ordinal_maps"].items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)

    nominal_cols = [c for c in artifacts["nominal_cols"] if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    for col, median_val in artifacts["saved_medians"].items():
        flag_col = f"{col}_missing"
        if col in df.columns:
            df[flag_col] = df[col].isna().astype(int)
            df[col] = df[col].fillna(median_val)
        else:
            # column wasn't provided at all in this request -> treat as missing
            df[col] = median_val
            df[flag_col] = 1

    drop_now = artifacts["leakage_cols"] + artifacts["unimportant_features"]
    df = df.drop(columns=[c for c in drop_now if c in df.columns], errors="ignore")

    final_cols = artifacts["final_feature_columns"]
    df = df.reindex(columns=final_cols, fill_value=0)

    return df