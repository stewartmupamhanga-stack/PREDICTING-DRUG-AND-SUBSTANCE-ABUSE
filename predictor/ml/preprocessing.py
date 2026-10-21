import os
import json

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Cleaning and encoding follow resultsAndFindings.ipynb (DATA PREPARATION,
# up to the first Random Forest fit): same typo fixes, same integer codes
# for ordinal answers, same mode imputation. The one addition is the last
# step, which shapes the result into the columns listed in
# artifacts["final_feature_columns"] (one-hot columns for nominal answers
# and *_missing flags), because that is what the saved models expect.
# ----------------------------------------------------------------------

# The notebook's 20 predictors, in its column order. Used as the output
# only when artifacts.json has no final_feature_columns.
FEATURE_COLUMNS = [
    "age", "gender", "highestEducation", "main_activity", "monthlyIncomeTotal",
    "employStatusHead", "strugle12MonthsHead", "livingSitu18Years",
    "parentsWorkedDiaspora6Months", "relativeProblemWithDrugs",
    "comfortableSharingPersonalProblems", "eatDinnerWithFamily",
    "guadianKnowMyWhereabouts", "outOf5FriendsUseDrugs", "howFriendsFeelTryingDrugs",
    "unsupervisedEveningsInAWeek", "participationInOrganizedProsocialActivitiesInAWeek",
    "hoursSpentOnSocialMedia", "mostUsedPlatform", "howTimesSeenDrugSupportContentIn30Days",
]

# Columns that become one-hot columns (e.g. gender_male) when the model's
# feature list contains such columns. artifacts["nominal_cols"] overrides.
CANDIDATE_NOMINAL_COLS = [
    "province", "gender", "main_activity", "employStatusHead", "livingSitu18Years",
    "parentsWorkedDiaspora6Months", "relativeProblemWithDrugs", "mostUsedPlatform",
]

# Category merges the one-hot feature set relies on. artifacts["typo_fixes"]
# overrides each of these if it has the same key.
DEFAULT_NOMINAL_FIXES = {
    "main_activity": {"formallyEmployed": "employed"},
    "parentsWorkedDiaspora6Months": {"yesLesftWithGuardian": "yesLeftWithGuardian"},
}
DEFAULT_RARE_PROVINCES = ["masvingo", "bulawayo", "mashCentral", "mashWest", "matNorth"]

_LIKERT_AGREE = {"stronglyDisagree": -2, "disagree": -1, "neutral": 0, "agree": 1, "stronglyAgree": 2}

# Integer codes copied 1:1 from the notebook.
ORDINAL_MAPS = {
    "age": {"21_24": 1, "18_20": 0, "25_29": 2, "30_35": 3},
    "gender": {"female": 0, "male": 1},
    "highestEducation": {"tetiary": 0, "aLevel": 1, "ordinaryLevel": 2, "someSecondary": 3},
    "main_activity": {
        "student": 0, "selfEmployed": 1, "employed": 2, "unemployedSeekingWork": 3,
        "pieceJob": 4, "formallyEmployed": 5, "neet": 6,
    },
    "monthlyIncomeTotal": {"251_500": 2, "500+": 3, "preferNotToSay": -1, "101_250": 1, "100-": 0},
    "employStatusHead": {"employed": 0, "selfEmployed": 1, "retired": 2, "unemployed": 3, "preferNotToSay": -1},
    "strugle12MonthsHead": {"rarely": 1, "never": 0, "sometimes": 2, "always": 3, "often": 4},
    "livingSitu18Years": {"bothBiologicalParents": 0, "livingWithSingleParent": 1, "livingWithGuardian": 2, "other": 3},
    "parentsWorkedDiaspora6Months": {
        "noLivedWithMe": 0, "yesLeftWithOtherParent": 1, "notApplicable": 2,
        "yesLesftWithGuardian": 3, "yesLeftWithGuardian": 4,
    },
    "relativeProblemWithDrugs": {"never": 0, "yesInThePast": 1, "yesCurrently": 2, "preferNotToSay": -1},
    "comfortableSharingPersonalProblems": _LIKERT_AGREE,
    "eatDinnerWithFamily": _LIKERT_AGREE,
    "guadianKnowMyWhereabouts": _LIKERT_AGREE,
    "outOf5FriendsUseDrugs": {"0": 0, "1_2": 1, "3_4": 2, "5": 3},
    "howFriendsFeelTryingDrugs": {
        "stronglyDisapprove": 0, "disapprove": 1, "neutral": 2, "approve": 3, "stronglyApprove": 4,
    },
    "unsupervisedEveningsInAWeek": {"0": 0, "1_2": 1, "3_4": 2, "5+": 3},
    "participationInOrganizedProsocialActivitiesInAWeek": {"0": 0, "1_2": 1, "3_4": 2, "5+": 3, "never": -1},
    "hoursSpentOnSocialMedia": {"1-": -1, "1_3": 0, "3_5": 1, "5+": 2},
    "mostUsedPlatform": {
        "whatsapp": 0, "tiktok": 1, "facebook": 2, "instagram": 3, "other": 4, "youtube": 5, "telegram": 6,
    },
    "howTimesSeenDrugSupportContentIn30Days": {"rarely": 0, "sometimes": 1, "often": 2, "daily": 3, "never": -1},
}

# Fill values for missing answers. The notebook fills with the column mode;
# a single live request has no mode of its own, so the training-set modes
# (read off the notebook's value_counts output, 136 rows) are stored here.
# artifacts["saved_modes"] and then artifacts["saved_medians"] override them.
TRAINING_MODES = {
    "age": 1, "gender": 1, "highestEducation": 0, "main_activity": 0,
    "monthlyIncomeTotal": 2, "employStatusHead": 0, "strugle12MonthsHead": 1,
    "livingSitu18Years": 0, "parentsWorkedDiaspora6Months": 0, "relativeProblemWithDrugs": 0,
    "comfortableSharingPersonalProblems": 1, "eatDinnerWithFamily": 1,
    "guadianKnowMyWhereabouts": 1, "outOf5FriendsUseDrugs": 0, "howFriendsFeelTryingDrugs": 0,
    "unsupervisedEveningsInAWeek": 1, "participationInOrganizedProsocialActivitiesInAWeek": 1,
    "hoursSpentOnSocialMedia": 2, "mostUsedPlatform": 0, "howTimesSeenDrugSupportContentIn30Days": 0,
}


def load_artifacts(artifact_dir: str) -> dict:
    with open(os.path.join(artifact_dir, "artifacts.json")) as f:
        return json.load(f)


def _as_clean_str(series: pd.Series) -> pd.Series:
    """Text used for dictionary lookups; unmatched or missing values map to NaN.
    Whole-number floats (5.0) are turned back into '5'."""
    s = series.astype(str).str.strip()
    return s.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)


def preprocess_raw_input(raw_df: pd.DataFrame, artifacts: dict | None = None) -> pd.DataFrame:
    """
    Raw survey responses -> feature matrix in the column order of
    artifacts["final_feature_columns"].

    Steps (notebook method):
      1. typo fixes on the raw text
      2. integer coding of ordinal answers with the notebook's maps
      3. mode imputation for missing answers
    then, only because the saved models ask for them:
      4. one-hot columns for nominal answers (province_harare, gender_male, ...)
      5. *_missing flags, set before imputation
    """
    artifacts = artifacts or {}
    df = raw_df.copy()

    final_given = bool(artifacts.get("final_feature_columns"))
    final_cols = artifacts["final_feature_columns"] if final_given else FEATURE_COLUMNS

    # Which columns become one-hot. Without a final column list, everything
    # stays integer-coded exactly as in the notebook.
    if final_given:
        nominal_cols = artifacts.get("nominal_cols") or [
            c for c in CANDIDATE_NOMINAL_COLS if any(f.startswith(c + "_") for f in final_cols)
        ]
    else:
        nominal_cols = []
    nominal_set = set(nominal_cols)

    tf = artifacts.get("typo_fixes", {})

    # ---- 1. typo fixes (notebook cells)
    if "age" in df.columns:
        df["age"] = (
            _as_clean_str(df["age"]).str.replace("-", "_", regex=False).str.replace("30_36", "30_35", regex=False)
        )
    if "strugle12MonthsHead" in df.columns:
        df["strugle12MonthsHead"] = _as_clean_str(df["strugle12MonthsHead"]).str.lower()
    for col in ["comfortableSharingPersonalProblems", "eatDinnerWithFamily", "guadianKnowMyWhereabouts"]:
        if col in df.columns:
            df[col] = _as_clean_str(df[col]).str.replace("stronlyDisagree", "stronglyDisagree", regex=False)
    if "howFriendsFeelTryingDrugs" in df.columns:
        df["howFriendsFeelTryingDrugs"] = _as_clean_str(df["howFriendsFeelTryingDrugs"]).str.replace(
            "stronlyDisapprove", "stronglyDisapprove", regex=False
        )

    # ---- 2. integer coding (ordinal columns only)
    for col, mapping in ORDINAL_MAPS.items():
        if col in nominal_set:
            continue
        if col in df.columns:
            df[col] = _as_clean_str(df[col]).map(mapping)
        else:
            df[col] = np.nan  # not supplied in this request -> imputed below

    # "preferNotToSay" income is one of the answers the model treats as missing
    # whenever it was given a monthlyIncomeTotal_missing flag.
    if "monthlyIncomeTotal_missing" in final_cols and "monthlyIncomeTotal" in df.columns:
        df.loc[df["monthlyIncomeTotal"] == -1, "monthlyIncomeTotal"] = np.nan

    # ---- 5. *_missing flags (must be set before imputation)
    for flag in [c for c in final_cols if c.endswith("_missing")]:
        base = flag[: -len("_missing")]
        if base in df.columns:
            df[flag] = df[base].isna().astype(int)

    # ---- 3. imputation
    fills = {**TRAINING_MODES, **artifacts.get("saved_modes", {}), **artifacts.get("saved_medians", {})}
    for col in ORDINAL_MAPS:
        if col in nominal_set or col not in df.columns:
            continue
        if col in fills:
            df[col] = df[col].fillna(fills[col])

    # ---- 4. one-hot columns for nominal answers
    present_nominal = [c for c in nominal_cols if c in df.columns]
    for col in present_nominal:
        s = df[col].astype("string").str.strip()
        if col == "province":
            rare = tf.get("rare_provinces_to_other", DEFAULT_RARE_PROVINCES)
            s = s.replace({p: "other" for p in rare})
        else:
            s = s.replace(tf.get(col, DEFAULT_NOMINAL_FIXES.get(col, {})))
        df[col] = s
    if present_nominal:
        df = pd.get_dummies(df, columns=present_nominal, prefix_sep="_", dtype=int)

    # ---- return what the model asks for
    dummy_prefixes = tuple(c + "_" for c in nominal_cols)
    unavailable = [c for c in final_cols if c not in df.columns and not c.startswith(dummy_prefixes)]
    if unavailable:
        raise ValueError(
            f"artifacts['final_feature_columns'] lists columns this preprocessing cannot build: {unavailable}"
        )
    # Categories that don't occur in this request have no dummy column yet -> 0.
    return df.reindex(columns=final_cols, fill_value=0).astype(float)