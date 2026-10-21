import os
from datetime import datetime

import pandas as pd
from django.conf import settings


REQUIRED_COLUMNS = [
    "id", "date", "province", "age", "gender", "highestEducation", "main_activity",
    "monthlyIncomeTotal", "employStatusHead", "strugle12MonthsHead", "livingSitu18Years",
    "parentsWorkedDiaspora6Months", "relativeProblemWithDrugs", "comfortableSharingPersonalProblems",
    "eatDinnerWithFamily", "guadianKnowMyWhereabouts", "outOf5FriendsUseDrugs", "howFriendsFeelTryingDrugs",
    "unsupervisedEveningsInAWeek", "participationInOrganizedProsocialActivitiesInAWeek",
    "hoursSpentOnSocialMedia", "mostUsedPlatform", "howTimesSeenDrugSupportContentIn30Days",
    "everUsedIllegalDrugs", "howManyTimesYouUsedLast30Days", "mostUsedDrug", "drugsUsed",
]

SAMPLE_DIR = os.path.join(settings.BASE_DIR, "predictor", "ml", "sample_data")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "sample_template.xlsx")

# Rename this if "scrape" was meant to be something else — this is the
# folder every successfully validated upload gets copied into.
SCRAPE_DIR = os.path.join(settings.BASE_DIR, "predictor", "ml", "pred_results")


def ensure_sample_file_exists():
    """
    Creates the downloadable sample_template.xlsx (header row only, with
    one blank example row so the format is self-explanatory) if it
    doesn't already exist. Requires openpyxl (pip install openpyxl).
    """
    if os.path.exists(SAMPLE_PATH):
        return
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    example_row = {col: "" for col in REQUIRED_COLUMNS}
    pd.DataFrame([example_row], columns=REQUIRED_COLUMNS).to_excel(SAMPLE_PATH, index=False)


def read_uploaded_file(uploaded_file):
    """
    Reads an uploaded CSV/XLSX/XLS into a DataFrame.
    Raises ValueError with a user-facing message on failure.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Upload a .csv, .xlsx, or .xls file.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}")


def validate_columns(df: pd.DataFrame):
    """
    Returns (is_valid, missing_columns, unexpected_columns), comparing
    the uploaded file's columns against REQUIRED_COLUMNS. Column order
    doesn't matter, but every required column must be present and no
    extra/misspelled columns are allowed.
    """
    uploaded_cols = set(df.columns)
    required_cols = set(REQUIRED_COLUMNS)

    missing = sorted(required_cols - uploaded_cols)
    unexpected = sorted(uploaded_cols - required_cols)

    is_valid = not missing and not unexpected
    return is_valid, missing, unexpected


def save_to_scrape_folder(uploaded_file):
    """
    Saves the raw uploaded file into SCRAPE_DIR with a timestamp
    prefixed onto its original name, so repeated uploads never
    overwrite each other. Returns the saved path.

    Call uploaded_file.seek(0) before this if the file was already
    read (e.g. by pandas for validation) earlier in the same request.
    """
    os.makedirs(SCRAPE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.basename(uploaded_file.name)  # strip any path components
    saved_path = os.path.join(SCRAPE_DIR, f"{timestamp}_{safe_name}")

    with open(saved_path, "wb") as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)

    return saved_path




# (keeps your existing read_uploaded_file / validate_columns / save_to_scrape_folder
#  as-is; these are new functions alongside them)

import shutil
import uuid

import numpy as np

TEMP_UPLOAD_DIR = os.path.join(settings.BASE_DIR, "predictor", "ml", "tmp_uploads")


def save_temp_upload(uploaded_file):
    """
    Streams the uploaded file to a temp path keyed by a random token, so
    it can be re-read on the confirm-step POST (a file input can't be
    redisplayed after a redirect). Returns (token, original_filename).
    """
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    token = uuid.uuid4().hex
    ext = os.path.splitext(uploaded_file.name)[1]
    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{token}{ext}")
    with open(temp_path, "wb+") as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)
    return token, uploaded_file.name


def get_temp_upload_path(token, original_name):
    ext = os.path.splitext(original_name)[1]
    return os.path.join(TEMP_UPLOAD_DIR, f"{token}{ext}")


def cleanup_temp_upload(path):
    try:
        os.remove(path)
    except OSError:
        pass


def read_uploaded_file_from_path(path):
    """Path-based counterpart to read_uploaded_file, for re-reading the temp file."""
    name = path.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(path)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Upload a .csv, .xlsx, or .xls file.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}")


def save_path_to_scrape_folder(temp_path, original_name):
    """
    Path-based counterpart to save_to_scrape_folder, for the confirm-step
    flow where the file is sitting in TEMP_UPLOAD_DIR rather than still
    being an in-request UploadedFile. Mirrors the same timestamp-prefix
    convention so bulk uploads never collide with each other or with
    single uploads of the same original filename.
    """
    os.makedirs(SCRAPE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.basename(original_name)
    dest_path = os.path.join(SCRAPE_DIR, f"{timestamp}_{safe_name}")

    shutil.copy(temp_path, dest_path)
    return dest_path


def _json_safe(value):
    """
    Converts a single cell value from df.iterrows() into something
    SQLite's JSON_VALID() will accept: NaN/NaT -> None, numpy scalar
    types -> native Python types. Without this, pandas' NaN for missing
    cells gets written as the bare token NaN by json.dumps (Python
    allows it, strict JSON does not), and SQLite's JSONField CHECK
    constraint rejects the whole insert.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_predictions(df: pd.DataFrame):
    """
    Runs predictor_service.predict() once per row. Returns a list of dicts:
    {"id": <external id or None>, "risk_level", "used_30_days", "error", "raw"}.
    A single bad row (e.g. an out-of-range value) is caught and marked as
    an error row rather than aborting the whole batch.
    """
    from predictor.ml import predictor_service  # local import avoids import cycles

    results = []
    for _, row in df.iterrows():
        raw = {k: _json_safe(v) for k, v in row.to_dict().items()}
        row_id = raw.pop("id", None)
        try:
            pred = predictor_service.predict(raw)
            results.append({
                "id": row_id,
                "risk_level": pred.get("risk_level"),
                "used_30_days": pred.get("used_30_days"),
                "error": None,
                "raw": raw,
            })
        except Exception as exc:
            results.append({
                "id": row_id,
                "risk_level": None,
                "used_30_days": None,
                "error": str(exc),
                "raw": raw,
            })
    return results