from django.shortcuts import render, redirect
from django.apps import apps
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse, FileResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

import json
import os
import pandas as pd

from .models import SurveyResponse, MLConfig
from .forms import AssessmentForm
from .ml import predictor_service
from .ml.pilot_training import run_pilot
from . import bulk_upload
from .research_stats import (
    get_survey_admin_stats, get_missing_value_report,
    get_class_balance_stats, compute_feature_risk_stats, load_raw_data
)


# ---------------------------------------------------------------------------
# Assessment form + result page
# ---------------------------------------------------------------------------

BAND_BOUNDS = {
    'low':      (0, 33),
    'moderate': (33, 66),
    'high':     (66, 100),
}


def compute_gauge_position(risk_level, confidence):
    band_start, band_end = BAND_BOUNDS[risk_level]
    band_width = band_end - band_start

    confidence = max(50, min(confidence, 100))
    frac = (confidence - 50) / 50

    if risk_level == 'low':
        position = band_end - (band_width * frac)
    elif risk_level == 'high':
        position = band_start + (band_width * frac)
    else:  # moderate — no natural direction, keep centered
        position = band_start + band_width / 2

    return round(position, 1)


def submit_assessment(request):
    if request.method == 'POST':
        form = AssessmentForm(request.POST)
        if form.is_valid():
            prediction = predictor_service.predict_details(form.cleaned_data)
            context = {
                'risk_level': prediction['risk_level'],
                'confidence': round(prediction['confidence'] * 100),
            }

            context['gauge_position'] = compute_gauge_position(
                context['risk_level'], context['confidence']
            )

            return render(request, 'predictor/result.html', context)
    else:
        form = AssessmentForm()

    return render(request, 'predictor/assessment.html', {'form': form})


def result_view(request, pk):
    survey_instance = SurveyResponse.objects.get(pk=pk)

    context = {
        'risk_level': survey_instance.predicted_risk_level,
        'survey': survey_instance,
    }
    return render(request, 'predictor/result.html', context)


# ---------------------------------------------------------------------------
# Research objective pages
# ---------------------------------------------------------------------------

def load_metrics():
    path = os.path.join(settings.BASE_DIR, 'predictor', 'ml', 'artifacts', 'metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# Results transcribed from Chapter 4, Table 4.2. Keep these separate from
# the most recent hyperparameter-search scores stored in metrics.json.
CHAPTER4_MODEL_RESULTS = {
    'sample_size': 468,
    'models': [
        {
            'name': 'Two-stage chained model',
            'target': 'Three-level risk (Low / Moderate / High)',
            'accuracy': 0.85,
            'macro_f1': 0.82,
        },
        {
            'name': 'Sub-model A: ever used',
            'target': 'Ever used (Yes / No)',
            'accuracy': 0.86,
            'macro_f1': 0.84,
        },
        {
            'name': 'Sub-model B: recent use',
            'target': 'Used in last 30 days (Yes / No)',
            'accuracy': 0.82,
            'macro_f1': 0.82,
        },
    ],
}


def build_pilot_display(pilot_metrics):
    """Prepare accuracy and macro-F1 values with their actual CV protocols."""
    if not pilot_metrics:
        return None

    ever = pilot_metrics.get('model_ever', {})
    recent = pilot_metrics.get('model_used30', {})
    ever_winner = ever.get('winner')
    recent_winner = recent.get('winner')
    ever_comparison = ever.get('comparison', {})
    recent_comparison = recent.get('nested_outer_cv_comparison', {})
    ever_result = ever_comparison.get(ever_winner, {})
    recent_result = recent_comparison.get(recent_winner, {})

    ever_candidates = [
        {
            'name': name,
            'accuracy': result.get('cv_accuracy_mean'),
            'macro_f1': result.get('cv_macro_f1_mean'),
            'winner': name == ever_winner,
        }
        for name, result in ever_comparison.items()
    ]
    recent_candidates = [
        {
            'name': name,
            'accuracy': result.get('accuracy'),
            'macro_f1': result.get('macro_f1'),
            'winner': name == recent_winner,
        }
        for name, result in recent_comparison.items()
    ]

    return {
        'trained_at': pilot_metrics.get('trained_at'),
        'ever': {
            'name': ever_winner,
            'accuracy': ever_result.get('cv_accuracy_mean'),
            'macro_f1': ever_result.get('cv_macro_f1_mean'),
            'validation': 'Mean 5-fold CV within the training split',
            'rows': ever_result.get('train_rows'),
        },
        'recent': {
            'name': recent_winner,
            'accuracy': recent_result.get('accuracy'),
            'macro_f1': recent_result.get('macro_f1'),
            'validation': 'Pooled 5-fold outer out-of-fold predictions',
            'rows': pilot_metrics.get('dataset', {}).get('recent_valid_rows'),
        },
        'ever_candidates': ever_candidates,
        'recent_candidates': recent_candidates,
    }


def objective1_survey(request):
    stats = get_survey_admin_stats()
    return render(request, 'predictor/survey_administration.html', {'stats': stats})


def objective2_preprocessing(request):
    missing_report = get_missing_value_report(threshold=5.0)
    balance = get_class_balance_stats()
    balance_rows = [
        {'label': k, 'count': v, 'pct': balance['pct'].get(k, 0)}
        for k, v in balance['counts'].items()
    ]
    return render(request, 'predictor/data_preproccessing.html', {
        'missing_report': missing_report,
        'balance': balance_rows,
    })


def objective3_training(request):
    metrics = load_metrics()
    pilot_config = MLConfig.get_solo()
    context = {
        'metrics': metrics,
        'chapter4_results': CHAPTER4_MODEL_RESULTS,
        'pilot_metrics': pilot_config.pilot_metrics,
        'pilot_display': build_pilot_display(pilot_config.pilot_metrics),
        'pilot_mode_enabled': pilot_config.pilot_mode_enabled,
    }
    return render(request, 'predictor/training_tuning.html', context)


def objective4_evaluation(request):
    metrics = load_metrics()
    return render(request, 'predictor/feature_importance_eval.html', {'metrics': metrics})


def objective5_recommendations(request):
    metrics = load_metrics()
    config = apps.get_app_config('predictor')
    config.ensure_models_loaded()

    top_features = metrics['top_5_features'] if metrics else []
    stats = compute_feature_risk_stats(top_features, config.artifacts) if top_features else []
    labels_by_feature = {
        item['feature']: item.get('label', item['feature'])
        for item in top_features
    }
    for stat in stats:
        stat['label'] = labels_by_feature.get(stat['feature'], stat['feature'])

    return render(request, 'predictor/recommendations.html', {
        'top_features': top_features,
        'stats': stats,
    })


# ---------------------------------------------------------------------------
# ML API endpoints (plain Django, no DRF)
# ---------------------------------------------------------------------------

def _parse_json_body(request):
    """
    Returns (data, error_response). If error_response is not None,
    the caller should return it immediately.
    """
    if not request.body:
        return None, JsonResponse({"detail": "Request body is empty."}, status=400)
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JsonResponse({"detail": "Invalid JSON."}, status=400)
    return data, None


@csrf_exempt
@require_http_methods(["POST"])
def predict_view(request):
    """POST one respondent's raw survey answers -> both predictions back."""
    data, error = _parse_json_body(request)
    if error:
        return error

    if not isinstance(data, dict):
        return JsonResponse(
            {"detail": "Expected a JSON object of raw survey answers."}, status=400
        )

    try:
        result = predictor_service.predict(data)
    except Exception as exc:
        return JsonResponse({"detail": str(exc)}, status=400)

    return JsonResponse(result, status=200)


@require_http_methods(["GET", "POST"])
def pilot_mode_toggle_view(request):
    """GET current state. POST {"enabled": true|false} to switch models."""
    config = MLConfig.get_solo()

    if request.method == "GET":
        return JsonResponse({
            "pilot_mode_enabled": config.pilot_mode_enabled,
            "pilot_model_ever_name": config.pilot_model_ever_name,
            "pilot_model_used30_name": config.pilot_model_used30_name,
            "pilot_trained_at": config.pilot_trained_at,
            "pilot_metrics": config.pilot_metrics,
        })

    data, error = _parse_json_body(request)
    if error:
        return error

    if "enabled" not in data or not isinstance(data["enabled"], bool):
        return JsonResponse(
            {"detail": "Expected JSON body: {\"enabled\": true|false}."}, status=400
        )

    previous_enabled = config.pilot_mode_enabled
    config.pilot_mode_enabled = data["enabled"]
    config.save()

    try:
        predictor_service.get_bundle(force_reload=True)  # apply immediately, no restart needed
    except Exception as exc:
        # Don't leave pilot mode on if there's nothing for it to load —
        # revert and tell the caller why.
        config.pilot_mode_enabled = previous_enabled
        config.save()
        return JsonResponse({"detail": str(exc)}, status=400)

    return JsonResponse({
        "pilot_mode_enabled": config.pilot_mode_enabled,
        "pilot_model_ever_name": config.pilot_model_ever_name,
        "pilot_model_used30_name": config.pilot_model_used30_name,
    })


@staff_member_required
@require_http_methods(["POST"])
def train_pilot_view(request):
    """
    POST {"data_path": "..."} to race the candidate models and switch
    pilot mode on automatically once a winner is picked for each target.
    Runs synchronously (fine for this dataset size) — move to Celery/RQ
    if the training set grows enough that this starts blocking requests.
    """
    data, error = _parse_json_body(request)
    if error:
        return error

    data_path = data.get("data_path")
    if not data_path or not isinstance(data_path, str):
        return JsonResponse(
            {"detail": "Expected JSON body: {\"data_path\": \"<path to CSV>\"}."},
            status=400,
        )

    # The web training endpoint is limited to the configured study dataset.
    # Use the management command for any separately authorized research file.
    requested_path = os.path.normcase(os.path.realpath(data_path))
    approved_path = os.path.normcase(os.path.realpath(settings.SURVEY_DATA_CSV))
    if requested_path != approved_path:
        return JsonResponse(
            {"detail": "Web pilot training is restricted to the configured study dataset."},
            status=400,
        )

    try:
        metrics = run_pilot(data_path)
    except Exception as exc:
        return JsonResponse({"detail": str(exc)}, status=400)

    predictor_service.get_bundle(force_reload=True)

    return JsonResponse(metrics, status=200)


# ---------------------------------------------------------------------------
# Bulk survey upload (sample template download + validate-and-store)
# ---------------------------------------------------------------------------

def download_sample_view(request):
    """Serves sample_template.xlsx, generating it first if it doesn't exist yet."""
    bulk_upload.ensure_sample_file_exists()
    return FileResponse(
        open(bulk_upload.SAMPLE_PATH, "rb"),
        as_attachment=True,
        filename="sample_template.xlsx",
    )

'''

def bulk_upload_view(request):
    """
    POST-only: validates an uploaded CSV/XLSX/XLS against REQUIRED_COLUMNS
    and, if it matches, copies the raw file into bulk_upload.SCRAPE_DIR.
    Redirects back to wherever the form was submitted from (via the
    hidden 'next' field), using Django messages to report success/failure.
    """
    redirect_to = request.POST.get("next") or "home"

    if request.method != "POST":
        return redirect(redirect_to)

    uploaded_file = request.FILES.get("bulk_file")
    if not uploaded_file:
        messages.error(request, "Please choose a file to upload.")
        return redirect(redirect_to)

    try:
        df = bulk_upload.read_uploaded_file(uploaded_file)
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect(redirect_to)

    is_valid, missing, unexpected = bulk_upload.validate_columns(df)
    if not is_valid:
        problems = []
        if missing:
            problems.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected columns: {', '.join(unexpected)}")
        messages.error(
            request,
            "Upload rejected — " + "; ".join(problems) +
            ". Download the sample template to check the required format."
        )
        return redirect(redirect_to)

    uploaded_file.seek(0)  # pandas consumed the stream while reading — reset before saving raw bytes
    saved_path = bulk_upload.save_to_scrape_folder(uploaded_file)

    messages.success(
        request,
        f"Uploaded {len(df)} row(s) successfully. Saved as {os.path.basename(saved_path)}."
    )
    return redirect(redirect_to)'''





# --- Replaces bulk_upload_view in bulk_uploaded_view.py, and adds three
#     new views (put them in the same file or wherever your other views live) ---

import csv
import io
import os

from django.http import JsonResponse
from django.urls import reverse

from predictor.models import BulkBatch, BulkAssessment
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.core.mail import EmailMessage

from predictor import bulk_upload
from predictor.models import BulkBatch, BulkAssessment, ForwardLog


def _actor_label(request):
    """Best-effort 'who did this' label for audit logs, with or without auth wired up."""
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        return getattr(user, "email", None) or user.get_username()
    return request.META.get("REMOTE_ADDR", "unknown")


# --- Replaces the bulk_upload_view from views_additions.py -------------
# (bulk_result_view / bulk_result_download_view / bulk_result_forward_view
#  are unchanged — those stay real pages, this only affects the upload
#  and confirm steps)



def bulk_upload_view(request):
    """
    JSON API for the two-stage bulk upload, driven entirely by fetch()
    from the frontend so the choice between Save-only and Predict & Save
    can be shown as an inline bar rather than a page navigation.

    stage=upload  -> validates the file, stages it to a temp path,
                      returns {status, token, original_name, row_count}
                      for the frontend to render the choice.
    stage=confirm -> re-reads the staged file by token and acts on the
                      chosen option. 'save_only' returns a message;
                      'predict_save' returns a redirect_url to the real
                      results page.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Invalid request method."}, status=405)

    stage = request.POST.get("stage", "upload")

    if stage == "upload":
        uploaded_file = request.FILES.get("bulk_file")
        if not uploaded_file:
            return JsonResponse({"status": "error", "message": "Please choose a file to upload."}, status=400)

        try:
            df = bulk_upload.read_uploaded_file(uploaded_file)
        except ValueError as exc:
            return JsonResponse({"status": "error", "message": str(exc)}, status=400)

        is_valid, missing, unexpected = bulk_upload.validate_columns(df)
        if not is_valid:
            problems = []
            if missing:
                problems.append(f"missing columns: {', '.join(missing)}")
            if unexpected:
                problems.append(f"unexpected columns: {', '.join(unexpected)}")
            return JsonResponse({
                "status": "error",
                "message": "Upload rejected — " + "; ".join(problems) +
                            ". Download the sample template to check the required format.",
            }, status=400)

        uploaded_file.seek(0)
        token, original_name = bulk_upload.save_temp_upload(uploaded_file)

        return JsonResponse({
            "status": "ok",
            "token": token,
            "original_name": original_name,
            "row_count": len(df),
        })

    elif stage == "confirm":
        token = request.POST.get("token")
        original_name = request.POST.get("original_name")
        action = request.POST.get("action")

        if not token or not original_name:
            return JsonResponse({"status": "error", "message": "Upload session expired. Please upload the file again."}, status=400)

        temp_path = bulk_upload.get_temp_upload_path(token, original_name)
        if not os.path.exists(temp_path):
            return JsonResponse({"status": "error", "message": "Upload session expired. Please upload the file again."}, status=400)

        try:
            df = bulk_upload.read_uploaded_file_from_path(temp_path)
        except ValueError as exc:
            bulk_upload.cleanup_temp_upload(temp_path)
            return JsonResponse({"status": "error", "message": str(exc)}, status=400)

        if action == "save_only":
            saved_path = bulk_upload.save_path_to_scrape_folder(temp_path, original_name)
            bulk_upload.cleanup_temp_upload(temp_path)

            BulkBatch.objects.create(
                original_filename=original_name,
                saved_path=saved_path,
                total_rows=len(df),
                mode="save_only",
                uploaded_by=_actor_label(request),
            )

            return JsonResponse({
                "status": "ok",
                "message": f"Saved {len(df)} row(s) as {os.path.basename(saved_path)}.",
            })

        elif action == "predict_save":
            saved_path = bulk_upload.save_path_to_scrape_folder(temp_path, original_name)
            bulk_upload.cleanup_temp_upload(temp_path)

            batch = BulkBatch.objects.create(
                original_filename=original_name,
                saved_path=saved_path,
                total_rows=len(df),
                mode="predict_save",
                uploaded_by=_actor_label(request),
            )

            results = bulk_upload.run_predictions(df)
            BulkAssessment.objects.bulk_create([
                BulkAssessment(
                    batch=batch,
                    external_id=str(r["id"]) if r.get("id") not in (None, "") else "",
                    risk_level=r.get("risk_level"),
                    used_30_days=r.get("used_30_days"),
                    error=r.get("error"),
                    raw_data=r.get("raw"),
                )
                for r in results
            ])

            failed = sum(1 for r in results if r.get("error"))

            return JsonResponse({
                "status": "ok",
                "redirect_url": reverse("bulk-result", args=[batch.id]),
                "failed_count": failed,
                "total": len(results),
            })

        else:
            return JsonResponse({"status": "error", "message": "Unknown action."}, status=400)

    else:
        return JsonResponse({"status": "error", "message": "Unknown upload stage."}, status=400)


# _actor_label is unchanged from views_additions.py — keep that function
# as-is, this view still calls it.


def bulk_result_view(request, batch_id):
    batch = get_object_or_404(BulkBatch, pk=batch_id)
    rows = batch.rows.all().order_by("id")
    return render(request, "predictor/result.html", {"batch": batch, "rows": rows})


def bulk_result_download_view(request, batch_id):
    batch = get_object_or_404(BulkBatch, pk=batch_id)
    rows = batch.rows.all().order_by("id")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="batch_{batch_id}_results.csv"'
    writer = csv.writer(response)
    writer.writerow(["ID", "Risk level", "Used in last 30 days", "Error"])
    for r in rows:
        writer.writerow([r.external_id, r.risk_level or "", "Yes" if r.used_30_days else ("No" if r.used_30_days is not None else ""), r.error or ""])
    return response


def bulk_result_forward_view(request, batch_id):
    """
    Emails the batch's CSV to a recipient and ALWAYS writes a ForwardLog
    row — this endpoint should never send without recording who sent
    what, to whom, and when.
    """
    batch = get_object_or_404(BulkBatch, pk=batch_id)
    if request.method != "POST":
        return redirect("bulk-result", batch_id=batch_id)

    recipient = request.POST.get("recipient_email", "").strip()
    if not recipient:
        messages.error(request, "Enter a recipient email address.")
        return redirect("bulk-result", batch_id=batch_id)

    rows = batch.rows.all().order_by("id")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Risk level", "Used in last 30 days"])
    for r in rows:
        writer.writerow([r.external_id, r.risk_level or "", "Yes" if r.used_30_days else ("No" if r.used_30_days is not None else "")])

    email = EmailMessage(
        subject=f"Risk assessment results — batch {batch.id}",
        body="Attached are the requested risk assessment results.",
        to=[recipient],
    )
    email.attach(f"batch_{batch.id}_results.csv", buf.getvalue(), "text/csv")
    email.send()

    ForwardLog.objects.create(
        batch=batch,
        sent_by=_actor_label(request),
        sent_to=recipient,
    )

    messages.success(request, f"Results forwarded to {recipient}.")
    return redirect("bulk-result", batch_id=batch_id)














