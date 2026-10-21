import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .ml import predictor_service



def _get_bundle_input(request):
    bundle = predictor_service.get_bundle()
    raw_df = pd.DataFrame([request.data])
    return bundle, predictor_service.prepare_input(raw_df, bundle)


def _predict_ever(X, model_ever):
    pred = model_ever.predict(X)[0]
    proba = model_ever.predict_proba(X)[0]
    classes = list(model_ever.classes_)
    return {
        "prediction": "ever_used" if pred == 1 else "never_used",
        "probability_ever_used": float(proba[classes.index(1)]),
    }


def _predict_used30(X, model_used30, threshold=None):
    proba = model_used30.predict_proba(X)[0]
    classes = list(model_used30.classes_)
    pred = (int(proba[classes.index(1)] >= threshold)
            if threshold is not None else model_used30.predict(X)[0])
    return {
        "prediction": "used_last_30_days" if pred == 1 else "not_used_last_30_days",
        "probability_used_last_30_days": float(proba[classes.index(1)]),
    }


class PredictEverUsedView(APIView):
    """Model A standalone — predicts 'ever used illegal drugs'."""

    def post(self, request):
        try:
            bundle, X = _get_bundle_input(request)
            result = _predict_ever(X, bundle["model_ever"])
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class PredictUsed30View(APIView):
    """
    Model B standalone. Trained only on the ever-used subset —
    calling it on a never-used person is out of its training
    distribution. Use PredictRiskView for the full, correct chain.
    """

    def post(self, request):
        try:
            bundle, X = _get_bundle_input(request)
            result = _predict_used30(
                X, bundle["model_used30"], bundle.get("used30_threshold")
            )
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class PredictRiskView(APIView):
    """
    Combined chain: Model A decides ever_used; Model B only runs
    if yes, to split moderate vs high. Full low/moderate/high output.
    """

    def post(self, request):
        try:
            bundle, X = _get_bundle_input(request)
            ever_result = _predict_ever(X, bundle["model_ever"])

            if ever_result["prediction"] == "never_used":
                return Response({
                    "risk_level": "low",
                    "ever_used": ever_result,
                    "used_30_days": None,
                })

            used30_result = _predict_used30(
                X, bundle["model_used30"], bundle.get("used30_threshold")
            )
            risk_level = "high" if used30_result["prediction"] == "used_last_30_days" else "moderate"

            return Response({
                "risk_level": risk_level,
                "ever_used": ever_result,
                "used_30_days": used30_result,
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ArtifactsInfoView(APIView):
    """Expose mappings and dropped columns for your interface to inspect."""

    def get(self, request):
        bundle = predictor_service.get_bundle()
        a = bundle["artifacts"]
        return Response({
            "input_mode": bundle.get("input_mode", "preprocessed"),
            "ordinal_maps": a.get("ordinal_maps", {}),
            "nominal_cols": a.get("nominal_cols", []),
            "unimportant_features_removed": a.get("unimportant_features", []),
            "leakage_cols_removed": a.get("leakage_cols", []),
            "expected_input_fields": list(a.get("ordinal_maps", {}).keys()) + a.get("nominal_cols", []),
            "final_feature_count": len(a.get("final_feature_columns", [])),
        })
