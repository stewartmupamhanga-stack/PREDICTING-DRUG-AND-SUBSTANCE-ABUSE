import os
from django.apps import AppConfig
from django.conf import settings


class PredictorConfig(AppConfig):
    default_app_config = 'predictor.PredictorConfig'
    name = 'predictor'

    model_ever = None
    model_used30 = None
    artifacts = None
    _loaded = False
    _loaded_signature = None

    def ready(self):
        # Don't load models here — this runs for EVERY manage.py command,
        # including the one that creates the artifacts in the first place.
        # Loading happens lazily via ensure_models_loaded() instead.
        pass

    @classmethod
    def ensure_models_loaded(cls, force_reload=False):
        import joblib
        from .ml.preprocessing import load_artifacts

        artifact_dir = os.path.join(settings.BASE_DIR, "predictor", "ml", "artifacts")
        artifacts_path = os.path.join(artifact_dir, "artifacts.json")
        model_ever_path = os.path.join(artifact_dir, "model_ever.joblib")
        model_used30_path = os.path.join(artifact_dir, "model_used30.joblib")

        if not all(os.path.exists(p) for p in [artifacts_path, model_ever_path, model_used30_path]):
            raise FileNotFoundError(
                "Model artifacts not found. Run 'python manage.py train_models --data <path>' first."
            )

        signature = tuple(
            (os.stat(p).st_mtime_ns, os.path.getsize(p))
            for p in [artifacts_path, model_ever_path, model_used30_path]
        )
        if cls._loaded and not force_reload and cls._loaded_signature == signature:
            return

        cls.artifacts = load_artifacts(artifact_dir)
        cls.model_ever = joblib.load(model_ever_path)
        cls.model_used30 = joblib.load(model_used30_path)
        cls._loaded_signature = signature
        cls._loaded = True
