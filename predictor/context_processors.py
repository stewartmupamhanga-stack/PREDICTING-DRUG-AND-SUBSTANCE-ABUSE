from .models import MLConfig


def model_mode(request):
    """Expose the persisted model selection to every page extending base.html."""
    config = MLConfig.objects.filter(pk=1).only(
        "pilot_mode_enabled",
        "pilot_model_ever_name",
        "pilot_model_used30_name",
    ).first()

    if config is None:
        return {
            "pilot_mode_enabled": False,
            "pilot_model_ever_name": "",
            "pilot_model_used30_name": "",
        }

    return {
        "pilot_mode_enabled": config.pilot_mode_enabled,
        "pilot_model_ever_name": config.pilot_model_ever_name or "",
        "pilot_model_used30_name": config.pilot_model_used30_name or "",
    }
