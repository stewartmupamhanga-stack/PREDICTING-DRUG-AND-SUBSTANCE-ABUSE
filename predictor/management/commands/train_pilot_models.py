from django.core.management.base import BaseCommand
from predictor.ml.pilot_training import run_pilot


class Command(BaseCommand):
    help = "Runs the notebook's lifetime and nested 30-day model comparisons; saves pilot-only models and metrics."

    def add_arguments(self, parser):
        parser.add_argument('--data', type=str, required=True, help='Path to the training CSV.')

    def handle(self, *args, **options):
        metrics = run_pilot(options['data'])
        self.stdout.write(self.style.SUCCESS(
            f"Model A winner: {metrics['model_ever']['winner']} "
            f"(f1_macro={metrics['model_ever']['cv_f1_macro']:.3f})"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Model B winner: {metrics['model_used30']['winner']} "
            f"(f1_macro={metrics['model_used30']['cv_f1_macro']:.3f})"
        ))
