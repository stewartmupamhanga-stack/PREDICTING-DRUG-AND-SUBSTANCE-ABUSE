from django.db import models

class SurveyResponse(models.Model):
    # Demographics & Economic
    age = models.CharField(max_length=10, choices=[('15-17', '15-17'), ('18-20', '18-20'), ('21_24', '21-24'), ('25_29', '25-29'), ('30-35', '30-35')])
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female')])
    highest_education = models.CharField(max_length=20, choices=[('ordinaryLevel', 'O Level'), ('aLevel', 'A Level'), ('tetiary', 'Tertiary'), ('someSecondary', 'Some Secondary')])
    main_activity = models.CharField(max_length=20, choices=[('student', 'Student'), ('pieceJob', 'Piece Job'), ('employed', 'Employed'), ('unemployed', 'Unemployed')])
    monthly_income = models.CharField(max_length=20, choices=[('0_100', '0-100'), ('101_250', '101-250'), ('251_500', '251-500'), ('500+', '500+'), ('preferNotToSay', 'Prefer not to say')])
    employ_status_head = models.CharField(max_length=20, choices=[('employed', 'Employed'), ('selfEmployed', 'Self Employed'), ('unemployed', 'Unemployed'), ('retired', 'Retired')])
    
    # Family & Peer Environment[cite: 1]
    struggle_12_months = models.CharField(max_length=15, choices=[('never', 'Never'), ('rarely', 'Rarely'), ('sometimes', 'Sometimes'), ('often', 'Often')])
    friends_feel_drugs = models.CharField(max_length=25, choices=[('stronglyDisapprove', 'Strongly Disapprove'), ('disapprove', 'Disapprove'), ('neutral', 'Neutral'), ('approve', 'Approve'), ('stronglyApprove', 'Strongly Approve')])
    unsupervised_evenings = models.CharField(max_length=10, choices=[('0', '0'), ('1_2', '1-2'), ('3_4', '3-4'), ('5+', '5+')])
    prosocial_activities = models.CharField(max_length=10, choices=[('never', 'Never'), ('1_2', '1-2'), ('3_4', '3-4'), ('5+', '5+')])
    
    # Digital Behavior[cite: 1]
    hours_social_media = models.CharField(max_length=10, choices=[('1-', 'Less than 1'), ('1_3', '1 to 3'), ('3_5', '3 to 5'), ('5+', '5+')])
    most_used_platform = models.CharField(max_length=15, choices=[('whatsapp', 'WhatsApp'), ('facebook', 'Facebook'), ('instagram', 'Instagram'), ('tiktok', 'TikTok'), ('youtube', 'YouTube')])
    seen_drug_content = models.CharField(max_length=15, choices=[('never', 'Never'), ('rarely', 'Rarely'), ('sometimes', 'Sometimes'), ('often', 'Often')])

    # Prediction Result
    predicted_risk_level = models.CharField(max_length=10, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Survey {self.id} - Risk: {self.predicted_risk_level}"




class MLConfig(models.Model):
    """
    Singleton row holding the pilot-mode ML settings. Always use
    MLConfig.get_solo() rather than querying this model directly.
    """
    pilot_mode_enabled = models.BooleanField(default=False)
    pilot_model_ever_name = models.CharField(max_length=255, blank=True, null=True)
    pilot_model_used30_name = models.CharField(max_length=255, blank=True, null=True)
    pilot_trained_at = models.DateTimeField(blank=True, null=True)
    pilot_metrics = models.JSONField(blank=True, null=True)

    class Meta:
        verbose_name = "ML Config"
        verbose_name_plural = "ML Config"

    def __str__(self):
        return "ML Config"

    def save(self, *args, **kwargs):
        self.pk = 1  # force singleton
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        pass  # prevent deletion

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj




class BulkBatch(models.Model):
    """
    One uploaded bulk file. Created on 'Save only' too, so every upload
    — predicted or not — has an audit trail of who uploaded what, when.
    """
    original_filename = models.CharField(max_length=255)
    saved_path = models.CharField(max_length=500)
    total_rows = models.IntegerField(default=0)
    mode = models.CharField(
        max_length=20,
        choices=[("save_only", "Save only"), ("predict_save", "Predict & Save")],
    )
    uploaded_by = models.CharField(max_length=255, blank=True, null=True)  # username/email or IP fallback
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Batch {self.id} ({self.original_filename}, {self.mode})"


class BulkAssessment(models.Model):
    """
    One row's prediction result within a BulkBatch. external_id is the
    'id' column from the uploaded file (the ID the school/police/rehab
    uses to track the individual) — NOT this row's own primary key.
    """
    batch = models.ForeignKey(BulkBatch, related_name="rows", on_delete=models.CASCADE)
    external_id = models.CharField(max_length=100, blank=True)
    risk_level = models.CharField(max_length=10, blank=True, null=True)
    used_30_days = models.BooleanField(blank=True, null=True)
    error = models.CharField(max_length=255, blank=True, null=True)  # set if this row's prediction failed
    raw_data = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Row {self.external_id} - {self.risk_level}"


class ForwardLog(models.Model):
    """
    Audit trail every time a batch's results are emailed out. This is
    intentionally mandatory (not optional logging) — results here are
    real people's risk classifications, so there should always be a
    record of who sent what to whom.
    """
    batch = models.ForeignKey(BulkBatch, related_name="forward_logs", on_delete=models.CASCADE)
    sent_by = models.CharField(max_length=255, blank=True, null=True)
    sent_to = models.EmailField()
    sent_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Batch {self.batch_id} forwarded to {self.sent_to} at {self.sent_at}"