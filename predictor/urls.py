from django.urls import path
from . import views
from . import api_views

urlpatterns = [
    #path('', views.home_view, name='home'),
    path('',views.submit_assessment,name='home'),
    
    path('objectives/survey/', views.objective1_survey, name='objective1'),
    path('objectives/preprocessing/', views.objective2_preprocessing, name='objective2'),
    path('objectives/training/', views.objective3_training, name='objective3'),
    path('objectives/evaluation/', views.objective4_evaluation, name='objective4'),
    path('objectives/recommendations/', views.objective5_recommendations, name='objective5'),


    path('result/<int:pk>/', views.result_view, name='result'),
    path('api/predict/ever-used/', api_views.PredictEverUsedView.as_view(), name='api_predict_ever_used'),
    path('api/predict/used-30-days/', api_views.PredictUsed30View.as_view(), name='api_predict_used30'),
    path('api/predict/risk/', api_views.PredictRiskView.as_view(), name='api_predict_risk'),
    path('api/artifacts/info/', api_views.ArtifactsInfoView.as_view(), name='api_artifacts_info'),



    path("predict/", views.predict_view, name="predict"),
    path("pilot-mode/", views.pilot_mode_toggle_view, name="pilot-mode"),
    path("train-pilot/", views.train_pilot_view, name="train-pilot"),


    path("download-sample/", views.download_sample_view, name="download-sample"),
    path("bulk-upload/", views.bulk_upload_view, name="bulk-upload"),


    # --- Add alongside your existing bulk-upload path in urls.py -----------

    path("bulk-result/<int:batch_id>/", views.bulk_result_view, name="bulk-result"),
    path("bulk-result/<int:batch_id>/download/", views.bulk_result_download_view, name="bulk-result-download"),
    path("bulk-result/<int:batch_id>/forward/", views.bulk_result_forward_view, name="bulk-result-forward"),


]
