from django.urls import path
from review_classifier import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_sentiment, name='predict_sentiment'),
]
