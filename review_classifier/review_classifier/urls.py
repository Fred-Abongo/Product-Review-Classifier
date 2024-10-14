from django.contrib import admin
from django.urls import path, include
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict-sentiment/', views.predict_sentiment, name='predict_sentiment'),
    path('review_classifier/', include('review_classifier.urls')),
]
