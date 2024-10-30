from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('', views.home, name='home'),
    path('predict_sentiment/', include('review_classifier_urls')),
    path('predict_sentiment/', views.predict_sentiment, name='predict_sentiment'),
]
