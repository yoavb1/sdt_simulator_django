from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('run-logic/', views.run_logic, name='run_logic'),
]