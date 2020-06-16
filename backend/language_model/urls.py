from django.conf.urls import url
from django.urls import path
from . import views


urlpatterns = [
    path('get_continue_by_input', views.get_continue_by_input)
]