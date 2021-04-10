from django.contrib import admin
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from feedback.views import RegisterAPI
from knox import views as knox_views
from feedback.views import LoginAPI
from . import views
urlpatterns = [
    path('register', RegisterAPI.as_view(), name='register'),
    path('login', LoginAPI.as_view(), name='login'),
    path('logout', knox_views.LogoutView.as_view(), name='logout'),
    path('logoutall', knox_views.LogoutAllView.as_view(), name='logoutall'),
    path('addtest', views.add_test),
    path('showtestlist', views.show_test_list),
    path('showtest', views.show_test),

]