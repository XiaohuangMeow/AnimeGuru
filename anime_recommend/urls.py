from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('People/', views.info, name='info'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='anime_recommend/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='anime_recommend/logout.html'), name='logout'),
    path('profile/', views.profile, name='profile')
]
