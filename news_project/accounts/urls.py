from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('login/', views.login_view, name='login'),  
    path('summarized', views.summarized_news, name='summarized'),
]
