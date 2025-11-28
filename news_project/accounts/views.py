from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
# accounts/views.py
from django.http import JsonResponse
import json
import os

def summarized_news(request):
    output_file = os.path.join('path_to_output_json', 'output.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        return JsonResponse(data)
    else:
        return JsonResponse({"summarized": []})

def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm = request.POST['confirmPassword']

        if password != confirm:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return redirect('signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('signup')

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, "Account created successfully! Please log in.")
        return redirect('login')

    return render(request, 'accounts/signup.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('login')

    return render(request, 'accounts/login.html')


@login_required
def dashboard_view(request):
    return render(request, 'accounts/dashboard.html')


def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('login')
