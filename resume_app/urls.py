"""
URL Configuration for resume_app

FIXES APPLIED:
  No bugs found in urls.py — file is clean.
  Kept identical to original.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('',                    views.index,           name='index'),
    path('upload/',             views.upload_resume,   name='upload_resume'),
    path('results/<int:pk>/',   views.results,         name='results'),
    path('history/',            views.history,         name='history'),
    path('dashboard/',          views.dashboard,       name='dashboard'),
    path('about/',              views.about,           name='about'),
    path('analyze-text/',       views.analyze_text,    name='analyze_text'),
    path('delete/<int:pk>/',    views.delete_analysis, name='delete_analysis'),
]
