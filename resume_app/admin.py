from django.contrib import admin
from .models import ResumeAnalysis


@admin.register(ResumeAnalysis)
class ResumeAnalysisAdmin(admin.ModelAdmin):
    list_display = ['file_name', 'ats_score', 'experience_years', 'status', 'uploaded_at']
    list_filter = ['status', 'uploaded_at']
    search_fields = ['file_name']
    readonly_fields = ['uploaded_at', 'raw_text', 'skills_json', 'sections_json',
                       'contact_json', 'recommendations_json', 'stats_json']
    ordering = ['-uploaded_at']
