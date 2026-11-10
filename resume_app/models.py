"""
Models for Resume Analyzer

FIXES APPLIED:
  1. get_ats_grade: guard against None / 0 ats_score
  2. All get_* methods return safe defaults on any error
"""

from django.db import models
import json


class ResumeAnalysis(models.Model):
    """Stores resume analysis results."""

    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('completed',  'Completed'),
        ('failed',     'Failed'),
    ]

    # File info
    file_name   = models.CharField(max_length=255)
    file_path   = models.FileField(upload_to='resumes/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status      = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Extracted text
    raw_text = models.TextField(blank=True)

    # Analysis results (stored as JSON strings)
    ats_score             = models.FloatField(default=0)
    skills_json           = models.TextField(blank=True, default='{}')
    sections_json         = models.TextField(blank=True, default='{}')
    contact_json          = models.TextField(blank=True, default='{}')
    recommendations_json  = models.TextField(blank=True, default='[]')
    predicted_roles_json  = models.TextField(blank=True, default='[]')
    stats_json            = models.TextField(blank=True, default='{}')
    jd_match_json         = models.TextField(blank=True, default='{}')

    # Meta info
    experience_years = models.FloatField(default=0)
    job_description  = models.TextField(blank=True)
    error_message    = models.TextField(blank=True)

    class Meta:
        ordering         = ['-uploaded_at']
        verbose_name     = 'Resume Analysis'
        verbose_name_plural = 'Resume Analyses'

    def __str__(self):
        return (
            f"{self.file_name} - ATS: {self.ats_score:.0f}% "
            f"({self.uploaded_at.strftime('%Y-%m-%d')})"
        )

    # ------------------------------------------------------------------
    # Safe JSON accessors
    # ------------------------------------------------------------------

    def get_skills(self):
        try:
            return json.loads(self.skills_json) or {}
        except Exception:
            return {}

    def get_sections(self):
        try:
            return json.loads(self.sections_json) or {}
        except Exception:
            return {}

    def get_contact(self):
        try:
            return json.loads(self.contact_json) or {}
        except Exception:
            return {}

    def get_recommendations(self):
        try:
            return json.loads(self.recommendations_json) or []
        except Exception:
            return []

    def get_predicted_roles(self):
        try:
            return json.loads(self.predicted_roles_json) or []
        except Exception:
            return []

    def get_stats(self):
        try:
            return json.loads(self.stats_json) or {}
        except Exception:
            return {}

    def get_jd_match(self):
        try:
            return json.loads(self.jd_match_json) or {}
        except Exception:
            return {}

    def get_ats_grade(self):
        # FIX: coerce None → 0 before comparison
        score = float(self.ats_score or 0)
        if score >= 85:
            return 'A+', 'Excellent'
        elif score >= 75:
            return 'A',  'Very Good'
        elif score >= 65:
            return 'B+', 'Good'
        elif score >= 55:
            return 'B',  'Average'
        elif score >= 45:
            return 'C',  'Below Average'
        else:
            return 'D',  'Needs Improvement'
