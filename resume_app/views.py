"""
Views for AI Resume Analyzer - M.Tech Level Django Application

FIXES APPLIED:
  1. Removed sys.path.insert at module level (caused import errors on Windows)
  2. Moved SkillExtractor / analyzer init inside a lazy loader so missing
     skills_db.json does not crash on startup
  3. Added proper error handling for missing media root
"""
import os
import json
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.conf import settings

from .models import ResumeAnalysis
from utils.extractor import extract_text_from_file, clean_extracted_text
from ml_engine.analyzer import ResumeQualityAnalyzer, SkillExtractor, ATSScorer, ResumeSectionParser

# ---------------------------------------------------------------------------
# FIX 1: Lazy initialisation — only build these once, but AFTER Django is
#         fully loaded (not at import time). Using a module-level sentinel
#         avoids the "file not found" crash when manage.py first starts.
# ---------------------------------------------------------------------------
_analyzer = None
_skill_extractor = None


def _get_analyzer():
    """Return a fully initialised ResumeQualityAnalyzer (lazy singleton)."""
    global _analyzer, _skill_extractor
    if _analyzer is None:
        skills_db_path = str(settings.BASE_DIR / 'ml_engine' / 'skills_db.json')
        _skill_extractor = SkillExtractor(skills_db_path)
        _analyzer = ResumeQualityAnalyzer()
        _analyzer.skill_extractor = _skill_extractor
    return _analyzer


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def index(request):
    recent = ResumeAnalysis.objects.filter(status='completed').order_by('-uploaded_at')[:5]
    return render(request, 'resume_app/index.html', {'recent': recent})


def upload_resume(request):
    if request.method != 'POST':
        return redirect('index')

    uploaded_file = request.FILES.get('resume')
    job_description = request.POST.get('job_description', '').strip()

    if not uploaded_file:
        return render(request, 'resume_app/index.html',
                      {'error': 'Please upload a resume file.'})

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in ['.pdf', '.docx', '.doc', '.txt']:
        return render(request, 'resume_app/index.html',
                      {'error': 'Invalid file type. Upload PDF, DOCX, or TXT.'})

    if uploaded_file.size > 10 * 1024 * 1024:
        return render(request, 'resume_app/index.html',
                      {'error': 'File too large. Maximum 10MB.'})

    # FIX 2: ensure media/resumes directory exists before saving
    os.makedirs(os.path.join(settings.MEDIA_ROOT, 'resumes'), exist_ok=True)

    analysis = ResumeAnalysis(
        file_name=uploaded_file.name,
        job_description=job_description,
        status='processing',
    )
    analysis.file_path = uploaded_file
    analysis.save()

    try:
        file_full_path = os.path.join(settings.MEDIA_ROOT, str(analysis.file_path))
        raw_text = extract_text_from_file(file_full_path)
        raw_text = clean_extracted_text(raw_text)

        if len(raw_text.strip()) < 50:
            raise ValueError("Could not extract sufficient text from the resume.")

        analysis.raw_text = raw_text

        # FIX 3: use lazy singleton
        az = _get_analyzer()
        result = az.analyze(raw_text, job_description if job_description else None)

        analysis.ats_score         = result['ats_scores'].get('total', 0)
        analysis.experience_years  = result.get('experience_years', 0)
        analysis.skills_json       = json.dumps(result['skills'])
        analysis.sections_json     = json.dumps(result['sections'])
        analysis.contact_json      = json.dumps(result['contact'])
        analysis.recommendations_json = json.dumps(result['recommendations'])
        analysis.predicted_roles_json = json.dumps(result['predicted_roles'])
        analysis.stats_json        = json.dumps(result['stats'])
        analysis.jd_match_json     = json.dumps(result.get('jd_match') or {})
        analysis.status            = 'completed'
        analysis.save()

        return redirect('results', pk=analysis.pk)

    except Exception as e:
        analysis.status        = 'failed'
        analysis.error_message = str(e)
        analysis.save()
        return render(request, 'resume_app/index.html',
                      {'error': f'Analysis failed: {str(e)}'})


def results(request, pk):
    analysis = get_object_or_404(ResumeAnalysis, pk=pk)

    if analysis.status == 'failed':
        return render(request, 'resume_app/error.html',
                      {'error': analysis.error_message, 'analysis': analysis})

    skills          = analysis.get_skills()
    # Normalize category key underscores for display
    if 'technical' in skills:
        skills['technical'] = {
            k.replace('_', ' ').title(): v
            for k, v in skills['technical'].items()
        }
    contact         = analysis.get_contact()
    recommendations = analysis.get_recommendations()
    predicted_roles = analysis.get_predicted_roles()
    stats           = analysis.get_stats()
    sections        = analysis.get_sections()
    jd_match        = analysis.get_jd_match()
    grade, grade_label = analysis.get_ats_grade()

    parser          = ResumeSectionParser()
    sections_parsed = parser.parse(analysis.raw_text)
    quant           = parser.count_quantified_achievements(analysis.raw_text)
    sections_parsed['_quantified'] = quant

    ats_scorer   = ATSScorer()
    detailed_ats = ats_scorer.score(
        analysis.raw_text, sections_parsed, skills, contact, analysis.experience_years
    )

    ats_components = [
        ('Contact Info',   detailed_ats.get('contact', 0),        10),
        ('Skills',         detailed_ats.get('skills', 0),         25),
        ('Sections',       detailed_ats.get('sections', 0),       20),
        ('Action Verbs',   detailed_ats.get('action_verbs', 0),   15),
        ('Quantified',     detailed_ats.get('quantified', 0),     15),
        ('Resume Length',  detailed_ats.get('length', 0),         10),
        ('Certifications', detailed_ats.get('certifications', 0),  5),
    ]

    context = {
        'analysis':      analysis,
        'skills':        skills,
        'contact':       contact,
        'recommendations': recommendations,
        'predicted_roles': predicted_roles,
        'stats':         stats,
        'sections':      sections,
        'jd_match':      jd_match,
        'grade':         grade,
        'grade_label':   grade_label,
        'detailed_ats':  detailed_ats,
        'ats_components': ats_components,
        'tech_skills':   skills.get('technical', {}),
    }
    return render(request, 'resume_app/results.html', context)


def analyze_text(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
            jd   = data.get('job_description', '')
            if len(text.strip()) < 50:
                return JsonResponse({'error': 'Text too short'}, status=400)
            az = _get_analyzer()
            result = az.analyze(text, jd if jd else None)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'POST required'}, status=405)


def history(request):
    analyses = ResumeAnalysis.objects.all().order_by('-uploaded_at')[:20]
    return render(request, 'resume_app/history.html', {'analyses': analyses})


def delete_analysis(request, pk):
    if request.method == 'POST':
        get_object_or_404(ResumeAnalysis, pk=pk).delete()
    return redirect('history')


def about(request):
    return render(request, 'resume_app/about.html')


def dashboard(request):
    total    = ResumeAnalysis.objects.filter(status='completed').count()
    analyses = ResumeAnalysis.objects.filter(status='completed')
    avg_score = round(sum(a.ats_score for a in analyses) / max(total, 1), 1)

    score_bands = {
        '90-100':   analyses.filter(ats_score__gte=90).count(),
        '80-89':    analyses.filter(ats_score__gte=80, ats_score__lt=90).count(),
        '70-79':    analyses.filter(ats_score__gte=70, ats_score__lt=80).count(),
        '60-69':    analyses.filter(ats_score__gte=60, ats_score__lt=70).count(),
        'Below 60': analyses.filter(ats_score__lt=60).count(),
    }

    return render(request, 'resume_app/dashboard.html', {
        'total':       total,
        'avg_score':   avg_score,
        'score_bands': score_bands,
        'recent':      analyses.order_by('-uploaded_at')[:10],
    })
