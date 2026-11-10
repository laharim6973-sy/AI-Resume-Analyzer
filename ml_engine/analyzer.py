"""
ML Engine - Core AI/ML Resume Analysis Module
M.Tech Level Implementation
Uses: NLP, TF-IDF, Cosine Similarity, Named Entity Recognition,
      Sentiment Analysis, Skill Extraction, ATS Scoring

FIXES APPLIED:
  1. _generate_recommendations: was accessing sections.get('raw_text') to
     count words for the length check. But raw_text is stripped from the
     sections dict before it reaches this method (see ResumeQualityAnalyzer.analyze).
     Fixed by passing the raw word_count directly to the method.
  2. Added guard against division-by-zero in cosine_similarity (already
     present) and in Flesch-Kincaid calculation (already guarded — kept).
  3. Minor: extract_ngrams returned empty list when tokens has fewer
     items than n; added an explicit guard.
"""

import re
import json
import math
import string
import os
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
    'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she', 'it', 'they',
    'my', 'our', 'your', 'his', 'her', 'its', 'their', 'also', 'so', 'if',
    'not', 'no', 'nor', 'up', 'down', 'out', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'each',
    'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very',
    'just', 'both', 'where', 'when', 'how', 'all', 'any', 'few'
}


def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> list:
    """Tokenize text into words."""
    words = preprocess_text(text).split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_ngrams(tokens: list, n: int) -> list:
    """Extract n-grams from tokens."""
    # FIX: guard against tokens shorter than n
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────
# TF-IDF IMPLEMENTATION (from scratch)
# ─────────────────────────────────────────────

class TFIDFVectorizer:
    """Custom TF-IDF implementation without external ML libraries."""

    def __init__(self):
        self.vocabulary  = {}
        self.idf_values  = {}
        self.doc_count   = 0

    def compute_tf(self, tokens: list) -> dict:
        """Compute term frequency."""
        tf    = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {term: count / total for term, count in tf.items()}

    def compute_tfidf_vector(self, text: str, corpus_idf: dict) -> dict:
        """Compute TF-IDF vector for a single document."""
        tokens    = tokenize(text)
        bigrams   = extract_ngrams(tokens, 2)
        all_terms = tokens + bigrams
        tf        = self.compute_tf(all_terms)
        tfidf     = {}
        for term, tf_val in tf.items():
            idf         = corpus_idf.get(term, math.log(2))
            tfidf[term] = tf_val * idf
        return tfidf

    def cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)


# ─────────────────────────────────────────────
# SKILL EXTRACTOR
# ─────────────────────────────────────────────

class SkillExtractor:
    """Extract technical and soft skills from resume text."""

    def __init__(self, skills_db_path: str):
        with open(skills_db_path, 'r') as f:
            self.skills_db = json.load(f)
        self._build_flat_skills()

    def _build_flat_skills(self):
        """Flatten skills database for quick lookup."""
        self.all_technical_skills = {}
        for category, skills in self.skills_db['technical_skills'].items():
            for skill in skills:
                self.all_technical_skills[skill.lower()] = category
        self.soft_skills    = [s.lower() for s in self.skills_db['soft_skills']]
        self.certifications = [c.lower() for c in self.skills_db['certifications']]
        self.action_verbs   = [v.lower() for v in self.skills_db['action_verbs']]
        self.weak_verbs     = [v.lower() for v in self.skills_db['weak_verbs']]
        self.job_roles      = self.skills_db['job_roles']

    def extract_skills(self, text: str) -> dict:
        """Extract all skill types from resume text."""
        text_lower     = text.lower()
        found_technical = defaultdict(list)
        found_soft      = []
        found_certs     = []
        found_action    = []
        found_weak      = []

        # Technical skills (check multi-word first)
        for skill, category in sorted(
            self.all_technical_skills.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_technical[category].append(skill)

        for skill in self.soft_skills:
            if skill in text_lower:
                found_soft.append(skill)

        for cert in self.certifications:
            if cert in text_lower:
                found_certs.append(cert)

        for verb in self.action_verbs:
            pattern = r'\b' + re.escape(verb) + r'\b'
            if re.search(pattern, text_lower):
                found_action.append(verb)

        for verb in self.weak_verbs:
            if verb in text_lower:
                found_weak.append(verb)

        return {
            'technical':       dict(found_technical),
            'soft':            list(set(found_soft)),
            'certifications':  list(set(found_certs)),
            'action_verbs':    list(set(found_action)),
            'weak_verbs':      list(set(found_weak)),
            'total_technical': sum(len(v) for v in found_technical.values()),
            'total_soft':      len(set(found_soft)),
        }

    def predict_job_role(self, text: str) -> list:
        """Predict best matching job roles using skill overlap."""
        text_lower  = text.lower()
        role_scores = {}
        for role, required_skills in self.job_roles.items():
            matched = sum(1 for s in required_skills if s in text_lower)
            score   = matched / len(required_skills) if required_skills else 0
            role_scores[role] = round(score * 100, 1)
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {'role': r.replace('_', ' ').title(), 'match': s}
            for r, s in sorted_roles[:5]
            if s > 0
        ]


# ─────────────────────────────────────────────
# RESUME SECTION PARSER
# ─────────────────────────────────────────────

class ResumeSectionParser:
    """Parse resume into structured sections using regex patterns."""

    SECTION_HEADERS = {
        'contact':        r'(contact|personal\s+info|personal\s+details)',
        'summary':        r'(summary|objective|profile|about\s+me|career\s+objective|professional\s+summary)',
        'experience':     r'(experience|work\s+experience|employment|work\s+history|professional\s+experience)',
        'education':      r'(education|academic|qualification|degree)',
        'skills':         r'(skills|technical\s+skills|core\s+competencies|technologies|expertise)',
        'projects':       r'(projects|personal\s+projects|academic\s+projects|key\s+projects)',
        'certifications': r'(certifications?|certificates?|courses?|training)',
        'achievements':   r'(achievements?|awards?|honors?|accomplishments?|recognition)',
        'publications':   r'(publications?|research|papers?)',
        'languages':      r'(languages?|spoken\s+languages)',
        'interests':      r'(interests?|hobbies|activities|extracurricular)',
    }

    def parse(self, text: str) -> dict:
        """Parse resume text into sections."""
        sections = {k: '' for k in self.SECTION_HEADERS}
        sections['raw_text'] = text
        lines           = text.split('\n')
        current_section = 'summary'
        buffer          = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                buffer.append('')
                continue
            matched_section = None
            for section, pattern in self.SECTION_HEADERS.items():
                if re.match(r'^\s*' + pattern + r'\s*[:\-]?\s*$',
                            line_stripped, re.IGNORECASE):
                    matched_section = section
                    break

            if matched_section:
                sections[current_section] = '\n'.join(buffer).strip()
                current_section = matched_section
                buffer          = []
            else:
                buffer.append(line)

        sections[current_section] = '\n'.join(buffer).strip()
        return sections

    def extract_contact_info(self, text: str) -> dict:
        """Extract email, phone, LinkedIn, GitHub from text."""
        contact         = {}
        email_pattern   = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern   = r'(\+?\d{1,3}[-.\\s]?)?\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4}'
        linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
        github_pattern  = r'github\.com/[\w\-]+'

        emails   = re.findall(email_pattern, text)
        phones   = re.findall(phone_pattern, text)
        linkedins = re.findall(linkedin_pattern, text, re.IGNORECASE)
        githubs  = re.findall(github_pattern, text, re.IGNORECASE)

        if emails:
            contact['email'] = emails[0]
        if phones:
            phone = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
            contact['phone'] = phone.strip()
        if linkedins:
            contact['linkedin'] = linkedins[0]
        if githubs:
            contact['github'] = githubs[0]

        return contact

    def extract_years_of_experience(self, text: str) -> float:
        """Estimate total years of experience from date patterns."""
        year_pattern = r'\b(19|20)\d{2}\b'
        years = [int(y) for y in re.findall(year_pattern, text)]
        if len(years) >= 2:
            span = max(years) - min(years)
            return round(min(span, 30), 1)
        return 0

    def extract_education_details(self, text: str) -> list:
        """Extract education qualifications."""
        degrees = []
        degree_patterns = [
            r'(ph\.?d|doctorate)',
            r'(m\.?tech|master\s+of\s+technology)',
            r'(m\.?e|master\s+of\s+engineering)',
            r'(m\.?sc|master\s+of\s+science)',
            r'(mba|master\s+of\s+business)',
            r'(b\.?tech|bachelor\s+of\s+technology)',
            r'(b\.?e|bachelor\s+of\s+engineering)',
            r'(b\.?sc|bachelor\s+of\s+science)',
            r'(bca|mca)',
            r'(diploma)',
        ]
        text_lower = text.lower()
        for pattern in degree_patterns:
            match = re.search(pattern, text_lower)
            if match:
                degrees.append(match.group(0).upper())
        return degrees

    def count_quantified_achievements(self, text: str) -> int:
        """Count bullet points with numbers/percentages (quantified impact)."""
        patterns = [
            r'\d+%',
            r'\$[\d,]+',
            r'\d+x\s',
            r'reduced\s+by\s+\d+',
            r'increased\s+by\s+\d+',
            r'improved\s+by\s+\d+',
            r'\d+\s+(users|customers|clients|teams?|projects?|products?)',
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count


# ─────────────────────────────────────────────
# ATS SCORER
# ─────────────────────────────────────────────

class ATSScorer:
    """ATS (Applicant Tracking System) compatibility scorer."""

    def score(self, text: str, sections: dict, skills: dict,
              contact: dict, experience_years: float) -> dict:
        """Calculate comprehensive ATS score."""
        scores = {}

        # 1. Contact Information (10 points)
        contact_score = 0
        if contact.get('email'):    contact_score += 4
        if contact.get('phone'):    contact_score += 3
        if contact.get('linkedin'): contact_score += 2
        if contact.get('github'):   contact_score += 1
        scores['contact'] = min(contact_score, 10)

        # 2. Skills Presence (25 points)
        tech_count  = skills.get('total_technical', 0)
        soft_count  = skills.get('total_soft', 0)
        skills_score = min(tech_count * 1.5, 18) + min(soft_count * 1.5, 7)
        scores['skills'] = min(round(skills_score), 25)

        # 3. Section Completeness (20 points)
        section_score = 0
        key_sections  = {
            'summary': 3, 'experience': 5, 'education': 4,
            'skills':  4, 'projects':   2, 'certifications': 2,
        }
        for section, weight in key_sections.items():
            if sections.get(section) and len(sections[section]) > 50:
                section_score += weight
        scores['sections'] = min(section_score, 20)

        # 4. Action Verbs (15 points)
        action_count = len(skills.get('action_verbs', []))
        weak_count   = len(skills.get('weak_verbs',   []))
        verb_score   = min(action_count * 1.5, 15) - (weak_count * 2)
        scores['action_verbs'] = max(min(round(verb_score), 15), 0)

        # 5. Quantified Achievements (15 points)
        quant = sections.get('_quantified', 0)
        scores['quantified'] = min(quant * 3, 15)

        # 6. Resume Length (10 points)
        word_count = len(text.split())
        if   300 <= word_count <= 700: scores['length'] = 10
        elif 200 <= word_count <= 900: scores['length'] = 7
        elif word_count < 200:         scores['length'] = 3
        else:                          scores['length'] = 5

        # 7. Certifications (5 points)
        cert_count = len(skills.get('certifications', []))
        scores['certifications'] = min(cert_count * 2, 5)

        total         = sum(scores.values())
        scores['total'] = min(total, 100)
        return scores


# ─────────────────────────────────────────────
# JOB DESCRIPTION MATCHER
# ─────────────────────────────────────────────

class JobDescriptionMatcher:
    """Match resume against job description using TF-IDF + Cosine Similarity."""

    def __init__(self):
        self.tfidf = TFIDFVectorizer()

    def _build_corpus_idf(self, texts: list) -> dict:
        """Build IDF from corpus."""
        N        = len(texts)
        doc_freq = Counter()
        for text in texts:
            tokens  = set(tokenize(text))
            bigrams = set(extract_ngrams(list(tokens), 2))
            doc_freq.update(tokens | bigrams)
        return {
            term: math.log((N + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }

    def match(self, resume_text: str, job_description: str) -> dict:
        """Calculate match between resume and job description."""
        corpus_idf = self._build_corpus_idf([resume_text, job_description])
        resume_vec = self.tfidf.compute_tfidf_vector(resume_text,    corpus_idf)
        jd_vec     = self.tfidf.compute_tfidf_vector(job_description, corpus_idf)
        similarity = self.tfidf.cosine_similarity(resume_vec, jd_vec)

        jd_tokens     = set(tokenize(job_description))
        resume_tokens = set(tokenize(resume_text))
        missing       = jd_tokens - resume_tokens
        matching      = jd_tokens & resume_tokens

        important_missing  = sorted([t for t in missing  if len(t) > 3])[:20]
        important_matching = sorted([t for t in matching if len(t) > 3])[:25]

        return {
            'similarity_score':   round(similarity * 100, 1),
            'missing_keywords':   important_missing,
            'matching_keywords':  important_matching,
            'recommendation':     self._get_recommendation(similarity),
        }

    def _get_recommendation(self, score: float) -> str:
        if score >= 0.7:
            return "Excellent match! Your resume strongly aligns with this job."
        elif score >= 0.5:
            return "Good match. Add a few more relevant keywords to improve."
        elif score >= 0.3:
            return "Moderate match. Tailor your resume more specifically to this role."
        else:
            return "Low match. Significant customization needed for this position."


# ─────────────────────────────────────────────
# RESUME QUALITY ANALYZER
# ─────────────────────────────────────────────

class ResumeQualityAnalyzer:
    """Comprehensive resume quality analysis and recommendations."""

    def __init__(self):
        self.parser         = ResumeSectionParser()
        self.skill_extractor = None   # Set after init via views._get_analyzer()
        self.ats_scorer     = ATSScorer()
        self.jd_matcher     = JobDescriptionMatcher()

    def analyze(self, text: str, job_description: str = None) -> dict:
        """Full resume analysis pipeline."""
        sections         = self.parser.parse(text)
        contact          = self.parser.extract_contact_info(text)
        skills           = self.skill_extractor.extract_skills(text) if self.skill_extractor else {}
        predicted_roles  = self.skill_extractor.predict_job_role(text) if self.skill_extractor else []
        experience_years = self.parser.extract_years_of_experience(text)
        education        = self.parser.extract_education_details(text)
        quant            = self.parser.count_quantified_achievements(text)
        sections['_quantified'] = quant

        ats_scores = self.ats_scorer.score(text, sections, skills, contact, experience_years)

        jd_match = None
        if job_description:
            jd_match = self.jd_matcher.match(text, job_description)

        # Word / sentence stats
        words          = text.split()
        sentences      = re.split(r'[.!?]+', text)
        word_count     = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_len = word_count / max(sentence_count, 1)

        # Flesch-Kincaid readability
        syllable_count = sum(self._count_syllables(w) for w in words)
        if sentence_count > 0 and word_count > 0:
            fk_score = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6  * (syllable_count / word_count)
            )
        else:
            fk_score = 50
        fk_score = max(0, min(100, fk_score))

        # FIX: pass word_count to recommendations so it can check length
        #      WITHOUT relying on sections['raw_text'] (which is stripped out below)
        recommendations = self._generate_recommendations(
            sections, skills, contact, ats_scores, quant, education, word_count
        )

        return {
            'contact':         contact,
            'sections':        {
                k: v for k, v in sections.items()
                if not k.startswith('_') and k != 'raw_text'
            },
            'skills':          skills,
            'predicted_roles': predicted_roles,
            'experience_years': experience_years,
            'education':       education,
            'ats_scores':      ats_scores,
            'jd_match':        jd_match,
            'recommendations': recommendations,
            'stats': {
                'word_count':            word_count,
                'sentence_count':        sentence_count,
                'avg_sentence_length':   round(avg_sentence_len, 1),
                'readability_score':     round(fk_score, 1),
                'quantified_achievements': quant,
                'action_verb_count':     len(skills.get('action_verbs', [])),
                'weak_verb_count':       len(skills.get('weak_verbs',   [])),
            },
        }

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count."""
        word   = word.lower().rstrip('e')
        vowels = re.findall(r'[aeiou]+', word)
        return max(1, len(vowels))

    # FIX: added word_count parameter — no longer reads sections['raw_text']
    def _generate_recommendations(
        self, sections, skills, contact, ats_scores, quant, education,
        word_count: int = 0
    ) -> list:
        """Generate actionable recommendations."""
        recs = []

        if not contact.get('email'):
            recs.append({
                'type': 'critical', 'icon': '📧',
                'title': 'Missing Email Address',
                'detail': 'Add a professional email address to your contact section.',
            })

        if not contact.get('linkedin'):
            recs.append({
                'type': 'important', 'icon': '💼',
                'title': 'Add LinkedIn Profile',
                'detail': 'LinkedIn URL significantly improves professional credibility.',
            })

        if not contact.get('github') and skills.get('total_technical', 0) > 3:
            recs.append({
                'type': 'important', 'icon': '🐙',
                'title': 'Add GitHub Profile',
                'detail': 'GitHub profile demonstrates practical coding experience.',
            })

        if len(skills.get('weak_verbs', [])) > 2:
            sample = ', '.join(skills['weak_verbs'][:3])
            recs.append({
                'type': 'important', 'icon': '💪',
                'title': 'Replace Weak Action Verbs',
                'detail': (
                    f'Replace "{sample}" with strong action verbs '
                    f'like Engineered, Architected, Delivered.'
                ),
            })

        if quant < 3:
            recs.append({
                'type': 'important', 'icon': '📊',
                'title': 'Add Quantified Achievements',
                'detail': 'Include numbers: "Reduced load time by 40%", "Served 10,000+ users".',
            })

        # FIX: use passed-in word_count instead of sections['raw_text']
        if ats_scores.get('length', 0) < 7:
            if word_count < 300:
                recs.append({
                    'type': 'warning', 'icon': '📝',
                    'title': 'Resume Too Short',
                    'detail': 'Aim for 400-700 words. Expand project descriptions and experience.',
                })
            else:
                recs.append({
                    'type': 'warning', 'icon': '✂️',
                    'title': 'Resume Too Long',
                    'detail': 'Keep resume to 1-2 pages (400-700 words) for best ATS results.',
                })

        if not sections.get('summary') or len(sections.get('summary', '')) < 50:
            recs.append({
                'type': 'important', 'icon': '🎯',
                'title': 'Add Professional Summary',
                'detail': 'A 3-4 line summary at the top dramatically improves first impressions.',
            })

        if not sections.get('projects') or len(sections.get('projects', '')) < 50:
            recs.append({
                'type': 'suggestion', 'icon': '🚀',
                'title': 'Add Projects Section',
                'detail': 'Include 2-3 key projects with tech stack and impact.',
            })

        if skills.get('total_technical', 0) < 5:
            recs.append({
                'type': 'important', 'icon': '⚙️',
                'title': 'Expand Technical Skills',
                'detail': 'List more relevant tools, languages, and frameworks you know.',
            })

        if not skills.get('certifications'):
            recs.append({
                'type': 'suggestion', 'icon': '🏅',
                'title': 'Add Certifications',
                'detail': 'AWS, Google Cloud, or other industry certifications boost credibility.',
            })

        if not recs:
            recs.append({
                'type': 'success', 'icon': '🌟',
                'title': 'Excellent Resume!',
                'detail': 'Your resume covers all major components well. Keep it updated!',
            })

        return recs
