"""
Utility functions for text extraction from PDF, DOCX, TXT files.
M.Tech Level - Resume Analyzer
"""

import os
import re


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        import pdfplumber
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return '\n'.join(text)
    except ImportError:
        pass
    try:
        import PyPDF2
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or '')
        return '\n'.join(text)
    except ImportError:
        pass
    return "PDF extraction library not available. Please install pdfplumber or PyPDF2."


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text)
        return '\n'.join(paragraphs)
    except ImportError:
        return "python-docx not available. Please install python-docx."
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"


def extract_text_from_file(file_path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        return "Unsupported file format. Please upload PDF, DOCX, or TXT."


def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Fix encoding issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove weird characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)
