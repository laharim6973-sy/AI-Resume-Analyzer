#!/usr/bin/env python3
"""
AI Resume Analyzer — One-Click Setup & Run Script
AI Resume Web Application  |  Windows 11 / macOS / Linux
"""

import os
import sys
import subprocess
import platform


def run(cmd, check=True):
    print(f"  ▶ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def main():
    print("=" * 60)
    print("   🤖 AI RESUME WEB APPLICATION — Setup")
    print("=" * 60)
    print()

    python  = sys.executable
    os_name = platform.system()
    print(f"  Platform : {os_name}")
    print(f"  Python   : {sys.version.split()[0]}")
    print()

    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        sys.exit(1)

    print("📦 Step 1: Installing dependencies...")
    run(f'"{python}" -m pip install --upgrade pip -q')
    run(f'"{python}" -m pip install -r requirements.txt -q')
    print("   ✅ Dependencies installed\n")

    print("📁 Step 2: Creating required directories...")
    os.makedirs("media/resumes", exist_ok=True)
    os.makedirs("staticfiles",   exist_ok=True)
    print("   ✅ Directories created\n")

    print("🗄️  Step 3: Setting up database...")
    run(f'"{python}" manage.py makemigrations --no-input')
    run(f'"{python}" manage.py migrate --no-input')
    print("   ✅ Database ready\n")

    print("=" * 60)
    print("   🚀 LAUNCHING SERVER")
    print("=" * 60)
    print()
    print("   🌐 Open your browser and go to:")
    print("   ➜  http://127.0.0.1:8000")
    print()
    print("   Press CTRL+C to stop the server")
    print()

    run(f'"{python}" manage.py runserver 8000', check=False)


if __name__ == "__main__":
    main()




#cd resume_analyzer_fixed