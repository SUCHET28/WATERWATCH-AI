@echo off
title WATERWATCH AI - Streamlit Dashboard
echo ===================================================
echo Starting WATERWATCH AI Streamlit Web Application...
echo ===================================================
cd /d "%~dp0"
python -m streamlit run app/app.py
pause
