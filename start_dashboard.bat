@echo off
cd /d "%~dp0"
echo Starting Customer Churn Prediction Dashboard...
python -m streamlit run app.py
pause
