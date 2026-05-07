@echo off
setlocal

set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python -m streamlit run src/demo/app.py
