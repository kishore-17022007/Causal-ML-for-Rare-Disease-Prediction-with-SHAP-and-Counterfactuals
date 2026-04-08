@REM ═══════════════════════════════════════════════════════════════════════════
@REM Master Control Script for Windows
@REM ═══════════════════════════════════════════════════════════════════════════
@REM Master Control Script: Full Project Initialization & Execution
@REM Run with: run_project.bat

@setlocal enabledelayedexpansion
@echo off
cls

echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                            ║
echo ║     nf RARE DISEASE ML - REAL-TIME CLINICAL DASHBOARD                     ║
echo ║                    Master Control Script (Windows)                         ║
echo ║                                                                            ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.

@REM Check Python Installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

@REM Check and Create Virtual Environment
echo.
echo [2/6] Setting up virtual environment...
if not exist ".venv\" (
    echo Creating new virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

@REM Activate Virtual Environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

@REM Install Dependencies
echo.
echo [3/6] Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] All dependencies installed

@REM Create Directories
echo.
echo [4/6] Setting up project directories...
if not exist "data\" mkdir data
if not exist "outputs\" mkdir outputs
if not exist "outputs\archive\" mkdir outputs\archive
echo [OK] Directories created

@REM Check and Train Models
echo.
echo [5/6] Checking ML models...
if not exist "outputs\trained_models.pkl" (
    echo [INFO] Models not found. Training ML pipeline...
    echo [INFO] This may take 2-5 minutes on first run...
    echo.
    
    if not exist "data\kaggle_data.csv" (
        echo [WARNING] Kaggle dataset not found at data\kaggle_data.csv
        echo [INFO] Using demo data generation mode
        echo.
    )
    
    python main.py
    if errorlevel 1 (
        echo Error: ML pipeline failed
        pause
        exit /b 1
    )
    echo [OK] ML pipeline completed successfully
) else (
    echo [OK] Trained models found
    
    set "MISSING=0"
    if not exist "outputs\trained_preprocessor.pkl" set MISSING=1
    if not exist "outputs\feature_names.pkl" set MISSING=1
    if not exist "outputs\best_threshold.pkl" set MISSING=1
    
    if !MISSING! equ 1 (
        echo [WARNING] Some dashboard artifacts missing. Regenerating...
        python main.py >nul 2>&1
        echo [OK] Artifacts regenerated
    )
)

@REM Validate Dashboard App
echo.
echo [6/6] Validating dashboard app...
if not exist "app.py" (
    echo Error: app.py not found
    pause
    exit /b 1
)
echo [OK] Dashboard app found (app.py)

@REM Launch Dashboard
echo.
echo ═══════════════════════════════════════════════════════════════════════════
echo Setup Complete! Starting Dashboard...
echo ═══════════════════════════════════════════════════════════════════════════
echo.
echo Dashboard Information:
echo   URL: http://localhost:8501
echo   Port: 8501
echo   Press Ctrl+C to stop
echo.
echo Quick Start:
echo   1. Open http://localhost:8501 in your browser
echo   2. Use the sidebar to set baseline patient inputs
echo   3. Use Simulation Lab to test input changes
echo   4. Check ⚠️ Alerts for any critical patients
echo.

python -m streamlit run app.py --logger.level=warning

pause
