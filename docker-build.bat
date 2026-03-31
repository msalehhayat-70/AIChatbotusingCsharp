@echo off
title TechBot - Docker Build Script
color 0A

echo.
echo  =====================================================
echo   TechBot AI Chatbot - Docker Container Builder
echo  =====================================================
echo.

:: Check Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Docker Desktop is not running!
    echo.
    echo  Please:
    echo    1. Open Docker Desktop
    echo    2. Wait for it to fully start (whale icon in taskbar)
    echo    3. Run this script again
    echo.
    pause
    exit /b 1
)

echo  [OK] Docker Desktop is running
echo.

:: Move to the directory where this script lives (project root)
cd /d "%~dp0"
echo  [INFO] Working directory: %CD%
echo.

:: Build the Docker image
echo  [STEP 1] Building Docker image: techbot
echo  -------------------------------------------------------
docker build -t techbot .
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Docker build failed. See errors above.
    pause
    exit /b 1
)

echo.
echo  [OK] Docker image 'techbot' built successfully!
echo.

:: Test the model directly (headless - no GUI needed)
echo  [STEP 2] Testing the AI model inside container...
echo  -------------------------------------------------------
docker run --rm techbot python3 Python/predict.py "what is machine learning"
echo.
docker run --rm techbot python3 Python/predict.py "what is a GPU"
echo.
docker run --rm techbot python3 Python/predict.py "hello"
echo.

echo  =====================================================
echo   SUCCESS! TechBot container is ready.
echo  =====================================================
echo.
echo  Useful Docker commands:
echo.
echo    List images:
echo      docker images
echo.
echo    Test AI model (headless):
echo      docker run --rm techbot python3 Python/predict.py "your question"
echo.
echo    Run container interactively:
echo      docker run -it techbot bash
echo.
echo    Remove the image:
echo      docker rmi techbot
echo.
pause
