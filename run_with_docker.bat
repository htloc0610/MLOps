@echo off
REM Run Vertex AI Pipeline using Python from Docker image with gcloud credentials

echo ======================================
echo Vertex AI Pipeline - Docker Execution
echo ======================================

REM Set variables
set GCLOUD_CONFIG=%APPDATA%\gcloud
set PROJECT_DIR=%~dp0

echo.
echo Checking gcloud credentials...
if not exist "%GCLOUD_CONFIG%\application_default_credentials.json" (
    echo ERROR: Application default credentials not found!
    echo Please run: gcloud auth application-default login
    pause
    exit /b 1
)

echo ✓ Credentials found
echo.
echo Running pipeline in Docker container...
echo.

docker run --rm ^
    -v "%PROJECT_DIR%:/workspace" ^
    -v "%GCLOUD_CONFIG%:/root/.config/gcloud:ro" ^
    -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json ^
    -e CLOUDSDK_CONFIG=/root/.config/gcloud ^
    -w /workspace ^
    training-image:latest ^
    -c "python /workspace/run_pipeline.py"

echo.
echo ======================================
echo Pipeline execution completed!
echo ======================================
pause
