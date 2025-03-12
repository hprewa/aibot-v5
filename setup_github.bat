@echo off
echo Setting up GitHub repository...

REM Check if git is installed
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Please install git and try again.
    exit /b 1
)

REM Initialize git repository if not already initialized
if not exist .git (
    echo Initializing git repository...
    git init
) else (
    echo Git repository already initialized.
)

REM Add all files to git
echo Adding files to git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Initial commit with cleaned codebase"

REM Prompt for GitHub repository URL
set /p github_url=Please enter your GitHub repository URL (e.g., https://github.com/username/repo.git): 

REM Add GitHub remote
echo Adding GitHub remote...
git remote add origin %github_url%

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main

echo GitHub repository setup complete!
echo Remember to set up the following secrets in your GitHub repository:
echo - GCP_SA_KEY: Your Google Cloud service account key (JSON)
echo - GCP_PROJECT_ID: Your Google Cloud project ID
echo - BQ_DATASET_ID: Your BigQuery dataset ID

echo Done!
pause 