# PowerShell script to run the MCP Slack Bot with proper virtual environment activation
# Usage: .\run_slack_bot.ps1 [--ngrok]

param(
    [switch]$ngrok = $false
)

# Error handling
$ErrorActionPreference = "Stop"

# Constants
$VENV_PATH = ".\.venv\Scripts\Activate.ps1"
$SCRIPT_PATH = "run_mcp_slack.py"

# Display banner
Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "                  Running MCP Slack Integration                     " -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "`n❌ Virtual environment not found at $VENV_PATH" -ForegroundColor Red
    Write-Host "Please create a virtual environment first:" -ForegroundColor Yellow
    Write-Host "python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "`n🔄 Activating virtual environment..." -ForegroundColor Blue
try {
    & $VENV_PATH
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment: $_" -ForegroundColor Red
    exit 1
}

# Check for required Python packages
Write-Host "`n🔄 Checking required packages..." -ForegroundColor Blue
$REQUIRED_PACKAGES = @("fastapi", "uvicorn", "requests")
$MISSING_PACKAGES = @()

foreach ($pkg in $REQUIRED_PACKAGES) {
    try {
        $null = python -c "import $pkg"
        Write-Host "✅ Package $pkg is installed" -ForegroundColor Green
    } catch {
        Write-Host "❌ Package $pkg is missing" -ForegroundColor Red
        $MISSING_PACKAGES += $pkg
    }
}

if ($MISSING_PACKAGES.Count -gt 0) {
    Write-Host "`n🔄 Installing missing packages..." -ForegroundColor Blue
    $packages = $MISSING_PACKAGES -join " "
    python -m pip install $packages
    
    # Verify installation
    $STILL_MISSING = @()
    foreach ($pkg in $MISSING_PACKAGES) {
        try {
            $null = python -c "import $pkg"
            Write-Host "✅ Package $pkg is now installed" -ForegroundColor Green
        } catch {
            Write-Host "❌ Failed to install $pkg" -ForegroundColor Red
            $STILL_MISSING += $pkg
        }
    }
    
    if ($STILL_MISSING.Count -gt 0) {
        Write-Host "`n❌ Some required packages could not be installed. Please install them manually:" -ForegroundColor Red
        Write-Host "python -m pip install $($STILL_MISSING -join ' ')" -ForegroundColor Yellow
        exit 1
    }
}

# Run the application
Write-Host "`n🚀 Starting MCP Slack Integration..." -ForegroundColor Blue

if ($ngrok) {
    Write-Host "With ngrok tunnel enabled" -ForegroundColor Cyan
    python $SCRIPT_PATH --ngrok
} else {
    python $SCRIPT_PATH
}

# Deactivate virtual environment (will happen automatically when script ends)
Write-Host "`n👋 Slack bot terminated. Deactivating virtual environment..." -ForegroundColor Blue
deactivate
Write-Host "Done." -ForegroundColor Green 