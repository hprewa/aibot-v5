# PowerShell script to run the MCP Slack App Home integration
param (
    [int]$Port = 8000,
    [switch]$UseNgrok
)

# Function to display banner
function Show-Banner {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host "         MCP SLACK APP HOME INTEGRATION LAUNCHER SCRIPT        " -ForegroundColor Cyan
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host ""
}

# Function to check and activate Python virtual environment
function Activate-VirtualEnv {
    $venvPath = ".\.venv"
    $activateScript = "$venvPath\Scripts\Activate.ps1"
    
    if (Test-Path $activateScript) {
        Write-Host "Activating Python virtual environment..." -ForegroundColor Yellow
        & $activateScript
        
        # Check if activation was successful
        if ($env:VIRTUAL_ENV) {
            Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
        } else {
            Write-Host "Failed to activate virtual environment." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Virtual environment not found at $venvPath." -ForegroundColor Yellow
        Write-Host "Creating a new virtual environment..." -ForegroundColor Yellow
        
        # Create virtual environment
        python -m venv .venv
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create virtual environment. Please check Python installation." -ForegroundColor Red
            exit 1
        }
        
        # Activate the new environment
        & $activateScript
        
        # Install requirements
        Write-Host "Installing requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to install requirements." -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Setup completed successfully!" -ForegroundColor Green
    }
}

# Function to check Slack environment variables
function Check-SlackEnvVars {
    $envFile = ".\.env"
    
    # Check if .env file exists
    if (-not (Test-Path $envFile)) {
        Write-Host ".env file not found. Creating a template..." -ForegroundColor Red
        
        if (Test-Path ".\.env.template") {
            Copy-Item -Path ".\.env.template" -Destination $envFile
            Write-Host "Created .env file from template. Please edit it to add your Slack credentials." -ForegroundColor Yellow
        } else {
            # Create a basic .env file
            @"
# Slack credentials
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# Other settings as needed
"@ | Out-File -FilePath $envFile -Encoding utf8
            
            Write-Host "Created basic .env file. Please edit it to add your Slack credentials." -ForegroundColor Yellow
        }
        
        Write-Host "You need to set up the following variables in your .env file:" -ForegroundColor Blue
        Write-Host "   - SLACK_BOT_TOKEN: Your Slack bot token (starts with xoxb-)" -ForegroundColor Blue
        Write-Host "   - SLACK_SIGNING_SECRET: Your Slack signing secret" -ForegroundColor Blue
        
        Write-Host "For instructions on how to set up your Slack app, see README_SLACK_SETUP.md" -ForegroundColor Blue
        
        $proceed = Read-Host "Would you like to continue anyway? (y/n)"
        if ($proceed -ne "y") {
            exit 0
        }
    }
    
    # Check for Slack token in environment
    if (-not $env:SLACK_BOT_TOKEN) {
        Write-Host "SLACK_BOT_TOKEN not found in environment. Make sure it's in your .env file." -ForegroundColor Yellow
    }
    
    # Check for Slack signing secret in environment
    if (-not $env:SLACK_SIGNING_SECRET) {
        Write-Host "SLACK_SIGNING_SECRET not found in environment. Make sure it's in your .env file." -ForegroundColor Yellow
    }
}

# Function to display setup reminder
function Show-SlackReminder {
    Write-Host ""
    Write-Host "REMINDER: Slack App Configuration" -ForegroundColor Green
    Write-Host "To use the App Home integration, make sure your Slack app has:" -ForegroundColor White
    Write-Host "1. The 'App Home' feature enabled" -ForegroundColor White
    Write-Host "2. The following scopes:" -ForegroundColor White
    Write-Host "   - chat:write" -ForegroundColor White
    Write-Host "   - im:history" -ForegroundColor White
    Write-Host "   - im:write" -ForegroundColor White
    Write-Host "   - app_home:update" -ForegroundColor White
    Write-Host "3. Event subscriptions for:" -ForegroundColor White
    Write-Host "   - app_home_opened" -ForegroundColor White
    Write-Host "   - message.im" -ForegroundColor White
    Write-Host ""
    Write-Host "If using ngrok, update your Event Subscription URL to:" -ForegroundColor White
    Write-Host "https://[your-ngrok-domain]/slack/events" -ForegroundColor White
    Write-Host ""
}

# Function to run the App Home integration
function Start-AppHome {
    param (
        [int]$Port = 8000,
        [switch]$UseNgrok
    )
    
    $ngrokParam = if ($UseNgrok) { "--ngrok" } else { "" }
    
    Write-Host "Starting MCP Slack App Home integration on port $Port" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    python run_mcp_app_home.py --port $Port $ngrokParam
}

# Main execution
Show-Banner
Activate-VirtualEnv
Check-SlackEnvVars
Show-SlackReminder
Start-AppHome -Port $Port -UseNgrok:$UseNgrok 