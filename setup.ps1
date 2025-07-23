# Cài đặt dependencies cho backend
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Cài đặt dependencies cho frontend
Write-Host "Installing Node.js dependencies..." -ForegroundColor Green
Set-Location frontend
npm install
Set-Location ..

# Test Google AI connection
Write-Host "Testing Google AI connection..." -ForegroundColor Yellow
Set-Location backend
python test_google_ai.py
Set-Location ..

Write-Host "Setup completed!" -ForegroundColor Green
Write-Host "Run start.bat to start the application" -ForegroundColor Yellow
Write-Host "Don't forget to add your GOOGLE_API_KEY to .env file" -ForegroundColor Cyan
