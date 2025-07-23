@echo off
echo Starting LegalSLM Data Tool...
echo.

echo [1/3] Starting Backend (Flask)...
cd backend
start cmd /k "python app.py"
cd ..

echo [2/3] Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo [3/3] Starting Frontend (React)...
cd frontend
start cmd /k "npm start"
cd ..

echo.
echo ================================
echo LegalSLM Data Tool is starting...
echo ================================
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul
