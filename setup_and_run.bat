@echo off
echo ============================================================
echo  AquaPredict - Water Quality Prediction Setup
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

echo [2/3] Training ML model (this may take 2-3 minutes)...
python train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Training failed
    pause
    exit /b 1
)
echo Model trained successfully!
echo.

echo [3/3] Starting Flask API server...
echo.
echo ============================================================
echo  API:      http://localhost:5000
echo  Frontend: Open frontend/index.html in your browser
echo  
echo  Press Ctrl+C to stop the server
echo ============================================================
echo.
python app.py
