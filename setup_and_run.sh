#!/bin/bash
echo "============================================================"
echo " AquaPredict - Water Quality Prediction Setup"
echo "============================================================"
echo ""

echo "[1/3] Installing Python dependencies..."
cd backend
pip install -r requirements.txt || { echo "ERROR: pip install failed"; exit 1; }
echo "Done!"
echo ""

echo "[2/3] Training ML model..."
python train_model.py || { echo "ERROR: Training failed"; exit 1; }
echo "Done!"
echo ""

echo "[3/3] Starting Flask API..."
echo ""
echo "============================================================"
echo " API:      http://localhost:5000"
echo " Frontend: Open frontend/index.html in your browser"
echo " Press Ctrl+C to stop"
echo "============================================================"
echo ""
python app.py
