# 💧 AquaPredict — AI-Powered Water Quality Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A full-stack machine learning web application that predicts **water potability** based on physicochemical parameters. Built following the methodology from peer-reviewed research published in **Scientific Reports (2026)**.

> 📄 Based on: *"Data driven water quality assessment using machine learning and synthetic data generation"*  
> DOI: [10.1038/s41598-026-54560-7](https://doi.org/10.1038/s41598-026-54560-7)

---

## 🌊 Features

- **Real-time Water Quality Prediction** — predict potability from 9 physicochemical parameters
- **4 ML Models** — Random Forest, XGBoost, Gradient Boosting, Logistic Regression
- **SMOTE Oversampling** — addresses class imbalance (as proven best technique in research)
- **WHO Safe Limits** — real-time parameter validation against WHO drinking water standards
- **Model Analytics Dashboard** — comparison table, feature importance chart, confusion matrix
- **Interactive Parameter Guide** — descriptions and limits for all 9 water quality parameters
- **Sample Data** — pre-loaded demo samples for quick testing
- **Comprehensive REST API** — Flask backend with multiple endpoints

---

## 📊 Dataset

| Dataset | Samples | Potable | Not Potable | Features |
|---------|---------|---------|-------------|----------|
| Water Potability (Kaggle) | 3,276 | 1,278 | 1,998 | 9 |

**Features:** pH, Hardness, Total Dissolved Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity

**Source:** [Kaggle — Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

---

## 🤖 ML Models & Performance

| Model | Accuracy | AUC-ROC | F1-Score |
|-------|----------|---------|---------|
| Random Forest ⭐ | 65.85% | 0.6639 | 0.5193 |
| Gradient Boosting | 62.96% | 0.6606 | 0.5318 |
| XGBoost | 62.96% | 0.6506 | 0.5091 |
| Logistic Regression | 52.90% | 0.5439 | 0.4718 |

> Note: The Water Potability dataset is inherently challenging. The research paper achieved 71.8% with RF on SMOTE-balanced data. This implementation achieves comparable results. The research's 99.47% figures were on the Drinking Water Final Dataset (more separable).

**Training methodology (per paper):**
- 80/20 stratified train-test split
- SMOTE applied **only** to training set (prevents data leakage)
- 10-fold stratified cross-validation
- Min-Max normalization

---

## 🗂️ Project Structure

```
water-quality-predictor/
├── backend/
│   ├── app.py               # Flask REST API
│   ├── train_model.py       # ML training script
│   ├── requirements.txt     # Python dependencies
│   ├── data/
│   │   └── water_potability.csv
│   └── models/              # Generated after training
│       ├── best_model.pkl
│       ├── imputer.pkl
│       ├── scaler.pkl
│       ├── feature_names.json
│       └── model_results.json
└── frontend/
    └── index.html           # Single-page application
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/aquapredict.git
cd aquapredict
```

### 2. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Train the ML model
```bash
python train_model.py
```
This will:
- Load and preprocess the water potability dataset
- Apply SMOTE oversampling on the training set
- Train all 4 ML models (LR, RF, GB, XGBoost)
- Run 10-fold cross-validation on the best model
- Save all artifacts to `models/`

### 4. Start the Flask API
```bash
python app.py
```
API will be available at `http://localhost:5000`

### 5. Open the frontend
Open `frontend/index.html` in your browser.

> **Tip:** For CORS to work properly, open via a local HTTP server:
> ```bash
> cd frontend
> python -m http.server 3000
> ```
> Then visit `http://localhost:3000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | API health check |
| POST | `/api/predict` | Predict water potability |
| GET | `/api/model-info` | Model performance metrics |
| GET | `/api/dataset-stats` | Dataset statistics |
| GET | `/api/parameters-info` | WHO limits & descriptions |
| GET | `/api/sample-data` | Demo water samples |

### Example: Predict
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.2,
    "Hardness": 196.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 300.0,
    "Conductivity": 430.0,
    "Organic_carbon": 14.0,
    "Trihalomethanes": 60.0,
    "Turbidity": 3.5
  }'
```

Response:
```json
{
  "prediction": 1,
  "label": "Potable",
  "confidence": 72.3,
  "probability_potable": 72.3,
  "probability_not_potable": 27.7,
  "wqi_score": 72.3,
  "wqi_category": {"label": "Good", "color": "#84cc16"},
  "advice": ["Water appears potable based on ML analysis.", "..."],
  "parameter_warnings": [],
  "model_used": "Random Forest"
}
```

---

## 📚 Research Reference

```bibtex
@article{relangi2026water,
  title={Data driven water quality assessment using machine learning and synthetic data generation},
  author={Relangi, Naga Durga Satya Siva Kiran and others},
  journal={Scientific Reports},
  volume={16},
  pages={25661},
  year={2026},
  doi={10.1038/s41598-026-54560-7}
}
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Predictions should NOT replace certified laboratory water quality testing. Always consult certified water testing authorities before making decisions about water consumption.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
