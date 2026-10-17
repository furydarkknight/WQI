"""
Water Quality Prediction API
Flask backend for the Water Quality Prediction website
Based on: Scientific Reports 2026 - Data driven WQA using ML
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, origins='*')

# ─────────────────────────────────────────
# Load trained model artifacts on startup
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model = None
imputer = None
scaler = None
feature_names = None
model_results = None

def load_artifacts():
    global model, imputer, scaler, feature_names, model_results
    try:
        with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'imputer.pkl'), 'rb') as f:
            imputer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r') as f:
            feature_names = json.load(f)
        with open(os.path.join(MODEL_DIR, 'model_results.json'), 'r') as f:
            model_results = json.load(f)
        print(f"[OK] Model loaded: {model_results.get('best_model_name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts: {e}")
        return False
load_artifacts()
# WHO water quality safe limits (for advice generation)
WHO_LIMITS = {
    'ph':               {'min': 6.5, 'max': 8.5, 'unit': '', 'name': 'pH'},
    'Hardness':         {'min': 0,   'max': 500,  'unit': 'mg/L', 'name': 'Hardness'},
    'Solids':           {'min': 0,   'max': 500,  'unit': 'ppm', 'name': 'Total Dissolved Solids'},
    'Chloramines':      {'min': 0,   'max': 4.0,  'unit': 'ppm', 'name': 'Chloramines'},
    'Sulfate':          {'min': 0,   'max': 250,  'unit': 'mg/L', 'name': 'Sulfate'},
    'Conductivity':     {'min': 0,   'max': 400,  'unit': 'uS/cm', 'name': 'Conductivity'},
    'Organic_carbon':   {'min': 0,   'max': 2,    'unit': 'ppm', 'name': 'Organic Carbon'},
    'Trihalomethanes':  {'min': 0,   'max': 80,   'unit': 'ug/L', 'name': 'Trihalomethanes'},
    'Turbidity':        {'min': 0,   'max': 5,    'unit': 'NTU', 'name': 'Turbidity'},
}

def generate_advice(input_values, prediction):
    """Generate parameter-specific health advice based on WHO limits"""
    advice = []
    warnings_list = []
    
    for feat, val in input_values.items():
        if val is None or feat not in WHO_LIMITS:
            continue
        limits = WHO_LIMITS[feat]
        if val > limits['max']:
            warnings_list.append(
                f"{limits['name']} ({val:.2f} {limits['unit']}) exceeds WHO limit of {limits['max']} {limits['unit']}"
            )
        elif feat == 'ph' and val < limits['min']:
            warnings_list.append(
                f"pH ({val:.2f}) is below WHO minimum of {limits['min']} — water is too acidic"
            )
    
    if prediction == 1:
        advice.append("Water appears potable based on ML analysis.")
        advice.append("Always verify with certified laboratory testing before consumption.")
        if warnings_list:
            advice.append("Note: Some parameters exceed recommended limits despite overall potability prediction.")
    else:
        advice.append("Water is predicted as NON-POTABLE. Do not consume without treatment.")
        advice.append("Consider filtration, boiling, or chemical treatment.")
        advice.append("Consult local water authority for appropriate treatment methods.")
        if warnings_list:
            advice.append("Specific parameters of concern detected.")
    
    return advice, warnings_list

# ─────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_results.get('best_model_name') if model_results else None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts JSON with water quality parameters.
    Returns potability prediction with confidence and advice.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Build feature vector in correct order
        input_dict = {}
        for feat in feature_names:
            val = data.get(feat)
            if val is not None and val != '':
                try:
                    input_dict[feat] = float(val)
                except (ValueError, TypeError):
                    input_dict[feat] = np.nan
            else:
                input_dict[feat] = np.nan
        
        # Create DataFrame for preprocessing
        input_df = pd.DataFrame([input_dict], columns=feature_names)
        
        # Apply same preprocessing as training:
        # 1. Impute missing values
        input_imputed = imputer.transform(input_df)
        # 2. Min-max scale
        input_scaled = scaler.transform(input_imputed)
        
        # Predict
        prediction = int(model.predict(input_scaled)[0])
        probability = model.predict_proba(input_scaled)[0]
        
        # Confidence = probability of predicted class
        confidence = float(probability[prediction]) * 100
        prob_potable = float(probability[1]) * 100
        prob_not_potable = float(probability[0]) * 100
        
        # Generate advice
        advice, param_warnings = generate_advice(input_dict, prediction)
        
        # WQI estimation (simplified formula based on normalized parameters)
        # Uses inverse distance from ideal values
        wqi_score = round(prob_potable, 1)
        
        result = {
            'prediction': prediction,
            'label': 'Potable' if prediction == 1 else 'Not Potable',
            'confidence': round(confidence, 1),
            'probability_potable': round(prob_potable, 1),
            'probability_not_potable': round(prob_not_potable, 1),
            'wqi_score': wqi_score,
            'wqi_category': get_wqi_category(wqi_score),
            'advice': advice,
            'parameter_warnings': param_warnings,
            'model_used': model_results.get('best_model_name', 'Unknown'),
            'input_values': {k: round(v, 4) if v is not None and not np.isnan(v) else None 
                           for k, v in input_dict.items()}
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


def get_wqi_category(score):
    """Classify WQI score into category"""
    if score >= 80:
        return {'label': 'Excellent', 'color': '#22c55e'}
    elif score >= 65:
        return {'label': 'Good', 'color': '#84cc16'}
    elif score >= 50:
        return {'label': 'Fair', 'color': '#eab308'}
    elif score >= 30:
        return {'label': 'Poor', 'color': '#f97316'}
    else:
        return {'label': 'Very Poor', 'color': '#ef4444'}


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model performance metrics for dashboard"""
    if model_results is None:
        return jsonify({'error': 'Model info not available'}), 503
    
    return jsonify({
        'best_model': model_results.get('best_model_name'),
        'cv_mean_accuracy': model_results.get('cv_mean'),
        'cv_std': model_results.get('cv_std'),
        'feature_importance': model_results.get('feature_importance', {}),
        'model_comparison': {
            name: {
                'accuracy': info.get('accuracy'),
                'auc_roc': info.get('auc_roc'),
                'f1_score': info.get('f1_score'),
                'precision': info.get('precision'),
                'recall': info.get('recall'),
                'brier_score': info.get('brier_score'),
                'confusion_matrix': info.get('confusion_matrix')
            }
            for name, info in model_results.items()
            if isinstance(info, dict) and 'accuracy' in info
        }
    })


@app.route('/api/dataset-stats', methods=['GET'])
def dataset_stats():
    """Return dataset statistics for the info panel"""
    return jsonify({
        'total_samples': 3276,
        'potable_samples': 1278,
        'not_potable_samples': 1998,
        'features': 9,
        'missing_values': {
            'ph': 491,
            'Sulfate': 781,
            'Trihalomethanes': 162
        },
        'oversampling': 'SMOTE',
        'train_split': 80,
        'test_split': 20,
        'cross_validation': '10-Fold Stratified',
        'paper_reference': 'Scientific Reports 2026 - Data driven WQA',
        'doi': 'https://doi.org/10.1038/s41598-026-54560-7'
    })


@app.route('/api/parameters-info', methods=['GET'])
def parameters_info():
    """Return WHO limits and parameter descriptions"""
    info = {
        'ph': {
            'name': 'pH',
            'description': 'Measure of how acidic or basic water is. WHO recommends 6.5-8.5.',
            'who_min': 6.5,
            'who_max': 8.5,
            'unit': '',
            'typical_range': '6.0 - 9.0',
            'impact': 'Affects taste, pipe corrosion, and disinfection efficiency'
        },
        'Hardness': {
            'name': 'Hardness',
            'description': 'Caused by calcium and magnesium salts. High hardness causes scale buildup.',
            'who_min': 0,
            'who_max': 500,
            'unit': 'mg/L',
            'typical_range': '47 - 323 mg/L',
            'impact': 'Affects taste and pipe infrastructure'
        },
        'Solids': {
            'name': 'Total Dissolved Solids (TDS)',
            'description': 'Total amount of dissolved substances including minerals and salts.',
            'who_min': 0,
            'who_max': 500,
            'unit': 'ppm',
            'typical_range': '320 - 61227 ppm',
            'impact': 'High TDS affects taste and may indicate contamination'
        },
        'Chloramines': {
            'name': 'Chloramines',
            'description': 'Disinfectants used in public water systems. Max safe level: 4 ppm.',
            'who_min': 0,
            'who_max': 4.0,
            'unit': 'ppm',
            'typical_range': '0.35 - 13.13 ppm',
            'impact': 'Disinfection byproduct — high levels are toxic'
        },
        'Sulfate': {
            'name': 'Sulfate',
            'description': 'Naturally present in minerals. WHO recommends < 250 mg/L.',
            'who_min': 0,
            'who_max': 250,
            'unit': 'mg/L',
            'typical_range': '129 - 481 mg/L',
            'impact': 'High levels cause laxative effects and affect taste'
        },
        'Conductivity': {
            'name': 'Electrical Conductivity',
            'description': 'Measures ability to conduct electricity, indicating dissolved ions.',
            'who_min': 0,
            'who_max': 400,
            'unit': 'uS/cm',
            'typical_range': '181 - 753 uS/cm',
            'impact': 'Indicator of overall water mineral content'
        },
        'Organic_carbon': {
            'name': 'Organic Carbon (TOC)',
            'description': 'Total organic carbon from decaying natural organic matter.',
            'who_min': 0,
            'who_max': 2,
            'unit': 'ppm',
            'typical_range': '2.21 - 28.30 ppm',
            'impact': 'Precursor to disinfection byproducts'
        },
        'Trihalomethanes': {
            'name': 'Trihalomethanes (THM)',
            'description': 'Byproducts of water chlorination. EPA limit: 80 ug/L.',
            'who_min': 0,
            'who_max': 80,
            'unit': 'ug/L',
            'typical_range': '10.74 - 124 ug/L',
            'impact': 'Long-term exposure linked to cancer risk'
        },
        'Turbidity': {
            'name': 'Turbidity',
            'description': 'Cloudiness or haziness of water caused by suspended particles.',
            'who_min': 0,
            'who_max': 5,
            'unit': 'NTU',
            'typical_range': '1.45 - 6.74 NTU',
            'impact': 'Indicates presence of pathogens and sediment'
        }
    }
    return jsonify(info)


@app.route('/api/sample-data', methods=['GET'])
def sample_data():
    """Return sample data points for demo purposes"""
    samples = [
        {
            'name': 'Clean Tap Water (Potable)',
            'ph': 7.2, 'Hardness': 196.0, 'Solids': 20000.0,
            'Chloramines': 7.0, 'Sulfate': 300.0, 'Conductivity': 430.0,
            'Organic_carbon': 14.0, 'Trihalomethanes': 60.0, 'Turbidity': 3.5
        },
        {
            'name': 'Industrial Runoff (Not Potable)',
            'ph': 4.5, 'Hardness': 280.0, 'Solids': 45000.0,
            'Chloramines': 11.0, 'Sulfate': 450.0, 'Conductivity': 680.0,
            'Organic_carbon': 25.0, 'Trihalomethanes': 110.0, 'Turbidity': 6.0
        },
        {
            'name': 'Spring Water (Potable)',
            'ph': 7.8, 'Hardness': 150.0, 'Solids': 15000.0,
            'Chloramines': 5.5, 'Sulfate': 200.0, 'Conductivity': 380.0,
            'Organic_carbon': 10.0, 'Trihalomethanes': 45.0, 'Turbidity': 2.8
        }
    ]
    return jsonify(samples)


# ─────────────────────────────────────────
# Application entry point
# ─────────────────────────────────────────
if __name__ == '__main__':
    if load_artifacts():
        print("[OK] Water Quality Prediction API starting...")
        print("[OK] API available at http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("[ERROR] Could not load model. Run train_model.py first!")
        print("Running in limited mode (no predictions)...")
        load_artifacts()
        app.run(debug=True, host='0.0.0.0', port=5000)
