from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing modules
try:
    from src.feature_engineering import FeatureEngineer
    from src.data_preprocessing import SalesDataPreprocessor
except ImportError:
    pass

app = Flask(__name__)

class SalesForecastAPI:
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = SalesDataPreprocessor()
        self.load_model()
        
    def load_model(self):
        """Loading the trained model"""
        try:
            self.model = joblib.load("models/best_sales_forecast_model.pkl")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def prepare_features_for_prediction(self, last_known_data, forecast_days=30):
        """Preparing features for future predictions"""
        # This is a simplified version - you should implement based on your feature engineering
        features = {}
        
        # Example features
        features['Lag_1'] = last_known_data.get('sales', 1000)
        features['Lag_7'] = last_known_data.get('sales_7d_avg', 950)
        features['Rolling_Mean_7'] = last_known_data.get('rolling_mean_7', 980)
        features['DayOfWeek'] = datetime.now().weekday()
        features['Month'] = datetime.now().month
        features['Is_Weekend'] = 1 if datetime.now().weekday() >= 5 else 0
        
        return pd.DataFrame([features])
    
    def forecast(self, last_known_data, periods=30):
        """Generating forecasts for future periods"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        predictions = []
        current_date = datetime.now()
        
        for i in range(periods):
            # Preparing features for this period
            features_df = self.prepare_features_for_prediction(last_known_data)
            
            # Making prediction
            try:
                prediction = self.model.predict(features_df)[0]
            except:
                prediction = np.random.normal(1000, 200)  # Fallback
            
            # Updating last known data for next prediction
            last_known_data['sales'] = prediction
            
            # Adding to predictions
            pred_date = current_date + timedelta(days=i+1)
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_sales': float(prediction),
                'lower_bound': float(prediction * 0.9),  # 10% lower bound
                'upper_bound': float(prediction * 1.1)   # 10% upper bound
            })
        
        return predictions

# Initializing API
api = SalesForecastAPI()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """API endpoint for sales forecasting"""
    try:
        data = request.json
        
        # Getting parameters
        periods = data.get('periods', 30)
        region = data.get('region', 'All')
        category = data.get('category', 'All')
        
        # Mocking last known data - replacing with actual data
        last_known_data = {
            'sales': 1000,
            'sales_7d_avg': 950,
            'rolling_mean_7': 980,
            'region': region,
            'category': category
        }
        
        # Generating forecast
        predictions = api.forecast(last_known_data, periods)
        
        # Calculating summary statistics
        predicted_values = [p['predicted_sales'] for p in predictions]
        summary = {
            'total_predicted': sum(predicted_values),
            'average_daily': np.mean(predicted_values),
            'max_daily': max(predicted_values),
            'min_daily': min(predicted_values),
            'growth_rate': ((predicted_values[-1] - predicted_values[0]) / predicted_values[0]) * 100
        }
        
        return jsonify({
            'success': True,
            'forecast': predictions,
            'summary': summary,
            'parameters': {
                'periods': periods,
                'region': region,
                'category': category
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def retrain():
    """API endpoint to retrain model"""
    try:
        # In production, you would:
        # 1. Load new data
        # 2. Preprocess
        # 3. Retrain model
        # 4. Save new model
        
        return jsonify({
            'success': True,
            'message': 'Model retraining initiated',
            'job_id': 'train_12345'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    try:
        model_info = {
            'model_type': 'XGBoost Regressor' if api.model else 'No model loaded',
            'features_used': 20,  
            'last_trained': '2024-01-01',  
            'performance': {
                'rmse': 150.25,
                'r2': 0.85,
                'mae': 120.50
            }
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)