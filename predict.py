import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import SalesDataPreprocessor
from src.feature_engineering import FeatureEngineer

class SalesPredictor:
    def __init__(self, model_path="models/best_sales_forecast_model.pkl"):
        self.model = joblib.load(model_path)
        self.preprocessor = SalesDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
    def predict_future(self, historical_data_path, forecast_days=30):
        """Predicting future sales based on historical data"""
        
        # Loading and preprocessing historical data
        df = self.preprocessor.load_data(historical_data_path)
        df_processed = self.preprocessor.preprocess(df)
        
        # Creating time series
        time_series = self.preprocessor.create_time_series_features(df_processed)
        
        # Engineer features
        df_features = self.feature_engineer.create_time_series_features(time_series)
        df_advanced = self.feature_engineer.create_advanced_features(df_processed, df_features)
        
        # Getting last N days for prediction
        last_date = df_advanced['Sale_Date'].max()
        start_date = last_date - timedelta(days=60)  # Using last 60 days as context
        
        recent_data = df_advanced[df_advanced['Sale_Date'] >= start_date].copy()
        
        # Generating predictions
        predictions = []
        current_data = recent_data.copy()
        
        for day in range(1, forecast_days + 1):
            # Preparing features for next day
            next_date = last_date + timedelta(days=day)
            
            # Creating feature row for prediction
            feature_row = self._create_prediction_row(current_data, next_date)
            
            # Making prediction
            features = feature_row.drop(['Sale_Date', 'Sales_Amount'], errors='ignore')
            prediction = self.model.predict(features.values.reshape(1, -1))[0]
            
            # Adding to predictions
            predictions.append({
                'Date': next_date,
                'Predicted_Sales': prediction,
                'Lower_Bound': prediction * 0.9,  # 10% lower bound
                'Upper_Bound': prediction * 1.1   # 10% upper bound
            })
            
            # Updating current data with prediction for next iteration
            feature_row['Sales_Amount'] = prediction
            current_data = pd.concat([current_data, pd.DataFrame([feature_row])], ignore_index=True)
            current_data = self.feature_engineer.create_time_series_features(
                current_data[['Sale_Date', 'Sales_Amount']].copy()
            )
        
        # Creating results dataframe
        results_df = pd.DataFrame(predictions)
        
        return results_df
    
    def _create_prediction_row(self, historical_data, prediction_date):
        """Create a feature row for prediction date"""
       
        last_row = historical_data.iloc[-1].copy()
        
        # Creating new row with prediction date
        new_row = last_row.copy()
        new_row['Sale_Date'] = prediction_date
        
        # Updating time-based features
        new_row['Year'] = prediction_date.year
        new_row['Month'] = prediction_date.month
        new_row['Day'] = prediction_date.day
        new_row['DayOfWeek'] = prediction_date.weekday()
        new_row['DayOfYear'] = prediction_date.timetuple().tm_yday
        new_row['WeekOfYear'] = prediction_date.isocalendar()[1]
        new_row['Is_Weekend'] = 1 if prediction_date.weekday() >= 5 else 0
        new_row['Is_Month_Start'] = 1 if prediction_date.day == 1 else 0
        new_row['Is_Month_End'] = 1 if prediction_date.day in [28, 29, 30, 31] else 0
        
        # Updating lag features (shift predictions)
        new_row['Lag_1'] = last_row['Sales_Amount']
        new_row['Lag_7'] = historical_data.iloc[-7]['Sales_Amount'] if len(historical_data) >= 7 else last_row['Sales_Amount']
        
        # Updating rolling features
        window_7 = historical_data['Sales_Amount'].tail(7)
        new_row['Rolling_Mean_7'] = window_7.mean()
        new_row['Rolling_Std_7'] = window_7.std()
        
        # Updating cyclical features
        new_row['Month_sin'] = np.sin(2 * np.pi * prediction_date.month/12)
        new_row['Month_cos'] = np.cos(2 * np.pi * prediction_date.month/12)
        new_row['DayOfYear_sin'] = np.sin(2 * np.pi * prediction_date.timetuple().tm_yday/365)
        new_row['DayOfYear_cos'] = np.cos(2 * np.pi * prediction_date.timetuple().tm_yday/365)
        
        return new_row
    
    def predict_single(self, input_features):
        """Predict sales for a single set of features"""
        # Ensuring features are in correct order
        feature_names = self.model.feature_names_in_
        input_df = pd.DataFrame([input_features])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        prediction = self.model.predict(input_df)[0]
        return prediction

def main():
    print("=" * 50)
    print("SALES PREDICTION")
    print("=" * 50)
    
    # Initializing predictor
    predictor = SalesPredictor()
    
    # Making predictions
    print("\nGenerating 30-day sales forecast...")
    predictions = predictor.predict_future("data/sales_data.csv", forecast_days=30)
    
    # Displaying results
    print("\n📊 SALES FORECAST RESULTS")
    print("-" * 60)
    
    # Summary statistics
    total_sales = predictions['Predicted_Sales'].sum()
    avg_daily = predictions['Predicted_Sales'].mean()
    max_daily = predictions['Predicted_Sales'].max()
    min_daily = predictions['Predicted_Sales'].min()
    growth = ((predictions.iloc[-1]['Predicted_Sales'] - predictions.iloc[0]['Predicted_Sales']) / 
              predictions.iloc[0]['Predicted_Sales']) * 100
    
    print(f"Total Predicted Sales (30 days): ${total_sales:,.2f}")
    print(f"Average Daily Sales: ${avg_daily:,.2f}")
    print(f"Maximum Daily Sales: ${max_daily:,.2f}")
    print(f"Minimum Daily Sales: ${min_daily:,.2f}")
    print(f"Growth Rate: {growth:.1f}%")
    
    # Showing detailed predictions
    print("\n📅 DETAILED FORECAST:")
    print("-" * 60)
    for idx, row in predictions.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')}: "
              f"${row['Predicted_Sales']:,.2f} "
              f"(${row['Lower_Bound']:,.2f} - ${row['Upper_Bound']:,.2f})")
    
    # Saving predictions
    predictions.to_csv('experiments/30_day_forecast.csv', index=False)
    print(f"\n✅ Forecast saved to: experiments/30_day_forecast.csv")
    
    # Example single prediction
    print("\n🧪 EXAMPLE SINGLE PREDICTION:")
    example_features = {
        'Lag_1': 1000,
        'Lag_7': 950,
        'Rolling_Mean_7': 980,
        'Rolling_Std_7': 50,
        'DayOfWeek': 1,  # Monday
        'Month': 1,      # January
        'Is_Weekend': 0
    }
    
    single_pred = predictor.predict_single(example_features)
    print(f"Input features: {example_features}")
    print(f"Predicted sales: ${single_pred:,.2f}")

if __name__ == "__main__":
    main()