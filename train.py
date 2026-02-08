import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import SalesDataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import SalesForecastingModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 50)
    print("SALES FORECASTING PIPELINE")
    print("=" * 50)
    
    # Initializing components
    preprocessor = SalesDataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_trainer = SalesForecastingModel()
    
    # Step 1: Loading and preprocessing data
    print("\n1. Loading and preprocessing data...")
    data_path = "D:\sales forecasting project\sales_data.csv"
    df = preprocessor.load_data(data_path)
    df_processed = preprocessor.preprocess(df)
    
    # Step 2: Creating time series data
    print("\n2. Creating time series features...")
    time_series = preprocessor.create_time_series_features(df_processed)
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    df_features = feature_engineer.create_time_series_features(time_series)
    df_advanced = feature_engineer.create_advanced_features(df_processed, df_features)
    
    # Step 4: Preparing train/test split
    print("\n4. Preparing train/test split...")
    X_train, X_test, y_train, y_test, train_df, test_df = feature_engineer.prepare_train_test(
        df_advanced, test_size=0.2
    )
    
    # Step 5: Training different models
    print("\n5. Training models...")
    
    # Training XGBoost
    print("\nTraining XGBoost model...")
    xgb_model, xgb_metrics = model_trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Training Random Forest
    print("\nTraining Random Forest model...")
    rf_model, rf_metrics = model_trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    # Training Gradient Boosting
    print("\nTraining Gradient Boosting model...")
    gb_model, gb_metrics = model_trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # Step 6: Hyperparameter tuning for best model
    print("\n6. Performing hyperparameter tuning...")
    best_model, best_params = model_trainer.hyperparameter_tuning(X_train, y_train, 'xgboost')
    
    # Training final model with best parameters
    print("\n7. Training final model with best parameters...")
    final_model, final_metrics = model_trainer.train_xgboost(
        X_train, y_train, X_test, y_test, best_params
    )
    
    # Step 7: Saving the best model
    print("\n8. Saving the best model...")
    model_trainer.save_model(final_model, "models/best_sales_forecast_model.pkl")
    
    # Step 8: Visualizing results
    visualize_results(train_df, test_df, X_test, final_model)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Best Model RMSE: {final_metrics['test_rmse']:.2f}")
    print(f"Best Model R2: {final_metrics['test_r2']:.3f}")
    print("=" * 50)
    print("\nTo start MLflow UI, run: mlflow ui")
    print("MLflow tracking URI: http://localhost:5000")

def visualize_results(train_df, test_df, X_test, model):
    """Create visualization of model performance"""
    os.makedirs("experiments", exist_ok=True)
    
    # Making predictions
    y_pred_test = model.predict(X_test)
    
    # Creating comparison dataframe
    results_df = test_df[['Sale_Date', 'Sales_Amount']].copy()
    results_df['Predicted'] = y_pred_test
    results_df['Error'] = results_df['Sales_Amount'] - results_df['Predicted']
    
    # Plotting actual vs predicted
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['Sale_Date'], results_df['Sales_Amount'], label='Actual', alpha=0.7)
    plt.plot(results_df['Sale_Date'], results_df['Predicted'], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['Sales_Amount'], results_df['Predicted'], alpha=0.6)
    plt.plot([results_df['Sales_Amount'].min(), results_df['Sales_Amount'].max()],
             [results_df['Sales_Amount'].min(), results_df['Sales_Amount'].max()], 
             'r--', label='Perfect Prediction')
    plt.title('Prediction Scatter Plot')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(results_df['Error'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    error_series = results_df.set_index('Sale_Date')['Error']
    error_series.rolling(window=7).mean().plot()
    plt.title('7-Day Rolling Mean of Errors')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Saving results to CSV
    results_df.to_csv('experiments/predictions_results.csv', index=False)
    print("Visualizations saved to 'experiments/' directory")

if __name__ == "__main__":
    # Creating necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    main()