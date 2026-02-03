import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

class SalesForecastingModel:
    def __init__(self, experiment_name="sales_forecasting"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setupinf MLflow tracking"""
        mlflow.set_experiment(self.experiment_name)
        print(f"MLflow experiment: {self.experiment_name}")
        
    def train_xgboost(self, X_train, y_train, X_test, y_test, params=None):
        """Training XGBoost model with MLflow tracking"""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        with mlflow.start_run(run_name="xgboost_forecast"):
            # Log parameters
            mlflow.log_params(params)
            
            # Training model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Making predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculating metrics
            metrics = self.calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.xgboost.log_model(model, "xgboost_model")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Saving feature importance as artifact
            importance_path = "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            print(f"XGBoost Model - Test RMSE: {metrics['test_rmse']:.2f}, R2: {metrics['test_r2']:.3f}")
            
            return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, params=None):
        """Training Random Forest model"""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        
        with mlflow.start_run(run_name="random_forest_forecast"):
            mlflow.log_params(params)
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            metrics = self.calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            print(f"Random Forest - Test RMSE: {metrics['test_rmse']:.2f}, R2: {metrics['test_r2']:.3f}")
            
            return model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, params=None):
        """Training Gradient Boosting model"""
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        
        with mlflow.start_run(run_name="gradient_boosting_forecast"):
            mlflow.log_params(params)
            
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            metrics = self.calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.sklearn.log_model(model, "gradient_boosting_model")
            
            print(f"Gradient Boosting - Test RMSE: {metrics['test_rmse']:.2f}, R2: {metrics['test_r2']:.3f}")
            
            return model, metrics
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='xgboost'):
        """Performing hyperparameter tuning"""
        if model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            model = xgb.XGBRegressor(random_state=42)
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        with mlflow.start_run(run_name=f"{model_type}_tuning"):
            grid_search = RandomizedSearchCV(
                model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_iter=10, random_state=42, n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", -grid_search.best_score_)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.2f}")
            
            return grid_search.best_estimator_, grid_search.best_params_
    
    def calculate_metrics(self, y_train, y_test, y_pred_train, y_pred_test):
        """Calculating evaluation metrics"""
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        return metrics
    
    def save_model(self, model, filepath):
        """Saving model to disk"""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Loading model from disk"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model