import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import json
from datetime import datetime

def track_experiments():
    """Tracking and analyzing MLflow experiments"""
    
    # Initializing MLflow client
    client = MlflowClient()
    
    # Getting all experiments
    experiments = client.search_experiments()
    
    print("=" * 60)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name}")
        print(f"Experiment ID: {exp.experiment_id}")
        
        # Getting runs for this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        if runs:
            # Creating DataFrame of runs
            run_data = []
            for run in runs:
                run_info = {
                    'Run ID': run.info.run_id,
                    'Run Name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'Start Time': datetime.fromtimestamp(run.info.start_time/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'Status': run.info.status,
                    'Test RMSE': run.data.metrics.get('test_rmse', None),
                    'Test R2': run.data.metrics.get('test_r2', None),
                    'Model Type': run.data.tags.get('model_type', 'N/A')
                }
                run_data.append(run_info)
            
            df_runs = pd.DataFrame(run_data)
            
            # Displaying summary
            print(f"Total runs: {len(runs)}")
            print("\nBest performing runs:")
            best_runs = df_runs.sort_values('Test R2', ascending=False).head(3)
            print(best_runs[['Run Name', 'Test RMSE', 'Test R2', 'Model Type']].to_string(index=False))
            
            # Saving to CSV
            df_runs.to_csv(f'experiments/{exp.name}_runs_summary.csv', index=False)
            print(f"Summary saved to: experiments/{exp.name}_runs_summary.csv")
    
    # Comparing models across experiments
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    all_runs_data = []
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for run in runs:
            if run.data.metrics.get('test_rmse'):
                all_runs_data.append({
                    'Experiment': exp.name,
                    'Run Name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'Model Type': run.data.tags.get('model_type', 'N/A'),
                    'Test RMSE': run.data.metrics.get('test_rmse'),
                    'Test R2': run.data.metrics.get('test_r2'),
                    'Train RMSE': run.data.metrics.get('train_rmse'),
                    'Parameters': json.dumps(run.data.params) if run.data.params else 'N/A'
                })
    
    if all_runs_data:
        comparison_df = pd.DataFrame(all_runs_data)
        
        # Grouping by model type
        model_comparison = comparison_df.groupby('Model Type').agg({
            'Test RMSE': ['mean', 'min', 'max'],
            'Test R2': ['mean', 'min', 'max']
        }).round(3)
        
        print("\nModel Performance Comparison:")
        print(model_comparison.to_string())
        
        # Saving comparison
        comparison_df.to_csv('experiments/all_models_comparison.csv', index=False)
        print(f"\nDetailed comparison saved to: experiments/all_models_comparison.csv")
        
        # Finding best model
        best_run = comparison_df.loc[comparison_df['Test R2'].idxmax()]
        print(f"\n🌟 BEST MODEL:")
        print(f"   Model Type: {best_run['Model Type']}")
        print(f"   Run Name: {best_run['Run Name']}")
        print(f"   Test R2: {best_run['Test R2']:.3f}")
        print(f"   Test RMSE: {best_run['Test RMSE']:.2f}")
        
        # Registering best model
        try:
            best_run_id = None
            for exp in experiments:
                runs = client.search_runs(experiment_ids=[exp.experiment_id])
                for run in runs:
                    if run.data.tags.get('mlflow.runName') == best_run['Run Name']:
                        best_run_id = run.info.run_id
                        break
                if best_run_id:
                    break
            
            if best_run_id:
                # Register model
                result = mlflow.register_model(
                    f"runs:/{best_run_id}/xgboost_model",
                    "best_sales_forecast_model"
                )
                print(f"\n✅ Model registered: {result.name} (Version {result.version})")
                
        except Exception as e:
            print(f"Note: Could not register model - {e}")

def serve_mlflow_ui():
    """Starting MLflow UI"""
    import subprocess
    import time
    
    print("\n" + "=" * 60)
    print("STARTING MLFLOW UI")
    print("=" * 60)
    print("\nMLflow UI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Starting MLflow UI
        subprocess.run(["mlflow", "ui", "--port", "5000", "--host", "0.0.0.0"])
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped")

if __name__ == "__main__":
    # Tracking experiments
    track_experiments()
    
    # Asking if user wants to start UI
    response = input("\nStart MLflow UI? (y/n): ")
    if response.lower() == 'y':
        serve_mlflow_ui()