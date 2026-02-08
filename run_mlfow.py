# run_mlflow.py
import subprocess
import sys

def main():
    """Start MLflow UI"""
    print("Starting MLflow UI on http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    try:
        # Start MLflow UI
        subprocess.run(["mlflow", "ui", "--port", "5000", "--host", "0.0.0.0"])
    except KeyboardInterrupt:
        print("\nMLflow UI stopped")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()