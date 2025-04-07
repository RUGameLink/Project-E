import os
import subprocess
import time

def run_script(script_name):
    """Run a Python script and time its execution."""
    print(f"\n{'=' * 80}")
    print(f"Running {script_name}...")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    subprocess.run(['python', script_name], check=True)
    end_time = time.time()
    
    print(f"\n{'=' * 80}")
    print(f"{script_name} completed in {end_time - start_time:.2f} seconds.")
    print(f"{'=' * 80}\n")

def main():
    # Make sure we're in the models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\nRadon Level Prediction Project")
    print("=" * 30)
    print("This script will run all components of the project in sequence:")
    print("1. Data Analysis")
    print("2. Model Training")
    print("3. Making Predictions")
    print("=" * 30)
    
    # Ask user whether to continue
    response = input("\nDo you want to continue? This may take some time to complete. (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return
    
    # Step 1: Analyze data
    print("\nStep 1: Analyzing radon data and relationships with temperature and pressure.")
    run_script('analyze_radon_data.py')
    
    # Step 2: Train model
    print("\nStep 2: Training neural network models. This may take some time...")
    run_script('neural_network_model.py')
    
    # Step 3: Make predictions
    print("\nStep 3: Making predictions with the best model.")
    run_script('predict_radon.py')
    
    print("\nAll processes completed successfully!")
    print("Results and trained models have been saved to the current directory.")
    print("\nTo see the analysis results and predictions, check the following files:")
    print("- time_series_analysis.png")
    print("- correlation_heatmap.png")
    print("- scatter_relationships.png")
    print("- lag_correlations.png")
    print("- *_training_history.png")
    print("- radon_prediction_result.png")

if __name__ == "__main__":
    main() 