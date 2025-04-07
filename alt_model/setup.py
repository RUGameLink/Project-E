import subprocess
import sys
import os
import platform


def check_python_version():
    """Check if Python version is at least 3.8."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current Python version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"Python version check passed: {current_version[0]}.{current_version[1]}")
    return True


def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        print("Installing required packages...")
        # Use the path relative to the script location
        requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Package installation completed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install required packages.")
        return False


def check_tensorflow():
    """Check if TensorFlow can be imported and is using GPU if available."""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        if tf.config.list_physical_devices('GPU'):
            print("TensorFlow is using GPU.")
        else:
            print("TensorFlow is using CPU only.")
            print("Note: For better performance, consider using a system with GPU support.")
        
        return True
    except ImportError:
        print("Error: Failed to import TensorFlow.")
        return False


def create_required_directories():
    """Create directories that are required but might not exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_output_dir = os.path.join(script_dir, "models")
    
    # Make sure models directory exists
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)
        print(f"Created directory: {models_output_dir}")
    else:
        print(f"Directory already exists: {models_output_dir}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("Radon Level Prediction Model - Setup")
    print("=" * 60 + "\n")
    
    print("System information:")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create required directories
    if not create_required_directories():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check TensorFlow
    if not check_tensorflow():
        return
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nYou can now run the project scripts:")
    print("1. To run all components in sequence:")
    print("   python models/run_all.py")
    print("\n2. To run components individually:")
    print("   python models/analyze_radon_data.py")
    print("   python models/neural_network_model.py")
    print("   python models/predict_radon.py")


if __name__ == "__main__":
    main() 