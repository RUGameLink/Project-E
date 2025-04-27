import os
import pickle
import json
from pathlib import Path

# Set paths
BASE_PATH = Path("model_save_preset")
HISTORY_PATH = BASE_PATH / "history"

def inspect_history_file(file_path):
    """Load and inspect a history file (pickle or JSON)."""
    try:
        if str(file_path).endswith('.pkl'):
            with open(file_path, 'rb') as f:
                history = pickle.load(f)
        elif str(file_path).endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            print(f"Unsupported file format: {file_path}")
            return
        
        # Basic info
        print(f"File: {file_path}")
        print(f"Type: {type(history)}")
        
        # If it's a dictionary, print the keys
        if isinstance(history, dict):
            print(f"Top-level keys: {sorted(history.keys())}")
            
            # Check for model name and group name
            if 'model_name' in history:
                print(f"Model name: {history['model_name']}")
            if 'group_name' in history:
                print(f"Group name: {history['group_name']}")
            
            # Check for history nested data
            if 'history' in history and isinstance(history['history'], dict):
                print(f"History metrics: {sorted(history['history'].keys())}")
                
                # Print a sample of each metric
                for key in sorted(history['history'].keys()):
                    if isinstance(history['history'][key], list):
                        print(f"  {key}: {history['history'][key][:3]}... (total: {len(history['history'][key])})")
            
            # Check for metrics data
            if 'metrics' in history and isinstance(history['metrics'], dict):
                print(f"Final metrics: {history['metrics']}")
            
            # Check for epoch info
            if 'epoch' in history:
                print(f"Epochs: {history['epoch']}")
            
            # Check for params info
            if 'params' in history and isinstance(history['params'], dict):
                print(f"Params: {sorted(history['params'].keys())}")
        
        # Check for Keras History object
        elif hasattr(history, 'history'):
            print(f"Keys: {sorted(history.history.keys())}")
            
            # Print sample metrics
            for key in sorted(history.history.keys()):
                if isinstance(history.history[key], list):
                    print(f"  {key}: {history.history[key][:3]}... (total: {len(history.history[key])})")
        else:
            print(f"Unknown history format. Attributes: {dir(history)[:10]}")
        
        print("-" * 80)
    except Exception as e:
        print(f"Error inspecting file {file_path}: {e}")
        print("-" * 80)

def main():
    # Scan all model groups
    for group_dir in sorted(os.listdir(HISTORY_PATH)):
        group_path = HISTORY_PATH / group_dir
        
        if os.path.isdir(group_path):
            print(f"\n=== Group: {group_dir} ===\n")
            
            # Get all history files in the directory
            all_files = [f for f in os.listdir(group_path) if f.endswith('.pkl') or f.endswith('.json')]
            
            # Inspect just the first file from each group to avoid too much output
            if all_files:
                file_path = group_path / all_files[0]
                inspect_history_file(file_path)
                
                # Show summary of all files in the group
                print(f"Total files in group: {len(all_files)}")
                for history_file in all_files:
                    print(f"  - {history_file}")
                print("-" * 80)

if __name__ == "__main__":
    main() 