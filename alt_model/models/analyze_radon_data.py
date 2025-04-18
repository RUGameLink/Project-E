import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def load_data(file_path):
    """Load and preprocess data from CSV file."""
    try:
        # Try different encodings
        data = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, delimiter=';', encoding='cp1252')
    
    # Convert columns to proper format
    data['Temperature (¡C)'] = data['Temperature (¡C)'].str.replace(',', '.').astype(float)
    data['Pressure (mBar)'] = data['Pressure (mBar)'].str.replace(',', '.').astype(float)
    
    # Convert datetime and set as index
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d.%m.%Y %H:%M')
    data.set_index('Datetime', inplace=True)
    
    # Fill missing values with mean for each column
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].mean(), inplace=True)
    
    return data

def plot_time_series(data):
    """Plot time series data for Radon, Temperature, and Pressure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Radon plot
    axes[0].plot(data.index, data['Radon (Bq.m3)'], 'b-')
    axes[0].set_title('Radon Levels Over Time')
    axes[0].set_ylabel('Radon (Bq.m3)')
    axes[0].grid(True)
    
    # Temperature plot
    axes[1].plot(data.index, data['Temperature (¡C)'], 'r-')
    axes[1].set_title('Temperature Over Time')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].grid(True)
    
    # Pressure plot
    axes[2].plot(data.index, data['Pressure (mBar)'], 'g-')
    axes[2].set_title('Pressure Over Time')
    axes[2].set_ylabel('Pressure (mBar)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png')
    plt.close()

def analyze_correlations(data):
    """Analyze correlations between variables."""
    # Calculate correlations
    corr_radon_temp, p_radon_temp = pearsonr(data['Radon (Bq.m3)'], data['Temperature (¡C)'])
    corr_radon_pressure, p_radon_pressure = pearsonr(data['Radon (Bq.m3)'], data['Pressure (mBar)'])
    corr_temp_pressure, p_temp_pressure = pearsonr(data['Temperature (¡C)'], data['Pressure (mBar)'])
    
    print(f"Correlation between Radon and Temperature: {corr_radon_temp:.4f} (p-value: {p_radon_temp:.4f})")
    print(f"Correlation between Radon and Pressure: {corr_radon_pressure:.4f} (p-value: {p_radon_pressure:.4f})")
    print(f"Correlation between Temperature and Pressure: {corr_temp_pressure:.4f} (p-value: {p_temp_pressure:.4f})")
    
    # Create correlation matrix
    corr_matrix = data.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    return corr_matrix

def plot_scatter_relationships(data):
    """Plot scatter plots to visualize relationships between variables."""
    plt.figure(figsize=(15, 5))
    
    # Radon vs Temperature
    plt.subplot(1, 3, 1)
    plt.scatter(data['Temperature (¡C)'], data['Radon (Bq.m3)'], alpha=0.5)
    plt.title('Radon vs Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Radon (Bq.m3)')
    
    # Radon vs Pressure
    plt.subplot(1, 3, 2)
    plt.scatter(data['Pressure (mBar)'], data['Radon (Bq.m3)'], alpha=0.5)
    plt.title('Radon vs Pressure')
    plt.xlabel('Pressure (mBar)')
    plt.ylabel('Radon (Bq.m3)')
    
    # Temperature vs Pressure
    plt.subplot(1, 3, 3)
    plt.scatter(data['Temperature (¡C)'], data['Pressure (mBar)'], alpha=0.5)
    plt.title('Temperature vs Pressure')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (mBar)')
    
    plt.tight_layout()
    plt.savefig('scatter_relationships.png')
    plt.close()

def analyze_statistics(data):
    """Calculate and print basic statistics."""
    stats = data.describe()
    print("\nBasic Statistics:")
    print(stats)
    
    # Save statistics to CSV
    stats.to_csv('data_statistics.csv')
    
    # Plot histograms
    data.hist(bins=30, figsize=(15, 5))
    plt.tight_layout()
    plt.savefig('data_histograms.png')
    plt.close()

def perform_lag_analysis(data, lag_days=7):
    """Analyze the effect of lagged variables."""
    # Create lagged features
    lag_hours = lag_days * 24
    
    # For demonstration, we'll create lags for temp and pressure to predict radon
    for lag in [1, 6, 12, 24, 48, lag_hours]:
        data[f'Temp_Lag_{lag}h'] = data['Temperature (¡C)'].shift(lag)
        data[f'Pressure_Lag_{lag}h'] = data['Pressure (mBar)'].shift(lag)
    
    # Drop NaN values created by shifting
    data_lag = data.dropna()
    
    # Calculate correlations with lagged variables
    corr_lag = data_lag.corr()['Radon (Bq.m3)'].sort_values(ascending=False)
    
    print("\nCorrelations with Radon (including lagged variables):")
    print(corr_lag)
    
    # Plot top correlations
    plt.figure(figsize=(12, 6))
    corr_lag.drop('Radon (Bq.m3)').plot(kind='bar')
    plt.title('Correlation of Variables with Radon Levels')
    plt.ylabel('Correlation Coefficient')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('lag_correlations.png')
    plt.close()
    
    return corr_lag

def main():
    # Load data
    print("Loading and preprocessing data...")
    data = load_data('../../data/data_fragment.csv')
    print("Data shape:", data.shape)
    print("Data columns:", data.columns)
    print("First few rows:")
    print(data.head())
    
    # Plot time series
    print("\nPlotting time series data...")
    plot_time_series(data)
    
    # Analyze correlations
    print("\nAnalyzing correlations...")
    corr_matrix = analyze_correlations(data)
    
    # Plot scatter relationships
    print("\nPlotting scatter relationships...")
    plot_scatter_relationships(data)
    
    # Analyze statistics
    print("\nAnalyzing basic statistics...")
    analyze_statistics(data)
    
    # Perform lag analysis
    print("\nPerforming lag analysis...")
    lag_correlations = perform_lag_analysis(data)
    
    print("\nAnalysis complete. Results saved to the current directory.")

if __name__ == "__main__":
    main() 