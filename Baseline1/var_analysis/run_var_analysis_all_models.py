"""
Comprehensive VaR Analysis for All Models
Runs filtered historical simulation VaR analysis on all volatility forecasting models.

Usage:
  python run_var_analysis_all_models.py
"""

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def find_model_outputs():
    """Find all model output files"""
    output_dir = "model_output"
    if not os.path.exists(output_dir):
        print(f"Model output directory {output_dir} not found!")
        return []
    
    # Find all prediction files (those ending with win*.csv but not metrics)
    prediction_files = []
    for file in glob(f"{output_dir}/*.csv"):
        if "win" in file and "metrics" not in file:
            prediction_files.append(file)
    
    return sorted(prediction_files)

def parse_model_info(filepath):
    """Extract model information from filename"""
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')
    
    # Extract index (SPX, NDX, RTY)
    index = parts[0]
    
    # Extract model type and window
    if "LSTM" in filename and len(parts) >= 4:
        # Format: SPX_LSTM_LSTM_win22.csv
        model_type = "LSTM_Baseline"
        window = parts[-1].replace('win', '')
    elif any(mode in filename for mode in ['single', 'dual', 'triple']):
        # Format: SPX_S_single_win22.csv or SPX_ST_dual_win22.csv
        model_type = f"GARCH_LSTM_{parts[2]}"  # single/dual/triple
        window = parts[-1].replace('win', '')
    else:
        return None
    
    return {
        'index': index,
        'model_type': model_type,
        'window': window,
        'filepath': filepath
    }

def run_var_analysis(returns_csv, forecast_csv, model_name, confidence_levels=[0.01, 0.05]):
    """Run VaR analysis for a single model"""
    
    print(f"Running VaR analysis for {model_name}...")
    
    cmd = [
        "python", "filtered_historical_simulation_var.py",
        "--returns_csv", returns_csv,
        "--volatility_forecasts", forecast_csv,
        "--model_name", model_name,
        "--confidence_levels"] + [str(cl) for cl in confidence_levels] + [
        "--out_prefix", "var_results",
        "--plot"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✓ Completed: {model_name}")
            return True
        else:
            print(f"✗ Failed: {model_name}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception for {model_name}: {e}")
        return False

def consolidate_var_results():
    """Consolidate all VaR results into summary tables"""
    
    # Find all VaR metrics files
    var_files = glob("var_results_*_metrics.csv")
    
    if not var_files:
        print("No VaR results found!")
        return
    
    # Combine all results
    all_results = []
    for file in var_files:
        df = pd.read_csv(file)
        all_results.append(df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save consolidated results
    combined_df.to_csv("var_results_consolidated.csv", index=False)
    print(f"Consolidated VaR results saved: var_results_consolidated.csv")
    
    # Create summary statistics
    create_var_summary(combined_df)
    
    return combined_df

def create_var_summary(df):
    """Create summary tables and rankings"""
    
    # Summary by model type
    summary_stats = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        for cl in df['confidence_level'].unique():
            cl_data = model_data[model_data['confidence_level'] == cl]
            if len(cl_data) > 0:
                summary_stats.append({
                    'model': model,
                    'confidence_level': cl,
                    'avg_violation_rate': cl_data['violation_rate'].mean(),
                    'avg_uc_pvalue': cl_data['uc_pvalue'].mean(),
                    'avg_violation_severity': cl_data['violation_severity'].mean(),
                    'avg_var_mean': cl_data['var_mean'].mean(),
                    'total_violations': cl_data['violations_count'].sum()
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv("var_summary_by_model.csv", index=False)
    print("VaR summary by model saved: var_summary_by_model.csv")
    
    # Model rankings
    create_model_rankings(df)

def create_model_rankings(df):
    """Create model performance rankings"""
    
    rankings = []
    
    for cl in df['confidence_level'].unique():
        cl_data = df[df['confidence_level'] == cl]
        
        # Score based on multiple criteria (lower is better for most metrics)
        cl_data['violation_error'] = abs(cl_data['violation_rate'] - cl)
        cl_data['uc_score'] = 1 - cl_data['uc_pvalue']  # Lower p-value is worse
        
        # Composite score (you can adjust weights)
        cl_data['composite_score'] = (
            0.4 * cl_data['violation_error'] +
            0.3 * cl_data['uc_score'] +
            0.3 * abs(cl_data['violation_severity'])
        )
        
        # Rank models
        cl_ranked = cl_data.sort_values('composite_score')[['model', 'violation_rate', 'uc_pvalue', 'violation_severity', 'composite_score']]
        cl_ranked['rank'] = range(1, len(cl_ranked) + 1)
        cl_ranked['confidence_level'] = cl
        
        rankings.append(cl_ranked)
    
    rankings_df = pd.concat(rankings, ignore_index=True)
    rankings_df.to_csv("var_model_rankings.csv", index=False)
    print("Model rankings saved: var_model_rankings.csv")
    
    # Print top performers
    print("\nTop 3 Models by Confidence Level:")
    for cl in df['confidence_level'].unique():
        cl_ranks = rankings_df[rankings_df['confidence_level'] == cl].head(3)
        print(f"\n{cl:.0%} VaR:")
        for _, row in cl_ranks.iterrows():
            print(f"  {row['rank']}. {row['model']} (Score: {row['composite_score']:.4f})")

def create_comparison_plots():
    """Create comparison plots across models"""
    
    # Load consolidated results
    if not os.path.exists("var_results_consolidated.csv"):
        print("No consolidated results found for plotting")
        return
    
    df = pd.read_csv("var_results_consolidated.csv")
    
    # Violation rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, cl in enumerate(sorted(df['confidence_level'].unique())):
        cl_data = df[df['confidence_level'] == cl]
        
        ax = axes[i]
        models = cl_data['model'].values
        violation_rates = cl_data['violation_rate'].values
        expected_rate = cl
        
        bars = ax.bar(range(len(models)), violation_rates, alpha=0.7)
        ax.axhline(y=expected_rate, color='red', linestyle='--', linewidth=2, 
                  label=f'Expected Rate ({expected_rate:.1%})')
        
        # Color bars based on how close they are to expected rate
        for j, (bar, rate) in enumerate(zip(bars, violation_rates)):
            error = abs(rate - expected_rate)
            if error < 0.01:  # Within 1%
                bar.set_color('green')
            elif error < 0.02:  # Within 2%
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_title(f'VaR {int(cl*100)}% - Violation Rates')
        ax.set_ylabel('Violation Rate')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('var_violation_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Violation rates comparison plot saved: var_violation_rates_comparison.png")

def main():
    print("=== Comprehensive VaR Analysis ===")
    
    # Create VaR results directory
    os.makedirs("var_results", exist_ok=True)
    
    # Find all model outputs
    model_files = find_model_outputs()
    print(f"Found {len(model_files)} model output files")
    
    if not model_files:
        print("No model outputs found! Make sure models have been trained.")
        return
    
    # Data mapping for each index
    data_files = {
        'SPX': 'data/SPX_RV22.csv',
        'NDX': 'data/NDX_RV22.csv', 
        'RTY': 'data/RTY_RV22.csv'
    }
    
    successful_runs = 0
    total_runs = 0
    
    # Run VaR analysis for each model
    for model_file in model_files:
        model_info = parse_model_info(model_file)
        if not model_info:
            print(f"Could not parse model info for {model_file}")
            continue
        
        index = model_info['index']
        if index not in data_files:
            print(f"No data file found for index {index}")
            continue
        
        model_name = f"{index}_{model_info['model_type']}_win{model_info['window']}"
        
        total_runs += 1
        if run_var_analysis(data_files[index], model_file, model_name):
            successful_runs += 1
    
    print(f"\n=== VaR Analysis Complete ===")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    
    if successful_runs > 0:
        # Consolidate results
        print("\nConsolidating results...")
        consolidate_var_results()
        
        # Create comparison plots
        print("\nCreating comparison plots...")
        create_comparison_plots()
        
        print("\n=== All Analysis Complete ===")
        print("Check the following files:")
        print("- var_results_consolidated.csv - All VaR metrics")
        print("- var_summary_by_model.csv - Summary statistics")
        print("- var_model_rankings.csv - Model performance rankings")
        print("- var_violation_rates_comparison.png - Comparison plot")
        print("- Individual VaR plots: var_results_*_backtest.png")
    else:
        print("No successful VaR analyses completed.")

if __name__ == "__main__":
    main()