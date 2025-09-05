"""
Analyze Volatility Forecasting Performance
Compare RMSE/MAE across all models to understand if complexity helps volatility prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

def load_all_metrics():
    """Load all model metrics files"""
    metrics_files = glob("model_output/*_metrics.csv")
    
    all_metrics = []
    
    for file in metrics_files:
        df = pd.read_csv(file)
        
        # Parse model information from filename
        filename = os.path.basename(file)
        parts = filename.replace('_metrics.csv', '').split('_')
        
        index = parts[0]  # SPX, NDX, RTY
        
        if "LSTM" in filename and len(parts) >= 3:
            model_type = "LSTM_Baseline"
        elif "single" in filename:
            model_type = "Single_GARCH_LSTM"  
        elif "dual" in filename:
            model_type = "Dual_GARCH_LSTM"
        elif "triple" in filename:
            model_type = "Triple_GARCH_LSTM"
        else:
            continue
            
        df['index'] = index
        df['model_type'] = model_type
        df['model_id'] = f"{index}_{model_type}"
        
        all_metrics.append(df)
    
    if not all_metrics:
        print("No metrics files found!")
        return None
        
    combined_df = pd.concat(all_metrics, ignore_index=True)
    return combined_df

def analyze_model_performance(df):
    """Analyze and rank models by volatility forecasting performance"""
    
    print("=== Volatility Forecasting Performance Analysis ===\n")
    
    # Group by index and model type
    results_summary = []
    
    for index in ['SPX', 'NDX', 'RTY']:
        print(f"--- {index} Results ---")
        index_data = df[df['index'] == index]
        
        if len(index_data) == 0:
            print(f"No data for {index}")
            continue
            
        # Calculate average performance across windows
        for model_type in ['LSTM_Baseline', 'Single_GARCH_LSTM', 'Dual_GARCH_LSTM', 'Triple_GARCH_LSTM']:
            model_data = index_data[index_data['model_type'] == model_type]
            
            if len(model_data) > 0:
                avg_rmse = model_data['RMSE'].mean()
                avg_mae = model_data['MAE'].mean() 
                avg_mape = model_data['MAPE'].mean()
                best_window = model_data.loc[model_data['RMSE'].idxmin(), 'window']
                best_rmse = model_data['RMSE'].min()
                
                results_summary.append({
                    'index': index,
                    'model_type': model_type,
                    'avg_rmse': avg_rmse,
                    'avg_mae': avg_mae,
                    'avg_mape': avg_mape,
                    'best_rmse': best_rmse,
                    'best_window': best_window
                })
                
                print(f"  {model_type}:")
                print(f"    Avg RMSE: {avg_rmse:.6f}")
                print(f"    Avg MAE:  {avg_mae:.6f}")
                print(f"    Best RMSE: {best_rmse:.6f} (window {best_window})")
        
        # Rank models for this index
        index_results = [r for r in results_summary if r['index'] == index]
        index_results.sort(key=lambda x: x['avg_rmse'])
        
        print(f"\n  {index} Rankings (by avg RMSE):")
        for i, result in enumerate(index_results, 1):
            print(f"    {i}. {result['model_type']}: {result['avg_rmse']:.6f}")
        print()
    
    return pd.DataFrame(results_summary)

def create_performance_comparison_plots(df, summary_df):
    """Create visualizations comparing model performance"""
    
    # Plot 1: RMSE comparison across models and indices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, index in enumerate(['SPX', 'NDX', 'RTY']):
        ax = axes[i]
        index_data = summary_df[summary_df['index'] == index]
        
        model_types = index_data['model_type'].values
        avg_rmse = index_data['avg_rmse'].values
        
        colors = ['red', 'orange', 'blue', 'green']
        bars = ax.bar(range(len(model_types)), avg_rmse, color=colors[:len(model_types)])
        
        ax.set_title(f'{index} - Average RMSE')
        ax.set_ylabel('RMSE')
        ax.set_xticks(range(len(model_types)))
        ax.set_xticklabels([mt.replace('_', '\n') for mt in model_types], rotation=0, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                   f'{val:.5f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('volatility_forecasting_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Performance comparison plot saved: volatility_forecasting_performance_comparison.png")
    
    # Plot 2: Window analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(['RMSE', 'MAE', 'MAPE']):
        ax = axes[i]
        
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            windows = model_data['window'].values
            values = model_data[metric].values
            ax.plot(windows, values, marker='o', label=model_type, linewidth=2)
        
        ax.set_title(f'{metric} by Window Size')
        ax.set_xlabel('Window Size')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_by_window_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Window analysis plot saved: performance_by_window_analysis.png")

def analyze_paper_discrepancy(summary_df):
    """Analyze why results differ from the original paper"""
    
    print("\n=== Analysis: Discrepancy with Original Paper ===")
    
    # Check if more complex models actually perform worse
    complexity_analysis = []
    
    for index in ['SPX', 'NDX', 'RTY']:
        index_data = summary_df[summary_df['index'] == index]
        
        if len(index_data) < 2:
            continue
            
        baseline = index_data[index_data['model_type'] == 'LSTM_Baseline']
        single = index_data[index_data['model_type'] == 'Single_GARCH_LSTM'] 
        dual = index_data[index_data['model_type'] == 'Dual_GARCH_LSTM']
        triple = index_data[index_data['model_type'] == 'Triple_GARCH_LSTM']
        
        if len(baseline) > 0 and len(single) > 0:
            baseline_rmse = baseline['avg_rmse'].iloc[0]
            single_rmse = single['avg_rmse'].iloc[0] if len(single) > 0 else np.nan
            dual_rmse = dual['avg_rmse'].iloc[0] if len(dual) > 0 else np.nan
            triple_rmse = triple['avg_rmse'].iloc[0] if len(triple) > 0 else np.nan
            
            print(f"\n{index} Performance Progression:")
            print(f"  Baseline LSTM:       {baseline_rmse:.6f}")
            if not np.isnan(single_rmse):
                improvement_single = ((baseline_rmse - single_rmse) / baseline_rmse) * 100
                print(f"  Single GARCH-LSTM:   {single_rmse:.6f} ({improvement_single:+.2f}%)")
            if not np.isnan(dual_rmse):
                improvement_dual = ((baseline_rmse - dual_rmse) / baseline_rmse) * 100
                print(f"  Dual GARCH-LSTM:     {dual_rmse:.6f} ({improvement_dual:+.2f}%)")
            if not np.isnan(triple_rmse):
                improvement_triple = ((baseline_rmse - triple_rmse) / baseline_rmse) * 100
                print(f"  Triple GARCH-LSTM:   {triple_rmse:.6f} ({improvement_triple:+.2f}%)")
    
    print("\n=== Possible Explanations for Discrepancy ===")
    print("1. MARKET CHARACTERISTICS:")
    print("   - Financial indices (SPX/NDX/RTY) vs Indian metals have different volatility structures")
    print("   - Developed vs emerging market dynamics")
    print("   - Index diversification vs single commodity exposure")
    
    print("\n2. DATA DIFFERENCES:")
    print("   - Time period and market conditions")
    print("   - Data frequency and quality")
    print("   - Volatility clustering patterns")
    
    print("\n3. MODEL COMPLEXITY:")
    print("   - Overfitting in more complex models")
    print("   - Diminishing returns from additional GARCH branches")
    print("   - Parameter estimation challenges with limited data")
    
    print("\n4. IMPLEMENTATION FACTORS:")
    print("   - Hyperparameter tuning differences")
    print("   - Different train/test split strategies")
    print("   - Scaling and preprocessing variations")

def main():
    # Load all metrics
    df = load_all_metrics()
    
    if df is None:
        return
    
    print(f"Loaded metrics for {len(df)} model configurations")
    print(f"Indices: {df['index'].unique()}")
    print(f"Model types: {df['model_type'].unique()}")
    print(f"Windows: {sorted(df['window'].unique())}")
    
    # Analyze performance
    summary_df = analyze_model_performance(df)
    
    # Save detailed results
    df.to_csv('volatility_forecasting_detailed_results.csv', index=False)
    summary_df.to_csv('volatility_forecasting_summary.csv', index=False)
    
    # Create visualizations
    create_performance_comparison_plots(df, summary_df)
    
    # Analyze discrepancy with paper
    analyze_paper_discrepancy(summary_df)
    
    print(f"\n=== Files Generated ===")
    print("- volatility_forecasting_detailed_results.csv")
    print("- volatility_forecasting_summary.csv")
    print("- volatility_forecasting_performance_comparison.png")
    print("- performance_by_window_analysis.png")

if __name__ == "__main__":
    main()