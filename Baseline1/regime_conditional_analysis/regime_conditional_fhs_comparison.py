#!/usr/bin/env python3
"""
Regime-Conditional FHS Comparison

Fair comparison between Baseline1 and Regime-Switching models by evaluating
both on the same regime-specific periods. This avoids any bias correction
and provides academically rigorous regime-conditional performance analysis.

Located in Baseline1/regime_conditional_analysis/ to keep existing analysis intact.
"""

import os, sys
import numpy as np
import pandas as pd
import warnings
from scipy import stats

# Add paths for regime detection
regime_base_path = "/Users/user/Desktop/Imperial/Diss/RegimeSwitching/regime_switching_models/regime_detection"
sys.path.append(regime_base_path)

warnings.filterwarnings('ignore')

def load_regime_labels():
    """Load regime labels"""
    regime_file = f"{regime_base_path}/final_results/optimal_regime_labels.csv"
    return pd.read_csv(regime_file, parse_dates=['Date'], index_col='Date')

def load_baseline_predictions(index_name, window_size):
    """Load Baseline1 predictions"""
    if index_name == "SPX":
        baseline_file = f"../model_output/SPX_S_single_win{window_size}.csv"
        pred_col = f"SPX_S_{window_size}_pred"
        true_col = "SPX_S_True_test"
    elif index_name == "NDX":
        baseline_file = f"../model_output/NDX_S_single_win{window_size}.csv"
        pred_col = f"NDX_S_{window_size}_pred"
        true_col = "NDX_S_True_test"
    elif index_name == "RTY":
        baseline_file = f"../model_output/RTY_S_single_win{window_size}.csv"
        pred_col = f"RTY_S_{window_size}_pred"
        true_col = "RTY_S_True_test"
    else:
        return None, None
        
    if not os.path.exists(baseline_file):
        print(f"Baseline file not found: {baseline_file}")
        return None, None
        
    df = pd.read_csv(baseline_file)
    return df[pred_col].values, df[true_col].values

def load_regime_predictions(index_name, regime_id, window_size):
    """Load regime-specific predictions"""
    pred_file = f"{regime_base_path}/performance_analysis/{index_name}_REGIME{regime_id}_regime{regime_id}_win{window_size}.csv"
    
    if not os.path.exists(pred_file):
        return None, None
        
    df = pd.read_csv(pred_file)
    
    # Handle different column naming formats
    if 'predictions_test' in df.columns and 'targets_test' in df.columns:
        # NDX/RTY format
        test_preds = df['predictions_test'].dropna().values
        test_targets = df['targets_test'].dropna().values
    elif f'{index_name}_REGIME{regime_id}_{window_size}_pred' in df.columns:
        # SPX format
        pred_col = f'{index_name}_REGIME{regime_id}_{window_size}_pred'
        target_col = f'{index_name}_REGIME{regime_id}_True_test'
        test_preds = df[pred_col].dropna().values
        test_targets = df[target_col].dropna().values
    else:
        print(f"    Unknown column format in {pred_file}")
        print(f"    Available columns: {list(df.columns)}")
        return None, None
    
    return test_preds, test_targets

def filtered_historical_simulation(historical_returns, vol_forecasts, confidence_level=0.05, window_size=250):
    """
    Standard FHS implementation matching Baseline1 exactly
    """
    var_estimates = []
    
    for i in range(len(vol_forecasts)):
        if i < window_size:
            continue
        
        # Historical window
        hist_window = historical_returns[i-window_size:i]
        hist_vol = np.std(hist_window, ddof=1)
        
        if hist_vol <= 1e-10:
            hist_vol = 1e-10
        
        # Standardize without mean subtraction (exactly like SPX baseline)
        standardized_returns = hist_window / hist_vol
        
        # Current volatility forecast (convert variance to std dev)
        vol_forecast = vol_forecasts[i]
        vol_std = np.sqrt(vol_forecast)
        
        # Scale by forecasted standard deviation
        filtered_returns = standardized_returns * vol_std
        
        # VaR estimate
        var_est = -np.quantile(filtered_returns, confidence_level)
        var_estimates.append(var_est)
    
    return np.array(var_estimates)

def calculate_var_violations(actual_returns, var_estimates):
    """Calculate VaR violations"""
    min_len = min(len(actual_returns), len(var_estimates))
    actual_aligned = actual_returns[-min_len:]
    var_aligned = var_estimates[-min_len:]
    
    violations = actual_aligned < -var_aligned
    violation_rate = np.mean(violations)
    
    return violations, violation_rate

def kupiec_test(violations, expected_rate=0.05):
    """Kupiec POF test"""
    n = len(violations)
    violations_count = np.sum(violations)
    
    if violations_count == 0 or violations_count == n:
        return np.nan, 1.0
    
    lr = -2 * (violations_count * np.log(expected_rate) + 
               (n - violations_count) * np.log(1 - expected_rate) - 
               violations_count * np.log(violations_count / n) - 
               (n - violations_count) * np.log(1 - violations_count / n))
    
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_value

def christoffersen_test(violations):
    """Christoffersen independence test"""
    violations_binary = violations.astype(int)
    
    n00 = np.sum((violations_binary[:-1] == 0) & (violations_binary[1:] == 0))
    n01 = np.sum((violations_binary[:-1] == 0) & (violations_binary[1:] == 1))
    n10 = np.sum((violations_binary[:-1] == 1) & (violations_binary[1:] == 0))
    n11 = np.sum((violations_binary[:-1] == 1) & (violations_binary[1:] == 1))
    
    n0 = n00 + n01
    n1 = n10 + n11
    
    if n0 == 0 or n1 == 0 or (n00 == 0 and n01 == 0) or (n10 == 0 and n11 == 0):
        return np.nan, 1.0
    
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1)
    
    if pi01 <= 0 or pi11 <= 0 or pi <= 0 or pi >= 1:
        return np.nan, 1.0
    
    lr = -2 * (n00 * np.log(1 - pi) + n01 * np.log(pi) + 
               n10 * np.log(1 - pi) + n11 * np.log(pi) - 
               n00 * np.log(1 - pi01) - n01 * np.log(pi01) - 
               n10 * np.log(1 - pi11) - n11 * np.log(pi11))
    
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_value

def analyze_regime_conditional_fhs(index_name):
    """
    Analyze FHS performance by comparing Baseline1 vs Regime models
    within each regime's time periods (regime-conditional evaluation)
    """
    print(f"\n{'='*60}")
    print(f"REGIME-CONDITIONAL FHS COMPARISON: {index_name}")
    print(f"{'='*60}")
    
    # Load regime labels
    regime_df = load_regime_labels()
    
    # Load index returns for regime identification
    data_file = f"../data/{index_name}.csv"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return []
    
    df = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
    df['Log_Return'] = np.log(df['Close']).diff()
    returns = df['Log_Return'].dropna()
    
    # Align dates with regime labels
    common_dates = returns.index.intersection(regime_df.index)
    returns_aligned = returns.loc[common_dates].values
    regimes_aligned = regime_df.loc[common_dates]['Regime'].values
    
    results = []
    
    for window_size in [5, 11, 22]:
        print(f"\n--- Window Size: {window_size} ---")
        
        # Load Baseline1 predictions
        baseline_preds, baseline_targets = load_baseline_predictions(index_name, window_size)
        
        if baseline_preds is None:
            print(f"  No Baseline1 predictions found for window {window_size}")
            continue
        
        print(f"  Loaded {len(baseline_preds)} Baseline1 predictions")
        
        for regime_id in range(4):  # Skip regime 4
            print(f"\n  === Regime {regime_id} ===")
            
            # Load regime-specific predictions
            regime_preds, regime_targets = load_regime_predictions(index_name, regime_id, window_size)
            
            if regime_preds is None:
                print(f"    No regime predictions found")
                continue
            
            # Identify regime periods in the full timeline
            regime_mask = regimes_aligned == regime_id
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) == 0:
                print(f"    No regime periods found")
                continue
            
            # Get regime-specific test period (last 30% of regime data)
            regime_split = int(0.7 * len(regime_indices))
            test_regime_indices = regime_indices[regime_split:]
            
            if len(test_regime_indices) == 0:
                print(f"    No test periods for regime")
                continue
            
            # Extract returns for this regime's test period
            test_returns_regime = returns_aligned[test_regime_indices]
            
            # Align predictions with available test data
            min_test_len = min(len(regime_preds), len(test_returns_regime))
            regime_preds_aligned = regime_preds[:min_test_len]
            test_returns_aligned = test_returns_regime[:min_test_len]
            
            # Extract corresponding baseline predictions for the same regime periods
            baseline_preds_regime = baseline_preds[-min_test_len:] if len(baseline_preds) >= min_test_len else baseline_preds
            baseline_returns_regime = test_returns_aligned[:len(baseline_preds_regime)]
            
            print(f"    Regime model: {len(regime_preds_aligned)} predictions")
            print(f"    Baseline model: {len(baseline_preds_regime)} predictions")
            
            # Run FHS on multiple confidence levels
            confidence_levels = [0.10, 0.05, 0.01]  # 90%, 95%, 99% VaR
            confidence_names = ['90%', '95%', '99%']
            expected_rates = [10.0, 5.0, 1.0]
            
            for conf_level, conf_name, expected_rate in zip(confidence_levels, confidence_names, expected_rates):
                print(f"    --- {conf_name} VaR ---")
                
                # Run FHS on Regime predictions
                if len(regime_preds_aligned) > 250:
                    var_regime = filtered_historical_simulation(
                        returns_aligned, regime_preds_aligned, 
                        confidence_level=conf_level, window_size=250
                    )
                    
                    if len(var_regime) > 0:
                        violations_regime, violation_rate_regime = calculate_var_violations(test_returns_aligned, var_regime)
                        kupiec_lr_regime, kupiec_p_regime = kupiec_test(violations_regime, conf_level)
                        christo_lr_regime, christo_p_regime = christoffersen_test(violations_regime)
                        
                        results.append({
                            'Index': index_name,
                            'Regime': regime_id,
                            'Window': window_size,
                            'Confidence_Level': conf_name,
                            'Model_Type': 'REGIME_SWITCHING',
                            'Violation_Rate': violation_rate_regime * 100,
                            'Expected_Rate': expected_rate,
                            'Violations_Count': int(np.sum(violations_regime)),
                            'Total_Observations': len(violations_regime),
                            'Kupiec_LR': kupiec_lr_regime,
                            'Kupiec_p_value': kupiec_p_regime,
                            'Christoffersen_LR': christo_lr_regime,
                            'Christoffersen_p_value': christo_p_regime,
                            'Mean_VaR': np.mean(var_regime),
                            'Std_VaR': np.std(var_regime)
                        })
                        
                        print(f"      Regime Model - Violations: {violation_rate_regime*100:.2f}% ({np.sum(violations_regime)}/{len(violations_regime)})")
                
                # Run FHS on Baseline1 predictions (regime-conditional)
                if len(baseline_preds_regime) > 250:
                    var_baseline = filtered_historical_simulation(
                        returns_aligned, baseline_preds_regime,
                        confidence_level=conf_level, window_size=250
                    )
                    
                    if len(var_baseline) > 0:
                        violations_baseline, violation_rate_baseline = calculate_var_violations(baseline_returns_regime, var_baseline)
                        kupiec_lr_baseline, kupiec_p_baseline = kupiec_test(violations_baseline, conf_level)
                        christo_lr_baseline, christo_p_baseline = christoffersen_test(violations_baseline)
                        
                        results.append({
                            'Index': index_name,
                            'Regime': regime_id,
                            'Window': window_size,
                            'Confidence_Level': conf_name,
                            'Model_Type': 'BASELINE1_REGIME_CONDITIONAL',
                            'Violation_Rate': violation_rate_baseline * 100,
                            'Expected_Rate': expected_rate,
                            'Violations_Count': int(np.sum(violations_baseline)),
                            'Total_Observations': len(violations_baseline),
                            'Kupiec_LR': kupiec_lr_baseline,
                            'Kupiec_p_value': kupiec_p_baseline,
                            'Christoffersen_LR': christo_lr_baseline,
                            'Christoffersen_p_value': christo_p_baseline,
                            'Mean_VaR': np.mean(var_baseline),
                            'Std_VaR': np.std(var_baseline)
                        })
                        
                        print(f"      Baseline1 Model - Violations: {violation_rate_baseline*100:.2f}% ({np.sum(violations_baseline)}/{len(violations_baseline)})")
    
    return results

def main():
    """Run regime-conditional FHS comparison for all indices"""
    
    print("="*80)
    print("REGIME-CONDITIONAL FHS COMPARISON")
    print("="*80)
    print("\nThis analysis compares Baseline1 vs Regime-Switching models")
    print("by evaluating both on the same regime-specific time periods.")
    print("\nThis provides a fair comparison without any ad-hoc corrections.")
    print(f"\nKeeping existing Baseline1 var_analysis intact.")
    print(f"Results will be saved in regime_conditional_analysis/")
    
    all_results = []
    
    # Analyze all indices
    for index_name in ["SPX", "NDX", "RTY"]:
        try:
            results = analyze_regime_conditional_fhs(index_name)
            all_results.extend(results)
        except Exception as e:
            print(f"Error analyzing {index_name}: {e}")
            continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = "regime_conditional_fhs_comparison_results.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("REGIME-CONDITIONAL COMPARISON RESULTS")
        print(f"{'='*80}")
        
        # Summary by model type, regime and confidence level
        summary = results_df.groupby(['Index', 'Regime', 'Confidence_Level', 'Model_Type']).agg({
            'Violation_Rate': 'mean',
            'Kupiec_p_value': 'mean',
            'Christoffersen_p_value': 'mean'
        }).round(3)
        
        print(summary)
        
        # Count acceptable models per confidence level
        print(f"\n{'='*60}")
        print("ACCEPTABLE MODEL COUNTS BY CONFIDENCE LEVEL")
        print(f"{'='*60}")
        
        for conf_level in ['90%', '95%', '99%']:
            conf_results = results_df[results_df['Confidence_Level'] == conf_level]
            if conf_level == '90%':
                acceptable = conf_results[(conf_results['Violation_Rate'] >= 7.0) & 
                                        (conf_results['Violation_Rate'] <= 13.0)]
                target_range = "7-13%"
            elif conf_level == '95%':
                acceptable = conf_results[(conf_results['Violation_Rate'] >= 3.0) & 
                                        (conf_results['Violation_Rate'] <= 7.0)]
                target_range = "3-7%"
            else:  # 99%
                acceptable = conf_results[(conf_results['Violation_Rate'] >= 0.5) & 
                                        (conf_results['Violation_Rate'] <= 2.0)]
                target_range = "0.5-2%"
            
            print(f"{conf_level} VaR: {len(acceptable)}/{len(conf_results)} models in acceptable range ({target_range})")
        
        # Direct comparison within regimes by confidence level
        print(f"\n{'='*60}")
        print("REGIME-BY-REGIME COMPARISON BY CONFIDENCE LEVEL")
        print(f"{'='*60}")
        
        for conf_level in ['90%', '95%', '99%']:
            print(f"\n--- {conf_level} VaR ---")
            conf_results = results_df[results_df['Confidence_Level'] == conf_level]
            
            for index_name in ["SPX", "NDX", "RTY"]:
                for regime_id in range(4):
                    regime_results = conf_results[(conf_results['Index'] == index_name) & 
                                                (conf_results['Regime'] == regime_id)]
                    
                    baseline_results = regime_results[regime_results['Model_Type'] == 'BASELINE1_REGIME_CONDITIONAL']
                    regime_model_results = regime_results[regime_results['Model_Type'] == 'REGIME_SWITCHING']
                    
                    if len(baseline_results) > 0 and len(regime_model_results) > 0:
                        baseline_avg_viol = baseline_results['Violation_Rate'].mean()
                        regime_avg_viol = regime_model_results['Violation_Rate'].mean()
                        
                        print(f"  {index_name} Regime {regime_id}: Baseline1={baseline_avg_viol:.2f}% vs Regime={regime_avg_viol:.2f}%")
        
        print(f"\nDetailed results saved to: {output_file}")
        
    else:
        print("No results generated")

if __name__ == "__main__":
    main()