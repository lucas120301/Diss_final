import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

def kupiec_test(violations, total_obs, confidence_level):
    """Kupiec unconditional coverage test"""
    expected_violations = total_obs * (1 - confidence_level)
    if violations == 0:
        return 0, 1.0
    
    lr_stat = 2 * (violations * np.log(violations / expected_violations) + 
                   (total_obs - violations) * np.log((total_obs - violations) / (total_obs - expected_violations)))
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    return lr_stat, p_value

def christoffersen_test(violations_series, confidence_level):
    """Christoffersen conditional coverage test"""
    if len(violations_series) < 10:
        return 0, 1.0
    
    # Calculate transition probabilities
    transitions = []
    for i in range(1, len(violations_series)):
        transitions.append((violations_series[i-1], violations_series[i]))
    
    n00 = sum(1 for t in transitions if t == (0, 0))
    n01 = sum(1 for t in transitions if t == (0, 1))
    n10 = sum(1 for t in transitions if t == (1, 0))
    n11 = sum(1 for t in transitions if t == (1, 1))
    
    if n00 + n01 == 0 or n10 + n11 == 0:
        return 0, 1.0
    
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    if pi_01 == 0 and pi_11 == 0:
        return 0, 1.0
    
    likelihood_ratio = ((1 - pi_01)**n00 * pi_01**n01 * (1 - pi_11)**n10 * pi_11**n11) / \
                       ((1 - pi)**(n00 + n10) * pi**(n01 + n11))
    
    if likelihood_ratio <= 0:
        return 0, 1.0
    
    lr_stat = -2 * np.log(likelihood_ratio)
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    return lr_stat, p_value

def analyze_original_paper_results():
    results_dir = "/Result Analysis/Model Output"
    data_file = "/Data/Final.csv"
    
    # Load the main data for returns
    print("Loading original paper data...")
    data = pd.read_csv(data_file)
    
    # Models and metals to analyze
    metals = ['Copper', 'Lead', 'Tin', 'Nickel']
    models = ['S-LSTM', 'E-LSTM', 'T-LSTM', 'ST-LSTM', 'SE-LSTM', 'ET-LSTM', 'SET-LSTM(512,256)']
    
    all_results = []
    
    # Map metals to their actual log return columns
    metal_mapping = {
        'Copper': 'LR_Cop',
        'Lead': 'LR_Lead', 
        'Tin': 'LR_Tin',
        'Nickel': 'LR_Nic'
    }
    
    for metal in metals:
        print(f"\\nAnalyzing {metal}...")
        
        # Get the log returns for this metal
        lr_col = metal_mapping.get(metal)
        if lr_col is None or lr_col not in data.columns:
            print(f"  Warning: {lr_col} not found, skipping {metal}")
            continue
            
        returns = data[lr_col].dropna()
        
        for model in models:
            model_file = f"{results_dir}/{metal}_{model}_test.csv"
            
            if not os.path.exists(model_file):
                print(f"  {model}: File not found, skipping")
                continue
                
            try:
                # Load model predictions
                model_data = pd.read_csv(model_file)
                
                # Find the columns (they have different window sizes)  
                true_col = f'Volatility_{metal}_True_test'
                pred_cols = [col for col in model_data.columns if 'test_real_LSTM' in col]
                
                if true_col not in model_data.columns or len(pred_cols) == 0:
                    print(f"  {model}: No suitable columns found (true_col: {true_col}, pred_cols: {len(pred_cols)})")
                    continue
                
                # Use the first prediction column (usually 22-day window)
                pred_col = pred_cols[0]
                window = '22'  # Default assumption
                if '5_test' in pred_col:
                    window = '5'
                elif '11_test' in pred_col:
                    window = '11'
                elif '22_test' in pred_col:
                    window = '22'
                
                actual_vol = model_data[true_col].dropna()
                pred_vol = model_data[pred_col].dropna()
                
                # Align lengths
                min_len = min(len(actual_vol), len(pred_vol))
                actual_vol = actual_vol[:min_len]
                pred_vol = pred_vol[:min_len]
                
                if len(actual_vol) < 50:
                    print(f"  {model}: Insufficient data ({len(actual_vol)} obs)")
                    continue
                
                # Simple FHS: Use historical returns to generate VaR
                # Get corresponding returns for the test period
                test_returns = returns[-len(actual_vol):]
                
                if len(test_returns) != len(pred_vol):
                    # Align with available returns
                    min_len = min(len(test_returns), len(pred_vol))
                    test_returns = test_returns[-min_len:]
                    pred_vol = pred_vol[:min_len]
                    actual_vol = actual_vol[:min_len]
                
                # Generate VaR estimates using FHS methodology
                var_results_95 = []
                var_results_99 = []
                
                window_size = 252  # Use 1-year rolling window
                
                for i in range(window_size, len(test_returns)):
                    # Get historical window
                    hist_window = test_returns[i-window_size:i]
                    
                    # Get predicted volatility for scaling
                    pred_vol_t = pred_vol[i] if i < len(pred_vol) else pred_vol[-1]
                    
                    # Standardize historical returns (simple approach)
                    hist_std = np.std(hist_window)
                    if hist_std > 0:
                        standardized_returns = hist_window / hist_std
                        
                        # Scale by predicted volatility
                        scaled_returns = standardized_returns * pred_vol_t
                        
                        # Calculate VaR percentiles
                        var_95 = np.percentile(scaled_returns, 5)  # 5th percentile for 95% VaR
                        var_99 = np.percentile(scaled_returns, 1)  # 1st percentile for 99% VaR
                        
                        var_results_95.append(var_95)
                        var_results_99.append(var_99)
                    else:
                        var_results_95.append(0)
                        var_results_99.append(0)
                
                if len(var_results_95) == 0:
                    print(f"  {model}: No VaR results generated")
                    continue
                
                # Get actual returns for violation testing
                actual_returns = test_returns[window_size:]
                min_len = min(len(actual_returns), len(var_results_95))
                actual_returns = actual_returns[:min_len]
                var_95 = np.array(var_results_95[:min_len])
                var_99 = np.array(var_results_99[:min_len])
                
                # Calculate violations
                violations_95 = (actual_returns < var_95).astype(int)
                violations_99 = (actual_returns < var_99).astype(int)
                
                violation_rate_95 = np.mean(violations_95)
                violation_rate_99 = np.mean(violations_99)
                
                # Statistical tests
                uc_stat_95, uc_pvalue_95 = kupiec_test(np.sum(violations_95), len(violations_95), 0.95)
                cc_stat_95, cc_pvalue_95 = christoffersen_test(violations_95, 0.95)
                
                uc_stat_99, uc_pvalue_99 = kupiec_test(np.sum(violations_99), len(violations_99), 0.99)
                cc_stat_99, cc_pvalue_99 = christoffersen_test(violations_99, 0.99)
                
                # Calculate violation severity (average excess)
                excess_95 = actual_returns[violations_95 == 1] - var_95[violations_95 == 1]
                excess_99 = actual_returns[violations_99 == 1] - var_99[violations_99 == 1]
                
                severity_95 = np.mean(excess_95) if len(excess_95) > 0 else 0
                severity_99 = np.mean(excess_99) if len(excess_99) > 0 else 0
                
                # Store results
                for conf_level, viol_rate, uc_stat, uc_pval, cc_stat, cc_pval, severity in [
                    (0.05, violation_rate_95, uc_stat_95, uc_pvalue_95, cc_stat_95, cc_pvalue_95, severity_95),
                    (0.01, violation_rate_99, uc_stat_99, uc_pvalue_99, cc_stat_99, cc_pvalue_99, severity_99)
                ]:
                    all_results.append({
                        'model': f'{metal}_{model}_win{window}',
                        'confidence_level': conf_level,
                        'violation_rate': viol_rate,
                        'expected_rate': conf_level,
                        'violations_count': int(np.sum(violations_95 if conf_level == 0.05 else violations_99)),
                        'total_observations': len(violations_95),
                        'uc_statistic': uc_stat,
                        'uc_pvalue': uc_pval,
                        'cc_statistic': cc_stat,
                        'cc_pvalue': cc_pval,
                        'violation_severity': severity,
                        'var_mean': np.mean(var_95 if conf_level == 0.05 else var_99),
                        'var_std': np.std(var_95 if conf_level == 0.05 else var_99)
                    })
                
                print(f"  {model} (window {window}): 95% VaR violations = {violation_rate_95:.1%}, 99% VaR violations = {violation_rate_99:.1%}")
                
            except Exception as e:
                print(f"  {model}: Error processing - {e}")
                continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = "/Users/user/Desktop/Imperial/Diss/ORIGINAL_PAPER_FHS_RESULTS.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\\n=== Summary Results ===")
        print(f"Total models analyzed: {len(results_df)//2}")
        print(f"Results saved to: {output_file}")
        
        # Quick summary
        print("\\n95% VaR Performance:")
        subset_95 = results_df[results_df['confidence_level'] == 0.05]
        print(f"Average violation rate: {subset_95['violation_rate'].mean():.1%} (target: 5.0%)")
        print(f"Kupiec test pass rate: {(subset_95['uc_pvalue'] > 0.05).mean():.1%}")
        
        print("\\n99% VaR Performance:")
        subset_99 = results_df[results_df['confidence_level'] == 0.01]
        print(f"Average violation rate: {subset_99['violation_rate'].mean():.1%} (target: 1.0%)")
        print(f"Kupiec test pass rate: {(subset_99['uc_pvalue'] > 0.05).mean():.1%}")
        
        return results_df
    else:
        print("\\nNo results generated - check file paths and data structure")
        return None


if __name__ == "__main__":
    original_results = analyze_original_paper_results()
    