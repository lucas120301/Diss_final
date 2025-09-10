import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

def load_returns_and_forecasts(returns_csv, forecast_csv, lr_col="LR"):
    print(f"Loading returns from {returns_csv}")
    print(f"Loading forecasts from {forecast_csv}")
    
    # Load returns
    returns_df = pd.read_csv(returns_csv, parse_dates=["Date"])
    returns_df = returns_df.dropna(subset=[lr_col]).sort_values("Date")
    
    # Load forecasts - detect column names
    forecast_df = pd.read_csv(forecast_csv)
    
    # Find prediction and actual columns
    pred_cols = [c for c in forecast_df.columns if "pred" in c.lower()]
    true_cols = [c for c in forecast_df.columns if "true" in c.lower()]
    
    if not pred_cols:
        raise ValueError(f"No prediction column found in {forecast_csv}")
    
    forecast_col = pred_cols[0]
    actual_col = true_cols[0] if true_cols else None
    
    print(f"Using prediction column: {forecast_col}")
    if actual_col:
        print(f"Using actual column: {actual_col}")
    
    # Get forecast data
    forecasts = forecast_df[forecast_col].values
    test_actuals = forecast_df[actual_col].values if actual_col else None
    
    # Use all historical returns for simulation
    historical_returns = returns_df[lr_col].values
    
    print(f"Loaded {len(historical_returns)} historical returns")
    print(f"Loaded {len(forecasts)} volatility forecasts")
    
    return historical_returns, forecasts, test_actuals

def filtered_historical_simulation_corrected(historical_returns, volatility_forecasts, confidence_levels, window=252):
    print(f"Running corrected FHS with rolling window={window}")
    
    var_results = {f"VaR_{int(cl*100)}": [] for cl in confidence_levels}
    n_forecasts = len(volatility_forecasts)
    
    # We need at least 'window' historical returns before forecast period
    if len(historical_returns) < window + n_forecasts:
        raise ValueError(f"Need at least {window + n_forecasts} historical returns for FHS")
    
    # The forecast period starts after we have enough history
    forecast_start_idx = len(historical_returns) - n_forecasts
    
    for t, vol_forecast in enumerate(volatility_forecasts):
        # Get rolling window of historical returns ending just before forecast period
        window_end = forecast_start_idx + t
        window_start = window_end - window
        
        if window_start < 0:
            # Not enough history, use all available
            hist_window = historical_returns[:window_end]
        else:
            hist_window = historical_returns[window_start:window_end]
        
        if len(hist_window) < 50:  # Minimum reasonable window
            print(f"Warning: Small window size {len(hist_window)} at forecast {t}")
            continue
        
        # Standardize historical returns by their empirical volatility
        hist_vol = np.std(hist_window)
        if hist_vol <= 0:
            print(f"Warning: Zero historical volatility at forecast {t}")
            hist_vol = 1e-8  # Small positive value to avoid division by zero
        
        standardized_returns = hist_window / hist_vol

        vol_std = np.sqrt(vol_forecast)
        
        # Scale by forecasted standard deviation to get filtered returns
        filtered_returns = standardized_returns * vol_std
        
        # Compute VaR quantiles (left tail for losses)
        for cl in confidence_levels:
            # For VaR, we want the cl-th quantile (e.g., 1% quantile for 1% VaR)
            var_quantile = np.percentile(filtered_returns, cl * 100)
            var_results[f"VaR_{int(cl*100)}"].append(var_quantile)
    
    return {k: np.array(v) for k, v in var_results.items()}

def backtest_var_corrected(actual_returns, var_estimates, confidence_level):
    """
    Improved VaR backtesting with proper statistical tests
    """
    # VaR violations (when actual return < VaR estimate)
    violations = actual_returns < var_estimates
    violation_rate = np.mean(violations)
    expected_rate = confidence_level
    
    n = len(actual_returns)
    violations_count = np.sum(violations)
    
    # Kupiec Unconditional Coverage Test
    if violations_count == 0:
        uc_stat = 2 * n * np.log(1 - confidence_level)
        uc_pvalue = 1 - stats.chi2.cdf(uc_stat, df=1)
    elif violations_count == n:
        uc_stat = -2 * n * np.log(confidence_level)
        uc_pvalue = 1 - stats.chi2.cdf(uc_stat, df=1)
    else:
        # Likelihood ratio test statistic
        p_hat = violations_count / n
        uc_stat = -2 * (
            violations_count * np.log(confidence_level) + 
            (n - violations_count) * np.log(1 - confidence_level) -
            violations_count * np.log(p_hat) -
            (n - violations_count) * np.log(1 - p_hat)
        )
        uc_pvalue = 1 - stats.chi2.cdf(uc_stat, df=1)
    
    # Christoffersen Independence Test  
    # Count transitions: 00, 01, 10, 11
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        elif violations[i-1] == 1 and violations[i] == 1:
            n11 += 1
    
    # Independence test
    if n01 + n00 == 0 or n10 + n11 == 0 or violations_count == 0 or violations_count == n:
        cc_stat = np.nan
        cc_pvalue = np.nan
    else:
        pi_01 = n01 / (n01 + n00)
        pi_11 = n11 / (n10 + n11)
        pi = violations_count / n
        
        if pi_01 == 0 or pi_11 == 0 or pi == 0 or pi == 1:
            cc_stat = np.nan  
            cc_pvalue = np.nan
        else:
            cc_stat = -2 * np.log(
                (pi**violations_count * (1-pi)**(n-violations_count)) /
                (pi_01**n01 * (1-pi_01)**n00 * pi_11**n11 * (1-pi_11)**n10)
            )
            cc_pvalue = 1 - stats.chi2.cdf(cc_stat, df=1)
    
    # Additional metrics
    violation_severity = np.mean(actual_returns[violations] - var_estimates[violations]) if np.any(violations) else 0
    
    return {
        'violation_rate': violation_rate,
        'expected_rate': expected_rate,
        'violations_count': violations_count,
        'total_observations': n,
        'uc_statistic': uc_stat,
        'uc_pvalue': uc_pvalue,
        'cc_statistic': cc_stat,
        'cc_pvalue': cc_pvalue,
        'violation_severity': violation_severity,
        'var_mean': np.mean(var_estimates),
        'var_std': np.std(var_estimates)
    }

def main():
    parser = argparse.ArgumentParser(description="Corrected Filtered Historical Simulation VaR Analysis")
    parser.add_argument("--returns_csv", required=True, help="CSV with historical returns")
    parser.add_argument("--volatility_forecasts", required=True, help="CSV with volatility forecasts")
    parser.add_argument("--model_name", required=True, help="Model name for identification")
    parser.add_argument("--lr_col", default="LR", help="Log returns column name")
    parser.add_argument("--confidence_levels", nargs="+", type=float, default=[0.01, 0.05], 
                       help="VaR confidence levels")
    parser.add_argument("--simulation_window", type=int, default=252, 
                       help="Historical simulation window")
    parser.add_argument("--out_prefix", default="var_results", help="Output file prefix")
    
    args = parser.parse_args()
    
    print(f"=== Corrected FHS VaR Analysis for {args.model_name} ===")
    print(f"Confidence levels: {args.confidence_levels}")
    print(f"Simulation window: {args.simulation_window}")
    
    # Load data
    historical_returns, vol_forecasts, test_actuals = load_returns_and_forecasts(
        args.returns_csv, args.volatility_forecasts, args.lr_col
    )
    
    # IMPORTANT: test_actuals contains RV_22 values (variance), NOT log returns!
    # For FHS backtesting, we need the corresponding log returns from the test period
    # Use the last N historical returns that correspond to the forecast period
    actual_returns = historical_returns[-len(vol_forecasts):]
    print(f"Using last {len(vol_forecasts)} historical log returns for FHS backtesting")
    print(f"Note: Ignoring 'actual' column from forecast file (contains variance, not returns)")
    
    print(f"Running FHS on {len(vol_forecasts)} forecasts vs {len(actual_returns)} actual returns")
    
    # Ensure same length
    min_len = min(len(vol_forecasts), len(actual_returns))
    vol_forecasts = vol_forecasts[-min_len:]
    actual_returns = actual_returns[-min_len:]
    
    print(f"Using {min_len} aligned observations")
    
    # Compute VaR using corrected filtered historical simulation
    var_results = filtered_historical_simulation_corrected(
        historical_returns, vol_forecasts, args.confidence_levels, args.simulation_window
    )
    
    # Backtest and evaluate
    all_metrics = []
    
    for cl in args.confidence_levels:
        var_key = f"VaR_{int(cl*100)}"
        var_estimates = var_results[var_key]
        
        print(f"\n=== {var_key} Results ===")
        
        # Ensure same length for backtesting
        test_len = min(len(var_estimates), len(actual_returns))
        var_test = var_estimates[-test_len:]
        returns_test = actual_returns[-test_len:]
        
        # Backtest VaR
        backtest_metrics = backtest_var_corrected(returns_test, var_test, cl)
        
        # Store metrics
        metrics = {
            'model': args.model_name,
            'confidence_level': cl,
            **backtest_metrics
        }
        all_metrics.append(metrics)
        
        # Print summary
        print(f"Violation Rate: {backtest_metrics['violation_rate']:.3f} (Expected: {cl:.3f})")
        print(f"Violations: {backtest_metrics['violations_count']}/{backtest_metrics['total_observations']}")
        print(f"UC Test: stat={backtest_metrics['uc_statistic']:.3f}, p-value={backtest_metrics['uc_pvalue']:.4f}")
        print(f"CC Test: stat={backtest_metrics['cc_statistic']:.3f}, p-value={backtest_metrics['cc_pvalue']:.4f}")
        print(f"Average VaR: {backtest_metrics['var_mean']:.6f}")
        print(f"Violation Severity: {backtest_metrics['violation_severity']:.6f}")
        
    
    # Save detailed results
    results_df = pd.DataFrame(all_metrics)
    results_path = f"{args.out_prefix}_{args.model_name}_metrics.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved: {results_path}")
    
    # Save VaR time series
    var_ts_data = {}
    min_len = min(len(actual_returns), min(len(var_results[f"VaR_{int(cl*100)}"]) for cl in args.confidence_levels))
    
    var_ts_data['actual_returns'] = actual_returns[-min_len:]
    for cl in args.confidence_levels:
        var_key = f"VaR_{int(cl*100)}"
        var_ts_data[var_key] = var_results[var_key][-min_len:]
    
    var_ts_df = pd.DataFrame(var_ts_data)
    var_ts_path = f"{args.out_prefix}_{args.model_name}_timeseries.csv"
    var_ts_df.to_csv(var_ts_path, index=False)
    print(f"VaR time series saved: {var_ts_path}")
    
    print(f"\n=== Corrected FHS VaR Analysis Complete ===")

if __name__ == "__main__":
    main()