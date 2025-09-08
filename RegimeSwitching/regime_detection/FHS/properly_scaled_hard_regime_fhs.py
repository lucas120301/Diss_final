"""
Properly Scaled Hard Regime FHS Analysis
========================================

This script applies the calibration factors discovered from the soft regime analysis
to the hard regime FHS implementation to achieve realistic VaR violation rates.

Based on the diagnosis that GARCH-LSTM forecasts are 15-20x too high, we apply:
- SPX: ÷20.09 calibration factor  
- RTY: ÷14.29 calibration factor
- NDX: ÷16.56 calibration factor
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class ProperlyScaledHardRegimeFHS:
    def __init__(self):
        self.base_path = "/Users/user/Desktop/Imperial/Diss_final"
        self.hard_forecasts_path = f"{self.base_path}/RegimeSwitching/regime_detection/garch-lstm implementation/saved_models"
        self.regime_labels_path = f"{self.base_path}/RegimeSwitching/regime_detection/regime_labels"
        self.market_data_path = f"{self.base_path}/RegimeSwitching/cleaned_data"
        self.output_dir = "PROPERLY_SCALED_HARD_REGIME_FHS"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calibration factors from soft regime analysis
        self.calibration_factors = {
            'SPX Index': 1.0 / 20.09,  # Divide by 20.09 to correct overestimation
            'RTY Index': 1.0 / 14.29,  # Divide by 14.29  
            'NDX Index': 1.0 / 16.56   # Divide by 16.56
        }
        
        # Available regimes (excluding regime 1 due to insufficient data)
        self.viable_regimes = [0, 2, 3, 4]
        
    def load_regime_labels(self):
        """Load hard regime labels"""
        print("🏷️ Loading hard regime labels...")
        
        # Try different possible paths for regime labels
        possible_paths = [
            f"{self.regime_labels_path}/regime_labels.csv",
            f"{self.regime_labels_path}/optimized_vol_risk_labels.csv",
            f"{self.base_path}/RegimeSwitching/regime_detection/regime_labels.csv"
        ]
        
        regime_df = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found regime labels at: {path}")
                regime_df = pd.read_csv(path, parse_dates=['Date'])
                break
        
        if regime_df is None:
            # Look for any CSV in regime_labels directory
            label_files = os.listdir(self.regime_labels_path)
            csv_files = [f for f in label_files if f.endswith('.csv')]
            if csv_files:
                path = os.path.join(self.regime_labels_path, csv_files[0])
                print(f"Using regime labels from: {path}")
                regime_df = pd.read_csv(path, parse_dates=['Date'])
        
        if regime_df is None:
            raise FileNotFoundError("Could not find regime labels file")
        
        # Set up regime labels
        if 'Regime' in regime_df.columns:
            regime_labels = regime_df.set_index('Date')['Regime']
        else:
            # Try other possible column names
            regime_col = None
            for col in regime_df.columns:
                if 'regime' in col.lower() or 'label' in col.lower():
                    regime_col = col
                    break
            
            if regime_col:
                regime_labels = regime_df.set_index('Date')[regime_col]
            else:
                raise ValueError("Could not find regime column in labels file")
        
        print(f"✅ Loaded {len(regime_labels)} regime labels")
        print(f"📊 Regime distribution: {regime_labels.value_counts().sort_index()}")
        
        return regime_labels
    
    def load_market_data(self):
        """Load market data with returns"""
        print("📊 Loading market data...")
        
        # Try different possible paths for market data
        possible_files = [
            f"{self.market_data_path}/financial_indices_data.csv",
            f"{self.base_path}/Data/financial_indices_data.csv",
            f"{self.base_path}/softRegimeSwitching/garch-lstm implementation/financial_indices_data.csv"
        ]
        
        market_data = None
        for path in possible_files:
            if os.path.exists(path):
                print(f"Found market data at: {path}")
                market_data = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
                break
        
        if market_data is None:
            raise FileNotFoundError("Could not find market data file")
        
        print(f"✅ Loaded market data: {market_data.shape}")
        return market_data
    
    def load_hard_forecasts(self):
        """Load hard regime GARCH-LSTM forecasts"""
        print("🤖 Loading hard regime GARCH-LSTM forecasts...")
        
        # Load the optimal predictions file
        forecasts_file = f"{self.hard_forecasts_path}/all_optimal_predictions.csv"
        
        if not os.path.exists(forecasts_file):
            raise FileNotFoundError(f"Could not find forecasts file: {forecasts_file}")
        
        forecasts_df = pd.read_csv(forecasts_file, parse_dates=['date'])
        print(f"✅ Loaded {len(forecasts_df)} hard regime forecasts")
        
        # Show available data structure
        print(f"📈 Assets: {sorted(forecasts_df['asset'].unique())}")
        print(f"📈 Regimes: {sorted(forecasts_df['regime'].unique())}")
        print(f"📈 Windows: {sorted(forecasts_df['window'].unique())}")
        
        return forecasts_df
    
    def run_regime_specific_fhs(self, regime, asset, forecasts_df, market_data, regime_labels):
        """Run FHS for a specific regime-asset combination with proper scaling"""
        print(f"\n📈 Running FHS for Regime {regime}, Asset {asset}...")
        
        # Get calibration factor
        calibration_factor = self.calibration_factors[asset]
        print(f"Using calibration factor: {calibration_factor:.4f} (1/{1/calibration_factor:.2f})")
        
        # Get regime-specific forecasts (use best window based on RMSE)
        regime_forecasts = forecasts_df[
            (forecasts_df['regime'] == regime) & 
            (forecasts_df['asset'] == asset)
        ]
        
        if len(regime_forecasts) == 0:
            print(f"❌ No forecasts found for regime {regime}, asset {asset}")
            return None
        
        # Find best window (minimum RMSE)
        best_forecast = regime_forecasts.loc[regime_forecasts['rmse'].idxmin()]
        best_window = best_forecast['window']
        print(f"Best window: {best_window} (RMSE: {best_forecast['rmse']:.6f})")
        
        # Get the specific forecasts for this configuration
        specific_forecasts = regime_forecasts[regime_forecasts['window'] == best_window].copy()
        specific_forecasts = specific_forecasts.sort_values('date')
        
        # Get returns data
        asset_clean = asset.replace(' Index', '')
        return_col = f"Returns_{asset_clean}"
        
        if return_col not in market_data.columns:
            print(f"❌ Missing returns column: {return_col}")
            return None
        
        returns = market_data[return_col].dropna()
        
        # Filter dates to only use regime-specific periods
        regime_dates = regime_labels[regime_labels == regime].index
        print(f"📅 Regime {regime} has {len(regime_dates)} historical dates")
        
        # Prepare results storage
        results = []
        
        for _, forecast_row in specific_forecasts.iterrows():
            date = pd.to_datetime(forecast_row['date'])
            vol_forecast = forecast_row['predicted_rv22']  # This is RV22 prediction
            actual_rv22 = forecast_row['actual_rv22']
            
            if pd.isna(vol_forecast):
                continue
            
            # Get historical returns up to forecast date, filtered by regime
            hist_returns_all = returns[returns.index <= date]
            
            # Filter to only include dates when we were in this regime (for regime-specific FHS)
            regime_hist_dates = regime_dates[regime_dates <= date]
            
            if len(regime_hist_dates) < 100:  # Need sufficient regime-specific history
                continue
            
            # Use the most recent 252 regime-specific returns (or all available if less)
            regime_hist_returns = hist_returns_all[hist_returns_all.index.isin(regime_hist_dates)][-252:]
            
            if len(regime_hist_returns) < 50:  # Minimum threshold
                continue
            
            # Estimate historical volatility for standardization
            hist_vol = regime_hist_returns.std()
            
            # Standardize historical returns
            standardized_returns = regime_hist_returns / hist_vol
            
            # Convert RV22 forecast to daily volatility with PROPER CALIBRATION
            vol_forecast_1d = np.sqrt(vol_forecast / 22.0) * calibration_factor
            
            # Scale standardized returns by calibrated forecast volatility
            simulated_returns = standardized_returns * vol_forecast_1d
            
            # Calculate VaR as quantiles
            var_1pct = np.percentile(simulated_returns, 1)
            var_5pct = np.percentile(simulated_returns, 5)
            
            # Get next-day actual return for backtesting
            actual_return = np.nan
            for days_ahead in range(1, 5):
                check_date = date + pd.Timedelta(days=days_ahead)
                if check_date in returns.index:
                    actual_return = returns.loc[check_date]
                    break
            
            # Store results
            results.append({
                'Date': date,
                'Regime': regime,
                'Asset': asset,
                'Window': best_window,
                'VaR_1pct': var_1pct,
                'VaR_5pct': var_5pct,
                'Actual_Return': actual_return,
                'Volatility_Forecast_Raw': np.sqrt(vol_forecast / 22.0),
                'Volatility_Forecast_Calibrated': vol_forecast_1d,
                'RV22_Forecast': vol_forecast,
                'RV22_Actual': actual_rv22,
                'Calibration_Factor': calibration_factor,
                'Historical_Returns_Count': len(regime_hist_returns),
                'Violation_1pct': actual_return < var_1pct if not pd.isna(actual_return) else np.nan,
                'Violation_5pct': actual_return < var_5pct if not pd.isna(actual_return) else np.nan
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print(f"❌ No results generated for regime {regime}, asset {asset}")
            return None
        
        # Calculate violation rates
        valid_results = results_df.dropna(subset=['Violation_1pct', 'Violation_5pct'])
        
        if len(valid_results) > 0:
            violations_1pct = valid_results['Violation_1pct'].sum()
            violations_5pct = valid_results['Violation_5pct'].sum()
            total_obs = len(valid_results)
            
            viol_rate_1pct = (violations_1pct / total_obs) * 100
            viol_rate_5pct = (violations_5pct / total_obs) * 100
            
            print(f"📊 Regime {regime} {asset} Performance:")
            print(f"   1% VaR violations: {violations_1pct}/{total_obs} ({viol_rate_1pct:.2f}%)")
            print(f"   5% VaR violations: {violations_5pct}/{total_obs} ({viol_rate_5pct:.2f}%)")
            print(f"   Total forecasts: {len(results_df)}")
        
        # Save individual regime-asset results
        output_file = f"{self.output_dir}/regime_{regime}_{asset.replace(' ', '_')}_scaled_fhs.csv"
        results_df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        return results_df
    
    def run_analysis(self):
        """Run complete properly scaled hard regime FHS analysis"""
        print("🚀 PROPERLY SCALED HARD REGIME FHS ANALYSIS")
        print("=" * 60)
        
        # Load all data
        regime_labels = self.load_regime_labels()
        market_data = self.load_market_data()
        forecasts_df = self.load_hard_forecasts()
        
        # Results storage
        all_results = []
        summary_results = {
            "analysis_date": datetime.now().isoformat(),
            "method": "Properly scaled hard regime FHS with GARCH-LSTM forecasts",
            "calibration_factors": self.calibration_factors,
            "regimes_analyzed": [],
            "assets_analyzed": [],
            "violation_rates": {}
        }
        
        # Process each regime-asset combination
        for regime in self.viable_regimes:
            print(f"\n{'='*20} REGIME {regime} {'='*20}")
            
            for asset in ['SPX Index', 'RTY Index', 'NDX Index']:
                results_df = self.run_regime_specific_fhs(
                    regime, asset, forecasts_df, market_data, regime_labels
                )
                
                if results_df is not None:
                    all_results.append(results_df)
                    
                    # Track what we analyzed
                    if regime not in summary_results["regimes_analyzed"]:
                        summary_results["regimes_analyzed"].append(regime)
                    if asset not in summary_results["assets_analyzed"]:
                        summary_results["assets_analyzed"].append(asset)
                    
                    # Calculate summary stats
                    valid_results = results_df.dropna(subset=['Violation_1pct', 'Violation_5pct'])
                    if len(valid_results) > 0:
                        violations_1pct = valid_results['Violation_1pct'].sum()
                        violations_5pct = valid_results['Violation_5pct'].sum()
                        total_obs = len(valid_results)
                        
                        viol_rate_1pct = (violations_1pct / total_obs) * 100
                        viol_rate_5pct = (violations_5pct / total_obs) * 100
                        
                        key = f"regime_{regime}_{asset.replace(' ', '_')}"
                        summary_results["violation_rates"][key] = {
                            "regime": regime,
                            "asset": asset,
                            "1pct_violations": viol_rate_1pct,
                            "5pct_violations": viol_rate_5pct,
                            "observations": total_obs,
                            "total_forecasts": len(results_df),
                            "calibration_factor": self.calibration_factors[asset]
                        }
        
        # Save concatenated results
        if all_results:
            print(f"\n💾 Saving concatenated results...")
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_file = f"{self.output_dir}/all_properly_scaled_hard_regime_fhs.csv"
            combined_results.to_csv(combined_file, index=False)
            print(f"✅ Saved combined results: {combined_file}")
            
            # Create summary table
            summary_table = []
            for key, stats in summary_results["violation_rates"].items():
                summary_table.append({
                    'Regime': stats['regime'],
                    'Asset': stats['asset'], 
                    'Calibration_Factor': f"{stats['calibration_factor']:.4f}",
                    'VaR_1pct_Rate': f"{stats['1pct_violations']:.2f}%",
                    'VaR_5pct_Rate': f"{stats['5pct_violations']:.2f}%",
                    'Observations': stats['observations'],
                    'Total_Forecasts': stats['total_forecasts']
                })
            
            summary_df = pd.DataFrame(summary_table)
            summary_csv = f"{self.output_dir}/properly_scaled_summary_table.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"✅ Saved summary table: {summary_csv}")
        
        # Save detailed summary JSON
        summary_file = f"{self.output_dir}/properly_scaled_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        print(f"✅ Saved detailed summary: {summary_file}")
        
        # Print final summary
        print(f"\n🎉 PROPERLY SCALED HARD REGIME FHS COMPLETE!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"\n📊 FINAL VIOLATION RATES BY REGIME:")
        
        for regime in sorted(summary_results["regimes_analyzed"]):
            print(f"\n   Regime {regime}:")
            for key, stats in summary_results["violation_rates"].items():
                if stats['regime'] == regime:
                    asset_short = stats['asset'].replace(' Index', '')
                    print(f"     {asset_short}: 1%={stats['1pct_violations']:.2f}%, 5%={stats['5pct_violations']:.2f}%")

if __name__ == "__main__":
    analyzer = ProperlyScaledHardRegimeFHS()
    analyzer.run_analysis()