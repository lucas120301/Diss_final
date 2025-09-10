"""
FHS - Scaled by the scaling factors
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class ProperlyScaledFHS:
    def __init__(self):
        self.base_path = "/Users/user/Desktop/Imperial/Diss_final"
        self.soft_forecasts_dir = "results"
        self.market_data_path = f"{self.base_path}/softRegimeSwitching/garch-lstm implementation/financial_indices_data.csv"
        self.output_dir = "properly_scaled_fhs_results"

        os.makedirs(self.output_dir, exist_ok=True)
        
        self.calibration_factors = {
            'SPX': 1.0 / 20.09,  
            'RTY': 1.0 / 14.29,  
            'NDX': 1.0 / 16.56  
        }
        
    def load_data(self):
        print("📊 Loading soft weighted forecasts and market data...")
        
        self.soft_forecasts = pd.read_csv(
            f"{self.soft_forecasts_dir}/all_soft_weighted_forecasts.csv",
            parse_dates=['Date']
        )
        print(f"Loaded soft forecasts: {self.soft_forecasts.shape}")
        
        self.market_data = pd.read_csv(
            self.market_data_path, 
            parse_dates=['Date'], 
            index_col='Date'
        )
        print(f"✅ Loaded market data: {self.market_data.shape}")
        
        return self
    
    def run_fhs_for_asset(self, asset):
        """Run FHS with proper calibration for a specific asset"""
        print(f"\nRunning properly calibrated FHS for {asset}...")
        
        calibration_factor = self.calibration_factors[asset]
        print(f"Using calibration factor: {calibration_factor:.4f} (1/{1/calibration_factor:.2f})")
        
        # Get asset data
        asset_forecasts = self.soft_forecasts[
            self.soft_forecasts['Asset'] == asset
        ].copy().dropna()
        
        return_col = f"Returns_{asset}"
        if return_col not in self.market_data.columns:
            print(f"Missing returns column: {return_col}")
            return None
        
        returns = self.market_data[return_col].dropna()
        
        # Prepare results storage
        results = []
        
        # Sort forecasts by date
        asset_forecasts = asset_forecasts.sort_values('Date')
        
        for _, row in asset_forecasts.iterrows():
            date = pd.to_datetime(row['Date'])
            vol_forecast = row['Soft_Weighted_Forecast']  # This is RV22
            
            if pd.isna(vol_forecast):
                continue
            
            # Get historical returns up to forecast date (252 days)
            hist_returns = returns[returns.index <= date][-252:]
            if len(hist_returns) < 252:
                continue
            
            # Estimate historical volatility for standardization
            hist_vol = hist_returns.std()
            
            # Standardize historical returns
            standardized_returns = hist_returns / hist_vol
            
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
                'VaR_1pct': var_1pct,
                'VaR_5pct': var_5pct,
                'Actual_Return': actual_return,
                'Volatility_Forecast_Raw': np.sqrt(vol_forecast / 22.0),
                'Volatility_Forecast_Calibrated': vol_forecast_1d,
                'RV22_Forecast': vol_forecast,
                'Calibration_Factor': calibration_factor,
                'Violation_1pct': actual_return < var_1pct if not pd.isna(actual_return) else np.nan,
                'Violation_5pct': actual_return < var_5pct if not pd.isna(actual_return) else np.nan
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print(f"No FHS results generated for {asset}")
            return None
        
        # Calculate violation rates
        valid_results = results_df.dropna(subset=['Violation_1pct', 'Violation_5pct'])
        if len(valid_results) > 0:
            violations_1pct = valid_results['Violation_1pct'].sum()
            violations_5pct = valid_results['Violation_5pct'].sum()
            total_obs = len(valid_results)
            
            viol_rate_1pct = (violations_1pct / total_obs) * 100
            viol_rate_5pct = (violations_5pct / total_obs) * 100
            
            print(f"📊 {asset} FHS Performance (Properly Calibrated):")
            print(f"   1% VaR violations: {violations_1pct}/{total_obs} ({viol_rate_1pct:.2f}%)")
            print(f"   5% VaR violations: {violations_5pct}/{total_obs} ({viol_rate_5pct:.2f}%)")
            print(f"   Total forecasts: {len(results_df)}")
            
            # Show some sample VaR values to check reasonableness
            sample_vars = valid_results[['VaR_1pct', 'VaR_5pct']].head()
            print(f"   Sample VaR values:")
            for idx, row in sample_vars.iterrows():
                print(f"     VaR 1%: {row['VaR_1pct']:.4f}, VaR 5%: {row['VaR_5pct']:.4f}")
        
        # Save individual asset results
        output_file = f"{self.output_dir}/properly_scaled_fhs_results_{asset}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        return results_df
    
    def run_analysis(self):
        """Run complete properly calibrated FHS analysis"""
        print("🚀 PROPERLY CALIBRATED SOFT-WEIGHTED FHS ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Results storage
        all_results = []
        summary_results = {
            "analysis_date": datetime.now().isoformat(),
            "method": "Properly calibrated FHS with soft-weighted GARCH-LSTM forecasts",
            "calibration_factors": self.calibration_factors,
            "assets_analyzed": [],
            "violation_rates": {}
        }
        
        # Process each asset
        for asset in ['SPX', 'RTY', 'NDX']:
            print(f"\n=== PROCESSING {asset} ===")
            
            asset_results = self.run_fhs_for_asset(asset)
            
            if asset_results is not None:
                all_results.append(asset_results)
                
                # Calculate summary statistics
                valid_results = asset_results.dropna(subset=['Violation_1pct', 'Violation_5pct'])
                if len(valid_results) > 0:
                    violations_1pct = valid_results['Violation_1pct'].sum()
                    violations_5pct = valid_results['Violation_5pct'].sum()
                    total_obs = len(valid_results)
                    
                    viol_rate_1pct = (violations_1pct / total_obs) * 100
                    viol_rate_5pct = (violations_5pct / total_obs) * 100
                    
                    summary_results["assets_analyzed"].append(asset)
                    summary_results["violation_rates"][asset] = {
                        "1pct_violations": viol_rate_1pct,
                        "5pct_violations": viol_rate_5pct,
                        "observations": total_obs,
                        "total_forecasts": len(asset_results),
                        "calibration_factor": self.calibration_factors[asset]
                    }
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_file = f"{self.output_dir}/all_properly_scaled_fhs_results.csv"
            combined_results.to_csv(combined_file, index=False)
            print(f"💾 Saved combined results: {combined_file}")

        summary_file = f"{self.output_dir}/properly_scaled_fhs_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        print(f"💾 Saved summary: {summary_file}")
        
        print(f"\n🎉 PROPERLY CALIBRATED FHS ANALYSIS COMPLETE!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"\n📊 FINAL VIOLATION RATES:")
        for asset in summary_results["assets_analyzed"]:
            rates = summary_results["violation_rates"][asset]
            calib = rates["calibration_factor"]
            print(f"   {asset} (calib={calib:.4f}): 1%={rates['1pct_violations']:.2f}%, 5%={rates['5pct_violations']:.2f}%")

if __name__ == "__main__":
    analyzer = ProperlyScaledFHS()
    analyzer.run_analysis()