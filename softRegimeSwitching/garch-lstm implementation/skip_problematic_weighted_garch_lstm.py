"""
Weighted GARCH-LSTM with Skip Problematic Combinations
====================================================

Enhanced version that skips problematic combinations instead of hanging:
- Progress tracking to identify exactly where hangs occur
- Skip individual runs that fail and continue with others
- Save partial results as we go
- Detailed logging of what was skipped and why
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
import json
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

def metrics(y_true, y_pred):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(1 - (y_pred / y_true)))
    return rmse, mae, mape

def make_windows(series_1d, window):
    """Convert 1D array to (X, y) for LSTM training"""
    X, y = [], []
    for i in range(len(series_1d) - window):
        X.append(series_1d[i:i+window])
        y.append(series_1d[i+window])
    X = np.asarray(X).reshape(-1, window, 1)
    y = np.asarray(y).reshape(-1, 1)
    return X, y

def build_hybrid_model(window):
    """Build GARCH-LSTM hybrid model"""
    # Historical branch (RV_22)
    hist_in = Input(shape=(window, 1), name="hist_in")
    hist_l = LSTM(20, activation='relu', return_sequences=False, name="hist_lstm")(hist_in)
    hist_d = Dropout(0.2, name="hist_drop")(hist_l)

    # GARCH branch (variance forecasts)
    garch_in = Input(shape=(1, 1), name="garch_in")
    garch_l = LSTM(20, activation='relu', return_sequences=False, name="garch_lstm")(garch_in)
    garch_d = Dropout(0.2, name="garch_drop")(garch_l)

    # Combine branches
    x = concatenate([hist_d, garch_d], name="concat")
    x = Dense(128, activation='relu', name="fc1")(x)
    x = Dense(64, activation='relu', name="fc2")(x)
    out = Dense(1, activation='linear', name="out")(x)

    model = Model(inputs=[hist_in, garch_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_garch_variance_forecasts_with_fallback(returns, window=252):
    """
    Generate GARCH forecasts with fallback for failed fits
    """
    print(f"      Generating GARCH VARIANCE forecasts with {window}-day rolling window...")
    
    out = []
    n = len(returns)
    scale = 100.0
    failed_count = 0

    for t in range(window, n):
        r_win = returns[t-window:t]
        r_win_s = r_win * scale

        try:
            am = arch_model(r_win_s, vol='GARCH', p=1, q=1, mean='Constant', dist='normal', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            
            forecast = res.forecast(horizon=1)
            variance_scaled = forecast.variance.iloc[-1, 0]
            variance_original = variance_scaled / (scale**2)
            
            out.append(variance_original)
            
        except Exception:
            failed_count += 1
            # Use rolling variance as fallback
            fallback_var = np.var(r_win)
            out.append(fallback_var)
    
    if failed_count > 0:
        print(f"      ⚠️ {failed_count} GARCH fits failed, used fallback variance")
    
    print(f"      Generated {len(out)} GARCH variance forecasts")
    return np.array(out)

def align_lengths(*arrays):
    """Align multiple arrays to same length from the end"""
    L = min(len(a) for a in arrays)
    return [a[-L:] for a in arrays]

def validate_data_quality(data, name):
    """Check for problematic data that might cause hangs"""
    if len(data) == 0:
        return False, "empty data"
    
    if np.any(np.isnan(data)):
        return False, "contains NaN"
    
    if np.any(np.isinf(data)):
        return False, "contains inf"
    
    if np.all(data == 0):
        return False, "all zeros"
    
    if np.std(data) == 0:
        return False, "no variance"
    
    if np.max(np.abs(data)) > 1e6:
        return False, "extreme values"
    
    return True, "ok"

class SkipProblematicWeightedGARCHLSTM:
    """Weighted GARCH-LSTM that skips problematic combinations"""
    
    def __init__(self, 
                 data_file='financial_indices_data.csv',
                 soft_probs_file='../regimeDetection/results/best_soft_probabilities.csv',
                 results_dir='results',
                 min_obs_threshold=200, 
                 windows=[5, 11, 22], 
                 n_runs=25):
        
        self.data_file = data_file
        self.soft_probs_file = soft_probs_file
        self.results_dir = results_dir
        self.min_obs_threshold = min_obs_threshold
        self.windows = windows
        self.n_runs = n_runs
        self.target_assets = ['SPX', 'RTY', 'NDX']
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
        
        # Storage
        self.data = None
        self.soft_probabilities = None
        self.viable_regimes = []
        self.regime_data = {}
        self.trained_models = {}
        self.weighted_forecasts = {}
        self.results = []
        
        # Tracking
        self.skipped_combinations = []
        self.progress_log = []
        
    def load_and_prepare_data(self):
        """Load market data and soft probabilities"""
        print("🔄 Loading market data and soft probabilities...")
        
        # Load market data
        self.data = pd.read_csv(self.data_file, parse_dates=['Date'], index_col='Date')
        print(f"✅ Loaded market data: {self.data.shape}")
        
        # Load soft probabilities
        self.soft_probabilities = pd.read_csv(self.soft_probs_file, parse_dates=['Date'], index_col='Date')
        print(f"✅ Loaded soft probabilities: {self.soft_probabilities.shape}")
        
        # Identify regimes
        prob_columns = [col for col in self.soft_probabilities.columns if col.startswith('P_regime_')]
        self.viable_regimes = [int(col.split('_')[-1]) for col in prob_columns]
        print(f"✅ Found regimes: {self.viable_regimes}")
        
        # Align dates
        common_dates = self.data.index.intersection(self.soft_probabilities.index)
        self.data = self.data.loc[common_dates]
        self.soft_probabilities = self.soft_probabilities.loc[common_dates]
        
        print(f"✅ Aligned data: {len(common_dates)} common dates")
        return self
        
    def create_regime_specific_data(self):
        """Create regime-specific training data using soft probabilities"""
        print(f"\n🔧 Creating regime-specific training data...")
        
        for regime in self.viable_regimes:
            prob_col = f'P_regime_{regime}'
            regime_probs = self.soft_probabilities[prob_col]
            
            # Probability-weighted sampling
            weights = regime_probs / regime_probs.sum()
            n_samples = max(self.min_obs_threshold, int(len(weights) * 0.8))
            
            sampled_indices = np.random.choice(
                len(weights), 
                size=n_samples, 
                p=weights, 
                replace=True
            )
            
            regime_dates = self.data.index[sampled_indices]
            print(f"\nRegime {regime}: {n_samples} probability-weighted samples")
            
            # Store regime data for each asset
            self.regime_data[regime] = {}
            
            for asset in self.target_assets:
                vol_col = f"Volatility_{asset}"
                ret_col = f"Returns_{asset}"
                
                if vol_col not in self.data.columns or ret_col not in self.data.columns:
                    print(f"    ❌ {asset}: Missing columns {vol_col} or {ret_col}")
                    continue
                
                vol_data = self.data.loc[regime_dates, vol_col].dropna()
                ret_data = self.data.loc[regime_dates, ret_col].dropna()
                
                if len(vol_data) >= 200 and len(ret_data) >= 200:
                    # Store with alignment
                    common_asset_dates = vol_data.index.intersection(ret_data.index)
                    vol_aligned = vol_data.loc[common_asset_dates].values
                    ret_aligned = ret_data.loc[common_asset_dates].values
                    
                    # Convert volatility to variance
                    rv_22_equiv = vol_aligned ** 2
                    
                    self.regime_data[regime][asset] = {
                        'rv_22': rv_22_equiv,
                        'log_returns': ret_aligned,
                        'dates': common_asset_dates
                    }
                    print(f"    ✅ {asset}: {len(rv_22_equiv)} aligned observations")
                else:
                    print(f"    ❌ {asset}: Insufficient data (Vol:{len(vol_data)}, Ret:{len(ret_data)})")
        
        return self
        
    def train_regime_asset_models_with_skip(self, regime, asset):
        """Train models with comprehensive skip logic"""
        combination_key = f"regime_{regime}_{asset}"
        
        if regime not in self.regime_data or asset not in self.regime_data[regime]:
            self.skipped_combinations.append(f"{combination_key}: no_data")
            return []
        
        regime_asset_data = self.regime_data[regime][asset]
        rv_series = regime_asset_data['rv_22']
        lr_series = regime_asset_data['log_returns']
        
        print(f"  🎯 Training {asset} models for regime {regime}...")
        print(f"    Data: {len(rv_series)} RV observations, {len(lr_series)} LR observations")
        
        # Validate input data quality
        rv_valid, rv_reason = validate_data_quality(rv_series, "RV")
        lr_valid, lr_reason = validate_data_quality(lr_series, "LR")
        
        if not rv_valid:
            print(f"    ❌ Skipping due to RV data issue: {rv_reason}")
            self.skipped_combinations.append(f"{combination_key}: rv_data_{rv_reason}")
            return []
        
        if not lr_valid:
            print(f"    ❌ Skipping due to LR data issue: {lr_reason}")
            self.skipped_combinations.append(f"{combination_key}: lr_data_{lr_reason}")
            return []
        
        try:
            # Generate GARCH forecasts with fallback
            garch_forecasts = generate_garch_variance_forecasts_with_fallback(lr_series, window=252)
            
            # Align data
            rv_aligned, garch_aligned = align_lengths(rv_series, garch_forecasts)
            
            if len(rv_aligned) < 100:
                print(f"    ❌ Insufficient aligned data: {len(rv_aligned)}")
                self.skipped_combinations.append(f"{combination_key}: insufficient_aligned_data")
                return []
            
            print(f"    Aligned series length: {len(rv_aligned)}")
            
            # Split and scale
            n = len(rv_aligned)
            split_idx = int(0.7 * n)
            train_rv, test_rv = rv_aligned[:split_idx], rv_aligned[split_idx:]
            train_garch, test_garch = garch_aligned[:split_idx], garch_aligned[split_idx:]
            
            # Validate training data
            train_rv_valid, _ = validate_data_quality(train_rv, "train_RV")
            train_garch_valid, _ = validate_data_quality(train_garch, "train_GARCH")
            
            if not train_rv_valid or not train_garch_valid:
                print(f"    ❌ Skipping due to training data quality issues")
                self.skipped_combinations.append(f"{combination_key}: bad_training_data")
                return []
            
            # Scale data
            sc_rv = MinMaxScaler()
            train_rv_scaled = sc_rv.fit_transform(train_rv.reshape(-1, 1)).ravel()
            test_rv_scaled = sc_rv.transform(test_rv.reshape(-1, 1)).ravel()
            
            sc_garch = MinMaxScaler()
            train_garch_scaled = sc_garch.fit_transform(train_garch.reshape(-1, 1)).ravel()
            test_garch_scaled = sc_garch.transform(test_garch.reshape(-1, 1)).ravel()
            
            regime_asset_results = []
            
            # Train models for each window
            for window in self.windows:
                window_key = f"{combination_key}_win{window}"
                print(f"    Window {window}...")
                
                try:
                    # Create windowed data
                    X_train_hist, y_train = make_windows(train_rv_scaled, window)
                    X_test_hist, y_test = make_windows(test_rv_scaled, window)
                    
                    train_garch_input = train_garch_scaled[window:].reshape(-1, 1, 1)
                    test_garch_input = test_garch_scaled[window:].reshape(-1, 1, 1)
                    
                    if len(X_train_hist) < 20 or len(X_test_hist) < 5:
                        print(f"      ❌ Insufficient windowed data")
                        self.skipped_combinations.append(f"{window_key}: insufficient_windowed_data")
                        continue
                    
                    # Validate windowed data
                    if np.any(np.isnan(X_train_hist)) or np.any(np.isnan(y_train)):
                        print(f"      ❌ NaN in windowed training data")
                        self.skipped_combinations.append(f"{window_key}: nan_in_windowed_data")
                        continue
                    
                    # Multiple runs with individual tracking
                    predictions_runs = []
                    run_failures = []
                    
                    for run in range(self.n_runs):
                        run_start_time = time.time()
                        
                        try:
                            print(f"      🔄 Run {run+1}/{self.n_runs}...", end=" ")
                            
                            model = build_hybrid_model(window)
                            early_stopping = EarlyStopping(
                                monitor='val_loss', 
                                patience=5, 
                                restore_best_weights=True, 
                                verbose=0
                            )
                            
                            # Train model
                            history = model.fit(
                                [X_train_hist, train_garch_input], 
                                y_train,
                                epochs=100,
                                batch_size=32,
                                validation_split=0.1,
                                verbose=0,
                                callbacks=[early_stopping]
                            )
                            
                            # Generate predictions
                            preds = model.predict([X_test_hist, test_garch_input], verbose=0).ravel()
                            
                            # Validate predictions
                            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                                print("❌ Invalid predictions")
                                run_failures.append(f"run_{run+1}: invalid_predictions")
                                continue
                            
                            predictions_runs.append(preds)
                            
                            run_time = time.time() - run_start_time
                            print(f"✅ ({run_time:.1f}s)")
                            
                        except Exception as e:
                            run_time = time.time() - run_start_time
                            print(f"❌ Failed ({run_time:.1f}s): {str(e)[:30]}...")
                            run_failures.append(f"run_{run+1}: {str(e)[:50]}")
                            continue
                    
                    # Check if we have enough successful runs
                    if len(predictions_runs) == 0:
                        print(f"      ❌ All {self.n_runs} runs failed for window {window}")
                        self.skipped_combinations.append(f"{window_key}: all_runs_failed")
                        continue
                    
                    if len(predictions_runs) < max(1, self.n_runs // 4):  # Need at least 25% success
                        print(f"      ❌ Too few successful runs: {len(predictions_runs)}/{self.n_runs}")
                        self.skipped_combinations.append(f"{window_key}: too_few_successful_runs")
                        continue
                    
                    # Average predictions
                    preds_mean = np.mean(np.vstack(predictions_runs), axis=0)
                    preds_mean = np.clip(preds_mean, 0.0, 1.0)
                    
                    # Inverse transform
                    y_test_inv = sc_rv.inverse_transform(y_test.reshape(-1, 1)).ravel()
                    preds_inv = sc_rv.inverse_transform(preds_mean.reshape(-1, 1)).ravel()
                    
                    # Calculate metrics
                    mask = np.isfinite(y_test_inv) & np.isfinite(preds_inv)
                    y_eval = y_test_inv[mask]
                    p_eval = preds_inv[mask]
                    
                    if len(y_eval) < 5:
                        print(f"      ❌ Insufficient finite predictions")
                        self.skipped_combinations.append(f"{window_key}: insufficient_finite_predictions")
                        continue
                    
                    rmse, mae, mape = metrics(y_eval, p_eval)
                    
                    result = {
                        'regime': regime,
                        'asset': asset,
                        'window': window,
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape,
                        'n_runs': len(predictions_runs),
                        'test_size': len(y_eval),
                        'run_failures': len(run_failures)
                    }
                    regime_asset_results.append(result)
                    
                    print(f"      ✅ RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f} ({len(predictions_runs)}/{self.n_runs} runs)")
                    
                    # Save progress
                    self.progress_log.append(f"✅ {window_key}: RMSE={rmse:.6f}")
                    
                except Exception as e:
                    print(f"      ❌ Window {window} failed: {e}")
                    self.skipped_combinations.append(f"{window_key}: window_exception: {str(e)[:50]}")
                    continue
            
            return regime_asset_results
            
        except Exception as e:
            print(f"    ❌ Complete failure for {asset} regime {regime}: {e}")
            self.skipped_combinations.append(f"{combination_key}: complete_failure: {str(e)[:50]}")
            return []
    
    def run_complete_training_with_skip(self):
        """Execute training with skip logic"""
        print("🚀 WEIGHTED GARCH-LSTM TRAINING (SKIP PROBLEMATIC)")
        print("=" * 70)
        print("Enhanced to skip problematic combinations and continue")
        print(f"Assets: {', '.join(self.target_assets)}")
        print(f"Windows: {self.windows}")
        print(f"Runs per model: {self.n_runs}")
        
        start_time = time.time()
        
        # Load and prepare data
        self.load_and_prepare_data()
        self.create_regime_specific_data()
        
        # Train models
        all_results = []
        total_combinations = len(self.viable_regimes) * len(self.target_assets)
        completed = 0
        
        for regime in self.viable_regimes:
            print(f"\n=== TRAINING REGIME {regime} MODELS ===")
            
            for asset in self.target_assets:
                completed += 1
                print(f"\n📊 Progress: {completed}/{total_combinations} combinations")
                
                try:
                    regime_asset_results = self.train_regime_asset_models_with_skip(regime, asset)
                    all_results.extend(regime_asset_results)
                    
                    # Save intermediate results
                    if regime_asset_results:
                        temp_df = pd.DataFrame(all_results)
                        temp_df.to_csv(f"{self.results_dir}/intermediate_results.csv", index=False)
                    
                except Exception as e:
                    print(f"  ❌ Fatal error for {asset} regime {regime}: {e}")
                    self.skipped_combinations.append(f"regime_{regime}_{asset}: fatal_error: {str(e)[:50]}")
        
        self.results = all_results
        
        # Final summary
        elapsed_time = time.time() - start_time
        self._save_final_summary(elapsed_time)
        
        return self
    
    def _save_final_summary(self, elapsed_time):
        """Save comprehensive final summary"""
        print(f"\n🎉 TRAINING COMPLETE!")
        print("=" * 50)
        print(f"⏱️ Total time: {elapsed_time/3600:.2f} hours")
        print(f"✅ Successful models: {len(self.results)}")
        print(f"❌ Skipped combinations: {len(self.skipped_combinations)}")
        
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_file = f"{self.results_dir}/final_training_results.csv"
            results_df.to_csv(results_file, index=False)
            print(f"💾 Results saved: {results_file}")
            
            # Summary by regime
            print(f"\n📊 RESULTS BY REGIME:")
            for regime in self.viable_regimes:
                regime_results = results_df[results_df['regime'] == regime]
                if len(regime_results) > 0:
                    avg_rmse = regime_results['rmse'].mean()
                    print(f"  Regime {regime}: {len(regime_results)} models, avg RMSE: {avg_rmse:.6f}")
        
        if self.skipped_combinations:
            print(f"\n🚨 SKIPPED COMBINATIONS ({len(self.skipped_combinations)}):")
            skip_reasons = {}
            for skip in self.skipped_combinations:
                reason = skip.split(':')[-1].strip()
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            
            for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} times")
        
        # Save detailed logs
        with open(f"{self.results_dir}/skip_log.txt", 'w') as f:
            f.write("SKIPPED COMBINATIONS:\n")
            for skip in self.skipped_combinations:
                f.write(f"{skip}\n")
        
        print(f"\n📁 All files saved in: {self.results_dir}")

def main():
    """Run weighted GARCH-LSTM training with skip logic"""
    trainer = SkipProblematicWeightedGARCHLSTM(
        windows=[5, 11, 22],
        n_runs=25
    )
    trainer.run_complete_training_with_skip()

if __name__ == "__main__":
    main()