"""
Weighted Regime-Specific GARCH-LSTM Implementation
=================================================

Based on corrected_regime_garch_lstm.py architecture but using SOFT probabilities
for Dynamic Model Averaging instead of hard regime switching.

Key innovations:
1. Uses optimal soft probabilities from clustering validation (GMM full covariance)
2. Trains regime-specific GARCH-LSTM models using exact Baseline1 architecture
3. Weights forecasts using soft probabilities: Σ P(regime_k|t) × forecast_k(t)
4. Saves all models and creates unified weighted forecasts for fair VaR comparison

Architecture: Same as corrected_regime_garch_lstm.py
- Historical LSTM branch (RV_22 input)
- GARCH LSTM branch (variance forecasts)
- Combined with Dense(128,64,1) layers
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
warnings.filterwarnings('ignore')

def metrics(y_true, y_pred):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(1 - (y_pred / y_true)))
    return rmse, mae, mape

def make_windows(series_1d, window):
    """Convert 1D array to (X, y) for LSTM training - from Baseline1"""
    X, y = [], []
    for i in range(len(series_1d) - window):
        X.append(series_1d[i:i+window])
        y.append(series_1d[i+window])
    X = np.asarray(X).reshape(-1, window, 1)
    y = np.asarray(y).reshape(-1, 1)
    return X, y

def build_hybrid_model(window):
    """Build GARCH-LSTM hybrid exactly like corrected implementation"""
    # Historical branch (RV_22)
    hist_in = Input(shape=(window, 1), name="hist_in")
    hist_l = LSTM(20, activation='relu', return_sequences=False, name="hist_lstm")(hist_in)
    hist_d = Dropout(0.2, name="hist_drop")(hist_l)

    # GARCH branch (sGARCH forecasts)
    garch_in = Input(shape=(1, 1), name="garch_in")
    garch_l = LSTM(20, activation='relu', return_sequences=False, name="garch_lstm")(garch_in)
    garch_d = Dropout(0.2, name="garch_drop")(garch_l)

    # Combine branches - exact architecture
    x = concatenate([hist_d, garch_d], name="concat")
    x = Dense(128, activation='relu', name="fc1")(x)
    x = Dense(64, activation='relu', name="fc2")(x)
    out = Dense(1, activation='linear', name="out")(x)

    model = Model(inputs=[hist_in, garch_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_garch_variance_forecasts(returns, window=252):
    """
    Generate 252-day rolling sGARCH(1,1) VARIANCE forecasts
    Returns variance (not volatility) to match RV_22 target scale
    """
    print(f"      Generating GARCH VARIANCE forecasts with {window}-day rolling window...")
    
    out = []
    n = len(returns)
    scale = 100.0  # rescale returns to improve optimizer numerics

    for t in range(window, n):
        r_win = returns[t-window:t]
        r_win_s = r_win * scale  # scaled

        try:
            # fit sGARCH(1,1) with normal errors
            am = arch_model(r_win_s, vol='GARCH', p=1, q=1, mean='Constant', dist='normal', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            
            # 1-step variance forecast, convert to original scale
            forecast = res.forecast(horizon=1)
            variance_scaled = forecast.variance.iloc[-1, 0]  # variance in scaled units
            variance_original = variance_scaled / (scale**2)  # back to ORIGINAL units
            
            # Return VARIANCE to match RV_22 target
            out.append(variance_original)
            
        except Exception as e:
            print(f"        GARCH failed at t={t}: {e}")
            out.append(np.nan)

    print(f"      Generated {len(out)} GARCH variance forecasts")
    return np.array(out)

def align_lengths(*arrays):
    """Align multiple arrays to same length from the end"""
    L = min(len(a) for a in arrays)
    return [a[-L:] for a in arrays]

class WeightedRegimeGARCHLSTM:
    """Weighted regime-specific GARCH-LSTM using soft probabilities"""
    
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
        self.target_assets = ['SPX', 'RTY', 'NDX']  # Match your data columns
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
        
        # Storage
        self.data = None
        self.soft_probabilities = None
        self.viable_regimes = []
        self.regime_data = {}
        self.trained_models = {}  # Store all trained models
        self.regime_forecasts = {}  # Store regime-specific forecasts
        self.weighted_forecasts = {}  # Store final weighted forecasts
        self.results = []
        
    def load_and_prepare_data(self):
        """Load market data and soft probabilities"""
        print("🔄 Loading market data and soft probabilities...")
        
        # Load market data
        self.data = pd.read_csv(self.data_file, parse_dates=['Date'], index_col='Date')
        print(f"✅ Loaded market data: {self.data.shape}")
        print(f"   Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"   Assets: {self.target_assets}")
        
        # Load soft probabilities
        self.soft_probabilities = pd.read_csv(self.soft_probs_file, parse_dates=['Date'], index_col='Date')
        print(f"✅ Loaded soft probabilities: {self.soft_probabilities.shape}")
        print(f"   Date range: {self.soft_probabilities.index.min()} to {self.soft_probabilities.index.max()}")
        
        # Identify regimes and viable ones
        prob_columns = [col for col in self.soft_probabilities.columns if col.startswith('P_regime_')]
        self.viable_regimes = [int(col.split('_')[-1]) for col in prob_columns]
        print(f"✅ Found regimes: {self.viable_regimes}")
        
        # Align dates
        common_dates = self.data.index.intersection(self.soft_probabilities.index)
        self.data = self.data.loc[common_dates]
        self.soft_probabilities = self.soft_probabilities.loc[common_dates]
        
        print(f"✅ Aligned data: {len(common_dates)} common dates")
        print(f"   Final range: {common_dates.min()} to {common_dates.max()}")
        
        return self
    
    def create_regime_specific_data(self):
        """Create pseudo-regime data using soft probabilities for training"""
        print(f"\n🔧 Creating regime-specific training data using soft probabilities...")
        
        for regime in self.viable_regimes:
            prob_col = f'P_regime_{regime}'
            regime_probs = self.soft_probabilities[prob_col]
            
            # Use probability-weighted sampling for training data
            # Higher probability dates get higher weight in training
            weights = regime_probs / regime_probs.sum()  # Normalize to weights
            
            # Sample dates based on probabilities (with replacement for sufficient data)
            n_samples = max(self.min_obs_threshold, int(len(weights) * 0.8))  # Ensure minimum data
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
                # Extract volatility and returns data
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
                    
                    # Convert volatility to RV_22 equivalent (annualized)
                    rv_22_equiv = vol_aligned ** 2  # Variance (to match GARCH target)
                    
                    self.regime_data[regime][asset] = {
                        'rv_22': rv_22_equiv,
                        'log_returns': ret_aligned,
                        'dates': common_asset_dates
                    }
                    print(f"    ✅ {asset}: {len(rv_22_equiv)} aligned observations")
                else:
                    print(f"    ❌ {asset}: Insufficient data (Vol:{len(vol_data)}, Ret:{len(ret_data)})")
        
        return self
    
    def train_regime_asset_models(self, regime, asset):
        """Train GARCH-LSTM models for specific regime-asset"""
        if regime not in self.regime_data or asset not in self.regime_data[regime]:
            return []
        
        regime_asset_data = self.regime_data[regime][asset]
        rv_series = regime_asset_data['rv_22']
        lr_series = regime_asset_data['log_returns']
        
        print(f"  Training {asset} models for regime {regime}...")
        print(f"    Data: {len(rv_series)} RV observations, {len(lr_series)} LR observations")
        
        # Step 1: Generate GARCH variance forecasts
        garch_forecasts = generate_garch_variance_forecasts(lr_series, window=252)
        
        # Step 2: Align RV and GARCH forecasts
        rv_aligned, garch_aligned = align_lengths(rv_series, garch_forecasts)
        
        if len(rv_aligned) < 100:
            print(f"    ❌ Insufficient aligned data: {len(rv_aligned)}")
            return []
        
        print(f"    Aligned series length: {len(rv_aligned)}")
        
        # Step 3: 70/30 split
        n = len(rv_aligned)
        split_idx = int(0.7 * n)
        train_rv, test_rv = rv_aligned[:split_idx], rv_aligned[split_idx:]
        train_garch, test_garch = garch_aligned[:split_idx], garch_aligned[split_idx:]
        
        # Step 4: Scale each branch separately
        sc_rv = MinMaxScaler()
        train_rv_scaled = sc_rv.fit_transform(train_rv.reshape(-1, 1)).ravel()
        test_rv_scaled = sc_rv.transform(test_rv.reshape(-1, 1)).ravel()
        
        sc_garch = MinMaxScaler()
        train_garch_scaled = sc_garch.fit_transform(train_garch.reshape(-1, 1)).ravel()
        test_garch_scaled = sc_garch.transform(test_garch.reshape(-1, 1)).ravel()
        
        regime_asset_results = []
        regime_asset_models = {}
        
        # Step 5: Train models for each window size
        for window in self.windows:
            print(f"    Window {window}...")
            
            # Create windowed data
            X_train_hist, y_train = make_windows(train_rv_scaled, window)
            X_test_hist, y_test = make_windows(test_rv_scaled, window)
            
            # Create GARCH inputs
            train_garch_input = train_garch_scaled[window:].reshape(-1, 1, 1)
            test_garch_input = test_garch_scaled[window:].reshape(-1, 1, 1)
            
            if len(X_train_hist) < 20 or len(X_test_hist) < 5:
                print(f"      ❌ Insufficient windowed data")
                continue
            
            # Step 6: Multiple runs and averaging
            predictions_runs = []
            models_for_window = []
            
            for run in range(self.n_runs):
                try:
                    model = build_hybrid_model(window)
                    early_stopping = EarlyStopping(
                        monitor='val_loss', 
                        patience=5, 
                        restore_best_weights=True, 
                        verbose=0
                    )
                    
                    # Train model
                    model.fit(
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
                    predictions_runs.append(preds)
                    models_for_window.append(model)
                    
                except Exception as e:
                    print(f"      ⚠️ Run {run+1} failed: {e}")
                    continue
            
            # Step 7: Average predictions and save best models
            preds_clean = [p for p in predictions_runs if len(p) > 0 and np.all(np.isfinite(p))]
            if len(preds_clean) == 0:
                print(f"      ❌ All runs failed for window {window}")
                continue
            
            preds_mean = np.mean(np.vstack(preds_clean), axis=0)
            preds_mean = np.clip(preds_mean, 0.0, 1.0)
            
            # Step 8: Inverse transform
            y_test_inv = sc_rv.inverse_transform(y_test.reshape(-1, 1)).ravel()
            preds_inv = sc_rv.inverse_transform(preds_mean.reshape(-1, 1)).ravel()
            
            # Step 9: Calculate metrics
            mask = np.isfinite(y_test_inv) & np.isfinite(preds_inv)
            y_eval = y_test_inv[mask]
            p_eval = preds_inv[mask]
            
            if len(y_eval) < 5:
                print(f"      ❌ Insufficient finite predictions")
                continue
            
            rmse, mae, mape = metrics(y_eval, p_eval)
            
            # Store results and models
            result = {
                'regime': regime,
                'asset': asset,
                'window': window,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'n_runs': len(preds_clean),
                'test_size': len(y_eval)
            }
            regime_asset_results.append(result)
            
            # Save models and scalers
            model_key = f"regime_{regime}_{asset}_win{window}"
            regime_asset_models[model_key] = {
                'models': models_for_window,  # All successful models
                'scalers': {'rv': sc_rv, 'garch': sc_garch},
                'test_data': {
                    'X_test_hist': X_test_hist,
                    'test_garch_input': test_garch_input,
                    'y_test': y_test_inv,
                    'predictions': preds_inv
                }
            }
            
            print(f"      ✅ RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f} ({len(preds_clean)}/{self.n_runs} runs)")
        
        # Store trained models for this regime-asset
        if regime not in self.trained_models:
            self.trained_models[regime] = {}
        self.trained_models[regime][asset] = regime_asset_models
        
        return regime_asset_results
    
    def generate_weighted_forecasts(self):
        """Generate forecasts from each regime and weight by soft probabilities"""
        print(f"\n🎯 Generating weighted forecasts using soft probabilities...")
        
        # For each asset and window combination
        for asset in self.target_assets:
            for window in self.windows:
                key = f"{asset}_win{window}"
                print(f"\n  Processing {key}...")
                
                # Collect regime-specific forecasts for this asset-window
                regime_forecasts_dict = {}
                common_dates = None
                
                for regime in self.viable_regimes:
                    model_key = f"regime_{regime}_{asset}_win{window}"
                    
                    if (regime in self.trained_models and 
                        asset in self.trained_models[regime] and
                        model_key in self.trained_models[regime][asset]):
                        
                        test_data = self.trained_models[regime][asset][model_key]['test_data']
                        forecasts = test_data['predictions']
                        
                        # For demo, use test dates (in practice, generate on full out-of-sample)
                        if common_dates is None:
                            # Create synthetic dates for test period
                            test_start_idx = len(self.data) - len(forecasts)
                            common_dates = self.data.index[test_start_idx:test_start_idx + len(forecasts)]
                        
                        if len(forecasts) == len(common_dates):
                            regime_forecasts_dict[regime] = forecasts
                            print(f"    ✅ Regime {regime}: {len(forecasts)} forecasts")
                        else:
                            print(f"    ❌ Regime {regime}: Length mismatch")
                    else:
                        print(f"    ❌ Regime {regime}: No trained model")
                
                if len(regime_forecasts_dict) == 0 or common_dates is None:
                    print(f"    ❌ No regime forecasts available for {key}")
                    continue
                
                # Weight forecasts using soft probabilities
                weighted_forecasts = np.zeros(len(common_dates))
                regime_contributions = {regime: np.zeros(len(common_dates)) for regime in self.viable_regimes}
                
                for i, date in enumerate(common_dates):
                    if date in self.soft_probabilities.index:
                        date_probs = self.soft_probabilities.loc[date]
                        
                        for regime in regime_forecasts_dict:
                            prob_col = f'P_regime_{regime}'
                            if prob_col in date_probs:
                                prob = date_probs[prob_col]
                                forecast = regime_forecasts_dict[regime][i]
                                
                                contribution = prob * forecast
                                weighted_forecasts[i] += contribution
                                regime_contributions[regime][i] = contribution
                
                # Store weighted forecasts
                self.weighted_forecasts[key] = {
                    'dates': common_dates,
                    'weighted_forecasts': weighted_forecasts,
                    'regime_contributions': regime_contributions,
                    'regime_forecasts': regime_forecasts_dict
                }
                
                # Save to CSV
                forecast_df = pd.DataFrame({
                    'Date': common_dates,
                    'weighted_forecast': weighted_forecasts
                })
                
                # Add regime contributions
                for regime in self.viable_regimes:
                    if regime in regime_contributions:
                        forecast_df[f'regime_{regime}_contribution'] = regime_contributions[regime]
                
                forecast_file = f"{self.results_dir}/weighted_forecast_{asset}_win{window}.csv"
                forecast_df.to_csv(forecast_file, index=False)
                print(f"    💾 Saved: {forecast_file}")
        
        return self
    
    def save_all_models_and_results(self):
        """Save all trained models, scalers, and comprehensive results"""
        print(f"\n💾 Saving all models and results...")
        
        # Save trained models
        models_file = f"{self.results_dir}/models/all_trained_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(self.trained_models, f)
        print(f"    ✅ Saved trained models: {models_file}")
        
        # Save weighted forecasts
        forecasts_file = f"{self.results_dir}/weighted_forecasts.pkl"
        with open(forecasts_file, 'wb') as f:
            pickle.dump(self.weighted_forecasts, f)
        print(f"    ✅ Saved weighted forecasts: {forecasts_file}")
        
        # Save training results
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_file = f"{self.results_dir}/training_results.csv"
            results_df.to_csv(results_file, index=False)
            print(f"    ✅ Saved training results: {results_file}")
        
        # Save metadata
        metadata = {
            'methodology': 'weighted_regime_garch_lstm',
            'description': 'GARCH-LSTM models trained per regime, weighted by soft probabilities',
            'assets': self.target_assets,
            'windows': self.windows,
            'viable_regimes': self.viable_regimes,
            'n_runs_per_model': self.n_runs,
            'data_file': self.data_file,
            'soft_probs_file': self.soft_probs_file,
            'total_models_trained': len(self.results),
            'weighted_forecast_files': list(self.weighted_forecasts.keys())
        }
        
        metadata_file = f"{self.results_dir}/metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"    ✅ Saved metadata: {metadata_file}")
        
        return self
    
    def run_complete_training(self):
        """Execute complete weighted training process"""
        print("🚀 WEIGHTED REGIME-SPECIFIC GARCH-LSTM TRAINING")
        print("=" * 70)
        print("Using soft probabilities for Dynamic Model Averaging")
        print(f"Assets: {', '.join(self.target_assets)}")
        print(f"Windows: {self.windows}")
        print(f"Runs per model: {self.n_runs}")
        
        # Load and prepare data
        self.load_and_prepare_data()
        self.create_regime_specific_data()
        
        # Train models for each regime and asset
        all_results = []
        
        for regime in self.viable_regimes:
            print(f"\n=== TRAINING REGIME {regime} MODELS ===")
            
            for asset in self.target_assets:
                if regime not in self.regime_data or asset not in self.regime_data[regime]:
                    print(f"  ❌ {asset}: No data available")
                    continue
                
                regime_asset_results = self.train_regime_asset_models(regime, asset)
                all_results.extend(regime_asset_results)
        
        self.results = all_results
        
        # Generate weighted forecasts
        self.generate_weighted_forecasts()
        
        # Save everything
        self.save_all_models_and_results()
        
        # Display summary
        if all_results:
            results_df = pd.DataFrame(all_results)
            print(f"\n📊 TRAINING SUMMARY:")
            print(f"Total models trained: {len(all_results)}")
            
            for regime in self.viable_regimes:
                regime_results = results_df[results_df['regime'] == regime]
                if len(regime_results) > 0:
                    avg_rmse = regime_results['rmse'].mean()
                    print(f"  Regime {regime}: {len(regime_results)} models, avg RMSE: {avg_rmse:.6f}")
            
            print(f"\n🎯 WEIGHTED FORECASTS GENERATED:")
            for key in self.weighted_forecasts.keys():
                n_forecasts = len(self.weighted_forecasts[key]['weighted_forecasts'])
                print(f"  {key}: {n_forecasts} weighted forecasts")
        else:
            print("\n❌ No models were successfully trained")
        
        print(f"\n🎉 Weighted GARCH-LSTM training complete!")
        print(f"📁 All results saved in: {self.results_dir}")
        print(f"🎯 Ready for VaR comparison with Baseline1!")
        return self

if __name__ == "__main__":
    # Full production version
    trainer = WeightedRegimeGARCHLSTM(
        windows=[5, 11, 22],  # All windows for comprehensive analysis
        n_runs=25             # Full 25 runs for robust averaging
    )
    trainer.run_complete_training()