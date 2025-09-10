
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
import warnings
warnings.filterwarnings('ignore')

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(1 - (y_pred / y_true)))
    return rmse, mae, mape

def make_windows(series_1d, window):
    X, y = [], []
    for i in range(len(series_1d) - window):
        X.append(series_1d[i:i+window])
        y.append(series_1d[i+window])
    X = np.asarray(X).reshape(-1, window, 1)
    y = np.asarray(y).reshape(-1, 1)
    return X, y

def build_hybrid_model(window):
    # Historical branch (RV_22)
    hist_in = Input(shape=(window, 1), name="hist_in")
    hist_l = LSTM(20, activation='relu', return_sequences=False, name="hist_lstm")(hist_in)
    hist_d = Dropout(0.2, name="hist_drop")(hist_l)

    # GARCH branch (sGARCH forecasts)
    garch_in = Input(shape=(1, 1), name="garch_in")
    garch_l = LSTM(20, activation='relu', return_sequences=False, name="garch_lstm")(garch_in)
    garch_d = Dropout(0.2, name="garch_drop")(garch_l)

    # Combine branches - exact Baseline1 architecture
    x = concatenate([hist_d, garch_d], name="concat")
    x = Dense(128, activation='relu', name="fc1")(x)
    x = Dense(64, activation='relu', name="fc2")(x)
    out = Dense(1, activation='linear', name="out")(x)

    model = Model(inputs=[hist_in, garch_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_garch_variance_forecasts(returns, window=252):
    print(f"      Generating GARCH VARIANCE forecasts with {window}-day rolling window...")
    
    out = []
    n = len(returns)
    scale = 100.0  # rescale returns to improve optimizer numerics (from Baseline1)

    for t in range(window, n):
        r_win = returns[t-window:t]
        r_win_s = r_win * scale  # scaled

        try:
            # fit sGARCH(1,1) with normal errors (exact Baseline1 setup)
            am = arch_model(r_win_s, vol='GARCH', p=1, q=1, mean='Constant', dist='normal', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            
            # 1-step variance forecast, convert to original scale
            forecast = res.forecast(horizon=1)
            variance_scaled = forecast.variance.iloc[-1, 0]  # variance in scaled units
            variance_original = variance_scaled / (scale**2)  # back to ORIGINAL units
            
            # CRITICAL FIX: Return VARIANCE (not sqrt) to match RV_22 target
            out.append(variance_original)
            
        except Exception as e:
            print(f"        GARCH failed at t={t}: {e}")
            out.append(np.nan)

    print(f"      Generated {len(out)} GARCH variance forecasts")
    return np.array(out)

def align_lengths(*arrays):
    """Align multiple arrays to same length from the end (Baseline1 approach)"""
    L = min(len(a) for a in arrays)
    return [a[-L:] for a in arrays]

class CorrectedRegimeGARCHLSTM:
    def __init__(self, 
                 regime_labels_file='/Users/user/Desktop/Imperial/Diss/RegimeSwitching/regime_switching_models/regime_detection/regime_detection/testing/final_results/optimized_vol_risk_labels.csv',
                 data_file='/Users/user/Desktop/Imperial/Diss/RegimeSwitching/regime_switching_models/cleaned_data/complete_dataset_with_ndx.csv',
                 results_dir='/Users/user/Desktop/Imperial/Diss/RegimeSwitching/regime_switching_models/regime_detection/results',
                 min_obs_threshold=200, 
                 windows=[5, 11, 22], 
                 n_runs=25):  # Back to Baseline1's 25 runs
        
        self.regime_labels_file = regime_labels_file
        self.data_file = data_file
        self.results_dir = results_dir
        self.min_obs_threshold = min_obs_threshold
        self.windows = windows
        self.n_runs = n_runs
        self.target_assets = ['SPX Index', 'RTY Index', 'NDX Index']
        
        # Storage
        self.regime_labels = None
        self.data = None
        self.viable_regimes = []
        self.regime_data = {}
        self.results = []
        
    def load_and_align_data(self):
        print("Loading regime labels and market data...")
        
        # Load regime labels
        regime_df = pd.read_csv(self.regime_labels_file, parse_dates=['Date'])
        regime_labels = regime_df.set_index('Date')['Regime']
        
        # Load market data
        data = pd.read_csv(self.data_file, parse_dates=[0], index_col=0)
        
        # Exact alignment (critical for correct results)
        common_dates = regime_labels.index.intersection(data.index)
        regime_labels = regime_labels.loc[common_dates]
        data = data.loc[common_dates]
        
        # Remove NaN regimes
        valid_regime_mask = regime_labels.notna()
        regime_labels = regime_labels[valid_regime_mask]
        data = data.loc[regime_labels.index]
        
        self.regime_labels = regime_labels
        self.data = data
        
        print(f"Loaded and aligned {len(self.regime_labels)} observations")
        print(f"   Date range: {self.regime_labels.index.min()} to {self.regime_labels.index.max()}")
        
        # Identify viable regimes
        regime_counts = self.regime_labels.value_counts().sort_index()
        self.viable_regimes = []
        
        for regime, count in regime_counts.items():
            if count >= self.min_obs_threshold:
                self.viable_regimes.append(regime)
                print(f"  Regime {regime}: {count} obs Viable")
            else:
                print(f"  Regime {regime}: {count} obs Insufficient data")
        
        print(f"\nTraining on {len(self.viable_regimes)} viable regimes: {self.viable_regimes}")
        return self
    
    def prepare_regime_data(self):
        print(f"\nPreparing regime data...")
        
        for regime in self.viable_regimes:
            regime_mask = self.regime_labels == regime
            regime_dates = self.regime_labels[regime_mask].index
            
            print(f"\nRegime {regime}: {np.sum(regime_mask)} observations")
            
            # Store regime data for each asset
            self.regime_data[regime] = {}
            
            for asset in self.target_assets:
                # Extract RV_22 and LR data for this asset
                rv_col = f"{asset}_RV_22"
                lr_col = f"{asset}_LR"
                
                if rv_col not in self.data.columns or lr_col not in self.data.columns:
                    print(f"    {asset}: Missing columns")
                    continue
                
                rv_data = self.data.loc[regime_dates, rv_col].dropna()
                lr_data = self.data.loc[regime_dates, lr_col].dropna()
                
                if len(rv_data) >= 300 and len(lr_data) >= 300:
                    # Store with exact alignment
                    common_asset_dates = rv_data.index.intersection(lr_data.index)
                    rv_aligned = rv_data.loc[common_asset_dates].values
                    lr_aligned = lr_data.loc[common_asset_dates].values
                    
                    self.regime_data[regime][asset] = {
                        'rv_22': rv_aligned,
                        'log_returns': lr_aligned,
                        'dates': common_asset_dates
                    }
                    print(f"    {asset}: {len(rv_aligned)} aligned observations")
                else:
                    print(f"    {asset}: Insufficient data (RV:{len(rv_data)}, LR:{len(lr_data)})")
        
        return self
    
    def train_regime_asset_models(self, regime, asset):
        if regime not in self.regime_data or asset not in self.regime_data[regime]:
            return []
        
        regime_asset_data = self.regime_data[regime][asset]
        rv_series = regime_asset_data['rv_22']
        lr_series = regime_asset_data['log_returns']
        
        print(f"  Training {asset} models...")
        print(f"    Data: {len(rv_series)} RV observations, {len(lr_series)} LR observations")
        
        # Step 1: Generate GARCH variance forecasts on full series (Baseline1 approach)
        garch_forecasts = generate_garch_variance_forecasts(lr_series, window=252)
        
        # Step 2: Align RV and GARCH forecasts (critical step)
        rv_aligned, garch_aligned = align_lengths(rv_series, garch_forecasts)
        
        if len(rv_aligned) < 100:
            print(f"   Insufficient aligned data: {len(rv_aligned)}")
            return []
        
        print(f"    Aligned series length: {len(rv_aligned)}")
        
        # Step 3: 70/30 split (Baseline1 approach)
        n = len(rv_aligned)
        split_idx = int(0.7 * n)
        train_rv, test_rv = rv_aligned[:split_idx], rv_aligned[split_idx:]
        train_garch, test_garch = garch_aligned[:split_idx], garch_aligned[split_idx:]
        
        # Step 4: Scale each branch separately on train only (Baseline1 approach)
        sc_rv = MinMaxScaler()
        train_rv_scaled = sc_rv.fit_transform(train_rv.reshape(-1, 1)).ravel()
        test_rv_scaled = sc_rv.transform(test_rv.reshape(-1, 1)).ravel()
        
        sc_garch = MinMaxScaler()
        train_garch_scaled = sc_garch.fit_transform(train_garch.reshape(-1, 1)).ravel()
        test_garch_scaled = sc_garch.transform(test_garch.reshape(-1, 1)).ravel()
        
        regime_asset_results = []
        
        # Step 5: Train models for each window size
        for window in self.windows:
            print(f"    Window {window}...")
            
            # Create windowed data for historical branch (Baseline1 approach)
            X_train_hist, y_train = make_windows(train_rv_scaled, window)
            X_test_hist, y_test = make_windows(test_rv_scaled, window)
            
            # Create GARCH inputs (1-step, aligned with y indices) (Baseline1 approach)
            train_garch_input = train_garch_scaled[window:].reshape(-1, 1, 1)
            test_garch_input = test_garch_scaled[window:].reshape(-1, 1, 1)
            
            if len(X_train_hist) < 20 or len(X_test_hist) < 5:
                print(f"     Insufficient windowed data")
                continue
            
            # Step 6: Multiple runs and averaging (Baseline1: 25 runs)
            predictions_runs = []
            
            for run in range(self.n_runs):
                try:
                    model = build_hybrid_model(window)
                    early_stopping = EarlyStopping(
                        monitor='val_loss', 
                        patience=5, 
                        restore_best_weights=True, 
                        verbose=0
                    )
                    
                    # Train model (exact Baseline1 parameters)
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
                    
                except Exception as e:
                    print(f"      Run {run+1} failed: {e}")
                    continue
            
            # Step 7: Safe averaging (filter out failed runs) (Baseline1 approach)
            preds_clean = [p for p in predictions_runs if len(p) > 0 and np.all(np.isfinite(p))]
            if len(preds_clean) == 0:
                print(f"      All runs failed for window {window}")
                continue
            
            preds_mean = np.mean(np.vstack(preds_clean), axis=0)
            
            # Optional: clamp to [0,1] since target was MinMax-scaled (Baseline1 approach)
            preds_mean = np.clip(preds_mean, 0.0, 1.0)
            
            # Step 8: Inverse transform to original RV units (Baseline1 approach)
            y_test_inv = sc_rv.inverse_transform(y_test.reshape(-1, 1)).ravel()
            preds_inv = sc_rv.inverse_transform(preds_mean.reshape(-1, 1)).ravel()
            
            # Step 9: Calculate metrics on finite values only (Baseline1 approach)
            mask = np.isfinite(y_test_inv) & np.isfinite(preds_inv)
            y_eval = y_test_inv[mask]
            p_eval = preds_inv[mask]
            
            if len(y_eval) < 5:
                print(f"      Insufficient finite predictions")
                continue
            
            rmse, mae, mape = metrics(y_eval, p_eval)
            
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
            
            print(f"      RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f} ({len(preds_clean)}/{self.n_runs} runs)")
        
        return regime_asset_results
    
    def run_complete_training(self):
        """Execute complete corrected training process"""
        print("🚀 CORRECTED REGIME-SPECIFIC GARCH-LSTM TRAINING")
        print("=" * 60)
        print("Using Baseline1 methodology with proper variance handling")
        print(f"Assets: {', '.join(self.target_assets)}")
        print(f"Windows: {self.windows}")
        print(f"Runs per model: {self.n_runs}")
        
        # Load and prepare data
        self.load_and_align_data()
        self.prepare_regime_data()
        
        # Train models for each viable regime and asset
        all_results = []
        
        for regime in self.viable_regimes:
            print(f"\n=== TRAINING REGIME {regime} ===")
            
            for asset in self.target_assets:
                if regime not in self.regime_data or asset not in self.regime_data[regime]:
                    print(f"  {asset}: No data available")
                    continue
                
                regime_asset_results = self.train_regime_asset_models(regime, asset)
                all_results.extend(regime_asset_results)
        
        self.results = all_results
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_file = os.path.join(self.results_dir, 'corrected_regime_garch_lstm_results.csv')
            results_df.to_csv(results_file, index=False)
            
            print(f"\n💾 Results saved: {results_file}")
            print(f"Total models trained: {len(all_results)}")
            
            # Display summary
            print(f"\n📊 TRAINING SUMMARY:")
            for regime in self.viable_regimes:
                regime_results = results_df[results_df['regime'] == regime]
                if len(regime_results) > 0:
                    avg_rmse = regime_results['rmse'].mean()
                    print(f"  Regime {regime}: {len(regime_results)} models, avg RMSE: {avg_rmse:.6f}")
        else:
            print("\nNo models were successfully trained")
        
        print("\nCorrected GARCH-LSTM training complete!")
        return self

if __name__ == "__main__":
    trainer = CorrectedRegimeGARCHLSTM(
        min_obs_threshold=200,
        windows=[5, 11, 22],
        n_runs=25  # Baseline1's original 25 runs
    )
    trainer.run_complete_training()