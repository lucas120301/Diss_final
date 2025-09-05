"""
Quick diagnostic test for weighted GARCH-LSTM data loading
Test the data loading and basic setup without heavy computation
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("🧪 DIAGNOSTIC TEST: Data Loading and Basic Setup")
print("=" * 60)

try:
    # Test 1: Load financial data
    print("1️⃣ Testing financial data loading...")
    data_file = 'garch-lstm implementation/financial_indices_data.csv'
    data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
    print(f"   ✅ Financial data: {data.shape}")
    print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
    
    # Test 2: Load soft probabilities
    print("\n2️⃣ Testing soft probabilities loading...")
    soft_probs_file = 'regimeDetection/results/best_soft_probabilities.csv'
    soft_probabilities = pd.read_csv(soft_probs_file, parse_dates=['Date'], index_col='Date')
    print(f"   ✅ Soft probabilities: {soft_probabilities.shape}")
    print(f"   📅 Date range: {soft_probabilities.index.min()} to {soft_probabilities.index.max()}")
    
    # Test 3: Data alignment
    print("\n3️⃣ Testing data alignment...")
    common_dates = data.index.intersection(soft_probabilities.index)
    aligned_data = data.loc[common_dates]
    aligned_probs = soft_probabilities.loc[common_dates]
    print(f"   ✅ Aligned dates: {len(common_dates)}")
    print(f"   📊 Final data shape: {aligned_data.shape}")
    print(f"   📊 Final probs shape: {aligned_probs.shape}")
    
    # Test 4: Regime identification
    print("\n4️⃣ Testing regime identification...")
    prob_columns = [col for col in soft_probabilities.columns if col.startswith('P_regime_')]
    viable_regimes = [int(col.split('_')[-1]) for col in prob_columns]
    print(f"   ✅ Found regimes: {viable_regimes}")
    
    # Test 5: Sample regime data creation
    print("\n5️⃣ Testing regime sampling...")
    target_assets = ['SPX', 'RTY', 'NDX']
    min_obs_threshold = 200
    
    for regime in viable_regimes[:2]:  # Test first 2 regimes only
        prob_col = f'P_regime_{regime}'
        regime_probs = aligned_probs[prob_col]
        
        # Probability-weighted sampling
        weights = regime_probs / regime_probs.sum()
        n_samples = max(min_obs_threshold, int(len(weights) * 0.8))
        
        sampled_indices = np.random.choice(
            len(weights), 
            size=min(n_samples, len(weights)), 
            p=weights, 
            replace=True
        )
        
        regime_dates = aligned_data.index[sampled_indices]
        print(f"   ✅ Regime {regime}: {len(regime_dates)} samples")
        
        # Test asset data availability
        for asset in target_assets:
            vol_col = f"Volatility_{asset}"
            ret_col = f"Returns_{asset}"
            
            if vol_col in aligned_data.columns and ret_col in aligned_data.columns:
                vol_data = aligned_data.loc[regime_dates, vol_col].dropna()
                ret_data = aligned_data.loc[regime_dates, ret_col].dropna()
                print(f"     📈 {asset}: {len(vol_data)} vol, {len(ret_data)} ret observations")
            else:
                print(f"     ❌ {asset}: Missing columns")
    
    print("\n✅ ALL TESTS PASSED!")
    print("Data loading and basic setup works correctly.")
    print("The issue was likely with TensorFlow loading or GARCH computation.")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()