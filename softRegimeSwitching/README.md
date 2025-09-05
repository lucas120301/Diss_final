# Soft Regime-Switching GARCH-LSTM

This directory implements the **proper** regime-switching methodology based on academic literature:

## 🎯 **Methodology**

Instead of hard regime assignments, this approach uses:

1. **Soft regime probabilities**: `P(regime_k|t)` for each date
2. **Weighted forecasts**: Combine regime-specific GARCH-LSTM predictions using probabilities
3. **Smooth transitions**: No sudden jumps between regimes
4. **Uncertainty incorporation**: Accounts for regime classification uncertainty

## 📊 **The Formula**

For each date t:
```
combined_forecast(t) = Σ P(regime_k|t) × GARCH_LSTM_k(t)
```

Where:
- `P(regime_k|t)`: Probability of being in regime k at time t
- `GARCH_LSTM_k(t)`: Volatility forecast from regime k's model at time t

## 🔧 **Files**

- `soft_regime_probabilities.py`: Generate soft probabilities from K-means
- `weighted_garch_lstm.py`: Create weighted GARCH-LSTM forecasts
- `soft_var_comparison.py`: Compare with Baseline1 using proper methodology
- `run_soft_regime_switching.py`: Main execution script

## 📈 **Expected Results**

This methodology should:
- ✅ Show realistic VaR violation rates (3-7%)
- ✅ Outperform Baseline1 through regime adaptation
- ✅ Provide smooth forecast transitions
- ✅ Follow academic literature best practices

## 🎓 **Literature Basis**

Based on:
- "Forecasting risk with Markov-switching GARCH models: A large-scale performance study"
- "Bayesian Model Averaging under Regime Switching"  
- "Asset allocation under multivariate regime switching"

This implements the **Dynamic Model Averaging (DMA)** approach for regime-switching VaR.