"""
train_lstm_baseline_paper.py
Baseline LSTM exactly as in the paper.

- LSTM(20), Dropout(0.2), Dense(1)
- Adam + MSE, epochs=100, EarlyStopping(patience=5, monitor='val_loss')
- Windows: 5, 11, 22
- 70/30 time split
- 25 runs averaged
- Input: RV_22 (scaled via MinMax fit on train only)

Usage:
  python train_lstm_baseline_paper.py --rv_csv extracted_0_1_RV22.csv --date_col Date \
      --rv_col RV_22 --start 2006-03-30 --end 2020-03-20 --out_prefix SPX_LSTM
"""

import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

from replication_adapter import make_windows

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(1 - (y_pred / y_true)))
    return rmse, mae, mape

def build_model(window):
    m = Sequential()
    m.add(LSTM(20, input_shape=(window,1), activation='relu', return_sequences=False))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='linear'))
    m.compile(optimizer='adam', loss='mean_squared_error')
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rv_csv", required=True)
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--rv_col", default="RV_22")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out_prefix", default="LSTM")
    args = ap.parse_args()

    df = pd.read_csv(args.rv_csv, parse_dates=[args.date_col]).sort_values(args.date_col)
    if args.start: df = df[df[args.date_col] >= pd.to_datetime(args.start)]
    if args.end:   df = df[df[args.date_col] <= pd.to_datetime(args.end)]
    df = df.dropna(subset=[args.rv_col])

    series = df[args.rv_col].values.astype(float)
    n = len(series)
    split_idx = int(0.7 * n)
    train, test = series[:split_idx], series[split_idx:]

    # Scale RV to [0,1] fit on train only (paper's standardization intent)
    sc = MinMaxScaler()
    train_sc = sc.fit_transform(train.reshape(-1,1)).ravel()
    test_sc  = sc.transform(test.reshape(-1,1)).ravel()

    windows = [5, 11, 22]
    os.makedirs("model_output", exist_ok=True)
    all_metrics = []

    for win in windows:
        Xtr, ytr = make_windows(train_sc, win)
        Xte, yte = make_windows(test_sc,  win)

        preds_runs = []
        for _ in range(25):
            m = build_model(win)
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            m.fit(Xtr, ytr, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[es])
            preds_runs.append(m.predict(Xte, verbose=0).ravel())

        preds_mean = np.mean(np.vstack(preds_runs), axis=0)
        # Inverse-scale back to original RV units for metrics (optional but fair)
        yte_inv    = sc.inverse_transform(yte).ravel()
        preds_inv  = sc.inverse_transform(preds_mean.reshape(-1,1)).ravel()

        rmse, mae, mape = metrics(yte_inv, preds_inv)
        all_metrics.append({"window": win, "RMSE": rmse, "MAE": mae, "MAPE": mape})

        out = pd.DataFrame({
            f"{args.out_prefix}_True_test": yte_inv,
            f"{args.out_prefix}_{win}_pred": preds_inv
        })
        out.to_csv(f"model_output/{args.out_prefix}_LSTM_win{win}.csv", index=False)

    pd.DataFrame(all_metrics).to_csv(f"model_output/{args.out_prefix}_LSTM_metrics.csv", index=False)
    print(pd.DataFrame(all_metrics))

if __name__ == "__main__":
    main()
