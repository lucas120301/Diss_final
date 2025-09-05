"""
train_hybrids_paper.py
Single / Dual / Triple GARCH + LSTM hybrids per the paper.

- Historical branch: RV_22 (scaled via MinMax fit on train only)
- GARCH branches: supply your precomputed 252-day rolling forecasts (scaled per-branch on train)
- Architecture (for ALL variants):
  * each branch: LSTM(20) + Dropout(0.2)
  * combiner: Dense(128, relu) -> Dense(64, relu) -> Dense(1, linear)
- Adam + MSE, epochs=100, EarlyStopping(patience=5, monitor='val_loss')
- 25 runs averaged
- Windows: 5, 11, 22
- 70/30 time split

Usage examples:
  Single:
    python train_hybrids_paper.py --rv_csv extracted_0_1_RV22.csv --garch_csv sGarch_SPX_252roll.csv \
        --mode single --out_prefix SPX_S

  Dual (sGARCH + tGARCH):
    python train_hybrids_paper.py --rv_csv extracted_0_1_RV22.csv \
        --garch_csv sGarch_SPX_252roll.csv --garch_csv2 tGarch_SPX_252roll.csv \
        --mode dual --out_prefix SPX_ST

  Triple (s, e, t):
    python train_hybrids_paper.py --rv_csv extracted_0_1_RV22.csv \
        --garch_csv sGarch_SPX_252roll.csv --garch_csv2 eGarch_SPX_252roll.csv \
        --garch_csv3 tGarch_SPX_252roll.csv --mode triple --out_prefix SPX_SET
"""

import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

from replication_adapter import make_windows

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(1 - (y_pred / y_true)))
    return rmse, mae, mape

def build_hybrid(win, n_garch_branches=1):
    # Historical branch
    hist_in = Input(shape=(win,1), name="hist_in")
    hist_l  = LSTM(20, activation='relu', return_sequences=False, name="hist_lstm")(hist_in)
    hist_d  = Dropout(0.2, name="hist_drop")(hist_l)

    inputs = [hist_in]
    merges = [hist_d]

    # GARCH branches
    for k in range(n_garch_branches):
        g_in = Input(shape=(1,1), name=f"g{k+1}_in")
        g_l  = LSTM(20, activation='relu', return_sequences=False, name=f"g{k+1}_lstm")(g_in)
        g_d  = Dropout(0.2, name=f"g{k+1}_drop")(g_l)
        inputs.append(g_in)
        merges.append(g_d)

    x = concatenate(merges, name="concat")
    x = Dense(128, activation='relu', name="fc1")(x)
    x = Dense(64, activation='relu', name="fc2")(x)
    out = Dense(1, activation='linear', name="out")(x)

    m = Model(inputs=inputs, outputs=out)
    m.compile(optimizer='adam', loss='mean_squared_error')
    return m

def load_series(csv_path, date_col="Date", col="RV_22", start=None, end=None):
    df = pd.read_csv(csv_path, parse_dates=[date_col]).sort_values(date_col)
    if start: df = df[df[date_col] >= pd.to_datetime(start)]
    if end:   df = df[df[date_col] <= pd.to_datetime(end)]
    df = df.dropna(subset=[col])
    return df[col].values.astype(float)

def load_garch_forecast(csv_path):
    """
    Expect a single-column CSV or a CSV with a column named 'x' or 'forecast' etc.
    We’ll auto-pick the 1st numeric column.
    """
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(f"No numeric column found in {csv_path}")
    s = df[num_cols[0]].values.astype(float)
    s = s[np.isfinite(s)]
    return s

def align_lengths(*arrays):
    L = min(len(a) for a in arrays)
    return [a[-L:] for a in arrays]  # align to most recent overlapping period

def run_mode(rv_series, g1=None, g2=None, g3=None, mode="single", out_prefix="OUT"):
    n = len(rv_series)
    split_idx = int(0.7 * n)
    train_rv, test_rv = rv_series[:split_idx], rv_series[split_idx:]

    # Scale per branch on train, transform test
    sc_rv = MinMaxScaler()
    tr_rv_sc = sc_rv.fit_transform(train_rv.reshape(-1,1)).ravel()
    te_rv_sc = sc_rv.transform(test_rv.reshape(-1,1)).ravel()

    # Prepare GARCH(s)
    garch_tr_sc, garch_te_sc, scs = [], [], []

    for g in [g for g in [g1,g2,g3] if g is not None]:
        # Align lengths with rv_series first
        g = g[-len(rv_series):]
        tr_g = g[:split_idx]
        te_g = g[split_idx:]
        sc = MinMaxScaler()
        tr_g_sc = sc.fit_transform(tr_g.reshape(-1,1)).ravel()
        te_g_sc = sc.transform(te_g.reshape(-1,1)).ravel()
        garch_tr_sc.append(tr_g_sc)
        garch_te_sc.append(te_g_sc)
        scs.append(sc)

    windows = [5, 11, 22]
    os.makedirs("model_output", exist_ok=True)
    all_metrics = []

    for win in windows:
        Xtr, ytr = make_windows(tr_rv_sc, win)
        Xte, yte = make_windows(te_rv_sc,  win)

        # Build GARCH input windows (1-step LSTM expects shape (n,1,1))
        def gwin(arr):
            # align sizes: same count as Xtr/Xte samples
            # For a 1-step branch, we take the current g_t to predict next step (shift by window).
            # Use the y-index alignment from the historical branch:
            return arr[win:].reshape(-1,1,1)

        gtr_list = [gwin(g) for g in garch_tr_sc]
        gte_list = [gwin(g) for g in garch_te_sc]

        preds_runs = []
        for _ in range(25):
            if mode == "single":
                m = build_hybrid(win, n_garch_branches=1)
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
                m.fit([Xtr, gtr_list[0]], ytr, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[es])
                preds = m.predict([Xte, gte_list[0]], verbose=0).ravel()

            elif mode == "dual":
                m = build_hybrid(win, n_garch_branches=2)
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
                m.fit([Xtr, gtr_list[0], gtr_list[1]], ytr, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[es])
                preds = m.predict([Xte, gte_list[0], gte_list[1]], verbose=0).ravel()

            elif mode == "triple":
                m = build_hybrid(win, n_garch_branches=3)
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
                m.fit([Xtr, gtr_list[0], gtr_list[1], gtr_list[2]], ytr, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[es])
                preds = m.predict([Xte, gte_list[0], gte_list[1], gte_list[2]], verbose=0).ravel()

            else:
                raise ValueError("mode must be 'single', 'dual', or 'triple'")

            preds_runs.append(preds)

                # ---- SAFE AVERAGING + METRICS (replace your whole block below this line) ----
        # Keep only prediction runs that are fully finite
        preds_clean = [p for p in preds_runs if np.all(np.isfinite(p))]
        if len(preds_clean) == 0:
            raise RuntimeError("All prediction runs produced NaN/Inf.")

        preds_mean = np.mean(np.vstack(preds_clean), axis=0)

        # Optional: clamp to [0,1] since the target branch was MinMax-scaled on train
        # (helps prevent tiny extrapolations from blowing up inverse transform)
        preds_mean = np.clip(preds_mean, 0.0, 1.0)

        # Inverse-scale to original RV units
        yte_inv   = sc_rv.inverse_transform(yte.reshape(-1, 1)).ravel()
        preds_inv = sc_rv.inverse_transform(preds_mean.reshape(-1, 1)).ravel()

        # Drop any non-finite pairs before scoring
        mask = np.isfinite(yte_inv) & np.isfinite(preds_inv)
        y_eval = yte_inv[mask]
        p_eval = preds_inv[mask]

        rmse, mae, mape = metrics(y_eval, p_eval)
        all_metrics.append({"mode": mode, "window": win, "RMSE": rmse, "MAE": mae, "MAPE": mape})

        # Save aligned test series actually used for metrics
        out = pd.DataFrame({
            f"{out_prefix}_True_test": y_eval,
            f"{out_prefix}_{win}_pred": p_eval
        })
        out.to_csv(f"model_output/{out_prefix}_{mode}_win{win}.csv", index=False)
        # ---- END SAFE BLOCK ----


    pd.DataFrame(all_metrics).to_csv(f"model_output/{out_prefix}_{mode}_metrics.csv", index=False)
    print(pd.DataFrame(all_metrics))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rv_csv", required=True)
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--rv_col", default="RV_22")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--garch_csv", default=None)
    ap.add_argument("--garch_csv2", default=None)
    ap.add_argument("--garch_csv3", default=None)
    ap.add_argument("--mode", choices=["single","dual","triple"], required=True)
    ap.add_argument("--out_prefix", default="HYB")
    args = ap.parse_args()

    rv = load_series(args.rv_csv, date_col=args.date_col, col=args.rv_col, start=args.start, end=args.end)

    g1 = load_garch_forecast(args.garch_csv)  if args.garch_csv  else None
    g2 = load_garch_forecast(args.garch_csv2) if args.garch_csv2 else None
    g3 = load_garch_forecast(args.garch_csv3) if args.garch_csv3 else None

    # Align lengths among RV and GARCH(s) if any (trim to shortest from the end)
    series_list = [x for x in [rv, g1, g2, g3] if x is not None]
    L = min(len(x) for x in series_list)
    rv  = rv[-L:]
    if g1 is not None: g1 = g1[-L:]
    if g2 is not None: g2 = g2[-L:]
    if g3 is not None: g3 = g3[-L:]

    run_mode(rv, g1, g2, g3, mode=args.mode, out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()
