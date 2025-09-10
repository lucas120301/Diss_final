"""
KMeans clustering on Vol+Risk feature set

This script:
- Reconstructs a volatility + risk feature set from combined data
- Scales features
- Tests clustering with multiple seeds and tolerance values
- Reports the best Silhouette score found
- Saves cluster labels and metadata
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
import json

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv(
    "../../cleaned_data/combined_cleaned.csv",
    parse_dates=["Date"],
    index_col="Date"
)

print("Running clustering on Vol+Risk feature set...\n")

# Recreate the Vol+Risk features
features_dict = {}

# Volatility features
vol_assets = ["SPX Index", "RTY Index", "USGG10yr", "MOVE", "SX5E", "HSI", "GC1", "EURUSD"]
for asset in vol_assets:
    lr_col = f"{asset}_LR"
    rv_col = f"{asset}_RV_22"
    if rv_col in data.columns:
        features_dict[f"{asset}_Vol"] = data[rv_col]
    elif lr_col in data.columns:
        vol = data[lr_col].rolling(22).std()
        features_dict[f"{asset}_Vol"] = vol

# Market stress
stress_assets = ["SPX Index_LR", "RTY Index_LR", "USGG10yr_LR", "EURUSD_LR"]
stress_components = []
for col in stress_assets:
    if col in data.columns:
        abs_returns = data[col].abs().rolling(15).mean()
        stress_components.append(abs_returns)

if stress_components:
    stress_df = pd.concat(stress_components, axis=1)
    features_dict["Market_Stress"] = stress_df.mean(axis=1)

# Risk metrics
risk_metrics = {
    "vix3M_Close": "VIX3M_Level",
    "MOVE_Close": "MOVE_Level",
    "SKEW_Close": "SKEW_Level",
    "vvix_Close": "VVIX_Level"
}
for col, feature_name in risk_metrics.items():
    if col in data.columns:
        features_dict[feature_name] = data[col]

# Final feature matrix
features = pd.DataFrame(features_dict, index=data.index)
features = features.ffill().bfill().dropna()

print(f"Features: {features.shape[1]}, Observations: {features.shape[0]}")
print(f"Feature list: {list(features.columns)}")

# Scale
X_scaled = StandardScaler().fit_transform(features)

# Clustering experiments
best_result = None
best_score = -1

print("\nTesting optimization strategies...")

# Strategy 1: Different seeds, high n_init
print("1. Testing multiple seeds with high n_init...")
for seed in range(1, 101, 5):
    kmeans = KMeans(n_clusters=5, init="k-means++", n_init=100,
                    random_state=seed, max_iter=1000)
    labels = kmeans.fit_predict(X_scaled)
    cluster_sizes = np.bincount(labels)
    min_cluster_pct = min(cluster_sizes) / len(labels) * 100

    if min_cluster_pct >= 0.5:
        sil = silhouette_score(X_scaled, labels)
        if sil > best_score:
            best_result = {
                "score": sil,
                "labels": labels,
                "distribution": cluster_sizes.tolist(),
                "seed": seed,
                "strategy": "high_n_init"
            }
            best_score = sil
            print(f"   Seed {seed}: {sil:.3f}, dist={cluster_sizes.tolist()}")

# Strategy 2: Different tolerance values
print("\n2. Testing different tolerance settings...")
for tol in [1e-8, 1e-4, 1e-3]:
    kmeans = KMeans(n_clusters=5, init="k-means++", n_init=50,
                    random_state=42, tol=tol, max_iter=1000)
    labels = kmeans.fit_predict(X_scaled)
    cluster_sizes = np.bincount(labels)
    min_cluster_pct = min(cluster_sizes) / len(labels) * 100

    if min_cluster_pct >= 0.5:
        sil = silhouette_score(X_scaled, labels)
        if sil > best_score:
            best_result = {
                "score": sil,
                "labels": labels,
                "distribution": cluster_sizes.tolist(),
                "seed": 42,
                "strategy": f"tolerance_{tol}"
            }
            best_score = sil
            print(f"   Tolerance {tol}: {sil:.3f}")

# Final results
print("\n=== FINAL RESULT ===")
if best_result:
    print(f"Strategy: {best_result['strategy']}")
    print(f"Silhouette: {best_result['score']:.3f}")
    print(f"Distribution: {best_result['distribution']}")
    print(f"Seed: {best_result['seed']}")

    if not os.path.exists("final_results"):
        os.makedirs("final_results")

    labels_df = pd.DataFrame({
        "Date": features.index,
        "Regime": best_result["labels"]
    })
    labels_df.to_csv("final_results/vol_risk_labels.csv", index=False)

    results_dict = {
        "methodology": "vol_risk_feature_clustering",
        "silhouette_score": float(best_result["score"]),
        "cluster_distribution": best_result["distribution"],
        "n_features": features.shape[1],
        "feature_names": list(features.columns),
        "optimization_strategy": best_result["strategy"],
        "random_seed": best_result["seed"]
    }
    with open("final_results/vol_risk_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print("\nFiles saved:")
    print("  - final_results/vol_risk_labels.csv")
    print("  - final_results/vol_risk_results.json")
else:
    print("No valid clustering result found.")
