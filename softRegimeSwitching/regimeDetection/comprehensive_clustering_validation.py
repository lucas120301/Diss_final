"""
Comprehensive Clustering Validation for Soft Regime-Switching
============================================================

This script tests multiple clustering methods and soft probability conversion techniques
to identify the optimal approach for regime-switching VaR modeling.

Methods tested:
1. K-means with various soft probability conversions
2. Gaussian Mixture Models with different covariance types
3. Multiple evaluation metrics to select the best approach

Based on academic literature:
- "Dynamic Model Averaging under Regime Switching"
- "Bayesian Model Averaging for regime-switching models"
- "Markov-switching GARCH models"
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class ClusteringValidator:
    """Comprehensive validation of clustering methods for soft regime probabilities"""
    
    def __init__(self, data_path, n_regimes=5, random_state=42, start_date='2012-01-01'):
        """
        Args:
            data_path: Path to cleaned data CSV
            n_regimes: Number of regimes to detect
            random_state: Random seed for reproducibility
            start_date: Start date for analysis (default 2012-01-01)
        """
        self.data_path = data_path
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.start_date = start_date
        self.scaler = StandardScaler()
        
        # Results storage
        self.results = {}
        self.feature_data = None
        self.features_scaled = None
        
        print(f"🎯 COMPREHENSIVE CLUSTERING VALIDATION")
        print("=" * 60)
        print(f"Testing {n_regimes} regimes with multiple methods")
        print(f"Using data from {start_date} onwards")
        print()
        
    def load_and_prepare_data(self):
        """Load data and create regime features"""
        print("📊 Loading and preparing data...")
        
        # Load the cleaned data
        data = pd.read_csv(self.data_path, parse_dates=['Date'])
        print(f"  ✅ Loaded {len(data)} observations")
        print(f"  📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"  🏦 Assets: {data['Name'].nunique()} unique")
        
        # Filter data from start_date onwards
        data = data[data['Date'] >= pd.to_datetime(self.start_date)]
        print(f"  📅 Filtered to {self.start_date}: {len(data)} observations")
        
        # Pivot to wide format
        pivot_data = data.pivot(index='Date', columns='Name', values='Price')
        print(f"  📈 Pivot shape: {pivot_data.shape}")
        
        # Calculate returns for each asset - don't dropna yet
        returns = pivot_data.pct_change()
        print(f"  💹 Returns shape: {returns.shape}")
        
        # Create regime features following academic literature
        features = self._create_regime_features(returns, pivot_data)
        
        # Store for analysis
        self.feature_data = features
        self.features_scaled = self.scaler.fit_transform(features)
        
        print(f"  ✅ Created {features.shape[1]} regime features")
        print(f"  📊 Features: {list(features.columns)}")
        
        return self
        
    def _create_regime_features(self, returns, prices):
        """Create comprehensive regime features"""
        print("    🔧 Creating regime detection features...")
        
        features_dict = {}
        
        # 1. Volatility features (most important for regime detection)
        # Use only assets with good coverage (4700+ observations)
        vol_assets = ['SPX Index', 'RTY Index', 'NDX Index', 'VIX', 'MOVE', 'SX5E', 'HSI']
        
        for asset in vol_assets:
            if asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) >= 1000:  # Ensure sufficient data
                    # Realized volatility (22-day)
                    rv = asset_returns.rolling(22).std() * np.sqrt(252)
                    features_dict[f'{asset}_RV22'] = rv
                    
                    # GARCH-like volatility (EWMA)
                    ewma_vol = asset_returns.ewm(span=22).std() * np.sqrt(252)
                    features_dict[f'{asset}_EWMA'] = ewma_vol
        
        # 2. Market stress indicators  
        stress_assets = ['SPX Index', 'RTY Index', 'EURUSD', 'GC1']
        stress_components = []
        
        for asset in stress_assets:
            if asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) >= 1000:
                    # Absolute returns (stress measure)
                    abs_returns = asset_returns.abs().rolling(15).mean()
                    stress_components.append(abs_returns)
        
        if stress_components:
            stress_df = pd.concat(stress_components, axis=1)
            features_dict['Market_Stress'] = stress_df.mean(axis=1, skipna=True)
        
        # 3. Risk premium features
        # Only use assets with good coverage
        risk_assets = ['VIX', 'MOVE', 'SKEW', 'vvix', 'GVZ', 'OVX']
        
        for asset in risk_assets:
            if asset in prices.columns:
                asset_prices = prices[asset].dropna()
                if len(asset_prices) >= 1000:
                    # Risk premium levels
                    features_dict[f'{asset}_Level'] = asset_prices
                    
                    # Risk premium changes
                    if asset in returns.columns:
                        asset_returns = returns[asset].dropna()
                        if len(asset_returns) >= 1000:
                            features_dict[f'{asset}_Change'] = asset_returns.rolling(5).mean()
        
        # 4. Correlation features (regime-dependent)
        # Check if key assets have sufficient data
        spx_returns = returns['SPX Index'].dropna() if 'SPX Index' in returns.columns else pd.Series()
        rty_returns = returns['RTY Index'].dropna() if 'RTY Index' in returns.columns else pd.Series()
        
        if len(spx_returns) >= 1000 and len(rty_returns) >= 1000:
            # Align the series
            common_dates = spx_returns.index.intersection(rty_returns.index)
            spx_aligned = spx_returns.loc[common_dates]
            rty_aligned = rty_returns.loc[common_dates]
            
            if len(common_dates) >= 1000:
                # Equity correlation
                equity_corr = spx_aligned.rolling(60).corr(rty_aligned)
                features_dict['Equity_Corr'] = equity_corr
                
                # Stock-bond correlation if bond data available
                if 'USGG10yr' in returns.columns:
                    bond_returns = returns['USGG10yr'].dropna()
                    if len(bond_returns) >= 1000:
                        bond_common = common_dates.intersection(bond_returns.index)
                        if len(bond_common) >= 1000:
                            correlation = spx_aligned.loc[bond_common].rolling(60).corr(bond_returns.loc[bond_common])
                            features_dict['Stock_Bond_Corr'] = correlation
        
        # 5. Momentum features
        momentum_assets = ['SPX Index', 'RTY Index', 'NDX Index']
        
        for asset in momentum_assets:
            if asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) >= 1000:
                    # 22-day momentum
                    momentum = asset_returns.rolling(22).sum()
                    features_dict[f'{asset}_Mom22'] = momentum
        
        # 6. Dispersion features
        equity_assets_available = []
        for asset in ['SPX Index', 'RTY Index', 'NDX Index']:
            if asset in returns.columns:
                asset_returns = returns[asset].dropna()
                if len(asset_returns) >= 1000:
                    equity_assets_available.append(asset)
        
        if len(equity_assets_available) >= 2:
            # Get aligned equity returns
            equity_series = []
            for asset in equity_assets_available:
                equity_series.append(returns[asset].dropna())
            
            # Find common dates
            common_equity_dates = equity_series[0].index
            for series in equity_series[1:]:
                common_equity_dates = common_equity_dates.intersection(series.index)
            
            if len(common_equity_dates) >= 1000:
                equity_aligned = pd.DataFrame({
                    asset: returns[asset].loc[common_equity_dates] 
                    for asset in equity_assets_available
                })
                
                # Cross-sectional volatility (dispersion)
                dispersion = equity_aligned.std(axis=1, skipna=True)
                features_dict['Equity_Dispersion'] = dispersion
        
        # Create DataFrame from features
        if not features_dict:
            print(f"      ❌ No features created - insufficient data coverage")
            return pd.DataFrame()
        
        # Find common index across all features
        common_index = None
        for feature_name, feature_series in features_dict.items():
            feature_index = feature_series.dropna().index
            if common_index is None:
                common_index = feature_index
            else:
                common_index = common_index.intersection(feature_index)
        
        if len(common_index) < 100:
            print(f"      ❌ Insufficient common dates: {len(common_index)}")
            return pd.DataFrame()
        
        # Align all features to common index
        aligned_features = {}
        for feature_name, feature_series in features_dict.items():
            aligned_features[feature_name] = feature_series.loc[common_index]
        
        features = pd.DataFrame(aligned_features, index=common_index)
        
        # Final cleanup - remove any remaining NaN
        features = features.dropna()
        
        print(f"      ✅ Created {len(features)} feature observations")
        print(f"      📅 Feature period: {features.index.min()} to {features.index.max()}")
        
        return features
    
    def test_kmeans_methods(self):
        """Test K-means with different soft probability conversion methods"""
        print("\n🎲 Testing K-means with soft probability methods...")
        
        kmeans_methods = {
            'inverse_distance': {
                'description': 'Inverse distance weighting: 1/(1+d)',
                'temperature': 1.0
            },
            'exponential_decay': {
                'description': 'Exponential decay: exp(-T*d)', 
                'temperature': 1.0
            },
            'gaussian_kernel': {
                'description': 'Gaussian kernel: exp(-0.5*(d/T)^2)',
                'temperature': 1.0
            },
            'softmax_distance': {
                'description': 'Softmax on negative distances: softmax(-T*d)',
                'temperature': 2.0
            },
            'power_distance': {
                'description': 'Power weighting: d^(-T)',
                'temperature': 2.0
            }
        }
        
        for method_name, config in kmeans_methods.items():
            print(f"  🔄 Testing {method_name}...")
            
            # Fit K-means
            kmeans = KMeans(
                n_clusters=self.n_regimes,
                init='k-means++',
                n_init=50,
                random_state=self.random_state,
                max_iter=1000
            )
            
            hard_labels = kmeans.fit_predict(self.features_scaled)
            distances = kmeans.transform(self.features_scaled)
            
            # Convert to soft probabilities
            soft_probs = self._convert_distances_to_probabilities(
                distances, method_name, config['temperature']
            )
            
            # Evaluate
            evaluation = self._evaluate_clustering(
                hard_labels, soft_probs, distances, method_name, config
            )
            
            self.results[f'kmeans_{method_name}'] = {
                'method_type': 'kmeans',
                'soft_method': method_name,
                'hard_labels': hard_labels,
                'soft_probabilities': soft_probs,
                'distances': distances,
                'cluster_centers': kmeans.cluster_centers_,
                'evaluation': evaluation,
                'config': config
            }
            
            print(f"      ✅ Completed {method_name}")
        
        return self
    
    def test_gmm_methods(self):
        """Test Gaussian Mixture Models with different covariance types"""
        print("\n🎭 Testing Gaussian Mixture Models...")
        
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        for cov_type in covariance_types:
            print(f"  🔄 Testing GMM with {cov_type} covariance...")
            
            try:
                # Fit GMM
                gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type=cov_type,
                    init_params='kmeans',
                    random_state=self.random_state,
                    max_iter=200
                )
                
                gmm.fit(self.features_scaled)
                
                # Get soft probabilities (GMM naturally produces these)
                soft_probs = gmm.predict_proba(self.features_scaled)
                hard_labels = gmm.predict(self.features_scaled)
                
                # Calculate distances to centroids for evaluation
                distances = np.zeros((len(self.features_scaled), self.n_regimes))
                for i in range(self.n_regimes):
                    distances[:, i] = np.linalg.norm(
                        self.features_scaled - gmm.means_[i], axis=1
                    )
                
                # Evaluate
                evaluation = self._evaluate_clustering(
                    hard_labels, soft_probs, distances, f'gmm_{cov_type}', 
                    {'description': f'GMM with {cov_type} covariance', 'cov_type': cov_type}
                )
                
                self.results[f'gmm_{cov_type}'] = {
                    'method_type': 'gmm',
                    'covariance_type': cov_type,
                    'hard_labels': hard_labels,
                    'soft_probabilities': soft_probs,
                    'distances': distances,
                    'means': gmm.means_,
                    'covariances': gmm.covariances_,
                    'weights': gmm.weights_,
                    'evaluation': evaluation,
                    'aic': gmm.aic(self.features_scaled),
                    'bic': gmm.bic(self.features_scaled),
                    'log_likelihood': gmm.score(self.features_scaled)
                }
                
                print(f"      ✅ Completed GMM {cov_type}")
                
            except Exception as e:
                print(f"      ❌ Failed GMM {cov_type}: {e}")
        
        return self
    
    def _convert_distances_to_probabilities(self, distances, method, temperature):
        """Convert distances to soft probabilities using specified method"""
        
        if method == 'inverse_distance':
            weights = 1.0 / (distances + 1e-8)
            
        elif method == 'exponential_decay':
            weights = np.exp(-temperature * distances)
            
        elif method == 'gaussian_kernel':
            weights = np.exp(-0.5 * (distances / temperature) ** 2)
            
        elif method == 'softmax_distance':
            weights = np.exp(-temperature * distances)
            
        elif method == 'power_distance':
            weights = np.power(distances + 1e-8, -temperature)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize to get probabilities (rows sum to 1)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def _evaluate_clustering(self, hard_labels, soft_probs, distances, method_name, config):
        """Comprehensive evaluation of clustering quality"""
        
        # 1. Traditional clustering metrics
        silhouette = silhouette_score(self.features_scaled, hard_labels)
        calinski_harabasz = calinski_harabasz_score(self.features_scaled, hard_labels)
        davies_bouldin = davies_bouldin_score(self.features_scaled, hard_labels)
        
        # 2. Soft probability quality metrics
        # Entropy (uncertainty measure)
        entropy = -(soft_probs * np.log(soft_probs + 1e-8)).sum(axis=1)
        avg_entropy = entropy.mean()
        max_entropy = np.log(self.n_regimes)
        
        # Probability concentration
        max_probs = soft_probs.max(axis=1)
        avg_max_prob = max_probs.mean()
        high_confidence = (max_probs > 0.7).mean()
        medium_confidence = (max_probs > 0.5).mean()
        
        # 3. Stability metrics
        regime_counts = np.bincount(hard_labels, minlength=self.n_regimes)
        regime_balance = regime_counts.std() / regime_counts.mean()  # Lower is better
        
        # 4. Temporal consistency (if we have dates)
        if hasattr(self, 'feature_data') and not self.feature_data.empty:
            # Calculate regime switching frequency
            regime_switches = np.sum(np.diff(hard_labels) != 0)
            switch_rate = regime_switches / len(hard_labels)
        else:
            switch_rate = np.nan
        
        # 5. Distance-based metrics
        inertia = np.sum(distances.min(axis=1) ** 2)  # Within-cluster sum of squares
        
        evaluation = {
            # Traditional metrics
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz, 
            'davies_bouldin_score': davies_bouldin,  # Lower is better
            
            # Soft probability metrics
            'avg_entropy': avg_entropy,
            'entropy_ratio': avg_entropy / max_entropy,  # 0-1, lower = more confident
            'avg_max_probability': avg_max_prob,  # Higher is better
            'high_confidence_pct': high_confidence,  # Prop. with >70% confidence
            'medium_confidence_pct': medium_confidence,  # Prop. with >50% confidence
            
            # Stability metrics
            'regime_balance': regime_balance,  # Lower = more balanced
            'regime_switch_rate': switch_rate,  # Moderate is better
            'inertia': inertia,  # Lower is better
            
            # Regime distribution
            'regime_counts': regime_counts.tolist(),
            'dominant_regime': int(np.argmax(regime_counts)),
            'smallest_regime_size': int(regime_counts.min()),
            'largest_regime_size': int(regime_counts.max()),
        }
        
        return evaluation
    
    def compare_all_methods(self):
        """Compare all tested methods and rank them"""
        print(f"\n📊 COMPARING ALL METHODS")
        print("=" * 50)
        
        if not self.results:
            print("❌ No results to compare")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        
        for method_name, result in self.results.items():
            eval_metrics = result['evaluation']
            
            # Calculate composite scores
            # Quality score (higher is better)
            quality_score = (
                eval_metrics['silhouette_score'] * 0.3 +
                eval_metrics['calinski_harabasz_score'] / 1000 * 0.2 +  # Normalize
                (1 - eval_metrics['davies_bouldin_score']) * 0.2 +  # Invert (lower is better)
                eval_metrics['avg_max_probability'] * 0.3
            )
            
            # Stability score (higher is better)
            stability_score = (
                (1 - eval_metrics['regime_balance'] / 2) * 0.4 +  # Normalize and invert
                (1 - eval_metrics['entropy_ratio']) * 0.3 +  # Less uncertainty
                eval_metrics['high_confidence_pct'] * 0.3
            )
            
            # Overall score
            overall_score = quality_score * 0.6 + stability_score * 0.4
            
            comparison_data.append({
                'method': method_name,
                'method_type': result['method_type'],
                'quality_score': quality_score,
                'stability_score': stability_score,
                'overall_score': overall_score,
                'silhouette': eval_metrics['silhouette_score'],
                'calinski_harabasz': eval_metrics['calinski_harabasz_score'],
                'davies_bouldin': eval_metrics['davies_bouldin_score'],
                'avg_max_prob': eval_metrics['avg_max_probability'],
                'entropy_ratio': eval_metrics['entropy_ratio'],
                'high_confidence': eval_metrics['high_confidence_pct'],
                'regime_balance': eval_metrics['regime_balance'],
                'switch_rate': eval_metrics['regime_switch_rate']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)
        
        # Display results
        print("🏆 TOP PERFORMING METHODS:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(comparison_df.head(5).iterrows()):
            print(f"{i+1}. {row['method']}")
            print(f"   Overall Score: {row['overall_score']:.3f}")
            print(f"   Quality: {row['quality_score']:.3f} | Stability: {row['stability_score']:.3f}")
            print(f"   Silhouette: {row['silhouette']:.3f} | Max Prob: {row['avg_max_prob']:.3f}")
            print(f"   Entropy Ratio: {row['entropy_ratio']:.3f} | High Confidence: {row['high_confidence']:.1%}")
            print()
        
        # Store comparison
        self.method_comparison = comparison_df
        
        # Recommend best method
        best_method = comparison_df.iloc[0]
        print(f"🎯 RECOMMENDED METHOD: {best_method['method']}")
        print(f"   → Best overall performance with {best_method['overall_score']:.3f} score")
        
        # Get description safely
        best_result = self.results[best_method['method']]
        if 'config' in best_result and 'description' in best_result['config']:
            print(f"   → {best_result['config']['description']}")
        elif best_method['method_type'] == 'gmm':
            cov_type = best_result.get('covariance_type', 'unknown')
            print(f"   → Gaussian Mixture Model with {cov_type} covariance")
        
        return self
    
    def create_visualizations(self, output_dir='regimeDetection/results'):
        """Create comprehensive visualizations"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n📊 Creating visualizations...")
            
            # 1. Method comparison heatmap
            if hasattr(self, 'method_comparison'):
                plt.figure(figsize=(14, 10))
                
                # Select key metrics for heatmap
                heatmap_cols = ['quality_score', 'stability_score', 'silhouette', 
                               'avg_max_prob', 'entropy_ratio', 'high_confidence']
                
                heatmap_data = self.method_comparison.set_index('method')[heatmap_cols].T
                
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                           center=0.5, cbar_kws={'label': 'Score'})
                plt.title('Clustering Method Comparison', fontsize=16, fontweight='bold')
                plt.xlabel('Methods', fontsize=12)
                plt.ylabel('Evaluation Metrics', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Top methods detailed comparison
                top_methods = self.method_comparison.head(3)['method'].tolist()
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Silhouette scores
                axes[0,0].bar(range(len(top_methods)), 
                             [self.results[m]['evaluation']['silhouette_score'] for m in top_methods])
                axes[0,0].set_title('Silhouette Score (Higher Better)')
                axes[0,0].set_xticks(range(len(top_methods)))
                axes[0,0].set_xticklabels([m.replace('_', '\n') for m in top_methods], rotation=45)
                
                # Entropy ratios
                axes[0,1].bar(range(len(top_methods)), 
                             [self.results[m]['evaluation']['entropy_ratio'] for m in top_methods])
                axes[0,1].set_title('Entropy Ratio (Lower Better)')
                axes[0,1].set_xticks(range(len(top_methods)))
                axes[0,1].set_xticklabels([m.replace('_', '\n') for m in top_methods], rotation=45)
                
                # Max probabilities
                axes[1,0].bar(range(len(top_methods)), 
                             [self.results[m]['evaluation']['avg_max_probability'] for m in top_methods])
                axes[1,0].set_title('Avg Max Probability (Higher Better)')
                axes[1,0].set_xticks(range(len(top_methods)))
                axes[1,0].set_xticklabels([m.replace('_', '\n') for m in top_methods], rotation=45)
                
                # High confidence percentages
                axes[1,1].bar(range(len(top_methods)), 
                             [self.results[m]['evaluation']['high_confidence_pct'] for m in top_methods])
                axes[1,1].set_title('High Confidence % (Higher Better)')
                axes[1,1].set_xticks(range(len(top_methods)))
                axes[1,1].set_xticklabels([m.replace('_', '\n') for m in top_methods], rotation=45)
                
                plt.suptitle('Top 3 Methods Detailed Comparison', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/top_methods_detailed.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Probability distribution visualization for best method
                best_method_name = self.method_comparison.iloc[0]['method']
                best_result = self.results[best_method_name]
                soft_probs = best_result['soft_probabilities']
                
                plt.figure(figsize=(12, 8))
                
                # Histogram of max probabilities
                plt.subplot(2, 2, 1)
                plt.hist(soft_probs.max(axis=1), bins=30, alpha=0.7, edgecolor='black')
                plt.title('Distribution of Max Probabilities')
                plt.xlabel('Max Probability')
                plt.ylabel('Frequency')
                
                # Regime probability time series (sample)
                plt.subplot(2, 2, 2)
                sample_indices = np.linspace(0, len(soft_probs)-1, 200, dtype=int)
                for i in range(min(3, self.n_regimes)):  # Show first 3 regimes
                    plt.plot(sample_indices, soft_probs[sample_indices, i], 
                            label=f'Regime {i}', alpha=0.7)
                plt.title('Regime Probabilities Over Time (Sample)')
                plt.xlabel('Time Index (Sampled)')
                plt.ylabel('Probability')
                plt.legend()
                
                # Entropy distribution
                plt.subplot(2, 2, 3)
                entropy = -(soft_probs * np.log(soft_probs + 1e-8)).sum(axis=1)
                plt.hist(entropy, bins=30, alpha=0.7, edgecolor='black')
                plt.title('Distribution of Entropy (Uncertainty)')
                plt.xlabel('Entropy')
                plt.ylabel('Frequency')
                
                # Regime assignment counts
                plt.subplot(2, 2, 4)
                hard_labels = best_result['hard_labels']
                unique, counts = np.unique(hard_labels, return_counts=True)
                plt.bar(unique, counts)
                plt.title('Regime Assignment Counts')
                plt.xlabel('Regime')
                plt.ylabel('Count')
                
                plt.suptitle(f'Best Method Analysis: {best_method_name}', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/best_method_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"    ✅ Saved visualizations to {output_dir}/")
            
        except Exception as e:
            print(f"    ⚠️ Visualization creation failed: {e}")
    
    def save_results(self, output_dir='regimeDetection/results'):
        """Save all results and recommendations"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving results to {output_dir}/...")
        
        # 1. Save method comparison
        if hasattr(self, 'method_comparison'):
            self.method_comparison.to_csv(f'{output_dir}/method_comparison.csv', index=False)
            print("    ✅ Method comparison saved")
        
        # 2. Save best method's soft probabilities
        if hasattr(self, 'method_comparison') and not self.method_comparison.empty:
            best_method_name = self.method_comparison.iloc[0]['method']
            best_result = self.results[best_method_name]
            
            # Create probabilities DataFrame with dates
            prob_columns = [f'P_regime_{i}' for i in range(self.n_regimes)]
            
            if hasattr(self, 'feature_data') and not self.feature_data.empty:
                prob_df = pd.DataFrame(
                    best_result['soft_probabilities'],
                    index=self.feature_data.index,
                    columns=prob_columns
                )
            else:
                prob_df = pd.DataFrame(
                    best_result['soft_probabilities'],
                    columns=prob_columns
                )
            
            prob_df.to_csv(f'{output_dir}/best_soft_probabilities.csv')
            print("    ✅ Best soft probabilities saved")
            
            # Save hard labels too
            labels_df = pd.DataFrame({
                'hard_regime': best_result['hard_labels']
            }, index=prob_df.index)
            labels_df.to_csv(f'{output_dir}/best_hard_labels.csv')
            print("    ✅ Best hard labels saved")
        
        # 3. Save comprehensive metadata
        metadata = {
            'validation_date': datetime.now().isoformat(),
            'data_path': self.data_path,
            'n_regimes': self.n_regimes,
            'random_state': self.random_state,
            'feature_count': self.feature_data.shape[1] if self.feature_data is not None else 0,
            'observation_count': len(self.features_scaled) if self.features_scaled is not None else 0,
            'methods_tested': list(self.results.keys()),
            'best_method': self.method_comparison.iloc[0]['method'] if hasattr(self, 'method_comparison') else None,
            'feature_names': list(self.feature_data.columns) if self.feature_data is not None else []
        }
        
        # Add detailed results for each method
        metadata['method_details'] = {}
        for method_name, result in self.results.items():
            metadata['method_details'][method_name] = {
                'method_type': result['method_type'],
                'evaluation_metrics': result['evaluation'],
                'config': result.get('config', {}),
                'aic': result.get('aic'),
                'bic': result.get('bic'),
                'log_likelihood': result.get('log_likelihood')
            }
        
        with open(f'{output_dir}/validation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print("    ✅ Validation metadata saved")
        
        # 4. Create summary report
        summary_lines = [
            "CLUSTERING VALIDATION SUMMARY",
            "=" * 50,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data: {self.data_path}",
            f"Regimes: {self.n_regimes}",
            f"Features: {self.feature_data.shape[1] if self.feature_data is not None else 0}",
            f"Observations: {len(self.features_scaled) if self.features_scaled is not None else 0}",
            "",
            "METHODS TESTED:",
            "-" * 20
        ]
        
        for i, method in enumerate(self.results.keys(), 1):
            summary_lines.append(f"{i}. {method}")
        
        if hasattr(self, 'method_comparison'):
            summary_lines.extend([
                "",
                "TOP 3 METHODS:",
                "-" * 20
            ])
            
            for i, (_, row) in enumerate(self.method_comparison.head(3).iterrows(), 1):
                summary_lines.append(f"{i}. {row['method']} (Score: {row['overall_score']:.3f})")
            
            best = self.method_comparison.iloc[0]
            summary_lines.extend([
                "",
                "RECOMMENDED METHOD:",
                "-" * 20,
                f"Method: {best['method']}",
                f"Overall Score: {best['overall_score']:.3f}",
                f"Quality Score: {best['quality_score']:.3f}",
                f"Stability Score: {best['stability_score']:.3f}",
                f"Silhouette Score: {best['silhouette']:.3f}",
                f"Avg Max Probability: {best['avg_max_prob']:.3f}",
                f"High Confidence %: {best['high_confidence']:.1%}",
                "",
                "FILES GENERATED:",
                "-" * 20,
                "• method_comparison.csv - Full comparison table",
                "• best_soft_probabilities.csv - Recommended soft probabilities", 
                "• best_hard_labels.csv - Hard regime assignments",
                "• validation_metadata.json - Complete validation details",
                "• method_comparison_heatmap.png - Visual comparison",
                "• top_methods_detailed.png - Top 3 methods analysis",
                "• best_method_analysis.png - Best method detailed analysis"
            ])
        
        with open(f'{output_dir}/SUMMARY.txt', 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print("    ✅ Summary report saved")
        
        return self

def main():
    """Run comprehensive clustering validation"""
    print("🎯 SOFT REGIME-SWITCHING CLUSTERING VALIDATION")
    print("=" * 60)
    print("Testing multiple clustering methods for optimal regime detection")
    print()
    
    # Initialize validator with 2012 start date for full historical coverage
    data_path = "/Users/user/Desktop/Imperial/Diss_final/softRegimeSwitching/terminal data 1_cleaned.csv"
    validator = ClusteringValidator(data_path, n_regimes=5, start_date='2012-01-01')
    
    try:
        # Run validation pipeline
        validator.load_and_prepare_data()
        validator.test_kmeans_methods()
        validator.test_gmm_methods()
        validator.compare_all_methods()
        validator.create_visualizations()
        validator.save_results()
        
        print(f"\n🎉 VALIDATION COMPLETE!")
        print("=" * 40)
        print("Check regimeDetection/results/ for:")
        print("  • Method comparison and rankings")
        print("  • Best soft probabilities")
        print("  • Comprehensive visualizations")
        print("  • Detailed metadata and summary")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()