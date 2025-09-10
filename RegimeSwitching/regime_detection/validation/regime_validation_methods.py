

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score, 
                           adjusted_rand_score, normalized_mutual_info_score)
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RegimeValidationMethods:
    def __init__(self, regime_labels_file='final_results/vol_risk_labels.csv'):
        """Initialize with the regime detection results to validate."""
        self.regime_labels_file = regime_labels_file
        self.regime_labels = None
        self.regime_dates = None
        self.features = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        
        # Major economic events for validation
        self.major_events = {
            '2015-08-11': 'China Yuan Devaluation',
            '2016-06-23': 'Brexit Vote', 
            '2016-11-08': 'Trump Election',
            '2018-02-05': 'VIX Spike / Volpocalypse',
            '2018-10-01': 'Trade War Escalation',
            '2020-03-11': 'COVID-19 Pandemic Declaration',
            '2020-03-23': 'COVID Market Bottom',
            '2022-02-24': 'Russia-Ukraine Conflict',
            '2022-03-16': 'Fed Rate Hike Cycle Begins',
            '2023-03-10': 'Silicon Valley Bank Collapse'
        }
        
    def load_regime_results(self):
        """Load the regime detection results and prepare features."""
        print("Loading regime detection results...")
        
        # Load regime labels
        regime_df = pd.read_csv(self.regime_labels_file, parse_dates=['Date'])
        self.regime_labels = regime_df['Regime'].values
        self.regime_dates = pd.to_datetime(regime_df['Date'])
        
        print(f"Loaded {len(self.regime_labels)} regime labels")
        print(f"Date range: {self.regime_dates.min()} to {self.regime_dates.max()}")
        print(f"Regimes found: {sorted(np.unique(self.regime_labels))}")
        
        # Load original data for features (using same as best replication)
        data = pd.read_csv('/Users/user/Desktop/Imperial/Diss/RegimeSwitching/regime_switching_models/cleaned_data/combined_cleaned.csv', parse_dates=['Date'], index_col='Date')
        feature_cols = ['SPX Index_RV_22', 'RTY Index_RV_22', 'USGG10yr_RV_22', 'vix3M_Close', 'MOVE_Close']
        
        self.features = data[feature_cols].ffill().bfill().dropna()
        
        # Align with regime labels dates
        common_dates = self.features.index.intersection(self.regime_dates)
        self.features = self.features.loc[common_dates]
        
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(self.features)
        
        print(f"Features prepared: {self.features.shape[1]} features, {len(self.features)} observations")
        return self
        
    def cross_method_validation(self):
        print("\n=== CROSS-METHOD VALIDATION ===")
        
        n_clusters = len(np.unique(self.regime_labels))
        
        # Alternative clustering methods
        alternative_methods = {
            'Hierarchical_Ward': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
            'Hierarchical_Complete': AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
            'Hierarchical_Average': AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
            'Gaussian_Mixture': GaussianMixture(n_components=n_clusters, random_state=42),
            'Spectral_RBF': SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42),
            'Spectral_Nearest': SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        }
        
        # DBSCAN with multiple epsilon values
        for eps in [0.3, 0.5, 0.7, 1.0]:
            alternative_methods[f'DBSCAN_eps{eps}'] = DBSCAN(eps=eps, min_samples=10)
        
        validation_results = {}
        ari_scores = []
        nmi_scores = []
        
        print("Testing alternative clustering methods...")
        
        for method_name, method in alternative_methods.items():
            try:
                if hasattr(method, 'fit_predict'):
                    alt_labels = method.fit_predict(self.X_scaled)
                else:  # For GaussianMixture
                    alt_labels = method.fit(self.X_scaled).predict(self.X_scaled)
                
                # Skip if DBSCAN finds only noise or single cluster
                if len(np.unique(alt_labels)) < 2:
                    print(f"  {method_name}: Insufficient clusters found")
                    continue
                
                # Calculate agreement metrics
                ari = adjusted_rand_score(self.regime_labels, alt_labels)
                nmi = normalized_mutual_info_score(self.regime_labels, alt_labels)
                
                validation_results[method_name] = {
                    'ari': float(ari),
                    'nmi': float(nmi),
                    'n_clusters_found': len(np.unique(alt_labels)),
                    'cluster_sizes': np.bincount(alt_labels).tolist()
                }
                
                ari_scores.append(ari)
                nmi_scores.append(nmi)
                
                print(f"  {method_name}: ARI={ari:.3f}, NMI={nmi:.3f}, Clusters={len(np.unique(alt_labels))}")
                
            except Exception as e:
                print(f"  {method_name}: Failed ({str(e)[:50]}...)")
                continue
        
        # Summary statistics
        if ari_scores:
            avg_ari = np.mean(ari_scores)
            max_ari = np.max(ari_scores)
            
            validation_summary = {
                'average_ari': float(avg_ari),
                'max_ari': float(max_ari),
                'average_nmi': float(np.mean(nmi_scores)),
                'methods_tested': len(validation_results),
                'perfect_agreement_count': sum(1 for ari in ari_scores if ari > 0.95)
            }
            
            print(f"\n📊 Cross-Method Validation Summary:")
            print(f"  Average ARI: {avg_ari:.3f}")
            print(f"  Maximum ARI: {max_ari:.3f}")  
            print(f"  Methods with ARI > 0.8: {sum(1 for ari in ari_scores if ari > 0.8)}/{len(ari_scores)}")
            print(f"  Perfect agreement (ARI > 0.95): {validation_summary['perfect_agreement_count']}")
            
            return validation_results, validation_summary
        else:
            print("No alternative methods succeeded")
            return {}, {}
    
    def economic_event_alignment(self):
        """
        Validate regime transitions against major market events.
        Check if regime changes align with known economic events within ±1 month.
        """
        print("\n=== ECONOMIC EVENT ALIGNMENT VALIDATION ===")
        
        # Find regime transition points
        regime_changes = []
        for i in range(1, len(self.regime_labels)):
            if self.regime_labels[i] != self.regime_labels[i-1]:
                regime_changes.append({
                    'date': self.regime_dates.iloc[i],
                    'from_regime': self.regime_labels[i-1],
                    'to_regime': self.regime_labels[i]
                })
        
        print(f"Found {len(regime_changes)} regime transitions")
        
        # Check alignment with economic events
        alignments = []
        aligned_events = 0
        
        for event_date_str, event_name in self.major_events.items():
            event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
            
            # Find closest regime transition within ±1 month
            closest_transition = None
            min_distance = timedelta(days=9999)  # Large number instead of infinity
            
            for transition in regime_changes:
                distance = abs((transition['date'] - event_date).total_seconds())
                if distance < min_distance.total_seconds():
                    min_distance = timedelta(seconds=distance)
                    closest_transition = transition
            
            # Check if within ±1 month (30 days)
            if closest_transition and min_distance <= timedelta(days=30):
                aligned_events += 1
                alignment_quality = 'perfect' if min_distance <= timedelta(days=7) else 'good'
                
                alignments.append({
                    'event_date': event_date_str,
                    'event_name': event_name,
                    'transition_date': closest_transition['date'].strftime('%Y-%m-%d'),
                    'days_difference': min_distance.days,
                    'regime_change': f"{closest_transition['from_regime']} → {closest_transition['to_regime']}",
                    'alignment_quality': alignment_quality
                })
                
                print(f"  {event_name} ({event_date_str})")
                print(f"      → Regime change: {closest_transition['from_regime']} → {closest_transition['to_regime']}")
                print(f"      → Distance: {min_distance.days} days")
            else:
                print(f"  {event_name} ({event_date_str}) - No nearby transition")
        
        # Calculate alignment statistics
        total_events = len(self.major_events)
        alignment_rate = aligned_events / total_events
        
        alignment_summary = {
            'total_events_tested': total_events,
            'aligned_events': aligned_events,
            'alignment_rate': float(alignment_rate),
            'major_transitions': len(regime_changes),
            'perfect_alignments': len([a for a in alignments if a['alignment_quality'] == 'perfect']),
            'threshold_met': alignment_rate >= 0.7  # 70% threshold from methodology
        }
        
        print(f"\nEconomic Event Alignment Summary:")
        print(f"  Events aligned: {aligned_events}/{total_events} ({alignment_rate:.1%})")
        print(f"  Threshold (70%): {'MET' if alignment_summary['threshold_met'] else 'NOT MET'}")
        print(f"  Perfect alignments (≤7 days): {alignment_summary['perfect_alignments']}")
        
        return alignments, alignment_summary
    
    def statistical_quality_metrics(self):
        """
        Calculate comprehensive statistical quality metrics for the clustering.
        """
        print("\n=== STATISTICAL QUALITY METRICS ===")
        
        # Primary clustering quality metrics
        silhouette = silhouette_score(self.X_scaled, self.regime_labels)
        calinski_harabasz = calinski_harabasz_score(self.X_scaled, self.regime_labels)
        davies_bouldin = davies_bouldin_score(self.X_scaled, self.regime_labels)
        
        # Additional metrics
        n_regimes = len(np.unique(self.regime_labels))
        regime_sizes = np.bincount(self.regime_labels)
        min_regime_size = np.min(regime_sizes)
        max_regime_size = np.max(regime_sizes)
        regime_balance = min_regime_size / max_regime_size  # Balance metric
        
        # Within-cluster sum of squares (inertia equivalent)
        centroids = []
        for regime in np.unique(self.regime_labels):
            regime_mask = self.regime_labels == regime
            centroid = np.mean(self.X_scaled[regime_mask], axis=0)
            centroids.append(centroid)
        
        # Calculate total within-cluster variance
        total_wcss = 0
        for regime in np.unique(self.regime_labels):
            regime_mask = self.regime_labels == regime
            regime_points = self.X_scaled[regime_mask]
            centroid = centroids[regime]
            wcss = np.sum((regime_points - centroid) ** 2)
            total_wcss += wcss
        
        metrics = {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_index': float(calinski_harabasz),
            'davies_bouldin_index': float(davies_bouldin),
            'n_regimes': int(n_regimes),
            'total_observations': len(self.regime_labels),
            'regime_sizes': regime_sizes.tolist(),
            'regime_balance': float(regime_balance),
            'within_cluster_sum_squares': float(total_wcss),
            'quality_assessment': {
                'silhouette_quality': 'excellent' if silhouette >= 0.4 else 'good' if silhouette >= 0.2 else 'acceptable',
                'cluster_separation': 'well_separated' if davies_bouldin < 1.0 else 'moderately_separated',
                'cluster_compactness': 'compact' if calinski_harabasz > 50 else 'moderate'
            }
        }
        
        print(f"📊 Statistical Quality Metrics:")
        print(f"  Silhouette Score: {silhouette:.3f} ({metrics['quality_assessment']['silhouette_quality']})")
        print(f"  Calinski-Harabasz Index: {calinski_harabasz:.1f} ({metrics['quality_assessment']['cluster_compactness']})")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.3f} ({metrics['quality_assessment']['cluster_separation']})")
        print(f"  Regime Balance: {regime_balance:.3f}")
        print(f"  Regime Sizes: {regime_sizes.tolist()}")
        
        return metrics
    
    def generate_validation_report(self, save_results=True):
        """
        Generate comprehensive validation report combining all validation methods.
        """
        print("🔍 COMPREHENSIVE REGIME VALIDATION REPORT")
        print("=" * 60)
        
        # Load results and prepare features
        self.load_regime_results()
        
        # Run all validation methods
        cross_method_results, cross_method_summary = self.cross_method_validation()
        alignment_results, alignment_summary = self.economic_event_alignment()
        statistical_metrics = self.statistical_quality_metrics()
        
        # Overall assessment
        validation_score = 0
        max_score = 3
        
        # Cross-method validation (1 point if avg ARI > 0.7)
        if cross_method_summary.get('average_ari', 0) >= 0.7:
            validation_score += 1
            
        # Economic alignment (1 point if >= 70% events aligned)
        if alignment_summary.get('alignment_rate', 0) >= 0.7:
            validation_score += 1
            
        # Statistical quality (1 point if silhouette >= 0.2)
        if statistical_metrics.get('silhouette_score', 0) >= 0.2:
            validation_score += 1
        
        # Comprehensive results
        comprehensive_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'regime_detection_source': self.regime_labels_file,
            'validation_score': f"{validation_score}/{max_score}",
            'overall_assessment': 'excellent' if validation_score == 3 else 'good' if validation_score >= 2 else 'needs_improvement',
            
            'cross_method_validation': {
                'summary': cross_method_summary,
                'detailed_results': cross_method_results
            },
            
            'economic_event_alignment': {
                'summary': alignment_summary, 
                'detailed_alignments': alignment_results
            },
            
            'statistical_quality_metrics': statistical_metrics,
            
            'robustness_assessment': {
                'method_robustness': cross_method_summary.get('average_ari', 0) >= 0.7,
                'economic_validity': alignment_summary.get('alignment_rate', 0) >= 0.7,
                'statistical_quality': statistical_metrics.get('silhouette_score', 0) >= 0.2,
                'recommended_for_production': validation_score >= 2
            }
        }
        
        # Print summary
        print(f"\n🎯 VALIDATION SUMMARY:")
        print(f"  Overall Score: {validation_score}/{max_score}")
        print(f"  Assessment: {comprehensive_results['overall_assessment'].upper()}")
        print(f"  Cross-Method Robustness: {'Y' if comprehensive_results['robustness_assessment']['method_robustness'] else 'N'}")
        print(f"  Economic Validity: {'Y' if comprehensive_results['robustness_assessment']['economic_validity'] else 'N'}")
        print(f"  Statistical Quality: {'Y' if comprehensive_results['robustness_assessment']['statistical_quality'] else 'N'}")
        print(f"  Production Ready: {'Y' if comprehensive_results['robustness_assessment']['recommended_for_production'] else 'N'}")
        
        # Save results
        if save_results:
            import os
            if not os.path.exists('final_results'):
                os.makedirs('final_results')
                
            with open('final_results/regime_validation_report.json', 'w') as f:
                json.dump(comprehensive_results, f, indent=2)
                
            print(f"\n📄 Validation report saved: final_results/regime_validation_report.json")
        
        return comprehensive_results


def main():
    """Main execution for comprehensive regime validation."""
    print("🔬 REGIME VALIDATION METHODS")
    print("Implementing comprehensive validation framework")
    print()
    
    validator = RegimeValidationMethods()
    results = validator.generate_validation_report()
    
    return validator, results


if __name__ == "__main__":
    validator, results = main()