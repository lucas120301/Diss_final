import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

plt.style.use('default')
sns.set_palette("Set2")

def load_data():
    """Load and prepare data"""
    probs = pd.read_csv('../best_soft_probabilities.csv')
    probs['Date'] = pd.to_datetime(probs['Date'])
    
    prices = pd.read_csv('../../../terminal data 1_cleaned.csv')
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    return probs, prices

def create_crisis_timeline_plot(probs, prices, save_path='crisis_regime_timeline.png'):
    """Focus on crisis periods and regime transitions"""
    
    spx_data = prices[prices['Name'] == 'SPX Index'].copy().sort_values('Date')
    spx_data['Returns'] = spx_data['Price'].pct_change()
    spx_data['RollingVol'] = spx_data['Returns'].rolling(22).std() * np.sqrt(252) * 100
    
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    
    # Define crisis periods
    crisis_periods = [
        {'start': '2013-05-01', 'end': '2013-09-01', 'name': 'Taper Tantrum', 'color': '#ff7f7f'},
        {'start': '2015-08-01', 'end': '2016-03-01', 'name': 'China Crisis', 'color': '#ffb347'},
        {'start': '2018-01-01', 'end': '2018-04-01', 'name': 'Vol Spike', 'color': '#ff6b6b'},
        {'start': '2018-10-01', 'end': '2018-12-31', 'name': 'Trade War', 'color': '#ff4757'},
        {'start': '2020-02-01', 'end': '2020-05-01', 'name': 'COVID-19', 'color': '#dc143c'},
    ]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: SPX with crisis shading
    ax1.plot(merged['Date'], merged['Price'], color='black', linewidth=1.5, alpha=0.8)
    
    for crisis in crisis_periods:
        start_date = pd.to_datetime(crisis['start'])
        end_date = pd.to_datetime(crisis['end'])
        ax1.axvspan(start_date, end_date, alpha=0.3, color=crisis['color'], label=crisis['name'])
    
    ax1.set_ylabel('S&P 500 Index', fontsize=12, fontweight='bold')
    ax1.set_title('Market Crises and Soft Regime Response', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Crisis regime (Regime 4) probability
    ax2.fill_between(merged['Date'], 0, merged['P_regime_4'], 
                     color='red', alpha=0.7, label='Crisis Regime (4)')
    ax2.fill_between(merged['Date'], merged['P_regime_4'], 
                     merged['P_regime_4'] + merged['P_regime_3'],
                     color='orange', alpha=0.7, label='High Vol Regime (3)')
    
    for crisis in crisis_periods:
        start_date = pd.to_datetime(crisis['start'])
        end_date = pd.to_datetime(crisis['end'])
        ax2.axvspan(start_date, end_date, alpha=0.2, color=crisis['color'])
    
    ax2.set_ylabel('Crisis/High Vol\nProbability', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Realized volatility
    ax3.plot(merged['Date'], merged['RollingVol'], color='purple', linewidth=1.5)
    ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.8, label='High Vol Threshold')
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.8, label='Crisis Vol Threshold')
    
    for crisis in crisis_periods:
        start_date = pd.to_datetime(crisis['start'])
        end_date = pd.to_datetime(crisis['end'])
        ax3.axvspan(start_date, end_date, alpha=0.2, color=crisis['color'])
    
    ax3.set_ylabel('Realized Volatility\n(% Annualized)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Format dates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def create_regime_transition_heatmap(probs, save_path='regime_transition_heatmap.png'):
    """Show how regimes transition over time"""
    
    # Create regime transition matrix
    regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    
    # Get dominant regime for each day
    dominant = probs[regime_cols].idxmax(axis=1)
    regime_map = {'P_regime_0': 0, 'P_regime_2': 2, 'P_regime_3': 3, 'P_regime_4': 4}
    dominant_num = dominant.map(regime_map)
    
    # Create transition matrix
    transitions = pd.crosstab(dominant_num.shift(1), dominant_num, normalize='index')
    transitions = transitions.fillna(0)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Transition matrix heatmap
    sns.heatmap(transitions, annot=True, fmt='.3f', cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Transition Probability'})
    ax1.set_title('Regime Transition Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('To Regime', fontsize=12)
    ax1.set_ylabel('From Regime', fontsize=12)
    
    # Regime persistence (diagonal elements)
    persistence = np.diag(transitions)
    regime_names = ['Regime 0\n(Low Vol)', 'Regime 2\n(Medium Vol)', 
                   'Regime 3\n(High Vol)', 'Regime 4\n(Crisis)']
    
    colors = ['skyblue', 'lightgreen', 'orange', 'red']
    bars = ax2.bar(regime_names, persistence, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Regime Persistence', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability of Staying in Same Regime', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, persistence):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def create_vol_regime_relationship(probs, prices, save_path='vol_regime_relationship.png'):
    """Clear relationship between volatility and regimes"""
    
    # Prepare data for all three indices
    assets = ['SPX Index', 'RTY Index', 'NDX Index']
    asset_names = ['S&P 500', 'Russell 2000', 'NASDAQ 100']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    regime_names = ['Low Vol', 'Medium Vol', 'High Vol', 'Crisis']
    colors = ['blue', 'green', 'orange', 'red']
    
    all_vol_data = []
    
    for idx, (asset, asset_name) in enumerate(zip(assets, asset_names)):
        asset_data = prices[prices['Name'] == asset].copy().sort_values('Date')
        asset_data['Returns'] = asset_data['Price'].pct_change()
        asset_data['Volatility'] = asset_data['Returns'].rolling(22).std() * np.sqrt(252) * 100
        
        merged = pd.merge(probs, asset_data, on='Date', how='inner')
        
        # Box plot showing volatility distribution by dominant regime
        dominant_regime = merged[regime_cols].idxmax(axis=1)
        regime_mapping = {'P_regime_0': 0, 'P_regime_2': 1, 'P_regime_3': 2, 'P_regime_4': 3}
        merged['Dominant_Regime'] = dominant_regime.map(regime_mapping)
        
        vol_by_regime = [merged[merged['Dominant_Regime'] == i]['Volatility'].dropna() 
                        for i in range(4)]
        
        bp = axes[idx].boxplot(vol_by_regime, labels=regime_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[idx].set_title(f'{asset_name} Volatility by Regime', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Annualized Volatility (%)', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Store data for summary
        for i, regime_name in enumerate(regime_names):
            if len(vol_by_regime[i]) > 0:
                all_vol_data.append({
                    'Asset': asset_name,
                    'Regime': regime_name,
                    'Mean_Vol': vol_by_regime[i].mean(),
                    'Median_Vol': vol_by_regime[i].median(),
                    'Count': len(vol_by_regime[i])
                })
    
    # Summary statistics table
    summary_df = pd.DataFrame(all_vol_data)
    pivot_mean = summary_df.pivot(index='Regime', columns='Asset', values='Mean_Vol')
    
    # Create summary table plot
    axes[3].axis('off')
    table_data = []
    for regime in regime_names:
        row_data = [regime]
        for asset in asset_names:
            val = pivot_mean.loc[regime, asset] if regime in pivot_mean.index else np.nan
            row_data.append(f'{val:.1f}%' if not pd.isna(val) else 'N/A')
        table_data.append(row_data)
    
    table = axes[3].table(cellText=table_data,
                         colLabels=['Regime'] + asset_names,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the regime column
    for i, color in enumerate(colors):
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(weight='bold', color='white')
    
    axes[3].set_title('Average Volatility by Regime', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def create_soft_vs_hard_comparison(probs, save_path='soft_vs_hard_regimes.png'):
    """Compare soft probabilities vs hard assignments"""
    
    regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    
    # Get hard assignments (dominant regime)
    hard_assignments = probs[regime_cols].idxmax(axis=1)
    regime_mapping = {'P_regime_0': 0, 'P_regime_2': 2, 'P_regime_3': 3, 'P_regime_4': 4}
    hard_regime_num = hard_assignments.map(regime_mapping)
    
    # Sample a period to show the difference clearly
    sample_period = slice(1000, 1200)  # 200 days sample
    sample_dates = probs['Date'].iloc[sample_period]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Hard regime assignments
    colors = ['blue', 'green', 'orange', 'red']
    regime_nums = [0, 2, 3, 4]
    
    for i, (regime_num, color) in enumerate(zip(regime_nums, colors)):
        mask = hard_regime_num.iloc[sample_period] == regime_num
        ax1.scatter(sample_dates[mask], [regime_num] * mask.sum(), 
                   c=color, alpha=0.8, s=30, label=f'Regime {regime_num}')
    
    ax1.set_ylabel('Hard Regime\nAssignment', fontsize=12, fontweight='bold')
    ax1.set_title('Hard vs Soft Regime Assignment Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_yticks([0, 2, 3, 4])
    ax1.set_yticklabels(['0 (Low)', '2 (Med)', '3 (High)', '4 (Crisis)'])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Soft probabilities (stacked)
    ax2.stackplot(sample_dates,
                  probs['P_regime_0'].iloc[sample_period],
                  probs['P_regime_2'].iloc[sample_period],
                  probs['P_regime_3'].iloc[sample_period],
                  probs['P_regime_4'].iloc[sample_period],
                  labels=['Regime 0', 'Regime 2', 'Regime 3', 'Regime 4'],
                  colors=colors, alpha=0.8)
    
    ax2.set_ylabel('Soft Regime\nProbabilities', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime uncertainty (entropy)
    # Calculate entropy as measure of uncertainty
    entropy = -np.sum(probs[regime_cols].iloc[sample_period] * 
                     np.log(probs[regime_cols].iloc[sample_period] + 1e-10), axis=1)
    max_entropy = np.log(len(regime_cols))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    ax3.fill_between(sample_dates, 0, normalized_entropy, 
                     color='purple', alpha=0.7, label='Regime Uncertainty')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, 
                label='High Uncertainty Threshold')
    
    ax3.set_ylabel('Regime\nUncertainty', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Format dates
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def main():
    """Generate all improved visualizations"""
    print("Loading data...")
    probs, prices = load_data()
    
    print("Creating crisis timeline plot...")
    create_crisis_timeline_plot(probs, prices)
    
    print("Creating regime transition analysis...")
    create_regime_transition_heatmap(probs)
    
    print("Creating volatility-regime relationship plot...")
    create_vol_regime_relationship(probs, prices)
    
    print("Creating soft vs hard regime comparison...")
    create_soft_vs_hard_comparison(probs)
    
    print("\n✅ All improved visualizations completed!")
    print("\nGenerated files:")
    print("- crisis_regime_timeline.png: Shows how regimes respond to major crises")
    print("- regime_transition_heatmap.png: Regime switching patterns and persistence")  
    print("- vol_regime_relationship.png: Clear vol-regime relationship across assets")
    print("- soft_vs_hard_regimes.png: Demonstrates soft regime advantages")

if __name__ == "__main__":
    main()