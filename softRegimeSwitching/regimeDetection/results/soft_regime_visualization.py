import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load soft regime probabilities and price data"""
    # Load soft probabilities
    probs = pd.read_csv('best_soft_probabilities.csv')
    probs['Date'] = pd.to_datetime(probs['Date'])
    
    # Load price data
    prices = pd.read_csv('../../terminal data 1_cleaned.csv')
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    return probs, prices

def create_market_events():
    """Define key market events for annotation"""
    events = [
        {'date': '2012-09-13', 'event': 'Fed QE3 Announcement', 'color': 'green'},
        {'date': '2013-05-22', 'event': 'Bernanke Taper Tantrum', 'color': 'red'},
        {'date': '2013-12-18', 'event': 'Fed Begins Tapering', 'color': 'orange'},
        {'date': '2014-10-15', 'event': 'Fed Ends QE3', 'color': 'purple'},
        {'date': '2015-12-16', 'event': 'First Rate Hike (2009)', 'color': 'blue'},
        {'date': '2016-06-23', 'event': 'Brexit Vote', 'color': 'red'},
        {'date': '2016-11-08', 'event': 'Trump Election', 'color': 'green'},
        {'date': '2018-02-05', 'event': 'Volatility Spike', 'color': 'red'},
        {'date': '2018-10-03', 'event': 'Trade War Escalation', 'color': 'red'},
        {'date': '2020-03-11', 'event': 'WHO Declares Pandemic', 'color': 'red'},
        {'date': '2020-03-23', 'event': 'Market Bottom', 'color': 'green'},
    ]
    
    for event in events:
        event['date'] = pd.to_datetime(event['date'])
    
    return events

def plot_soft_regimes_with_spx(probs, prices, events, save_path='soft_regimes_spx_timeline.png'):
    """Create comprehensive soft regime visualization with SPX overlay"""
    
    # Filter for SPX data
    spx_data = prices[prices['Name'] == 'SPX Index'].copy()
    spx_data = spx_data.sort_values('Date')
    
    # Merge probabilities with SPX data
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                       gridspec_kw={'height_ratios': [2, 3, 1], 'hspace': 0.3})
    
    # Plot 1: SPX Price with market events
    ax1.plot(merged['Date'], merged['Price'], linewidth=1.5, color='black', alpha=0.8)
    ax1.set_ylabel('SPX Index Level', fontsize=12, fontweight='bold')
    ax1.set_title('S&P 500 Index with Soft Regime Probabilities and Market Events', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add market events as vertical lines
    for event in events:
        if merged['Date'].min() <= event['date'] <= merged['Date'].max():
            ax1.axvline(x=event['date'], color=event['color'], linestyle='--', 
                       alpha=0.6, linewidth=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    
    # Plot 2: Soft regime probabilities as stacked area
    regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    regime_labels = ['Regime 0 (Low Vol)', 'Regime 2 (Medium Vol)', 
                    'Regime 3 (High Vol)', 'Regime 4 (Crisis)']
    
    # Create stacked area plot
    ax2.stackplot(merged['Date'], 
                  merged['P_regime_0'], merged['P_regime_2'],
                  merged['P_regime_3'], merged['P_regime_4'],
                  labels=regime_labels,
                  alpha=0.8)
    
    ax2.set_ylabel('Regime Probability', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax2.grid(True, alpha=0.3)
    
    # Add market events
    for event in events:
        if merged['Date'].min() <= event['date'] <= merged['Date'].max():
            ax2.axvline(x=event['date'], color=event['color'], linestyle='--', 
                       alpha=0.6, linewidth=2)
    
    # Plot 3: Dominant regime (for clarity)
    dominant_regime = merged[regime_cols].idxmax(axis=1)
    regime_mapping = {'P_regime_0': 0, 'P_regime_2': 2, 'P_regime_3': 3, 'P_regime_4': 4}
    dominant_regime_num = dominant_regime.map(regime_mapping)
    
    # Create color map for regimes
    colors = ['blue', 'green', 'orange', 'red']
    for i, regime in enumerate([0, 2, 3, 4]):
        mask = dominant_regime_num == regime
        if mask.any():
            ax3.scatter(merged.loc[mask, 'Date'], [regime]*mask.sum(), 
                       c=colors[i], alpha=0.6, s=20)
    
    ax3.set_ylabel('Dominant\nRegime', fontsize=10, fontweight='bold')
    ax3.set_yticks([0, 2, 3, 4])
    ax3.set_yticklabels(['0 (Low)', '2 (Med)', '3 (High)', '4 (Crisis)'])
    ax3.grid(True, alpha=0.3)
    
    # Add market events
    for event in events:
        if merged['Date'].min() <= event['date'] <= merged['Date'].max():
            ax3.axvline(x=event['date'], color=event['color'], linestyle='--', 
                       alpha=0.6, linewidth=2)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    
    # Only show x-axis labels on bottom plot
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig

def plot_regime_characteristics(probs, prices, save_path='regime_characteristics.png'):
    """Create regime characterization plots"""
    
    # Get all assets
    assets = ['SPX Index', 'RTY Index', 'NDX Index']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Calculate rolling statistics for each regime
    regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    
    for idx, asset in enumerate(assets):
        if idx >= len(axes) - 1:
            break
            
        asset_data = prices[prices['Name'] == asset].copy()
        asset_data = asset_data.sort_values('Date')
        
        # Calculate returns
        asset_data['Returns'] = asset_data['Price'].pct_change()
        asset_data['Volatility'] = asset_data['Returns'].rolling(22).std() * np.sqrt(252)
        
        # Merge with probabilities
        merged = pd.merge(probs, asset_data, on='Date', how='inner')
        
        # Create volatility vs regime probability scatter
        for i, (regime_col, regime_num) in enumerate(zip(regime_cols, [0, 2, 3, 4])):
            mask = merged[regime_col] > 0.1  # Only show when regime has >10% probability
            if mask.any():
                axes[idx].scatter(merged.loc[mask, regime_col], 
                                merged.loc[mask, 'Volatility'],
                                alpha=0.6, label=f'Regime {regime_num}', s=20)
        
        axes[idx].set_xlabel(f'Regime Probability')
        axes[idx].set_ylabel('Realized Volatility (Annualized)')
        axes[idx].set_title(f'{asset.replace(" Index", "")} Volatility vs Regime Probability')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Use last subplot for regime duration analysis
    regime_durations = []
    dominant_regime = probs[regime_cols].idxmax(axis=1)
    
    current_regime = dominant_regime.iloc[0]
    duration = 1
    
    for i in range(1, len(dominant_regime)):
        if dominant_regime.iloc[i] == current_regime:
            duration += 1
        else:
            regime_durations.append(duration)
            current_regime = dominant_regime.iloc[i]
            duration = 1
    
    axes[3].hist(regime_durations, bins=20, alpha=0.7, edgecolor='black')
    axes[3].set_xlabel('Regime Duration (Days)')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Distribution of Regime Durations')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig

def create_event_annotation_plot(probs, prices, events, save_path='regime_events_detailed.png'):
    """Create detailed plot with event annotations"""
    
    spx_data = prices[prices['Name'] == 'SPX Index'].copy()
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Create background colors for dominant regimes
    regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    dominant_regime = merged[regime_cols].idxmax(axis=1)
    regime_mapping = {'P_regime_0': 0, 'P_regime_2': 2, 'P_regime_3': 3, 'P_regime_4': 4}
    dominant_regime_num = dominant_regime.map(regime_mapping)
    
    # Plot SPX
    ax.plot(merged['Date'], merged['Price'], linewidth=2, color='black', label='SPX Index')
    
    # Add regime background colors
    colors = ['lightblue', 'lightgreen', 'orange', 'red']
    alphas = [0.2, 0.2, 0.3, 0.4]
    
    for i, (regime, color, alpha) in enumerate(zip([0, 2, 3, 4], colors, alphas)):
        mask = dominant_regime_num == regime
        if mask.any():
            for start, end in get_continuous_periods(merged.loc[mask, 'Date']):
                ax.axvspan(start, end, alpha=alpha, color=color, label=f'Regime {regime}' if i < 4 else '')
    
    # Add event annotations
    for event in events:
        if merged['Date'].min() <= event['date'] <= merged['Date'].max():
            # Find closest price point
            closest_idx = (merged['Date'] - event['date']).abs().idxmin()
            price_at_event = merged.loc[closest_idx, 'Price']
            
            ax.annotate(event['event'], 
                       xy=(event['date'], price_at_event),
                       xytext=(10, 30), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=event['color'], alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color=event['color'], lw=1.5),
                       fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('SPX Index Level', fontsize=12, fontweight='bold')
    ax.set_title('Soft Regime-Switching with Key Market Events', fontsize=14, fontweight='bold')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig

def get_continuous_periods(dates):
    """Helper function to get continuous date periods"""
    if len(dates) == 0:
        return []
    
    dates = sorted(dates)
    periods = []
    start = dates[0]
    end = dates[0]
    
    for i in range(1, len(dates)):
        if (dates[i] - end).days <= 2:  # Allow small gaps
            end = dates[i]
        else:
            periods.append((start, end))
            start = dates[i]
            end = dates[i]
    
    periods.append((start, end))
    return periods

def main():
    """Main execution function"""
    print("Loading data...")
    probs, prices = load_data()
    events = create_market_events()
    
    print("Creating soft regime visualization with SPX...")
    plot_soft_regimes_with_spx(probs, prices, events)
    
    print("Creating regime characteristics plot...")
    plot_regime_characteristics(probs, prices)
    
    print("Creating detailed event annotation plot...")
    create_event_annotation_plot(probs, prices, events)
    
    print("\nAll visualizations completed!")
    print("Generated files:")
    print("- soft_regimes_spx_timeline.png")
    print("- regime_characteristics.png") 
    print("- regime_events_detailed.png")

if __name__ == "__main__":
    main()