import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load and prepare data"""
    probs = pd.read_csv('../best_soft_probabilities.csv')
    probs['Date'] = pd.to_datetime(probs['Date'])
    
    prices = pd.read_csv('../../../terminal data 1_cleaned.csv')
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    return probs, prices

def create_covid_crisis_zoom(probs, prices, save_path='covid_crisis_detailed.png'):
    """Detailed view of COVID-19 crisis period"""
    
    # Focus on COVID period
    start_date = pd.to_datetime('2019-12-01')
    end_date = pd.to_datetime('2020-08-01')
    
    spx_data = prices[prices['Name'] == 'SPX Index'].copy().sort_values('Date')
    spx_data['Returns'] = spx_data['Price'].pct_change()
    spx_data['RollingVol'] = spx_data['Returns'].rolling(10).std() * np.sqrt(252) * 100
    
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    covid_data = merged[(merged['Date'] >= start_date) & (merged['Date'] <= end_date)]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Key COVID events
    events = [
        {'date': '2020-01-21', 'event': 'First US Case', 'color': 'orange'},
        {'date': '2020-02-24', 'event': 'Market Drop Begins', 'color': 'red'},
        {'date': '2020-03-11', 'event': 'WHO Pandemic', 'color': 'red'},
        {'date': '2020-03-16', 'event': 'Circuit Breakers', 'color': 'darkred'},
        {'date': '2020-03-23', 'event': 'Market Bottom', 'color': 'green'},
        {'date': '2020-03-27', 'event': 'CARES Act', 'color': 'blue'},
        {'date': '2020-04-06', 'event': 'Fed Unlimited QE', 'color': 'green'},
    ]
    
    # Plot 1: SPX Price
    ax1.plot(covid_data['Date'], covid_data['Price'], color='black', linewidth=2)
    for event in events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            ax1.axvline(x=event_date, color=event['color'], linestyle='--', alpha=0.8)
            ax1.text(event_date, covid_data['Price'].max() * 0.95, event['event'], 
                    rotation=90, fontsize=8, ha='right', va='top')
    
    ax1.set_ylabel('S&P 500 Index', fontsize=12, fontweight='bold')
    ax1.set_title('COVID-19 Crisis: Regime Response (Dec 2019 - Aug 2020)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All regime probabilities including regime 1
    regime_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    regime_labels = ['Regime 0 (Low Vol)', 'Regime 1 (Rare)', 'Regime 2 (Med Vol)', 
                    'Regime 3 (High Vol)', 'Regime 4 (Crisis)']
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    
    # Check if regime 1 has any significant probabilities
    if covid_data['P_regime_1'].max() > 0.01:
        ax2.stackplot(covid_data['Date'],
                      covid_data['P_regime_0'], covid_data['P_regime_1'],
                      covid_data['P_regime_2'], covid_data['P_regime_3'], 
                      covid_data['P_regime_4'],
                      labels=regime_labels, colors=colors, alpha=0.8)
    else:
        # Skip regime 1 if it's negligible
        ax2.stackplot(covid_data['Date'],
                      covid_data['P_regime_0'], covid_data['P_regime_2'],
                      covid_data['P_regime_3'], covid_data['P_regime_4'],
                      labels=['Regime 0 (Low Vol)', 'Regime 2 (Med Vol)', 
                             'Regime 3 (High Vol)', 'Regime 4 (Crisis)'],
                      colors=['blue', 'green', 'orange', 'red'], alpha=0.8)
    
    for event in events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            ax2.axvline(x=event_date, color=event['color'], linestyle='--', alpha=0.6)
    
    ax2.set_ylabel('Regime Probabilities', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility with regime coloring
    ax3.plot(covid_data['Date'], covid_data['RollingVol'], color='purple', linewidth=2)
    
    # Color background based on dominant regime
    dominant_regime = covid_data[['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']].idxmax(axis=1)
    for i, (date, regime) in enumerate(zip(covid_data['Date'], dominant_regime)):
        if regime == 'P_regime_4':
            ax3.axvspan(date, date + pd.Timedelta(days=1), alpha=0.3, color='red')
        elif regime == 'P_regime_3':
            ax3.axvspan(date, date + pd.Timedelta(days=1), alpha=0.2, color='orange')
    
    for event in events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            ax3.axvline(x=event_date, color=event['color'], linestyle='--', alpha=0.6)
    
    ax3.set_ylabel('10-Day Volatility (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Format dates
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def create_taper_tantrum_zoom(probs, prices, save_path='taper_tantrum_detailed.png'):
    """Detailed view of 2013 Taper Tantrum"""
    
    start_date = pd.to_datetime('2013-04-01')
    end_date = pd.to_datetime('2013-10-01')
    
    spx_data = prices[prices['Name'] == 'SPX Index'].copy().sort_values('Date')
    spx_data['Returns'] = spx_data['Price'].pct_change()
    spx_data['RollingVol'] = spx_data['Returns'].rolling(10).std() * np.sqrt(252) * 100
    
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    taper_data = merged[(merged['Date'] >= start_date) & (merged['Date'] <= end_date)]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Key Taper Tantrum events
    events = [
        {'date': '2013-05-01', 'event': 'Strong Jobs Data', 'color': 'orange'},
        {'date': '2013-05-22', 'event': 'Bernanke Testimony', 'color': 'red'},
        {'date': '2013-06-19', 'event': 'Fed Meeting', 'color': 'red'},
        {'date': '2013-07-10', 'event': 'Bernanke Speech', 'color': 'blue'},
        {'date': '2013-09-18', 'event': 'Fed No Taper', 'color': 'green'},
    ]
    
    # Plot 1: SPX Price
    ax1.plot(taper_data['Date'], taper_data['Price'], color='black', linewidth=2)
    for event in events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            ax1.axvline(x=event_date, color=event['color'], linestyle='--', alpha=0.8)
            ax1.text(event_date, taper_data['Price'].max() * 0.98, event['event'], 
                    rotation=90, fontsize=9, ha='right', va='top')
    
    ax1.set_ylabel('S&P 500 Index', fontsize=12, fontweight='bold')
    ax1.set_title('Taper Tantrum: Regime Response (Apr - Oct 2013)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual regime probabilities
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    regime_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    regime_names = ['Low Vol', 'Rare', 'Med Vol', 'High Vol', 'Crisis']
    
    for i, (col, color, name) in enumerate(zip(regime_cols, colors, regime_names)):
        if taper_data[col].max() > 0.01:  # Only plot if regime has significant probability
            ax2.plot(taper_data['Date'], taper_data[col], color=color, 
                    linewidth=2, label=f'{name} (R{i})', alpha=0.8)
    
    for event in events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            ax2.axvline(x=event_date, color=event['color'], linestyle='--', alpha=0.6)
    
    ax2.set_ylabel('Individual Regime\nProbabilities', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime uncertainty and volatility
    regime_active_cols = [col for col in regime_cols if taper_data[col].max() > 0.01]
    entropy = -np.sum(taper_data[regime_active_cols] * 
                     np.log(taper_data[regime_active_cols] + 1e-10), axis=1)
    max_entropy = np.log(len(regime_active_cols))
    normalized_entropy = entropy / max_entropy
    
    ax3_twin = ax3.twinx()
    
    # Volatility
    line1 = ax3.plot(taper_data['Date'], taper_data['RollingVol'], 
                     color='purple', linewidth=2, label='Volatility')
    ax3.set_ylabel('10-Day Volatility (%)', fontsize=12, fontweight='bold', color='purple')
    
    # Uncertainty
    line2 = ax3_twin.plot(taper_data['Date'], normalized_entropy, 
                         color='brown', linewidth=2, label='Regime Uncertainty', linestyle=':')
    ax3_twin.set_ylabel('Regime Uncertainty', fontsize=12, fontweight='bold', color='brown')
    ax3_twin.set_ylim(0, 1)
    
    for event in events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            ax3.axvline(x=event_date, color=event['color'], linestyle='--', alpha=0.6)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Format dates
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def create_all_regimes_overview(probs, prices, save_path='all_regimes_complete.png'):
    """Complete overview including all regimes with better clarity"""
    
    spx_data = prices[prices['Name'] == 'SPX Index'].copy().sort_values('Date')
    spx_data['Returns'] = spx_data['Price'].pct_change()
    spx_data['RollingVol'] = spx_data['Returns'].rolling(22).std() * np.sqrt(252) * 100
    
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
    
    # Plot 1: SPX Price
    ax1.plot(merged['Date'], merged['Price'], color='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('S&P 500 Index', fontsize=12, fontweight='bold')
    ax1.set_title('Complete Soft Regime-Switching Analysis (All Regimes)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All regime probabilities separately
    regime_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3', 'P_regime_4']
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    regime_names = ['Regime 0 (Low Vol)', 'Regime 1 (Rare)', 'Regime 2 (Med Vol)', 
                   'Regime 3 (High Vol)', 'Regime 4 (Crisis)']
    
    for col, color, name in zip(regime_cols, colors, regime_names):
        if merged[col].max() > 0.001:  # Only plot if regime appears
            ax2.plot(merged['Date'], merged[col], color=color, linewidth=1.5, 
                    label=name, alpha=0.8)
    
    ax2.set_ylabel('Individual Regime\nProbabilities', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime statistics
    # Create regime frequency and duration analysis
    dominant_regime = merged[regime_cols].idxmax(axis=1)
    regime_counts = dominant_regime.value_counts()
    regime_percentages = regime_counts / len(dominant_regime) * 100
    
    # Convert to numbers for plotting
    regime_mapping = {'P_regime_0': 0, 'P_regime_1': 1, 'P_regime_2': 2, 
                     'P_regime_3': 3, 'P_regime_4': 4}
    dominant_regime_num = dominant_regime.map(regime_mapping)
    
    # Plot regime timeline
    scatter_colors = []
    for regime in dominant_regime_num:
        scatter_colors.append(colors[regime] if regime < len(colors) else 'gray')
    
    ax3.scatter(merged['Date'], dominant_regime_num, c=scatter_colors, alpha=0.6, s=8)
    ax3.set_ylabel('Dominant Regime', fontsize=12, fontweight='bold')
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(['0 (Low)', '1 (Rare)', '2 (Med)', '3 (High)', '4 (Crisis)'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility with regime context
    ax4.plot(merged['Date'], merged['RollingVol'], color='purple', linewidth=1.5, alpha=0.8)
    
    # Add horizontal lines for regime thresholds
    vol_percentiles = merged['RollingVol'].quantile([0.25, 0.5, 0.75, 0.9])
    ax4.axhline(y=vol_percentiles[0.5], color='green', linestyle=':', alpha=0.7, label='Median Vol')
    ax4.axhline(y=vol_percentiles[0.75], color='orange', linestyle=':', alpha=0.7, label='75th Percentile')
    ax4.axhline(y=vol_percentiles[0.9], color='red', linestyle=':', alpha=0.7, label='90th Percentile')
    
    ax4.set_ylabel('22-Day Volatility\n(% Annualized)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Format dates
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add regime statistics text box
    stats_text = "Regime Frequencies:\n"
    for regime, percentage in regime_percentages.items():
        regime_num = regime_mapping[regime]
        stats_text += f"Regime {regime_num}: {percentage:.1f}%\n"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def create_crisis_comparison(probs, prices, save_path='crisis_periods_comparison.png'):
    """Compare regime response across different crisis periods"""
    
    spx_data = prices[prices['Name'] == 'SPX Index'].copy().sort_values('Date')
    spx_data['Returns'] = spx_data['Price'].pct_change()
    spx_data['RollingVol'] = spx_data['Returns'].rolling(10).std() * np.sqrt(252) * 100
    merged = pd.merge(probs, spx_data, on='Date', how='inner')
    
    # Define crisis periods
    crisis_periods = [
        {'name': 'Taper Tantrum', 'start': '2013-05-01', 'end': '2013-08-31', 'color': 'red'},
        {'name': 'China Crisis', 'start': '2015-08-01', 'end': '2016-01-31', 'color': 'orange'},
        {'name': 'COVID-19', 'start': '2020-02-15', 'end': '2020-05-15', 'color': 'darkred'},
    ]
    
    fig, axes = plt.subplots(len(crisis_periods), 3, figsize=(18, 12))
    
    for i, crisis in enumerate(crisis_periods):
        start_date = pd.to_datetime(crisis['start'])
        end_date = pd.to_datetime(crisis['end'])
        crisis_data = merged[(merged['Date'] >= start_date) & (merged['Date'] <= end_date)]
        
        # Price plot
        axes[i, 0].plot(crisis_data['Date'], crisis_data['Price'], 
                       color='black', linewidth=2)
        axes[i, 0].set_title(f'{crisis["name"]} - Price', fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)
        if i == len(crisis_periods) - 1:
            axes[i, 0].set_xlabel('Date')
        axes[i, 0].set_ylabel('SPX Index')
        
        # Regime probabilities
        regime_cols = ['P_regime_0', 'P_regime_2', 'P_regime_3', 'P_regime_4']
        colors = ['blue', 'green', 'orange', 'red']
        
        axes[i, 1].stackplot(crisis_data['Date'],
                            crisis_data['P_regime_0'], crisis_data['P_regime_2'],
                            crisis_data['P_regime_3'], crisis_data['P_regime_4'],
                            colors=colors, alpha=0.8)
        axes[i, 1].set_title(f'{crisis["name"]} - Regimes', fontweight='bold')
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].grid(True, alpha=0.3)
        if i == len(crisis_periods) - 1:
            axes[i, 1].set_xlabel('Date')
        axes[i, 1].set_ylabel('Regime Prob.')
        
        # Volatility
        axes[i, 2].plot(crisis_data['Date'], crisis_data['RollingVol'], 
                       color='purple', linewidth=2)
        axes[i, 2].set_title(f'{crisis["name"]} - Volatility', fontweight='bold')
        axes[i, 2].grid(True, alpha=0.3)
        if i == len(crisis_periods) - 1:
            axes[i, 2].set_xlabel('Date')
        axes[i, 2].set_ylabel('10-Day Vol (%)')
        
        # Format dates
        for j in range(3):
            axes[i, j].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            plt.setp(axes[i, j].xaxis.get_majorticklabels(), rotation=45)
    
    # Add overall title
    fig.suptitle('Crisis Period Comparisons: Regime Response Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

def main():
    """Generate all crisis-focused visualizations"""
    print("Loading data...")
    probs, prices = load_data()
    
    print("Creating COVID-19 crisis detailed view...")
    create_covid_crisis_zoom(probs, prices)
    
    print("Creating Taper Tantrum detailed view...")
    create_taper_tantrum_zoom(probs, prices)
    
    print("Creating complete regime overview...")
    create_all_regimes_overview(probs, prices)
    
    print("Creating crisis periods comparison...")
    create_crisis_comparison(probs, prices)
    
    print("\n✅ All crisis-focused visualizations completed!")
    print("\nGenerated files:")
    print("- covid_crisis_detailed.png: Deep dive into COVID-19 period")
    print("- taper_tantrum_detailed.png: Deep dive into 2013 Taper Tantrum")
    print("- all_regimes_complete.png: Complete overview including all 5 regimes")
    print("- crisis_periods_comparison.png: Side-by-side crisis comparisons")

if __name__ == "__main__":
    main()