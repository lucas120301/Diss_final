#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Processing for Financial Indices (SPX, RTY, NDX)
Loads and preprocesses data from Excel terminal data file
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_terminal_data(excel_file_path):
    """
    Load financial data from Bloomberg terminal Excel file
    
    Parameters:
    excel_file_path: path to the Excel file
    
    Returns:
    dict: Dictionary with processed dataframes for each index
    """
    print("Loading data from Excel file...")
    
    # Read the Excel file
    df_raw = pd.read_excel(excel_file_path, header=None)
    
    data_dict = {}
    
    # Process each financial index
    target_securities = ['SPX Index', 'NDX Index', 'RTY Index']
    
    # Define the correct column mappings based on debug output
    security_mappings = {
        'SPX Index': {'date_col': 0, 'price_col': 1},
        'NDX Index': {'date_col': 3, 'price_col': 4}, 
        'RTY Index': {'date_col': 6, 'price_col': 7}
    }
    
    for security in target_securities:
        if security in security_mappings:
            print(f"Processing {security}...")
            
            date_col = security_mappings[security]['date_col']
            price_col = security_mappings[security]['price_col']
            
            # Extract dates and prices from correct columns, starting from row 3
            dates_raw = df_raw.iloc[3:, date_col].dropna()
            prices_raw = df_raw.iloc[3:, price_col].dropna()
            
            print(f"  Raw data: {len(dates_raw)} dates, {len(prices_raw)} prices")
            
            # Ensure same length
            min_len = min(len(dates_raw), len(prices_raw))
            dates_raw = dates_raw.iloc[:min_len]
            prices_raw = prices_raw.iloc[:min_len]
            
            print(f"  After alignment: {min_len} records")
            print(f"  Sample dates: {dates_raw.head(3).tolist()}")
            print(f"  Sample prices: {prices_raw.head(3).tolist()}")
            
            # Convert Excel serial dates to datetime
            try:
                # Convert to numeric first, then to datetime
                dates_numeric = pd.to_numeric(dates_raw, errors='coerce')
                # Excel epoch starts at 1899-12-30 (accounting for Excel's leap year bug)
                dates = pd.to_datetime('1899-12-30') + pd.to_timedelta(dates_numeric, unit='D')
                print(f"  Date conversion successful")
                print(f"  Converted date range: {dates.min()} to {dates.max()}")
            except Exception as e:
                print(f"  Date conversion failed: {e}")
                continue
            
            # Ensure prices are numeric
            try:
                prices_numeric = pd.to_numeric(prices_raw, errors='coerce')
                print(f"  Price conversion successful")
                print(f"  Price range: {prices_numeric.min():.2f} to {prices_numeric.max():.2f}")
            except Exception as e:
                print(f"  Price conversion failed: {e}")
                continue
            
            # Create dataframe
            df_processed = pd.DataFrame({
                'Date': dates,
                'Price': prices_numeric
            })
            
            # Remove any remaining NaN values
            df_processed = df_processed.dropna()
            
            # Sort by date
            df_processed = df_processed.sort_values('Date').reset_index(drop=True)
            
            data_dict[security] = df_processed
            print(f"  - Final: {len(df_processed)} records from {df_processed['Date'].min()} to {df_processed['Date'].max()}")
    
    return data_dict

def calculate_returns_and_volatility(data_dict, window=22):
    """
    Calculate log returns and rolling volatility for all indices
    
    Parameters:
    data_dict: Dictionary of dataframes from load_terminal_data
    window: Rolling window for volatility calculation (default 22 trading days)
    
    Returns:
    dict: Updated dictionary with returns and volatility calculations
    """
    processed_data = {}
    
    for security, df in data_dict.items():
        print(f"Calculating returns and volatility for {security}...")
        
        df_proc = df.copy()
        
        # Ensure Price column is numeric
        df_proc['Price'] = pd.to_numeric(df_proc['Price'], errors='coerce')
        
        # Remove any rows where Price is NaN
        df_proc = df_proc.dropna(subset=['Price'])
        
        if len(df_proc) < 2:
            print(f"  Warning: Insufficient data for {security}")
            continue
        
        # Calculate log returns
        try:
            df_proc['Returns'] = np.log(df_proc['Price'] / df_proc['Price'].shift(1))
        except Exception as e:
            print(f"  Error calculating returns for {security}: {e}")
            continue
        
        # Calculate rolling variance and volatility
        df_proc['Rolling_Var'] = df_proc['Returns'].rolling(window=window, center=False).var()
        df_proc['Volatility'] = df_proc['Rolling_Var'] * (252**0.5) * 100
        
        # Drop NaN values
        df_proc = df_proc.dropna().reset_index(drop=True)
        
        processed_data[security] = df_proc
        print(f"  - {len(df_proc)} valid records after processing")
        print(f"  - Returns: mean={df_proc['Returns'].mean():.6f}, std={df_proc['Returns'].std():.6f}")
        print(f"  - Volatility: mean={df_proc['Volatility'].mean():.2f}%, std={df_proc['Volatility'].std():.2f}%")
    
    return processed_data

def save_processed_data(data_dict, output_file='financial_indices_data.csv'):
    """
    Save processed data to CSV file in format expected by models
    
    Parameters:
    data_dict: Dictionary of processed dataframes
    output_file: Output CSV filename
    """
    print(f"Saving processed data to {output_file}...")
    
    # Combine all data into single dataframe
    combined_data = None
    
    for security, df in data_dict.items():
        # Rename columns to include security name
        security_name = security.replace(' Index', '').replace(' ', '_')
        
        df_renamed = df[['Date', 'Price', 'Returns', 'Volatility']].copy()
        df_renamed.columns = ['Date', f'Price_{security_name}', f'Returns_{security_name}', f'Volatility_{security_name}']
        
        if combined_data is None:
            combined_data = df_renamed
        else:
            # Merge on date
            combined_data = pd.merge(combined_data, df_renamed, on='Date', how='outer')
    
    if combined_data is None:
        print("Error: No data was successfully processed!")
        return None
    
    # Sort by date and save
    combined_data = combined_data.sort_values('Date').reset_index(drop=True)
    combined_data.to_csv(output_file, index=False)
    
    print(f"Data saved with {len(combined_data)} records")
    print(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
    
    return combined_data

def create_garch_input_data(data_dict, output_prefix='garch_data'):
    """
    Create separate CSV files for each index for GARCH modeling
    Format: single column of returns scaled by 1000 (following original approach)
    
    Parameters:
    data_dict: Dictionary of processed dataframes
    output_prefix: Prefix for output files
    """
    if not data_dict:
        print("No data available for GARCH input files.")
        return
        
    print("Creating GARCH input files...")
    
    for security, df in data_dict.items():
        security_name = security.replace(' Index', '').replace(' ', '_')
        
        # Scale returns by 1000 (following original methodology)
        scaled_returns = df['Returns'] * 1000
        
        # Save to CSV
        output_file = f"{output_prefix}_{security_name}.csv"
        scaled_returns.to_csv(output_file, index=False, header=['returns'])
        
        print(f"  - Saved {security} returns to {output_file}")

def main():
    """Main execution function"""
    # Example usage
    excel_file = "terminal data 1.xlsx"  # Update with your file path
    
    # Load and process data
    raw_data = load_terminal_data(excel_file)
    
    if not raw_data:
        print("No data was loaded successfully!")
        return False
    
    processed_data = calculate_returns_and_volatility(raw_data)
    
    if not processed_data:
        print("No data was processed successfully!")
        return False
    
    # Save processed data
    final_data = save_processed_data(processed_data)
    
    if final_data is None:
        print("Data saving failed!")
        return False
    
    # Create GARCH input files
    create_garch_input_data(processed_data)
    
    print("\nData processing complete!")
    print("Files created:")
    print("- financial_indices_data.csv (main dataset)")
    print("- garch_data_SPX.csv (for GARCH modeling)")
    print("- garch_data_NDX.csv (for GARCH modeling)")
    print("- garch_data_RTY.csv (for GARCH modeling)")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Data processing completed successfully!")
    else:
        print("\n❌ Data processing failed!")