#!/usr/bin/env python3
"""
Extract and combine data from sim_ftpl.csv and sim_non_ftpl.csv files.
Maps FRB/US variable names to the expected column names for the streamlit app.
"""

import pandas as pd
import os
import pandas
from numpy import array, cumprod

from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import sim_plot
from pyfrbus.load_data import load_data

def ftpl_simulation():
    
    # FRB/US FTPL
    # Load data
    data = load_data("../data/LONGBASE.TXT")

    # Load model
    frbus = Frbus("../models/model_FRBUS_FTPL.xml")

    # Specify dates 
    start = "2000Q1"
    init_start = "2000Q1"
    end =  "2024Q4"  
    end_1 =  "2025Q1"  
    start_period = pandas.Period(start)
    end_period = pandas.Period(end) 

    # Standard configuration, use surplus ratio targeting
    # data.loc[start:end, "gfdrt"] = 1
    data.loc[start:end, "dfpdbt"] = 0
    data.loc[start:end, "dfpsrp"] = 0

    # Enable customize fiscal policy
    data.loc[start:end, "dfpex"] = 1

    # Disable non-inertial Taylor rule
    data.loc[start:end, "dmptay"] = 0
    data.loc[start:end, "dmpintay"] = 0
    data.loc[start:end, "dmpintay"] = 0

    # For pure FTPL, consider using interest rate peg instead
    data.loc[start:end, "dmptay"] = 0      # Disable Taylor rule
    # data.loc[start:end, "dmpex"] = 1       # Use exogenous funds rate  
    # data.loc[start:end, "rfffix"] = 5.0    # Fixed interest rate
    
    # Enable thresholds
    data.loc[start:end, "dmptrsh"] = 1
    # Arbitrary threshold values
    data.loc[start:end, "lurtrsh"] = 6.0
    data.loc[start:end, "pitrsh"] = 3.0
    sim_ftpl = data.copy()
    sim_ftpl = frbus.init_trac(init_start, end_period, sim_ftpl)
    sim_ftpl.loc[init_start, "trptx"] = 0.137738886015378
    sim_ftpl.loc[init_start, "trcit"] = 0.344613522605853
    frbus_to_expected_mapping = { 
        # Core economic indicators
        'ugfdbtp': 'debt_to_gdp',
        'hggdp': 'gdp_growth',           # GDP growth rate
        'pcpi': 'inflation',             # Inflation (CPI)
        'lur': 'unemployment',           # Unemployment rate
        'xgdp': 'real_gdp',             # Real GDP
        'xgdpn': 'nominal_gdp',         # Nominal GDP
        # Tax and fiscal variables
        'tpn': 'personal_tax',          # Personal tax
        'tcin': 'corporate_tax',        # Corporate tax
        'trptx': 'personal_tax_rates',  # Personal tax rates
        'trcit': 'corporate_tax_rates', # Corporate tax rates
        # Trade variables
        'exn': 'exports',               # Exports
        'emn': 'imports',               # Imports
        'fcbn': 'trade_balance',        # Trade balance
        'fynin': 'net_foreign_investment_income',  # Net foreign investment income
        # Government and debt variables
        'gfdbtn': 'debt_to_gdp',        # Debt to GDP ratio
        'rff': 'interest_rate',         # Federal funds rate
        'gtrt': 'transfer_payments_ratio',  # Transfer payments ratio
        'egfe': 'federal_expenditures', # Federal expenditures
        'gtn': 'government_transfer_payments',  # Government transfer payments
        'gfsrpn': 'federal_surplus',    # Federal surplus
    }

    # Run simulation
    for current_quarter in pd.date_range(start=start, end=end_1, freq='Q'):
        try:
            array_values = []
            q = (current_quarter.month - 1) // 3 + 1
            quarter_str = f"{current_quarter.year}q{q}".lower()  
            previous_quarter = pd.Period(quarter_str) - 1 

            print(f'\nBefore solve Personal Income Tax Rates: Previous Quarter {previous_quarter}: {sim_ftpl.loc[previous_quarter, "trptx"]} | Quarter {quarter_str} {sim_ftpl.loc[quarter_str, "trptx"]}')
            print(f'\nBefore solve Corporate Tax Rates: Previous Quarter {previous_quarter}: {sim_ftpl.loc[previous_quarter, "trcit"]} | Quarter {quarter_str} {sim_ftpl.loc[quarter_str, "trcit"]}')
            
            print('='*100) 
            # for var in frbus_to_expected_mapping.keys():
            #     if var == 'quarter':
            #         continue
            #     array_values.append(f'Quarter: {quarter_str} | {var}: {sim_ftpl.loc[quarter_str, var]}') 
            # print(', '.join(array_values))

            sim_ftpl = frbus.solve(previous_quarter, quarter_str, sim_ftpl)

            # array_values = []
            # print('\nAfter solve: \n')
            # print('='*100)
            # for var in frbus_to_expected_mapping.keys():
            #     if var == 'quarter':
            #         continue
            #     array_values.append(f'Quarter: {quarter_str} | {var}: {sim_ftpl.loc[quarter_str, var]}') 
            # print(', '.join(array_values))

            
            print(f'\nAfter solve Personal Income Tax Rates: Previous Quarter {previous_quarter}: {sim_ftpl.loc[previous_quarter, "trptx"]} | Quarter {quarter_str} {sim_ftpl.loc[quarter_str, "trptx"]}')
            print(f'\nAfter solve Corporate Tax Rates: Previous Quarter {previous_quarter}: {sim_ftpl.loc[previous_quarter, "trcit"]} | Quarter {quarter_str} {sim_ftpl.loc[quarter_str, "trcit"]}')
            print('='*100) 
        except Exception as e:
            print(f"Error: {e}")
            print(f"Quarter: {quarter_str}")
            print(f"Previous quarter: {previous_quarter}")
            for var in frbus_to_expected_mapping.keys():
                if var == 'quarter':
                    continue
                print(f"{var}: {sim_ftpl.loc[previous_quarter:quarter_str, var]}")
            raise e

    ftpl_data = sim_ftpl.loc[start:end, :] 
    ftpl_data.to_csv('sim_ftpl.csv', index=True)  
    return ftpl_data

def non_ftpl_simulation():

    # FRB/US Baseline
    # Load data
    data_non_ftpl = load_data("../data/LONGBASE.TXT")

    # Load model
    frbus_original = Frbus("../models/model.xml")

    # Specify dates
    start = pandas.Period("2000Q1")
    end =  pandas.Period("2024Q4")

    # Standard configuration, use surplus ratio targeting
    data_non_ftpl.loc[start:end, "dfpdbt"] = 0
    data_non_ftpl.loc[start:end, "dfpsrp"] = 0

    # Use non-inertial Taylor rule
    data_non_ftpl.loc[start:end, "dmptay"] = 1
    data_non_ftpl.loc[start:end, "dmpintay"] = 0

    # Enable thresholds
    data_non_ftpl.loc[start:end, "dmptrsh"] = 1
    # Arbitrary threshold values
    data_non_ftpl.loc[start:end, "lurtrsh"] = 6.0
    data_non_ftpl.loc[start:end, "pitrsh"] = 3.0


    # Solve to baseline with adds
    with_adds_non_ftpl = frbus_original.init_trac(start, end, data_non_ftpl)

    # # Scenario based on 2021Q3 Survey of Professional Forecasters
    # with_adds.loc[start:end, "lurnat"] = 3.78

    # Run mcontrol
    sim_non_ftpl = frbus_original.solve(start, end, with_adds_non_ftpl)

    non_ftpl_data = sim_non_ftpl.loc[start:end, :] 
    non_ftpl_data.to_csv('sim_non_ftpl.csv', index=True) 
    return non_ftpl_data

def calculate_error_metrics(ftpl_data, non_ftpl_data):
    """
    Calculate MSE and MAE between FTPL and non-FTPL simulation data for key variables.
    """
    import numpy as np
    
    print("\n" + "="*60)
    print("ERROR METRICS: FTPL vs Non-FTPL Simulations")
    print("="*60)
    
    # Key variables to compare
    key_variables = [
        'picxfe',      # Core inflation
        'rff',         # Federal funds rate
        'xgap2',       # Output gap
        'gfsrpn',      # Federal surplus
        'gfdbtnp',     # Government debt
        'xgdp',        # Real GDP
        'lur',         # Unemployment rate
        'pieci',       # Wage inflation
        'rrff',        # Real federal funds rate
        'ugfdbtp',     # Government debt ratio
        'egfe',       # Government expenditures
        'egfet',       # Government expenditures trend,
        'trcit',       # Corporate tax rates
        'trptx',       # Personal tax rates
    ]
    
    results = {}
    
    for var in key_variables:
        if var in ftpl_data.columns and var in non_ftpl_data.columns:
            # Get the data (ensure same length)
            min_len = min(len(ftpl_data[var]), len(non_ftpl_data[var]))
            ftpl_values = ftpl_data[var].iloc[:min_len].values
            non_ftpl_values = non_ftpl_data[var].iloc[:min_len].values
            
            # Remove any NaN values
            mask = ~(np.isnan(ftpl_values) | np.isnan(non_ftpl_values))
            ftpl_clean = ftpl_values[mask]
            non_ftpl_clean = non_ftpl_values[mask]
            
            if len(ftpl_clean) > 0:
                # Calculate errors
                diff = ftpl_clean - non_ftpl_clean
                mse = np.mean(diff**2)
                mae = np.mean(np.abs(diff))
                rmse = np.sqrt(mse)
                
                # Calculate relative errors (as percentage of non-FTPL mean)
                non_ftpl_mean = np.mean(np.abs(non_ftpl_clean))
                if non_ftpl_mean > 0:
                    relative_mae = (mae / non_ftpl_mean) * 100
                    relative_rmse = (rmse / non_ftpl_mean) * 100
                else:
                    relative_mae = float('inf')
                    relative_rmse = float('inf')
                
                results[var] = {
                    'MSE': mse,
                    'MAE': mae, 
                    'RMSE': rmse,
                    'Relative_MAE_pct': relative_mae,
                    'Relative_RMSE_pct': relative_rmse,
                    'N_obs': len(ftpl_clean),
                    'FTPL_mean': np.mean(ftpl_clean),
                    'Non_FTPL_mean': np.mean(non_ftpl_clean)
                }
                
                print(f"\n{var.upper()}:")
                print(f"  MSE:           {mse:.6f}")
                print(f"  MAE:           {mae:.6f}")
                print(f"  RMSE:          {rmse:.6f}")
                print(f"  Relative MAE:  {relative_mae:.2f}%")
                print(f"  Relative RMSE: {relative_rmse:.2f}%")
                print(f"  FTPL Mean:     {np.mean(ftpl_clean):.4f}")
                print(f"  Non-FTPL Mean: {np.mean(non_ftpl_clean):.4f}")
                print(f"  Observations:  {len(ftpl_clean)}")
        else:
            print(f"\nWarning: Variable '{var}' not found in one or both datasets")
    
    # Summary statistics
    if results:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Variables with highest differences
        mae_sorted = sorted(results.items(), key=lambda x: x[1]['Relative_MAE_pct'], reverse=True)
        print("\nVariables with highest relative MAE (%):")
        for var, metrics in mae_sorted[:100]:
            print(f"  {var:12}: {metrics['Relative_MAE_pct']:8.2f}%")
        
        # Overall statistics
        all_mae = [v['MAE'] for v in results.values()]
        all_mse = [v['MSE'] for v in results.values()]
        all_rel_mae = [v['Relative_MAE_pct'] for v in results.values() if v['Relative_MAE_pct'] != float('inf')]
        
        print(f"\nOverall Statistics:")
        print(f"  Average MAE:          {np.mean(all_mae):.6f}")
        print(f"  Average MSE:          {np.mean(all_mse):.6f}")
        print(f"  Average Relative MAE: {np.mean(all_rel_mae):.2f}%")
        print(f"  Max Relative MAE:     {np.max(all_rel_mae):.2f}%")
        print(f"  Min Relative MAE:     {np.min(all_rel_mae):.2f}%")
        
        # Save detailed results
        results_df = pd.DataFrame(results).T
        results_df.to_csv('error_metrics_ftpl_vs_non_ftpl.csv', index=True)
        print(f"\nDetailed results saved to: error_metrics_ftpl_vs_non_ftpl.csv")
    
    print("="*60)
def extract_and_combine_simulation_data():
    """
    Extract specified columns from FTPL and non-FTPL CSV files and combine them.
    """
        

    ftpl_data = ftpl_simulation()
    non_ftpl_data = non_ftpl_simulation()

    # Calculate MSE and MAE between FTPL and non-FTPL data
    calculate_error_metrics(ftpl_data, non_ftpl_data)
         
    # View results

    # Define the mapping from FRB/US variable names to expected column names
    # Based on the mapping found in streamlit-app.py lines 1365-1390
    frbus_to_expected_mapping = {
        # Quarter column - assuming it exists or we'll create from index
        'quarter': 'quarter',
        # Core economic indicators
        'hggdp': 'gdp_growth',           # GDP growth rate
        'pcpi': 'inflation',             # Inflation (CPI)
        'lur': 'unemployment',           # Unemployment rate
        'xgdp': 'real_gdp',             # Real GDP
        'xgdpn': 'nominal_gdp',         # Nominal GDP
        # Tax and fiscal variables
        'tpn': 'personal_tax',          # Personal tax
        'tcin': 'corporate_tax',        # Corporate tax
        'trptx': 'personal_tax_rates',  # Personal tax rates
        'trcit': 'corporate_tax_rates', # Corporate tax rates
        # Trade variables
        'exn': 'exports',               # Exports
        'emn': 'imports',               # Imports
        'fcbn': 'trade_balance',        # Trade balance
        'fynin': 'net_foreign_investment_income',  # Net foreign investment income
        # Government and debt variables
        'gfdbtn': 'debt_to_gdp',        # Debt to GDP ratio
        'rff': 'interest_rate',         # Federal funds rate
        'gtrt': 'transfer_payments_ratio',  # Transfer payments ratio
        'egfe': 'federal_expenditures', # Federal expenditures
        'gtn': 'government_transfer_payments',  # Government transfer payments
        'gfsrpn': 'federal_surplus',    # Federal surplus
    }
    output_file = '/home/dominus/RL_FRBUS/RL_FRBUS_Frontend/combined_sim_data_ftpl_vs_non_ftpl.csv'
    
    print(f"FTPL data shape: {ftpl_data.shape}")
    print(f"Non-FTPL data shape: {non_ftpl_data.shape}") 
    
    # Extract only the columns we need from the mapping
    def extract_columns(df, simulation_type_name):
        """Extract and rename columns according to the mapping"""
        extracted_data = {} 
        # Create quarter column if it doesn't exist (use index) 
        if 'quarter' not in df.columns and len(df) > 0:
            # Assume data starts from a specific quarter, we'll use a placeholder
            # You might need to adjust this based on the actual data structure
            extracted_data['quarter'] = [df.index[i] for i in range(len(df))]
        
        # Extract the columns that exist in the dataframe
        for frbus_col, expected_col in frbus_to_expected_mapping.items():
            if frbus_col in df.columns:
                extracted_data[expected_col] = df[frbus_col].values
            else:
                if frbus_col == 'quarter':
                    continue
                print(f"Warning: Column '{frbus_col}' not found in {simulation_type_name} data")
                # Fill with NaN for missing columns
                extracted_data[expected_col] = [None] * len(df)
        
        # Add simulation type
        extracted_data['simulation_type'] = [simulation_type_name] * len(df)
        
        return pd.DataFrame(extracted_data)
    
    # Extract data from both datasets
    print("Extracting FTPL data...")
    ftpl_extracted = extract_columns(ftpl_data, "FRB/US FTPL")
    
    print("Extracting non-FTPL data...")
    non_ftpl_extracted = extract_columns(non_ftpl_data, "FRB/US")
    
    # Combine the datasets
    print("Combining datasets...")
    combined_data = pd.concat([ftpl_extracted, non_ftpl_extracted], ignore_index=True)
    
    # Save to CSV
    print(f"Saving combined data to combined_sim_data_ftpl_vs_non_ftpl.csv...")
    combined_data.to_csv(output_file, index=False)
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Combined data columns: {list(combined_data.columns)}")
    print(f"Simulation types: {combined_data['simulation_type'].unique()}")
    # print("\nFirst few rows:")
    # print(combined_data)
    
    print(f"\nData successfully saved to: combined_sim_data_ftpl_vs_non_ftpl.csv")

if __name__ == "__main__":
    extract_and_combine_simulation_data()