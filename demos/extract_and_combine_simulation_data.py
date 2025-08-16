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


def extract_and_combine_simulation_data():
    """
    Extract specified columns from FTPL and non-FTPL CSV files and combine them.
    """
        

    # FRB/US FTPL
    # Load data
    data = load_data("../data/LONGBASE.TXT")

    # Load model
    frbus = Frbus("../models/model_FRBUS_FTPL.xml")

    # Specify dates
    start = pandas.Period("2000Q1")
    end =  pandas.Period("2024Q1")

    data.loc[:, "dftpl"] = 1

    # Standard configuration, use surplus ratio targeting
    data.loc[start:end, "dfpdbt"] = 0
    data.loc[start:end, "dfpsrp"] = 1

    # Use non-inertial Taylor rule
    data.loc[start:end, "dmptay"] = 1
    data.loc[start:end, "dmpintay"] = 0

    # Enable thresholds
    data.loc[start:end, "dmptrsh"] = 1
    # Arbitrary threshold values
    data.loc[start:end, "lurtrsh"] = 6.0
    data.loc[start:end, "pitrsh"] = 3.0


    # Solve to baseline with adds
    with_adds = frbus.init_trac(start, end, data)

    # # Scenario based on 2021Q3 Survey of Professional Forecasters
    # with_adds.loc[start:end, "lurnat"] = 3.78

    # Run mcontrol
    sim_ftpl = frbus.solve(start, end, with_adds)

    ftpl_data = sim_ftpl.loc[start:end, :] 
    ftpl_data.to_csv('sim_ftpl.csv', index=False)  
    

        
    # FRB/US Baseline
    # Load data
    data = load_data("../data/LONGBASE.TXT")

    # Load model
    frbus = Frbus("../models/model.xml")

    # Specify dates
    start = pandas.Period("2000Q1")
    end =  pandas.Period("2024Q1")

    # Standard configuration, use surplus ratio targeting
    data.loc[start:end, "dfpdbt"] = 0
    data.loc[start:end, "dfpsrp"] = 1

    # Use non-inertial Taylor rule
    data.loc[start:end, "dmptay"] = 1
    data.loc[start:end, "dmpintay"] = 0

    # Enable thresholds
    data.loc[start:end, "dmptrsh"] = 1
    # Arbitrary threshold values
    data.loc[start:end, "lurtrsh"] = 6.0
    data.loc[start:end, "pitrsh"] = 3.0


    # Solve to baseline with adds
    with_adds = frbus.init_trac(start, end, data)

    # # Scenario based on 2021Q3 Survey of Professional Forecasters
    # with_adds.loc[start:end, "lurnat"] = 3.78

    # Run mcontrol
    sim_non_ftpl = frbus.solve(start, end, with_adds)

    non_ftpl_data = sim_non_ftpl.loc[start:end, :] 
    non_ftpl_data.to_csv('sim_non_ftpl.csv', index=False)  
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
        # Keep pcpi as well for the additional column
        'pcpi': 'pcpi'
    }
    output_file = '/home/dominus/RL_FRBUS/RL_FRBUS_Frontend/combined_sim_data_ftpl_vs_non_ftpl.csv'
    
    print(f"FTPL data shape: {ftpl_data.shape}")
    print(f"Non-FTPL data shape: {non_ftpl_data.shape}")
    print(f"FTPL columns: {list(ftpl_data.columns)}")
    print(f"Non-FTPL columns: {list(non_ftpl_data.columns)}")
    
    # Extract only the columns we need from the mapping
    def extract_columns(df, simulation_type_name):
        """Extract and rename columns according to the mapping"""
        extracted_data = {}
        print(f"Columns in df: {df.columns}")
        # Create quarter column if it doesn't exist (use index)
        print(f"First element of df: {df.loc[pd.Period('2000Q1')]}")
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
    print("\nFirst few rows:")
    print(combined_data.head())
    
    print(f"\nData successfully saved to: combined_sim_data_ftpl_vs_non_ftpl.csv")

if __name__ == "__main__":
    extract_and_combine_simulation_data()