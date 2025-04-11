# from streamer import EconomicStream
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from websocket import create_connection
from websocket._app import WebSocketApp
import threading
import queue
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
from plotly.subplots import make_subplots  # Add this import
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

# Define muted red palette
MUTED_REDS = {
    'bright': '#c44f4f',  # Muted version of FF4B4B
    'medium': '#9A1A3C',  # Muted version of B91D47
    'dark': '#d62728',    # Muted version of 871B2D
    'darkest': '#3A1919',  # Muted version of 441D1D
    'light': '#F5AE98', # light
    'lightest': '#F39C7F', # light orange
}
 
def mean_absolute_error(df, df_without_tariff, df_without_rl, title, year_range=None, small_value=False, dark_mode=False):
    # Helper function to calculate mean absolute error
    
    # Filter by year range if provided
    if year_range is not None:
        min_year, max_year = year_range
        
        # Extract year from quarter string (e.g., "2020Q1" -> 2020)
        def extract_year(quarter_str):
            # Handle both formats: "2020Q1" and "2020q1"
            return int(quarter_str.split('q')[0].split('Q')[0])
        
        # Create year columns for filtering
        df_years = df['quarter'].apply(extract_year)
        df_without_tariff_years = df_without_tariff['quarter'].apply(extract_year)
        df_without_rl_years = df_without_rl['quarter'].apply(extract_year)
        
        # Filter dataframes
        df_filtered = df[df_years.between(min_year, max_year)].reset_index(drop=True)
        df_without_tariff_filtered = df_without_tariff[df_without_tariff_years.between(min_year, max_year)].reset_index(drop=True)
        df_without_rl_filtered = df_without_rl[df_without_rl_years.between(min_year, max_year)].reset_index(drop=True)
    else:
        # Use all data if no year range specified
        df_filtered = df
        df_without_tariff_filtered = df_without_tariff
        df_without_rl_filtered = df_without_rl
    
    def calculate_mae(df, df_without_tariff, df_without_rl):
        mae_rl = []
        mae_tariff = [] 

        mae_rl_gdp_growth = np.abs(df['gdp_growth'] - df_without_tariff['gdp_growth']) 
        mae_rl_inflation = np.abs(df['inflation'] - df_without_tariff['inflation']) 
        mae_rl_unemployment = np.abs(df['unemployment'] - df_without_tariff['unemployment']) 
        mae_rl_real_gdp = np.abs(df['real_gdp'] - df_without_tariff['real_gdp']) 
        mae_rl_nominal_gdp = np.abs(df['nominal_gdp'] - df_without_tariff['nominal_gdp']) 
        mae_rl_personal_tax = np.abs(df['personal_tax'] - df_without_tariff['personal_tax']) 
        mae_rl_corporate_tax = np.abs(df['corporate_tax'] - df_without_tariff['corporate_tax']) 
        mae_rl_exports = np.abs(df['exports'] - df_without_tariff['exports']) 
        mae_rl_imports = np.abs(df['imports'] - df_without_tariff['imports'])  
        mae_rl_debt_to_gdp = np.abs(df['debt_to_gdp'] - df_without_tariff['debt_to_gdp']) 
        mae_rl_interest_rate = np.abs(df['interest_rate'] - df_without_tariff['interest_rate']) 
        mae_rl_pcpi = np.abs(df['pcpi'] - df_without_tariff['pcpi']) 
        mae_rl_transfer_payments_ratio = np.abs(df['transfer_payments_ratio'] - df_without_tariff['transfer_payments_ratio']) 
        mae_rl_federal_expenditures = np.abs(df['federal_expenditures'] - df_without_tariff['federal_expenditures']) 
        mae_rl_personal_tax_rates = np.abs(df['personal_tax_rates'] - df_without_tariff['personal_tax_rates']) 
        mae_rl_corporate_tax_rates = np.abs(df['corporate_tax_rates'] - df_without_tariff['corporate_tax_rates']) 
        mae_rl_government_transfer_payments = np.abs(df['government_transfer_payments'] - df_without_tariff['government_transfer_payments']) 
        mae_rl_federal_surplus = np.abs(df['federal_surplus'] - df_without_tariff['federal_surplus']) 

        mae_tariff_gdp_growth = np.abs(df_without_rl['gdp_growth'] - df_without_tariff['gdp_growth']) 
        mae_tariff_inflation = np.abs(df_without_rl['inflation'] - df_without_tariff['inflation']) 
        mae_tariff_unemployment = np.abs(df_without_rl['unemployment'] - df_without_tariff['unemployment']) 
        mae_tariff_real_gdp = np.abs(df_without_rl['real_gdp'] - df_without_tariff['real_gdp']) 
        mae_tariff_nominal_gdp = np.abs(df_without_rl['nominal_gdp'] - df_without_tariff['nominal_gdp']) 
        mae_tariff_personal_tax = np.abs(df_without_rl['personal_tax'] - df_without_tariff['personal_tax']) 
        mae_tariff_corporate_tax = np.abs(df_without_rl['corporate_tax'] - df_without_tariff['corporate_tax']) 
        mae_tariff_exports = np.abs(df_without_rl['exports'] - df_without_tariff['exports'])  
        mae_tariff_imports = np.abs(df_without_rl['imports'] - df_without_tariff['imports']) 
        mae_tariff_debt_to_gdp = np.abs(df_without_rl['debt_to_gdp'] - df_without_tariff['debt_to_gdp']) 
        mae_tariff_interest_rate = np.abs(df_without_rl['interest_rate'] - df_without_tariff['interest_rate']) 
        mae_tariff_pcpi = np.abs(df_without_rl['pcpi'] - df_without_tariff['pcpi']) 
        mae_tariff_transfer_payments_ratio = np.abs(df_without_rl['transfer_payments_ratio'] - df_without_tariff['transfer_payments_ratio']) 
        mae_tariff_federal_expenditures = np.abs(df_without_rl['federal_expenditures'] - df_without_tariff['federal_expenditures']) 
        mae_tariff_personal_tax_rates = np.abs(df_without_rl['personal_tax_rates'] - df_without_tariff['personal_tax_rates']) 
        mae_tariff_corporate_tax_rates = np.abs(df_without_rl['corporate_tax_rates'] - df_without_tariff['corporate_tax_rates']) 
        mae_tariff_government_transfer_payments = np.abs(df_without_rl['government_transfer_payments'] - df_without_tariff['government_transfer_payments']) 
        mae_tariff_federal_surplus = np.abs(df_without_rl['federal_surplus'] - df_without_tariff['federal_surplus']) 
        
        
        mae_tariff = [mae_tariff_gdp_growth, mae_tariff_inflation, mae_tariff_unemployment, mae_tariff_real_gdp, mae_tariff_nominal_gdp, mae_tariff_personal_tax, mae_tariff_corporate_tax, mae_tariff_exports, mae_tariff_imports, mae_tariff_debt_to_gdp, mae_tariff_interest_rate, mae_tariff_pcpi, mae_tariff_transfer_payments_ratio, mae_tariff_federal_expenditures, mae_tariff_personal_tax_rates, mae_tariff_corporate_tax_rates, mae_tariff_government_transfer_payments, mae_tariff_federal_surplus]
        mae_rl = [mae_rl_gdp_growth, mae_rl_inflation, mae_rl_unemployment, mae_rl_real_gdp, mae_rl_nominal_gdp, mae_rl_personal_tax, mae_rl_corporate_tax, mae_rl_exports, mae_rl_imports, mae_rl_debt_to_gdp, mae_rl_interest_rate, mae_rl_pcpi, mae_rl_transfer_payments_ratio, mae_rl_federal_expenditures, mae_rl_personal_tax_rates, mae_rl_corporate_tax_rates, mae_rl_government_transfer_payments, mae_rl_federal_surplus]
        return mae_tariff, mae_rl
    
    # Use filtered data for MAE calculation
    mae_tariff, mae_rl = calculate_mae(df_filtered, df_without_tariff_filtered, df_without_rl_filtered)
    
    # Create dataframe for visualization
    metric_names = [
        'GDP Growth', 'Inflation', 'Unemployment', 'Real GDP', 'Nominal GDP', 
        'Personal Tax', 'Corporate Tax', 'Exports', 'Imports', 'Debt to GDP',
        'Interest Rate', 'PCPI', 'Transfer Payments Ratio', 'Federal Expenditures',
        'Personal Tax Rates', 'Corporate Tax Rates', 'Government Transfer Payments', 
        'Federal Surplus'
    ]
    
    # Calculate mean MAE for each series
    mae_rl_means = [round(np.mean(series), 2)  for series in mae_rl]
    mae_tariff_means = [round(np.mean(series), 2)  for series in mae_tariff]
    if not small_value:
        mae_rl_valid_indices = [i for i, series in enumerate(mae_rl_means) if series > 1.0 and mae_tariff_means[i] > 1.0]
        metric_names = [metric_names[i] for i in mae_rl_valid_indices]
        mae_rl_means = [mae_rl_means[i] for i in mae_rl_valid_indices]
        mae_tariff_means = [mae_tariff_means[i] for i in mae_rl_valid_indices] 
    if small_value:
        mae_rl_valid_indices = [i for i, series in enumerate(mae_rl_means) if series <= 1.0 and mae_tariff_means[i] <= 1.0]
        metric_names = [metric_names[i] for i in mae_rl_valid_indices]
        mae_rl_means = [mae_rl_means[i] for i in mae_rl_valid_indices]
        mae_tariff_means = [mae_tariff_means[i] for i in mae_rl_valid_indices] 
    # Create dataframe for plotting
    mae_df = pd.DataFrame({
        'Metric': metric_names,
        'RL - FRBUS vs Historical Data': mae_rl_means,
        'FRB/US model vs Historical Data': mae_tariff_means
    })
    
    # Add year range to title if provided
    title_with_year = title
    if year_range:
        title_with_year = f"{title} ({min_year}-{max_year})"
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=mae_df['Metric'],
        y=mae_df['RL - FRBUS vs Historical Data'],
        name='RL - FRBUS vs Historical Data',
        marker_color=MUTED_REDS['dark']
    ))
    
    fig.add_trace(go.Bar(
        x=mae_df['Metric'],
        y=mae_df['FRB/US model vs Historical Data'],
        name='FRB/US model vs Historical Data',
        marker_color=MUTED_REDS['light']
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Mean Absolute Error Comparison: {title_with_year}',
        xaxis_title='Economic Metrics',
        yaxis_title='Mean Absolute Error',
        # Font for y axis title
        yaxis_title_font=dict(size=20, color='black' if not dark_mode else 'white'),
        # Font for x axis title
        xaxis_title_font=dict(size=20, color='black' if not dark_mode else 'white'),
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20)
        )
    )
    
    
    # Improve readability of x-axis labels
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=18, color='black' if not dark_mode else 'white'),  # Increased from 10 to 18
        gridcolor='black' if not dark_mode else 'white',
        zerolinecolor='#303030'
    )
    
    fig.update_yaxes(
        tickfont=dict(size=18, color='black' if not dark_mode else 'white'),  # Added explicit font size for y-axis
        gridcolor='black' if not dark_mode else 'white',
        zerolinecolor='#303030'
    )
    
    return fig  

def root_mean_square_deviation(df, df_without_tariff, df_without_rl, title, year_range=None, small_value=False, dark_mode=False):
    
    # Filter by year range if provided
    if year_range is not None:
        min_year, max_year = year_range
        
        # Extract year from quarter string (e.g., "2020Q1" -> 2020)
        def extract_year(quarter_str):
            # Handle both formats: "2020Q1" and "2020q1"
            return int(quarter_str.split('q')[0].split('Q')[0])
        
        # Create year columns for filtering
        df_years = df['quarter'].apply(extract_year)
        df_without_tariff_years = df_without_tariff['quarter'].apply(extract_year)
        df_without_rl_years = df_without_rl['quarter'].apply(extract_year)
        
        # Filter dataframes
        df_filtered = df[df_years.between(min_year, max_year)].reset_index(drop=True)
        df_without_tariff_filtered = df_without_tariff[df_without_tariff_years.between(min_year, max_year)].reset_index(drop=True)
        df_without_rl_filtered = df_without_rl[df_without_rl_years.between(min_year, max_year)].reset_index(drop=True)
    else:
        # Use all data if no year range specified
        df_filtered = df
        df_without_tariff_filtered = df_without_tariff
        df_without_rl_filtered = df_without_rl
    
    def calculate_rmse(df, df_without_tariff, df_without_rl):
        rmse_rl = []
        rmse_tariff = [] 

        rmse_rl_gdp_growth = (np.abs(df['gdp_growth'] - df_without_tariff['gdp_growth'])**2)
        rmse_rl_inflation = (np.abs(df['inflation'] - df_without_tariff['inflation'])**2)
        rmse_rl_unemployment = (np.abs(df['unemployment'] - df_without_tariff['unemployment'])**2)
        rmse_rl_real_gdp = (np.abs(df['real_gdp'] - df_without_tariff['real_gdp'])**2)
        rmse_rl_nominal_gdp = (np.abs(df['nominal_gdp'] - df_without_tariff['nominal_gdp'])**2)
        rmse_rl_personal_tax = (np.abs(df['personal_tax'] - df_without_tariff['personal_tax'])**2)
        rmse_rl_corporate_tax = (np.abs(df['corporate_tax'] - df_without_tariff['corporate_tax'])**2)
        rmse_rl_exports = (np.abs(df['exports'] - df_without_tariff['exports'])**2)
        rmse_rl_imports = (np.abs(df['imports'] - df_without_tariff['imports'])**2)
        rmse_rl_debt_to_gdp = (np.abs(df['debt_to_gdp'] - df_without_tariff['debt_to_gdp'])**2)
        rmse_rl_interest_rate = (np.abs(df['interest_rate'] - df_without_tariff['interest_rate'])**2)
        rmse_rl_pcpi = (np.abs(df['pcpi'] - df_without_tariff['pcpi'])**2)
        rmse_rl_transfer_payments_ratio = (np.abs(df['transfer_payments_ratio'] - df_without_tariff['transfer_payments_ratio'])**2)
        rmse_rl_federal_expenditures = (np.abs(df['federal_expenditures'] - df_without_tariff['federal_expenditures'])**2)
        rmse_rl_personal_tax_rates = (np.abs(df['personal_tax_rates'] - df_without_tariff['personal_tax_rates'])**2)
        rmse_rl_corporate_tax_rates = (np.abs(df['corporate_tax_rates'] - df_without_tariff['corporate_tax_rates'])**2)
        rmse_rl_government_transfer_payments = (np.abs(df['government_transfer_payments'] - df_without_tariff['government_transfer_payments'])**2)
        rmse_rl_federal_surplus = (np.abs(df['federal_surplus'] - df_without_tariff['federal_surplus'])**2)

        rmse_tariff_gdp_growth = (np.abs(df_without_rl['gdp_growth'] - df_without_tariff['gdp_growth'])**2)
        rmse_tariff_inflation = (np.abs(df_without_rl['inflation'] - df_without_tariff['inflation'])**2)
        rmse_tariff_unemployment = (np.abs(df_without_rl['unemployment'] - df_without_tariff['unemployment'])**2)
        rmse_tariff_real_gdp = (np.abs(df_without_rl['real_gdp'] - df_without_tariff['real_gdp'])**2)
        rmse_tariff_nominal_gdp = (np.abs(df_without_rl['nominal_gdp'] - df_without_tariff['nominal_gdp'])**2)
        rmse_tariff_personal_tax = (np.abs(df_without_rl['personal_tax'] - df_without_tariff['personal_tax'])**2)
        rmse_tariff_corporate_tax = (np.abs(df_without_rl['corporate_tax'] - df_without_tariff['corporate_tax'])**2)
        rmse_tariff_exports = (np.abs(df_without_rl['exports'] - df_without_tariff['exports'])**2)
        rmse_tariff_imports = (np.abs(df_without_rl['imports'] - df_without_tariff['imports'])**2)
        rmse_tariff_debt_to_gdp = (np.abs(df_without_rl['debt_to_gdp'] - df_without_tariff['debt_to_gdp'])**2)
        rmse_tariff_interest_rate = (np.abs(df_without_rl['interest_rate'] - df_without_tariff['interest_rate'])**2)
        rmse_tariff_pcpi = (np.abs(df_without_rl['pcpi'] - df_without_tariff['pcpi'])**2)
        rmse_tariff_transfer_payments_ratio = (np.abs(df_without_rl['transfer_payments_ratio'] - df_without_tariff['transfer_payments_ratio'])**2)
        rmse_tariff_federal_expenditures = (np.abs(df_without_rl['federal_expenditures'] - df_without_tariff['federal_expenditures'])**2)
        rmse_tariff_personal_tax_rates = (np.abs(df_without_rl['personal_tax_rates'] - df_without_tariff['personal_tax_rates'])**2)
        rmse_tariff_corporate_tax_rates = (np.abs(df_without_rl['corporate_tax_rates'] - df_without_tariff['corporate_tax_rates'])**2)
        rmse_tariff_government_transfer_payments = (np.abs(df_without_rl['government_transfer_payments'] - df_without_tariff['government_transfer_payments'])**2)
        rmse_tariff_federal_surplus = (np.abs(df_without_rl['federal_surplus'] - df_without_tariff['federal_surplus'])**2)
        
        
        rmse_tariff = [rmse_tariff_gdp_growth, rmse_tariff_inflation, rmse_tariff_unemployment, rmse_tariff_real_gdp, rmse_tariff_nominal_gdp, rmse_tariff_personal_tax, rmse_tariff_corporate_tax, rmse_tariff_exports, rmse_tariff_imports, rmse_tariff_debt_to_gdp, rmse_tariff_interest_rate, rmse_tariff_pcpi, rmse_tariff_transfer_payments_ratio, rmse_tariff_federal_expenditures, rmse_tariff_personal_tax_rates, rmse_tariff_corporate_tax_rates, rmse_tariff_government_transfer_payments, rmse_tariff_federal_surplus]
        rmse_rl = [rmse_rl_gdp_growth, rmse_rl_inflation, rmse_rl_unemployment, rmse_rl_real_gdp, rmse_rl_nominal_gdp, rmse_rl_personal_tax, rmse_rl_corporate_tax, rmse_rl_exports, rmse_rl_imports, rmse_rl_debt_to_gdp, rmse_rl_interest_rate, rmse_rl_pcpi, rmse_rl_transfer_payments_ratio, rmse_rl_federal_expenditures, rmse_rl_personal_tax_rates, rmse_rl_corporate_tax_rates, rmse_rl_government_transfer_payments, rmse_rl_federal_surplus]
        return rmse_tariff, rmse_rl
    
    # Use filtered data for MAE calculation
    rmse_tariff, rmse_rl = calculate_rmse(df_filtered, df_without_tariff_filtered, df_without_rl_filtered)
    
    # Create dataframe for visualization
    metric_names = [
        'GDP Growth', 'Inflation', 'Unemployment', 'Real GDP', 'Nominal GDP', 
        'Personal Tax', 'Corporate Tax', 'Exports', 'Imports', 'Debt to GDP',
        'Interest Rate', 'PCPI', 'Transfer Payments Ratio', 'Federal Expenditures',
        'Personal Tax Rates', 'Corporate Tax Rates', 'Government Transfer Payments', 
        'Federal Surplus'
    ]
    
    # Calculate mean MAE for each series
    rmse_rl_means = [round(np.sqrt(np.mean(series)), 2) for series in rmse_rl]
    rmse_tariff_means = [round(np.sqrt(np.mean(series)), 2) for series in rmse_tariff]
    if not small_value:
        rmse_rl_valid_indices = [i for i, series in enumerate(rmse_rl_means) if series > 1.0 and rmse_tariff_means[i] > 1.0]
        metric_names = [metric_names[i] for i in rmse_rl_valid_indices]
        rmse_rl_means = [rmse_rl_means[i] for i in rmse_rl_valid_indices]
        rmse_tariff_means = [rmse_tariff_means[i] for i in rmse_rl_valid_indices] 

    if small_value:
        rmse_rl_valid_indices = [i for i, series in enumerate(rmse_rl_means) if series <= 1.0 and rmse_tariff_means[i] <= 1.0]
        metric_names = [metric_names[i] for i in rmse_rl_valid_indices]
        rmse_rl_means = [rmse_rl_means[i] for i in rmse_rl_valid_indices]
        rmse_tariff_means = [rmse_tariff_means[i] for i in rmse_rl_valid_indices] 
    # Create dataframe for plotting
    rmse_df = pd.DataFrame({
        'Metric': metric_names,
        'RL - FRBUS vs Historical Data': rmse_rl_means,
        'FRB/US model vs Historical Data': rmse_tariff_means
    })
    
    # Add year range to title if provided
    title_with_year = title
    if year_range:
        title_with_year = f"{title} ({min_year}-{max_year})"
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=rmse_df['Metric'],
        y=rmse_df['RL - FRBUS vs Historical Data'],
        name='RL - FRBUS vs Historical Data',
        marker_color=MUTED_REDS['dark']
    ))
    
    fig.add_trace(go.Bar(
        x=rmse_df['Metric'],
        y=rmse_df['FRB/US model vs Historical Data'],
        name='FRB/US model vs Historical Data',
        marker_color=MUTED_REDS['light']
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Root Mean Square Error Comparison: {title_with_year}',
        xaxis_title='Economic Metrics',
        yaxis_title='Root Mean Square Error',
        # Font for y axis title
        yaxis_title_font=dict(size=20, color='black' if not dark_mode else 'white'),
        # Font for x axis title
        xaxis_title_font=dict(size=20, color='black' if not dark_mode else 'white'),
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20)
        )
    )
    
    
    # Improve readability of x-axis labels
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=18, color='black' if not dark_mode else 'white'),  # Increased from 10 to 18
        gridcolor='black' if not dark_mode else 'white',
        zerolinecolor='#303030'
    )
    
    fig.update_yaxes(
        tickfont=dict(size=18, color='black' if not dark_mode else 'white'),  # Added explicit font size for y-axis
        gridcolor='black' if not dark_mode else 'white',
        zerolinecolor='#303030'
    )
    
    return fig 


def symmetric_mean_absolute_percentage_error(df, df_without_tariff, df_without_rl, title, year_range=None, small_value=False, dark_mode=False):

    # Filter by year range if provided
    if year_range is not None:
        min_year, max_year = year_range
        
        # Extract year from quarter string (e.g., "2020Q1" -> 2020)
        def extract_year(quarter_str):
            # Handle both formats: "2020Q1" and "2020q1"
            return int(quarter_str.split('q')[0].split('Q')[0])
        
        # Create year columns for filtering
        df_years = df['quarter'].apply(extract_year)
        df_without_tariff_years = df_without_tariff['quarter'].apply(extract_year)
        df_without_rl_years = df_without_rl['quarter'].apply(extract_year)
        
        # Filter dataframes
        df_filtered = df[df_years.between(min_year, max_year)].reset_index(drop=True)
        df_without_tariff_filtered = df_without_tariff[df_without_tariff_years.between(min_year, max_year)].reset_index(drop=True)
        df_without_rl_filtered = df_without_rl[df_without_rl_years.between(min_year, max_year)].reset_index(drop=True)
    else:
        # Use all data if no year range specified
        df_filtered = df
        df_without_tariff_filtered = df_without_tariff
        df_without_rl_filtered = df_without_rl
    
    def calculate_smape(df, df_without_tariff, df_without_rl):
        smape_rl = []
        smape_tariff = [] 

        smape_rl_gdp_growth = 100 * (np.abs(df['gdp_growth'] - df_without_tariff['gdp_growth']) / (np.abs(df['gdp_growth']) + np.abs(df_without_tariff['gdp_growth']) * 2))
        smape_rl_inflation = 100 * (np.abs(df['inflation'] - df_without_tariff['inflation']) / (np.abs(df['inflation']) + np.abs(df_without_tariff['inflation']) * 2))
        smape_rl_unemployment = 100 * (np.abs(df['unemployment'] - df_without_tariff['unemployment']) / (np.abs(df['unemployment']) + np.abs(df_without_tariff['unemployment']) * 2))
        smape_rl_real_gdp = 100 * (np.abs(df['real_gdp'] - df_without_tariff['real_gdp']) / (np.abs(df['real_gdp']) + np.abs(df_without_tariff['real_gdp']) * 2))
        smape_rl_nominal_gdp = 100 * (np.abs(df['nominal_gdp'] - df_without_tariff['nominal_gdp']) / (np.abs(df['nominal_gdp']) + np.abs(df_without_tariff['nominal_gdp']) * 2))
        smape_rl_personal_tax = 100 * (np.abs(df['personal_tax'] - df_without_tariff['personal_tax']) / (np.abs(df['personal_tax']) + np.abs(df_without_tariff['personal_tax']) * 2))
        smape_rl_corporate_tax = 100 * (np.abs(df['corporate_tax'] - df_without_tariff['corporate_tax']) / (np.abs(df['corporate_tax']) + np.abs(df_without_tariff['corporate_tax']) * 2))
        smape_rl_exports = 100 * (np.abs(df['exports'] - df_without_tariff['exports']) / (np.abs(df['exports']) + np.abs(df_without_tariff['exports']) * 2))
        smape_rl_imports = 100 * (np.abs(df['imports'] - df_without_tariff['imports']) / (np.abs(df['imports']) + np.abs(df_without_tariff['imports']) * 2))
        smape_rl_debt_to_gdp = 100 * (np.abs(df['debt_to_gdp'] - df_without_tariff['debt_to_gdp']) / (np.abs(df['debt_to_gdp']) + np.abs(df_without_tariff['debt_to_gdp']) * 2))
        smape_rl_interest_rate = 100 * (np.abs(df['interest_rate'] - df_without_tariff['interest_rate']) / (np.abs(df['interest_rate']) + np.abs(df_without_tariff['interest_rate']) * 2))
        smape_rl_pcpi = 100 * (np.abs(df['pcpi'] - df_without_tariff['pcpi']) / (np.abs(df['pcpi']) + np.abs(df_without_tariff['pcpi']) * 2))
        smape_rl_transfer_payments_ratio = 100 * (np.abs(df['transfer_payments_ratio'] - df_without_tariff['transfer_payments_ratio']) / (np.abs(df['transfer_payments_ratio']) + np.abs(df_without_tariff['transfer_payments_ratio']) * 2))
        smape_rl_federal_expenditures = 100 * (np.abs(df['federal_expenditures'] - df_without_tariff['federal_expenditures']) / (np.abs(df['federal_expenditures']) + np.abs(df_without_tariff['federal_expenditures']) * 2))
        smape_rl_personal_tax_rates = 100 * (np.abs(df['personal_tax_rates'] - df_without_tariff['personal_tax_rates']) / (np.abs(df['personal_tax_rates']) + np.abs(df_without_tariff['personal_tax_rates']) * 2))
        smape_rl_corporate_tax_rates = 100 * (np.abs(df['corporate_tax_rates'] - df_without_tariff['corporate_tax_rates']) / (np.abs(df['corporate_tax_rates']) + np.abs(df_without_tariff['corporate_tax_rates']) * 2))
        smape_rl_government_transfer_payments = 100 * (np.abs(df['government_transfer_payments'] - df_without_tariff['government_transfer_payments']) / (np.abs(df['government_transfer_payments']) + np.abs(df_without_tariff['government_transfer_payments']) * 2))
        smape_rl_federal_surplus = 100 * (np.abs(df['federal_surplus'] - df_without_tariff['federal_surplus']) / (np.abs(df['federal_surplus']) + np.abs(df_without_tariff['federal_surplus']) * 2))

        smape_tariff_gdp_growth = 100 * (np.abs(df_without_rl['gdp_growth'] - df_without_tariff['gdp_growth']) / (np.abs(df_without_rl['gdp_growth']) + np.abs(df_without_tariff['gdp_growth']) * 2))
        smape_tariff_inflation = 100 * (np.abs(df_without_rl['inflation'] - df_without_tariff['inflation']) / (np.abs(df_without_rl['inflation']) + np.abs(df_without_tariff['inflation']) * 2))
        smape_tariff_unemployment = 100 * (np.abs(df_without_rl['unemployment'] - df_without_tariff['unemployment']) / (np.abs(df_without_rl['unemployment']) + np.abs(df_without_tariff['unemployment']) * 2))
        smape_tariff_real_gdp = 100 * (np.abs(df_without_rl['real_gdp'] - df_without_tariff['real_gdp']) / (np.abs(df_without_rl['real_gdp']) + np.abs(df_without_tariff['real_gdp']) * 2))
        smape_tariff_nominal_gdp = 100 * (np.abs(df_without_rl['nominal_gdp'] - df_without_tariff['nominal_gdp']) / (np.abs(df_without_rl['nominal_gdp']) + np.abs(df_without_tariff['nominal_gdp']) * 2))
        smape_tariff_personal_tax = 100 * (np.abs(df_without_rl['personal_tax'] - df_without_tariff['personal_tax']) / (np.abs(df_without_rl['personal_tax']) + np.abs(df_without_tariff['personal_tax']) * 2))
        smape_tariff_corporate_tax = 100 * (np.abs(df_without_rl['corporate_tax'] - df_without_tariff['corporate_tax']) / (np.abs(df_without_rl['corporate_tax']) + np.abs(df_without_tariff['corporate_tax']) * 2))
        smape_tariff_exports = 100 * (np.abs(df_without_rl['exports'] - df_without_tariff['exports']) / (np.abs(df_without_rl['exports']) + np.abs(df_without_tariff['exports']) * 2))
        smape_tariff_imports = 100 * (np.abs(df_without_rl['imports'] - df_without_tariff['imports']) / (np.abs(df_without_rl['imports']) + np.abs(df_without_tariff['imports']) * 2))
        smape_tariff_debt_to_gdp = 100 * (np.abs(df_without_rl['debt_to_gdp'] - df_without_tariff['debt_to_gdp']) / (np.abs(df_without_rl['debt_to_gdp']) + np.abs(df_without_tariff['debt_to_gdp']) * 2))
        smape_tariff_interest_rate = 100 * (np.abs(df_without_rl['interest_rate'] - df_without_tariff['interest_rate']) / (np.abs(df_without_rl['interest_rate']) + np.abs(df_without_tariff['interest_rate']) * 2))
        smape_tariff_pcpi = 100 * (np.abs(df_without_rl['pcpi'] - df_without_tariff['pcpi']) / (np.abs(df_without_rl['pcpi']) + np.abs(df_without_tariff['pcpi']) * 2))
        smape_tariff_transfer_payments_ratio = 100 * (np.abs(df_without_rl['transfer_payments_ratio'] - df_without_tariff['transfer_payments_ratio']) / (np.abs(df_without_rl['transfer_payments_ratio']) + np.abs(df_without_tariff['transfer_payments_ratio']) * 2))
        smape_tariff_federal_expenditures = 100 * (np.abs(df_without_rl['federal_expenditures'] - df_without_tariff['federal_expenditures']) / (np.abs(df_without_rl['federal_expenditures']) + np.abs(df_without_tariff['federal_expenditures']) * 2))
        smape_tariff_personal_tax_rates = 100 * (np.abs(df_without_rl['personal_tax_rates'] - df_without_tariff['personal_tax_rates']) / (np.abs(df_without_rl['personal_tax_rates']) + np.abs(df_without_tariff['personal_tax_rates']) * 2))
        smape_tariff_corporate_tax_rates = 100 * (np.abs(df_without_rl['corporate_tax_rates'] - df_without_tariff['corporate_tax_rates']) / (np.abs(df_without_rl['corporate_tax_rates']) + np.abs(df_without_tariff['corporate_tax_rates']) * 2))
        smape_tariff_government_transfer_payments = 100 * (np.abs(df_without_rl['government_transfer_payments'] - df_without_tariff['government_transfer_payments']) / (np.abs(df_without_rl['government_transfer_payments']) + np.abs(df_without_tariff['government_transfer_payments']) * 2))
        smape_tariff_federal_surplus = 100 * (np.abs(df_without_rl['federal_surplus'] - df_without_tariff['federal_surplus']) / (np.abs(df_without_rl['federal_surplus']) + np.abs(df_without_tariff['federal_surplus']) * 2))
        
        
        smape_tariff = [smape_tariff_gdp_growth, smape_tariff_inflation, smape_tariff_unemployment, smape_tariff_real_gdp, smape_tariff_nominal_gdp, smape_tariff_personal_tax, smape_tariff_corporate_tax, smape_tariff_exports, smape_tariff_imports, smape_tariff_debt_to_gdp, smape_tariff_interest_rate, smape_tariff_pcpi, smape_tariff_transfer_payments_ratio, smape_tariff_federal_expenditures, smape_tariff_personal_tax_rates, smape_tariff_corporate_tax_rates, smape_tariff_government_transfer_payments, smape_tariff_federal_surplus]
        smape_rl = [smape_rl_gdp_growth, smape_rl_inflation, smape_rl_unemployment, smape_rl_real_gdp, smape_rl_nominal_gdp, smape_rl_personal_tax, smape_rl_corporate_tax, smape_rl_exports, smape_rl_imports, smape_rl_debt_to_gdp, smape_rl_interest_rate, smape_rl_pcpi, smape_rl_transfer_payments_ratio, smape_rl_federal_expenditures, smape_rl_personal_tax_rates, smape_rl_corporate_tax_rates, smape_rl_government_transfer_payments, smape_rl_federal_surplus]
        return smape_tariff, smape_rl
    
    # Use filtered data for MAE calculation
    smape_tariff, smape_rl = calculate_smape(df_filtered, df_without_tariff_filtered, df_without_rl_filtered)
    
    # Create dataframe for visualization
    metric_names = [
        'GDP Growth', 'Inflation', 'Unemployment', 'Real GDP', 'Nominal GDP', 
        'Personal Tax', 'Corporate Tax', 'Exports', 'Imports', 'Debt to GDP',
        'Interest Rate', 'PCPI', 'Transfer Payments Ratio', 'Federal Expenditures',
        'Personal Tax Rates', 'Corporate Tax Rates', 'Government Transfer Payments', 
        'Federal Surplus'
    ]
    
    # Calculate mean MAE for each series
    smape_rl_means = [round(np.mean(series), 2) for series in smape_rl]
    smape_tariff_means = [round(np.mean(series), 2) for series in smape_tariff]
    if not small_value:
        smape_rl_valid_indices = [i for i, series in enumerate(smape_rl_means) if series > 1.0 and smape_tariff_means[i] > 1.0]
        metric_names = [metric_names[i] for i in smape_rl_valid_indices]
        smape_rl_means = [smape_rl_means[i] for i in smape_rl_valid_indices]
        smape_tariff_means = [smape_tariff_means[i] for i in smape_rl_valid_indices] 

    if small_value:
        smape_rl_valid_indices = [i for i, series in enumerate(smape_rl_means) if series <= 1.0 and smape_tariff_means[i] <= 1.0]
        metric_names = [metric_names[i] for i in smape_rl_valid_indices]
        smape_rl_means = [smape_rl_means[i] for i in smape_rl_valid_indices]
        smape_tariff_means = [smape_tariff_means[i] for i in smape_rl_valid_indices] 
    
    # Create dataframe for plotting
    smape_df = pd.DataFrame({
        'Metric': metric_names,
        'RL - FRBUS vs Historical Data': smape_rl_means,
        'FRB/US model vs Historical Data': smape_tariff_means
    })
    
    # Add year range to title if provided
    title_with_year = title
    if year_range:
        title_with_year = f"{title} ({min_year}-{max_year})"
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=smape_df['Metric'],
        y=smape_df['RL - FRBUS vs Historical Data'],
        name='RL - FRBUS vs Historical Data',
        marker_color=MUTED_REDS['dark'],
        textfont=dict(size=15, color='black' if not dark_mode else 'white')
    ))
    
    fig.add_trace(go.Bar(
        x=smape_df['Metric'],
        y=smape_df['FRB/US model vs Historical Data'],
        name='FRB/US model vs Historical Data',
        marker_color=MUTED_REDS['light'],
        textfont=dict(size=15, color='black' if not dark_mode else 'white')
    ))
    # Update layout
    fig.update_layout(
        title=f'Symmetric mean absolute percentage error Comparison: {title_with_year}',
        xaxis_title='Economic Metrics',
        yaxis_title='Symmetric mean absolute percentage error',
        title_font_color='black' if not dark_mode else 'white',
        # Font for y axis title
        yaxis_title_font=dict(size=15, color='black' if not dark_mode else 'white'),
        # Font for x axis title
        xaxis_title_font=dict(size=20, color='black' if not dark_mode else 'white'),
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20)
        )
    )
    
    
    # Improve readability of x-axis labels
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=18, color='black' if not dark_mode else 'white'),  # Increased from 10 to 18
        gridcolor='black' if not dark_mode else 'white',
        zerolinecolor='#303030'
    )
    
    fig.update_yaxes(
        tickfont=dict(size=18, color='black' if not dark_mode else 'white'),  # Added explicit font size for y-axis
        gridcolor='black' if not dark_mode else 'white',
        zerolinecolor='#303030'
    )
    return fig 

def render_actions_charts(df, title):
    """Render tax-related charts with rates and amounts"""
    fig = go.Figure()
    
    # Create subplots: top for tax rates, bottom for tax amounts
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=('Actions Rates', 'Expenditures'),
                       vertical_spacing=0.2) 
    
    fig.add_trace(
        go.Scatter(x=df['quarter'], y=df['interest_rate'], 
                  name='Federal Funds Rate', line=dict(color=MUTED_REDS['dark'])),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['quarter'], y=df['transfer_payments_ratio'], 
                  name='Trend Ratio of Transfer Payments to GDP', line=dict(color=MUTED_REDS['light'], dash='dash')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['quarter'], y=df['personal_tax_rates'], 
                  name='Personal Tax Revenues Rates', line=dict(color=MUTED_REDS['bright'], dash='dash')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['quarter'], y=df['corporate_tax_rates'], 
                  name='Corporate Tax Revenues Rates', line=dict(color=MUTED_REDS['lightest'], dash='dash')),
        row=2, col=1
    )

    # Plot Tax Amounts (bottom subplot)
    fig.add_trace(
        go.Scatter(x=df['quarter'], y=df['federal_expenditures'], 
                  name='Trend Level of Federal Government Expenditures', line=dict(color=MUTED_REDS['medium'], dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['quarter'], y=df['government_transfer_payments'], 
                  name='Government Transfer Payments', line=dict(color=MUTED_REDS['lightest'], dash='dash')),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,  # Increased height for better visibility
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue (Percentage of GDP)", row=2, col=1)
    fig.update_yaxes(title_text="Revenue (Billions of $)", row=3, col=1)
    
    return fig

def render_gdp_charts(df, title):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components
    fig.add_trace(go.Bar(x=df['quarter'], y=df['real_gdp'], name='Real GDP',
                        marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['nominal_gdp'], name='Nominal GDP',
                        marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=0))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay'
    )
    
    return fig

def render_tax_charts(df, title):
    """Render tax-related charts"""
    fig = go.Figure()
    
    # Add tax components
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['personal_tax'], name='Personal Tax', line=dict(color=MUTED_REDS['dark'])))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['corporate_tax'], name='Corporate Tax', line=dict(color=MUTED_REDS['light'])))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Value',
        # barmode='stack'
    )
    
    return fig

def render_trade_charts(df, title):
    """Render trade balance charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width  
    
    # Add trade components
    fig.add_trace(go.Bar(x=df['quarter'], y=df['trade_balance'], name='Trade Balance',
                        marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['exports'], name='Exports',
                        marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['imports'], name='Imports',
                        marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['net_foreign_investment_income'], name='Net Foreign Investment Income',
                        marker_color=MUTED_REDS['lightest'],
                        width=bar_width,
                        offset=2*bar_width))
    
    fig.update_layout(
        title=title,
        barmode='overlay',
        xaxis_title='Quarter',
        yaxis_title='Value',
        legend=dict(font=dict(size=25))
    )
    
    return fig

def update_dashboard(metrics_data, placeholder):
    """Update dashboard with new metrics"""
    with placeholder.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Real-time Metrics")
            try:
                # Display current values
                st.metric(
                    "GDP Growth",
                    f"{metrics_data['rl_decision']['metrics']['hggdp']:.2f}%"
                )
                
                st.metric(
                    "Inflation Rate",
                    f"{metrics_data['rl_decision']['metrics']['pcpi']:.2f}%"
                )
                
                st.metric(
                    "Unemployment",
                    f"{metrics_data['rl_decision']['metrics']['lur']:.2f}%"
                )
            except Exception as e:
                st.error(f"Error displaying metrics: {str(e)}")

# Add visualization functions
def render_overview_charts(df, title):
    """Render overview charts"""
    fig = go.Figure()
    
    gdp_growth_rl = ((df['real_gdp'] - df['real_gdp'].shift(1)) / 
                     df['real_gdp'].shift(1) * 400.0)  # *400 for annualized rate
    
    # Add traces for main economic indicators
    fig.add_trace(go.Scatter(x=df['quarter'], y=gdp_growth_rl, name='Real GDP Growth', line=dict(color=MUTED_REDS['dark'])))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['unemployment'], name='Unemployment', line=dict(color=MUTED_REDS['bright'])))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['interest_rate'], name='Federal Funds Rate', line=dict(color=MUTED_REDS['light'])))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Percentage',
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    return fig

# Add visualization functions
def render_overview_tax_rates_charts(df, title):
    """Render overview charts"""
    fig = go.Figure()
    
    # Add traces for main economic indicators
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['personal_tax_rates'], name='Personal Tax Rates', line=dict(color=MUTED_REDS['dark'])))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['corporate_tax_rates'], name='Corporate Tax Rates', line=dict(color=MUTED_REDS['bright'])))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Percentage',
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    return fig

# Add visualization functions
def render_overview_inflation_charts(df, title):
    """Render overview charts"""
    fig = go.Figure()
    
    # Add traces for main economic indicators
    fig.add_trace(go.Scatter(x=df['quarter'], y=((df['pcpi'] - df['pcpi'].shift(1)) / df['pcpi'].shift(1)) * 100.0, name='Inflation', line=dict(color=MUTED_REDS['dark'])))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Percentage (%)',
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    return fig

# Common comparison charts
def render_comparison_chart(df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff, metric, title, y_axis_title, chart_type='bar'):
    """Render comparison charts for different metrics
    Args:
        df_rl_tariff (DataFrame): Data for AI with tariff
        df_without_tariff (DataFrame): Data for AI without tariff
        df_base_simulation (DataFrame): Data for base simulation
        df_base_simulation_with_tariff (DataFrame): Data for base simulation with tariff
        metric (str): Column name of the metric to plot
        title (str): Chart title
        y_axis_title (str): Y-axis label
        chart_type (str): 'bar' or 'scatter' for different visualization types
    """
    fig = go.Figure()
    
    # Common trace parameters
    traces_bar = [
        {
            'x': df_rl_tariff['quarter'],
            'y': df_rl_tariff[metric],
            'name': f'{title} (RL-FRB/US Agent)',
            'marker_color': MUTED_REDS['dark'],
            'width': 0.2,
            'offset': -0.2
        },
        {
            'x': df_without_tariff['quarter'],
            'y': df_without_tariff[metric],
            'name': f'{title} ({historical_label_or_hypothetical_label})',
            'marker_color': MUTED_REDS['bright'],
            'width': 0.2,
            'offset': 0
        },
        {
            'x': df_base_simulation['quarter'],
            'y': df_base_simulation[metric],
            'name': f'{title} (FRB/US)',
            'marker_color': MUTED_REDS['light'],
            'width': 0.2,
            'offset': 0.2
        },
        {
            'x': df_base_simulation_with_tariff['quarter'],
            'y': df_base_simulation_with_tariff[metric],
            'name': f'{title} (FRB/US - With Tariff)',
            'marker_color': MUTED_REDS['lightest'],
            'width': 0.2,
            'offset': 0.4
        }
    ]
    traces_scatter = [
        {
            'x': df_rl_tariff['quarter'],
            'y': df_rl_tariff[metric],
            'name': f'{title} (RL-FRB/US Agent)',
            'marker_color': MUTED_REDS['dark']
        },
        {
            'x': df_without_tariff['quarter'],
            'y': df_without_tariff[metric],
            'name': f'{title} ({historical_label_or_hypothetical_label})',
            'marker_color': MUTED_REDS['bright']
        },
        {
            'x': df_base_simulation['quarter'],
            'y': df_base_simulation[metric],
            'name': f'{title} (FRB/US)',
            'marker_color': MUTED_REDS['light']
        },
        {
            'x': df_base_simulation_with_tariff['quarter'],
            'y': df_base_simulation_with_tariff[metric],
            'name': f'{title} (FRB/US - With Tariff)',
            'marker_color': MUTED_REDS['lightest']
        }
    ]
    if chart_type == 'bar':
        for trace in traces_bar:
            # Add bar-specific parameters
            fig.add_trace(go.Bar(**trace))
    else:  
        # scatter
        for trace in traces_scatter:
            fig.add_trace(go.Scatter(**trace))
    
    # Update layout
    fig.update_layout(
        title=f'{title} Comparison',
        xaxis_title='Quarter',
        yaxis_title=y_axis_title,
        barmode='overlay' if chart_type == 'bar' else None,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

def render_inflation_rate_comparison_charts(df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff):
    """Render inflation comparison charts"""
    fig = go.Figure()
    
    # Calculate inflation for each scenario and round to 2 decimal places
    inflation_rl = ((df_rl_tariff['pcpi'] - df_rl_tariff['pcpi'].shift(1)) / df_rl_tariff['pcpi'].shift(1) * 100.0)
    inflation_without = ((df_without_tariff['pcpi'] - df_without_tariff['pcpi'].shift(1)) / df_without_tariff['pcpi'].shift(1) * 100.0)
    inflation_base = ((df_base_simulation['pcpi'] - df_base_simulation['pcpi'].shift(1)) / df_base_simulation['pcpi'].shift(1) * 100.0)
    inflation_base_with_tariff = ((df_base_simulation_with_tariff['pcpi'] - df_base_simulation_with_tariff['pcpi'].shift(1)) / df_base_simulation_with_tariff['pcpi'].shift(1) * 100.0)

    # Add traces for each scenario
    fig.add_trace(go.Scatter(
        x=df_rl_tariff['quarter'], 
        y=inflation_rl, 
        name='Inflation - RL-FRB/US Agent', 
        line=dict(color=MUTED_REDS['dark'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_without_tariff['quarter'], 
        y=inflation_without, 
        name=f'Inflation - {historical_label_or_hypothetical_label}', 
        line=dict(color=MUTED_REDS['bright'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation['quarter'], 
        y=inflation_base, 
        name='Inflation - FRB/US', 
        line=dict(color=MUTED_REDS['light'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation_with_tariff['quarter'], 
        y=inflation_base_with_tariff, 
        name='Inflation - FRB/US - With Tariff', 
        line=dict(color=MUTED_REDS['lightest'])
    ))

    fig.update_layout(
        title='Inflation Comparison',
        xaxis_title='Quarter',
        yaxis_title='Percentage (%)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.5
        )
    )
    
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

def render_unemployment_comparison_charts(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render unemployment comparison charts"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['unemployment'], name='Unemployment - RL-FRB/US Agent', marker_color=MUTED_REDS['dark']))
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['unemployment'], name=f'Unemployment - {historical_label_or_hypothetical_label}', marker_color=MUTED_REDS['bright']))
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['unemployment'], name='Unemployment - FRB/US', marker_color=MUTED_REDS['light']))

    fig.update_layout(
        title='Unemployment Comparison',
        xaxis_title='Quarter',
        yaxis_title='Percentage (%)',
        hovermode='x unified'
    )
    return fig

def render_gdp_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.2  # Adjust this value to control bar width
    # Add GDP components for RL-FRB/US Agent
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['real_gdp'], 
                        name='Real GDP (RL-FRB/US Agent)', 
                        marker_color=MUTED_REDS['dark'],  
                        width=bar_width,
                        offset=-bar_width)) 
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['real_gdp'], 
                        name=f'Real GDP ({historical_label_or_hypothetical_label})', 
                        marker_color=MUTED_REDS['bright'],  
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRB/US
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['real_gdp'], 
                        name='Real GDP (FRB/US)', 
                        marker_color=MUTED_REDS['light'],  
                        width=bar_width,
                        offset=bar_width))
    
    # Add GDP components for FRB/US - With Tariff
    fig.add_trace(go.Bar(x=df_base_simulation_with_tariff['quarter'], y=df_base_simulation_with_tariff['real_gdp'], 
                        name='Real GDP (FRB/US - With Tariff)', 
                        marker_color=MUTED_REDS['lightest'],  
                        width=bar_width,
                        offset=2*bar_width))
    
    # Update layout with dark mode colors
    fig.update_layout(
        title='Real GDP Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay',  # Light gray text
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1  # Semi-transparent black background
        )
    )
    
    # Update axes for dark mode
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

def render_gdp_charts_comparison_nominal(df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.2  # Adjust this value to control bar width
    
    # Add GDP components for RL-FRB/US Agent
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['nominal_gdp'], 
                        name='Nominal GDP (RL-FRB/US Agent)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['nominal_gdp'], 
                        name=f'Nominal GDP ({historical_label_or_hypothetical_label})', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRB/US
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['nominal_gdp'], 
                        name='Nominal GDP (FRB/US)', marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width)) 
    
    # Add GDP components for FRB/US - With Tariff
    fig.add_trace(go.Bar(x=df_base_simulation_with_tariff['quarter'], y=df_base_simulation_with_tariff['nominal_gdp'], 
                        name='Nominal GDP (FRB/US - With Tariff)', marker_color=MUTED_REDS['lightest'],
                        width=bar_width,
                        offset=2*bar_width))
    
    fig.update_layout(
        title='Nominal GDP Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay',  # Light gray text
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1  # Semi-transparent black background
        )
    )
    
    return fig

def render_personal_tax_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for RL-FRB/US Agent
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['personal_tax'], 
                        name='Personal Tax (RL-FRB/US Agent)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['personal_tax'], 
                        name=f'Personal Tax ({historical_label_or_hypothetical_label})', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRB/US
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['personal_tax'], 
                        name='Personal Tax (FRB/US)', marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width)) 
    
    fig.update_layout(
        title='Personal Tax Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Value',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def render_corporate_tax_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for RL-FRB/US Agent
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['corporate_tax'], 
                        name='Corporate Tax (RL-FRB/US Agent)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['corporate_tax'], 
                        name=f'Corporate Tax ({historical_label_or_hypothetical_label})', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRB/US
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['corporate_tax'], 
                        name='Corporate Tax (FRB/US)', marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width))
    
    fig.update_layout(
        title='Corporate Tax Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Value',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def render_government_transfer_payments_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render government transfer payments charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for RL-FRB/US Agent
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['government_transfer_payments'], 
                        name='Government Transfer Payments (RL-FRB/US Agent)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['government_transfer_payments'], 
                        name=f'Government Transfer Payments ({historical_label_or_hypothetical_label})', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRB/US
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['government_transfer_payments'], 
                        name='Government Transfer Payments (FRB/US)', marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width)) 
    
    fig.update_layout(
        title='Government Transfer Payments Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def render_government_debt_to_gdp_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render government debt to GDP charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for RL-FRB/US Agent
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['debt_to_gdp'], 
                        name='Government Debt to GDP (RL-FRB/US Agent)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['debt_to_gdp'], 
                        name=f'Government Debt to GDP ({historical_label_or_hypothetical_label})', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRB/US
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['debt_to_gdp'], 
                        name='Government Debt to GDP (FRB/US)', marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width)) 
    
    fig.update_layout(
        title='Government Debt to GDP Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def render_real_gdp_growth_comparison_charts(df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff):
    """Render Real GDP growth rate comparison charts"""
    fig = go.Figure()
    
    # Calculate GDP growth rates for each scenario (quarter-over-quarter, annualized)
    gdp_growth_rl = ((df_rl_tariff['real_gdp'] - df_rl_tariff['real_gdp'].shift(1)) / 
                     df_rl_tariff['real_gdp'].shift(1) * 400.0)  # *400 for annualized rate
    
    gdp_growth_without = ((df_without_tariff['real_gdp'] - df_without_tariff['real_gdp'].shift(1)) / 
                         df_without_tariff['real_gdp'].shift(1) * 400.0)
    
    gdp_growth_base = ((df_base_simulation['real_gdp'] - df_base_simulation['real_gdp'].shift(1)) / 
                       df_base_simulation['real_gdp'].shift(1) * 400.0)
    
    gdp_growth_base_with_tariff = ((df_base_simulation_with_tariff['real_gdp'] - df_base_simulation_with_tariff['real_gdp'].shift(1)) / 
                       df_base_simulation_with_tariff['real_gdp'].shift(1) * 400.0)

    # Add traces for each scenario
    fig.add_trace(go.Scatter(
        x=df_rl_tariff['quarter'], 
        y=gdp_growth_rl, 
        name='Real GDP Growth - RL-FRB/US Agent', 
        line=dict(color=MUTED_REDS['dark'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_without_tariff['quarter'], 
        y=gdp_growth_without, 
        name=f'Real GDP Growth - {historical_label_or_hypothetical_label}', 
        line=dict(color=MUTED_REDS['bright'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation['quarter'], 
        y=gdp_growth_base, 
        name='Real GDP Growth - FRB/US', 
        line=dict(color=MUTED_REDS['light'])
    ))

    fig.add_trace(go.Scatter(
        x=df_base_simulation_with_tariff['quarter'], 
        y=gdp_growth_base_with_tariff, 
        name='Real GDP Growth - FRB/US - With Tariff', 
        line=dict(color=MUTED_REDS['lightest'])
    ))

    fig.update_layout(
        title='Real GDP Growth Rate Comparison (Annualized %)',
        xaxis_title='Quarter',
        yaxis_title='Growth Rate (%)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

def render_nominal_gdp_growth_comparison_charts(df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff):
    """Render Nominal GDP growth rate comparison charts"""
    fig = go.Figure()
    
    # Calculate GDP growth rates for each scenario (quarter-over-quarter, annualized)
    gdp_growth_rl = ((df_rl_tariff['nominal_gdp'] - df_rl_tariff['nominal_gdp'].shift(1)) / 
                     df_rl_tariff['nominal_gdp'].shift(1) * 400.0)  # *400 for annualized rate
    
    gdp_growth_without = ((df_without_tariff['nominal_gdp'] - df_without_tariff['nominal_gdp'].shift(1)) / 
                         df_without_tariff['nominal_gdp'].shift(1) * 400.0)
    
    gdp_growth_base = ((df_base_simulation['nominal_gdp'] - df_base_simulation['nominal_gdp'].shift(1)) / 
                       df_base_simulation['nominal_gdp'].shift(1) * 400.0)
    
    gdp_growth_base_with_tariff = ((df_base_simulation_with_tariff['nominal_gdp'] - df_base_simulation_with_tariff['nominal_gdp'].shift(1)) / 
                       df_base_simulation_with_tariff['nominal_gdp'].shift(1) * 400.0)
    # Add traces for each scenario
    fig.add_trace(go.Scatter(
        x=df_rl_tariff['quarter'], 
        y=gdp_growth_rl, 
        name='Nominal GDP Growth - RL-FRB/US Agent', 
        line=dict(color=MUTED_REDS['dark'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_without_tariff['quarter'], 
        y=gdp_growth_without, 
        name=f'Nominal GDP Growth - {historical_label_or_hypothetical_label}', 
        line=dict(color=MUTED_REDS['bright'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation['quarter'], 
        y=gdp_growth_base, 
        name='Nominal GDP Growth - FRB/US', 
        line=dict(color=MUTED_REDS['light'])
    ))

    fig.add_trace(go.Scatter(
        x=df_base_simulation_with_tariff['quarter'], 
        y=gdp_growth_base_with_tariff, 
        name='Nominal GDP Growth - FRB/US - With Tariff', 
        line=dict(color=MUTED_REDS['lightest'])
    ))

    fig.update_layout(
        title='Nominal GDP Growth Rate Comparison (Annualized %)',
        xaxis_title='Quarter',
        yaxis_title='Growth Rate (%)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

# Create connection controls
class EconomicStream:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.lock = threading.Lock()
        self.url = "ws://localhost:8000/ws/metrics" 
        columns = ['quarter', 'gdp_growth', 'inflation', 'unemployment',
            'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
            'exports', 'imports', 'debt_to_gdp', 'interest_rate', 'pcpi', 'transfer_payments_ratio', 
            'federal_expenditures', 'personal_tax_rates', 'corporate_tax_rates', 'government_transfer_payments',
            'federal_surplus']
        
        self.metrics_history_rl_tariff = pd.DataFrame(columns=columns)
        self.metrics_history_without_tariff = pd.DataFrame(columns=columns)
        self.metrics_history_base_simulation = pd.DataFrame(columns=columns)
        self.metrics_history_base_simulation_with_tariff = pd.DataFrame(columns=columns)

        
        # Initialize metrics storage locally
        self.last_message = None

    def handle_message(self, message_data):
        """Process incoming message and update metrics"""
        self.lock.acquire()
        try:
            # Extract metrics from message
            metrics = message_data['rl_decision']
            metrics_without_tariff = message_data['rl_decision_without_tariff']
            metrics_base_simulation = message_data['base_simulation']
            metrics_base_simulation_with_tariff = message_data['base_simulation_with_tariff']
            
            # Create new row for DataFrame
            new_data = {
                'quarter': metrics['quarter'].upper(),
                'gdp_growth': metrics['metrics']['hggdp'],
                'inflation': metrics['metrics']['pcpi'],
                'unemployment': metrics['metrics']['lur'],
                'real_gdp': metrics['metrics']['xgdp'],
                'nominal_gdp': metrics['metrics']['xgdpn'],
                'personal_tax': metrics['metrics']['tpn'],
                'corporate_tax': metrics['metrics']['tcin'],
                'exports': metrics['metrics']['exn'],
                'imports': metrics['metrics']['emn'],
                'debt_to_gdp': metrics['metrics']['gfdbtn'],
                'interest_rate': metrics['metrics']['rff'],
                'pcpi': metrics['metrics']['pcpi'],
                'transfer_payments_ratio': metrics['metrics']['gtrt'],
                'federal_expenditures': metrics['metrics']['egfe'],
                'personal_tax_rates': metrics['metrics']['trptx'],
                'corporate_tax_rates': metrics['metrics']['trcit'],
                'government_transfer_payments': metrics['metrics']['gtn'],
                'federal_surplus': metrics['metrics']['gfsrpn'],
                'trade_balance': metrics['metrics']['fcbn'],
                "net_foreign_investment_income": metrics['metrics']['fynin']
            }
            
            new_data_without_tariff = {
                'quarter': metrics_without_tariff['quarter'].upper(),
                'gdp_growth': metrics_without_tariff['metrics']['hggdp'],
                'inflation': metrics_without_tariff['metrics']['pcpi'],
                'unemployment': metrics_without_tariff['metrics']['lur'],
                'real_gdp': metrics_without_tariff['metrics']['xgdp'],
                'nominal_gdp': metrics_without_tariff['metrics']['xgdpn'],
                'personal_tax': metrics_without_tariff['metrics']['tpn'],
                'corporate_tax': metrics_without_tariff['metrics']['tcin'],
                'exports': metrics_without_tariff['metrics']['exn'],
                'imports': metrics_without_tariff['metrics']['emn'],
                'debt_to_gdp': metrics_without_tariff['metrics']['gfdbtn'],
                'interest_rate': metrics_without_tariff['metrics']['rff'],
                'pcpi': metrics_without_tariff['metrics']['pcpi'],
                'transfer_payments_ratio': metrics_without_tariff['metrics']['gtrt'],
                'federal_expenditures': metrics_without_tariff['metrics']['egfe'],
                'personal_tax_rates': metrics_without_tariff['metrics']['trptx'],
                'corporate_tax_rates': metrics_without_tariff['metrics']['trcit'],
                'government_transfer_payments': metrics_without_tariff['metrics']['gtn'],
                'federal_surplus': metrics_without_tariff['metrics']['gfsrpn'],
                'trade_balance': metrics_without_tariff['metrics']['fcbn'],
                "net_foreign_investment_income": metrics_without_tariff['metrics']['fynin']
            }
            
            new_data_base_simulation = {
                'quarter': metrics_base_simulation['quarter'].upper(),
                'gdp_growth': metrics_base_simulation['metrics']['hggdp'],
                'inflation': metrics_base_simulation['metrics']['pcpi'],
                'unemployment': metrics_base_simulation['metrics']['lur'],
                'real_gdp': metrics_base_simulation['metrics']['xgdp'],
                'nominal_gdp': metrics_base_simulation['metrics']['xgdpn'],
                'personal_tax': metrics_base_simulation['metrics']['tpn'],
                'corporate_tax': metrics_base_simulation['metrics']['tcin'],
                'exports': metrics_base_simulation['metrics']['exn'],
                'imports': metrics_base_simulation['metrics']['emn'],
                'debt_to_gdp': metrics_base_simulation['metrics']['gfdbtn'],
                'interest_rate': metrics_base_simulation['metrics']['rff'],
                'pcpi': metrics_base_simulation['metrics']['pcpi'],
                'transfer_payments_ratio': metrics_base_simulation['metrics']['gtrt'],
                'federal_expenditures': metrics_base_simulation['metrics']['egfe'],
                'personal_tax_rates': metrics_base_simulation['metrics']['trptx'],
                'corporate_tax_rates': metrics_base_simulation['metrics']['trcit'],
                'government_transfer_payments': metrics_base_simulation['metrics']['gtn'],
                'federal_surplus': metrics_base_simulation['metrics']['gfsrpn'],
                'trade_balance': metrics_base_simulation['metrics']['fcbn'],
                "net_foreign_investment_income": metrics_base_simulation['metrics']['fynin']
            }
            
            new_data_base_simulation_with_tariff = {
                'quarter': metrics_base_simulation_with_tariff['quarter'].upper(),
                'gdp_growth': metrics_base_simulation_with_tariff['metrics']['hggdp'],
                'inflation': metrics_base_simulation_with_tariff['metrics']['pcpi'],
                'unemployment': metrics_base_simulation_with_tariff['metrics']['lur'],
                'real_gdp': metrics_base_simulation_with_tariff['metrics']['xgdp'],
                'nominal_gdp': metrics_base_simulation_with_tariff['metrics']['xgdpn'],
                'personal_tax': metrics_base_simulation_with_tariff['metrics']['tpn'],
                'corporate_tax': metrics_base_simulation_with_tariff['metrics']['tcin'],
                'exports': metrics_base_simulation_with_tariff['metrics']['exn'],
                'imports': metrics_base_simulation_with_tariff['metrics']['emn'],
                'debt_to_gdp': metrics_base_simulation_with_tariff['metrics']['gfdbtn'],
                'interest_rate': metrics_base_simulation_with_tariff['metrics']['rff'],
                'pcpi': metrics_base_simulation_with_tariff['metrics']['pcpi'],
                'transfer_payments_ratio': metrics_base_simulation_with_tariff['metrics']['gtrt'],
                'federal_expenditures': metrics_base_simulation_with_tariff['metrics']['egfe'],
                'personal_tax_rates': metrics_base_simulation_with_tariff['metrics']['trptx'],
                'corporate_tax_rates': metrics_base_simulation_with_tariff['metrics']['trcit'],
                'government_transfer_payments': metrics_base_simulation_with_tariff['metrics']['gtn'],
                'federal_surplus': metrics_base_simulation_with_tariff['metrics']['gfsrpn'],
                'trade_balance': metrics_base_simulation_with_tariff['metrics']['fcbn'],
                "net_foreign_investment_income": metrics_base_simulation_with_tariff['metrics']['fynin']
            }
            
            # Update metrics history
            self.metrics_history_rl_tariff = pd.concat([
                self.metrics_history_rl_tariff,
                pd.DataFrame([new_data])
            ]).reset_index(drop=True)
            
            self.metrics_history_without_tariff = pd.concat([
                self.metrics_history_without_tariff,
                pd.DataFrame([new_data_without_tariff])
            ]).reset_index(drop=True)
            
            self.metrics_history_base_simulation = pd.concat([
                self.metrics_history_base_simulation,
                pd.DataFrame([new_data_base_simulation])
            ]).reset_index(drop=True)

            self.metrics_history_base_simulation_with_tariff = pd.concat([
                self.metrics_history_base_simulation_with_tariff,
                pd.DataFrame([new_data_base_simulation_with_tariff])
            ]).reset_index(drop=True)
            
            # Add sorting key
            def quarter_to_sortable(q):
                year, quarter = q.split('q')
                return int(year) * 10 + int(quarter)
            
            # Sort by quarter using the custom sorting key
            self.metrics_history_rl_tariff['sort_key'] = self.metrics_history_rl_tariff['quarter'].apply(quarter_to_sortable)
            self.metrics_history_rl_tariff = self.metrics_history_rl_tariff.sort_values('sort_key')
            self.metrics_history_rl_tariff = self.metrics_history_rl_tariff.drop('sort_key', axis=1)
            
            self.metrics_history_without_tariff['sort_key'] = self.metrics_history_without_tariff['quarter'].apply(quarter_to_sortable)
            self.metrics_history_without_tariff = self.metrics_history_without_tariff.sort_values('sort_key')
            self.metrics_history_without_tariff = self.metrics_history_without_tariff.drop('sort_key', axis=1)
            
            self.metrics_history_base_simulation['sort_key'] = self.metrics_history_base_simulation['quarter'].apply(quarter_to_sortable)
            self.metrics_history_base_simulation = self.metrics_history_base_simulation.sort_values('sort_key')
            self.metrics_history_base_simulation = self.metrics_history_base_simulation.drop('sort_key', axis=1)
            
            self.metrics_history_base_simulation_with_tariff['sort_key'] = self.metrics_history_base_simulation_with_tariff['quarter'].apply(quarter_to_sortable)
            self.metrics_history_base_simulation_with_tariff = self.metrics_history_base_simulation_with_tariff.sort_values('sort_key')
            self.metrics_history_base_simulation_with_tariff = self.metrics_history_base_simulation_with_tariff.drop('sort_key', axis=1)

            # Keep only last 50 records
            if len(self.metrics_history_rl_tariff) > 50:
                self.metrics_history_rl_tariff = self.metrics_history_rl_tariff.tail(50)
            
            if len(self.metrics_history_without_tariff) > 50:
                self.metrics_history_without_tariff = self.metrics_history_without_tariff.tail(50)
                
            if len(self.metrics_history_base_simulation) > 50:
                self.metrics_history_base_simulation = self.metrics_history_base_simulation.tail(50)

            if len(self.metrics_history_base_simulation_with_tariff) > 50:
                self.metrics_history_base_simulation_with_tariff = self.metrics_history_base_simulation_with_tariff.tail(50)

            # Reset index after all operations
            self.metrics_history_rl_tariff = self.metrics_history_rl_tariff.reset_index(drop=True)
            self.metrics_history_without_tariff = self.metrics_history_without_tariff.reset_index(drop=True)
            self.metrics_history_base_simulation = self.metrics_history_base_simulation.reset_index(drop=True)
            self.metrics_history_base_simulation_with_tariff = self.metrics_history_base_simulation_with_tariff.reset_index(drop=True)
            
            # Store last message
            self.last_message = message_data
            
            # Force Streamlit to rerun
            st.rerun()
        except Exception as e:
            logger.error(f"EconomicStream: Error handling message: {str(e)}")
            st.error(f"EconomicStream: Error handling message: {str(e)}")
        finally:
            self.lock.release()

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            message_data = json.loads(message)
            logger.info(f"Received message for quarter: {message_data['rl_decision']['quarter']}")
            
            # Process message
            self.handle_message(message_data)
            
            # Update dashboard in a new thread
            thread = threading.Thread(
                target=update_dashboard,
                args=(message_data, self.placeholder)
            )
            thread.daemon = True
            # Add Streamlit script run context to thread
            add_script_run_ctx(thread)
            thread.start()
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            st.error(f"Error processing message: {str(e)}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {str(error)}")
        st.error(f"WebSocket error: {str(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        st.warning("WebSocket connection closed")

    def on_open(self, ws):
        logger.info("WebSocket connected")
        st.success("Connected to simulation")

    def connect(self):
        """Create and run WebSocket connection"""
        ws = WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Store WebSocket connection in session state
        st.session_state.ws = ws
        ws.run_forever()


# Main app setup
st.set_page_config(
    page_title="Economic Simulation Dashboard",
    page_icon=":material/stacked_line_chart:",
    layout="wide"
)
# Add Material Icons CSS in the header
st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <style>
        .material-icons {
            font-family: 'Material Icons';
            font-weight: normal;
            font-style: normal;
            display: inline-block;
            line-height: 1;
            text-transform: none;
            letter-spacing: normal;
            word-wrap: normal;
            white-space: nowrap;
            direction: ltr;
        }
    </style>
""", unsafe_allow_html=True)
st.title("Economic Simulation Dashboard")

# Add view selector in sidebar
st.sidebar.header("Dashboard Views")
selected_view = st.sidebar.radio(
    "Select View",
    ["Overview", "GDP Metrics", "Revenue and Expenditure Metrics", "Trade Balance"]
)

# Create placeholder for real-time updates
placeholder = st.empty()
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
# Create connection controls
connection_status = st.sidebar.empty()
simulation_status = st.sidebar.empty()
col0, col1, col2, col3 = st.sidebar.columns(4)

# Initialize stream object if not exists
if 'stream' not in st.session_state:
    st.session_state.stream = EconomicStream(placeholder)

# Add in the sidebar, before current controls
st.sidebar.header("Simulation Settings")
# Add simulation type selector
st.session_state.simulation_type = st.sidebar.radio(
    "Simulation Type",
    ["Historical Simulation", "Hypothetical Simulation"],
    index=1  # Default to hypothetical
)

# Add year range sliders based on simulation type
if st.session_state.simulation_type == "Historical Simulation":
    start_year, end_year = st.sidebar.slider(
        "Historical Year Range",
        min_value=1970,
        max_value=2024,
        value=(2000, 2024),  # Default range
        step=1
    )
    st.session_state.start_year = start_year
    st.session_state.end_year = end_year
    # Historical simulation doesn't use tariff rate
    tariff_rate = 0.0
    st.sidebar.info("Historical simulation uses actual economic data from the selected period.")
else:  # Hypothetical Simulation
    start_year, end_year = st.sidebar.slider(
        "Future Year Range",
        min_value=2024,
        max_value=2075,
        value=(2024, 2030),  # Default range
        step=1
    )
    st.session_state.start_year = start_year
    st.session_state.end_year = end_year
    # Add tariff rate for hypothetical simulation
    tariff_rate = st.sidebar.slider(
        "Tariff Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,  # Default tariff
        step=0.5
    )
    st.session_state.tariff_rate = tariff_rate
    st.sidebar.info("Hypothetical simulation projects economic outcomes into the future with the specified tariff rate.")

# Add a divider
st.sidebar.divider()
st.sidebar.header("Connection Controls")
historical_label_or_hypothetical_label = "Historical Data" if st.session_state.simulation_type == "Historical Simulation" else "RL-FRB/US Agent without Tariff"

def clear_simulation_data(): 
    st.cache_data.clear()
    if hasattr(st.session_state.stream, 'metrics_history_rl_tariff'):
        st.session_state.stream.metrics_history_rl_tariff = pd.DataFrame()
    if hasattr(st.session_state.stream, 'metrics_history_without_tariff'):
        st.session_state.stream.metrics_history_without_tariff = pd.DataFrame()
    if hasattr(st.session_state.stream, 'metrics_history_base_simulation'):
        st.session_state.stream.metrics_history_base_simulation = pd.DataFrame()
    if hasattr(st.session_state.stream, 'metrics_history_base_simulation_with_tariff'):
        st.session_state.stream.metrics_history_base_simulation_with_tariff = pd.DataFrame()
    if hasattr(st.session_state.stream, 'ws'):
        st.session_state.stream.ws.close()
        connection_status.warning("Disconnected")
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Connection controls
with col0: 
    if st.button('Effective Training', use_container_width=True):
        # Run simulation by calling the API endpoint
        response = requests.get('http://localhost:8000/run_simulation_training_effective_relocation')
        if response.status_code == 200:
            simulation_status.success("Effective training started")
        else:
            simulation_status.error("Failed to start effective training")
with col1:
    if st.button('Training', use_container_width=True):
        # Run simulation by calling the API endpoint
        response = requests.get('http://localhost:8000/run_simulation_training')
        if response.status_code == 200:
            simulation_status.success("Training completed")
        else:
            simulation_status.error("Failed to start training")
with col2:
    if st.button('Connect', use_container_width=True):
        thread = threading.Thread(target=st.session_state.stream.connect)
        thread.daemon = True
        add_script_run_ctx(thread)
        thread.start()
        connection_status.success("Connecting...")

with col3:
    if st.button('Refresh', use_container_width=True): 
        clear_simulation_data()

# # Replace the existing Start Simulation button with this
# if st.sidebar.button('Start Simulation PPO', use_container_width=True):
#     thread = threading.Thread(target=st.session_state.stream.connect)
#     thread.daemon = True
#     add_script_run_ctx(thread)
#     thread.start()
    
#     # Build URL with parameters
#     base_url = 'http://localhost:8000/run_simulation'
#     params = {
#         'simulation_type': 'historical' if st.session_state.simulation_type == "Historical Simulation" else 'hypothetical',
#         'start_year': start_year,
#         'end_year': end_year,
#         'tariff_rate': tariff_rate if st.session_state.simulation_type == "Hypothetical Simulation" else 0.0
#     }
    
#     # Run simulation by calling the API endpoint with parameters
#     response = requests.get(base_url, params=params)
    
#     if response.status_code == 200 or response.status_code == 204:
#         simulation_status.success("Simulation started")     
#     else:
#         simulation_status.error(f"Failed to start simulation: {response.text}")

# # Replace the existing Start Simulation button with this
# if st.sidebar.button('Start Simulation PPO Active Learning', use_container_width=True):
#     thread = threading.Thread(target=st.session_state.stream.connect)
#     thread.daemon = True
#     add_script_run_ctx(thread)
#     thread.start()
    
#     # Build URL with parameters
#     base_url = 'http://localhost:8000/run_simulation'
#     params = {
#         'simulation_type': 'historical' if st.session_state.simulation_type == "Historical Simulation" else 'hypothetical',
#         'start_year': start_year,
#         'end_year': end_year,
#         'tariff_rate': tariff_rate if st.session_state.simulation_type == "Hypothetical Simulation" else 0.0,
#         'active_learning': True
#     }
    
#     # Run simulation by calling the API endpoint with parameters
#     response = requests.get(base_url, params=params)
    
#     if response.status_code == 200 or response.status_code == 204:
#         simulation_status.success("Simulation started")     
#     else:
#         simulation_status.error(f"Failed to start simulation: {response.text}")

# Replace the existing Start Simulation Effective Relocation button with this
if st.sidebar.button('Start Simulation Effective Relocation', use_container_width=True):
    thread = threading.Thread(target=st.session_state.stream.connect)
    thread.daemon = True
    add_script_run_ctx(thread)
    thread.start()
    
    # Build URL with parameters
    base_url = 'http://localhost:8000/run_simulation_effective_relocation'
    params = {
        'simulation_type': 'historical' if st.session_state.simulation_type == "Historical Simulation" else 'hypothetical',
        'start_year': start_year,
        'end_year': end_year,
        'tariff_rate': tariff_rate if st.session_state.simulation_type == "Hypothetical Simulation" else 0.0
    }
    
    # Run simulation by calling the API endpoint with parameters
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200 or response.status_code == 204:
        simulation_status.success("Simulation started")     
    else:
        simulation_status.error(f"Failed to start simulation: {response.text}")

if st.sidebar.button('Save Simulation', use_container_width=True, disabled=(not (hasattr(st.session_state.stream, 'metrics_history_rl_tariff') and not st.session_state.stream.metrics_history_rl_tariff.empty))):
    # Save the simulation data to a CSV file
    df_rl_tariff = st.session_state.stream.metrics_history_rl_tariff
    df_rl_tariff['simulation_type'] = 'RL-FRB/US Agent'
    df_without_tariff = st.session_state.stream.metrics_history_without_tariff
    df_without_tariff['simulation_type'] = historical_label_or_hypothetical_label
    df_base_simulation = st.session_state.stream.metrics_history_base_simulation
    df_base_simulation['simulation_type'] = 'FRB/US'
    df_base_simulation_with_tariff = st.session_state.stream.metrics_history_base_simulation_with_tariff
    df_base_simulation_with_tariff['simulation_type'] = 'FRB/US - With Tariff'

    df_rl_tariff.to_csv('simulation_data_rl_tariff.csv', index=False)
    df_without_tariff.to_csv('simulation_data_without_tariff.csv', index=False)
    df_base_simulation.to_csv('simulation_data_base_simulation.csv', index=False)
    df_base_simulation_with_tariff.to_csv('simulation_data_base_simulation_with_tariff.csv', index=False)
    # Combine all dataframes
    if st.session_state.simulation_type == "Historical Simulation":
        combined_df = pd.concat([df_rl_tariff, df_without_tariff, df_base_simulation], 
                            ignore_index=True)
    else:
        combined_df = pd.concat([df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff], 
                            ignore_index=True)
    
    # Save combined data to a single CSV
    filename = f'combined_simulation_data_{st.session_state.start_year}_{st.session_state.end_year}.csv' if st.session_state.simulation_type == "Historical Simulation" else f'combined_simulation_data_with_tariff_{st.session_state.start_year}_{st.session_state.end_year}.csv'
    combined_df.to_csv(filename, index=False)
    simulation_status.success("Simulation data saved to CSV files") 

    st.sidebar.download_button(
        label="Download Combined Simulation Data",
        data=combined_df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        key=f"download_combined_simulation_data",
        use_container_width=True
    )



# Main dashboard area
if hasattr(st.session_state.stream, 'metrics_history_rl_tariff') and not st.session_state.stream.metrics_history_rl_tariff.empty:
    df = st.session_state.stream.metrics_history_rl_tariff
    df_without_tariff = st.session_state.stream.metrics_history_without_tariff
    df_base_simulation = st.session_state.stream.metrics_history_base_simulation
    df_base_simulation_with_tariff = st.session_state.stream.metrics_history_base_simulation_with_tariff
    # Display charts based on selected view
    if selected_view == "Overview":
        dark_mode = st.checkbox("Dark Mode", value=False)
        def extract_year(quarter_str):
            # Handle both formats: "2020Q1" and "2020q1"
            return int(quarter_str.split('q')[0].split('Q')[0])

        all_years = []
        for element in [df, df_without_tariff, df_base_simulation]:
            years = element['quarter'].apply(extract_year).unique()
            all_years.extend(years)

        min_year = min(all_years)
        max_year = max(all_years)

        # Add the slider to your UI
        st.subheader("Filter by Year Range")
        selected_year_range = st.slider(
            "Select Year Range", 
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),  # Default to full range
            step=1
        )

        # Now call the MAE function with the selected year range
        mae_fig = mean_absolute_error(
            df,
            df_without_tariff,
            df_base_simulation,
            "Economic Metrics", 
            year_range=selected_year_range,
            dark_mode=dark_mode
        )

        mae_fig_small = mean_absolute_error(
            df,
            df_without_tariff,
            df_base_simulation,
            "Economic Metrics", 
            year_range=selected_year_range,
            small_value=True,
            dark_mode=dark_mode
        )
        # Display the figure
        st.plotly_chart(mae_fig, use_container_width=True)
        st.plotly_chart(mae_fig_small, use_container_width=True)
        rmse_fig = root_mean_square_deviation(
            df,
            df_without_tariff,
            df_base_simulation,
            "Economic Metrics", 
            year_range=selected_year_range,
            dark_mode=dark_mode
        )

        rmse_fig_small = root_mean_square_deviation(
            df,
            df_without_tariff,
            df_base_simulation,
            "Economic Metrics", 
            year_range=selected_year_range,
            small_value=True
        )
        # Display the figure
        st.plotly_chart(rmse_fig, use_container_width=True)
        st.plotly_chart(rmse_fig_small, use_container_width=True)
        smape_fig = symmetric_mean_absolute_percentage_error(
            df,
            df_without_tariff,
            df_base_simulation,
            "Economic Metrics", 
            year_range=selected_year_range,
            dark_mode=dark_mode
        )

        smape_fig_small = symmetric_mean_absolute_percentage_error(
            df,
            df_without_tariff,
            df_base_simulation,
            "Economic Metrics", 
            year_range=selected_year_range,
            small_value=True,
            dark_mode=dark_mode
        )
        # Display the figure
        st.plotly_chart(smape_fig, use_container_width=True)
        st.plotly_chart(smape_fig_small, use_container_width=True)
        st.plotly_chart(render_overview_charts(df, "Key Economic Indicators Over Time - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_without_tariff, f"Key Economic Indicators Over Time - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_base_simulation, "Key Economic Indicators Over Time - FRB/US"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_base_simulation_with_tariff, "Key Economic Indicators Over Time - FRB/US - With Tariff"), use_container_width=True)

        st.plotly_chart(render_overview_tax_rates_charts(df, "Tax Rates - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_without_tariff, f"Tax Rates - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_base_simulation, "Tax Rates - FRB/US"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_base_simulation_with_tariff, "Tax Rates - FRB/US - With Tariff"), use_container_width=True)

        st.plotly_chart(render_overview_inflation_charts(df, "Inflation - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_overview_inflation_charts(df_without_tariff, f"Inflation - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_overview_inflation_charts(df_base_simulation, "Inflation - FRB/US"), use_container_width=True)
        st.plotly_chart(render_overview_inflation_charts(df_base_simulation_with_tariff, "Inflation - FRB/US - With Tariff"), use_container_width=True)

        st.plotly_chart(render_inflation_rate_comparison_charts(df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff), use_container_width=True)
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,    
            'pcpi', 'Inflation', 'Price Index'
        ), use_container_width=True)
        
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'unemployment', 'Unemployment', 'Percentage (%)'
        ), use_container_width=True)        
        
        st.plotly_chart(render_real_gdp_growth_comparison_charts(df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff), use_container_width=True)
        st.plotly_chart(render_nominal_gdp_growth_comparison_charts(df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff), use_container_width=True)
        
        st.plotly_chart(render_actions_charts(df, "Actions - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_actions_charts(df_without_tariff, f"Actions - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_actions_charts(df_base_simulation, "Actions - FRB/US"), use_container_width=True)
        st.plotly_chart(render_actions_charts(df_base_simulation_with_tariff, "Actions - FRB/US - With Tariff"), use_container_width=True)
    elif selected_view == "GDP Metrics":
        st.plotly_chart(render_gdp_charts(df, "GDP Components - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_without_tariff, f"GDP Components - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_base_simulation, "GDP Components - FRB/US"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_base_simulation_with_tariff, "GDP Components - FRB/US - With Tariff"), use_container_width=True)
        
        st.plotly_chart(render_gdp_charts_comparison(df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff), use_container_width=True)
        st.plotly_chart(render_gdp_charts_comparison_nominal(df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff), use_container_width=True)
        # Additional GDP metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Real GDP - RL-FRB/US Agent",
                    f"${df['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['real_gdp'].iloc[-1] - df['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Real GDP - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['real_gdp'].iloc[-1] - df_without_tariff['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - FRB/US",
                    f"${df_base_simulation['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['real_gdp'].iloc[-1] - df_base_simulation['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['real_gdp'].iloc[-1] - df_base_simulation_with_tariff['real_gdp'].iloc[-2]):,.2f}B"
                )
            with col2:
                st.metric(
                    "Nominal GDP - RL-FRB/US Agent",
                    f"${df['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['nominal_gdp'].iloc[-1] - df['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Nominal GDP - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['nominal_gdp'].iloc[-1] - df_without_tariff['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - FRB/US",
                    f"${df_base_simulation['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['nominal_gdp'].iloc[-1] - df_base_simulation['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['nominal_gdp'].iloc[-1] - df_base_simulation_with_tariff['nominal_gdp'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying GDP metrics: {str(e)}")
    elif selected_view == "Revenue and Expenditure Metrics":
        st.plotly_chart(render_tax_charts(df, "Tax Revenue Components - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_without_tariff, f"Tax Revenue Components - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_base_simulation, "Tax Revenue Components - FRB/US"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_base_simulation_with_tariff, "Tax Revenue Components - FRB/US - With Tariff"), use_container_width=True)
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'personal_tax', 'Personal Tax', 'Value'
        ), use_container_width=True)
        
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'corporate_tax', 'Corporate Tax', 'Value'
        ), use_container_width=True)
        
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'debt_to_gdp', 'Government Debt to GDP', 'Billions of $'
        ), use_container_width=True)
        
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'federal_surplus', 'Federal Surplus', 'Billions of $',
            chart_type='scatter'
        ), use_container_width=True)
        
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'federal_expenditures', 'Federal Expenditures', 'Value'
        ), use_container_width=True)
        
        st.plotly_chart(render_comparison_chart(
            df, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff,
            'transfer_payments_ratio', 'Transfer Payments Ratio', 'Value'
        ), use_container_width=True)        
        # Additional Revenue and Expenditure Metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Personal Tax Revenue - RL-FRB/US Agent",
                    f"${df['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df['personal_tax'].iloc[-1] - df['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Personal Tax Revenue - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['personal_tax'].iloc[-1] - df_without_tariff['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - FRB/US",
                    f"${df_base_simulation['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['personal_tax'].iloc[-1] - df_base_simulation['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['personal_tax'].iloc[-1] - df_base_simulation_with_tariff['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - RL-FRB/US Agent",
                    f"${df['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df['corporate_tax'].iloc[-1] - df['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Corporate Tax Revenue - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['corporate_tax'].iloc[-1] - df_without_tariff['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - FRB/US",
                    f"${df_base_simulation['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['corporate_tax'].iloc[-1] - df_base_simulation['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['corporate_tax'].iloc[-1] - df_base_simulation_with_tariff['corporate_tax'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying Revenue and Expenditure Metrics: {str(e)}")
    elif selected_view == "Trade Balance":
        st.plotly_chart(render_trade_charts(df, "Trade Balance and Components - RL-FRB/US Agent"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_without_tariff, f"Trade Balance and Components - {historical_label_or_hypothetical_label}"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_base_simulation, "Trade Balance and Components - FRB/US"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_base_simulation_with_tariff, "Trade Balance and Components - FRB/US - With Tariff"), use_container_width=True)
        # Additional trade metrics
        col1, col2, col3, col4 = st.columns(4)
        try:
            with col1:
                trade_balance = df['trade_balance'].iloc[-1]
                prev_trade_balance = df['trade_balance'].iloc[-2]
                trade_balance_without_tariff = df_without_tariff['trade_balance'].iloc[-1]
                prev_trade_balance_without_tariff = df_without_tariff['trade_balance'].iloc[-2]
                trade_balance_base_simulation = df_base_simulation['trade_balance'].iloc[-1]
                prev_trade_balance_base_simulation = df_base_simulation['trade_balance'].iloc[-2]
                trade_balance_base_simulation_with_tariff = df_base_simulation_with_tariff['trade_balance'].iloc[-1]
                prev_trade_balance_base_simulation_with_tariff = df_base_simulation_with_tariff['trade_balance'].iloc[-2]
                st.metric(
                    "Trade Balance - RL-FRB/US Agent",
                    f"${trade_balance:,.2f}B",
                    f"{(trade_balance - prev_trade_balance):,.2f}B"
                )
                st.metric(
                    f"Trade Balance - {historical_label_or_hypothetical_label}",
                    f"${trade_balance_without_tariff:,.2f}B",
                    f"{(trade_balance_without_tariff - prev_trade_balance_without_tariff):,.2f}B"
                )
                st.metric(
                    "Trade Balance - FRB/US",
                    f"${trade_balance_base_simulation:,.2f}B",
                    f"{(trade_balance_base_simulation - prev_trade_balance_base_simulation):,.2f}B"
                )
                st.metric(
                    "Trade Balance - FRB/US - With Tariff",
                    f"${trade_balance_base_simulation_with_tariff:,.2f}B",
                    f"{(trade_balance_base_simulation_with_tariff - prev_trade_balance_base_simulation_with_tariff):,.2f}B"
                )
            with col2:
                st.metric(
                    "Exports - RL-FRB/US Agent",
                    f"${df['exports'].iloc[-1]:,.2f}B",
                    f"{(df['exports'].iloc[-1] - df['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Exports - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['exports'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['exports'].iloc[-1] - df_without_tariff['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - FRB/US",
                    f"${df_base_simulation['exports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['exports'].iloc[-1] - df_base_simulation['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['exports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['exports'].iloc[-1] - df_base_simulation_with_tariff['exports'].iloc[-2]):,.2f}B"
                )

            with col3:
                st.metric(
                    "Imports - RL-FRB/US Agent",
                    f"${df['imports'].iloc[-1]:,.2f}B",
                    f"{(df['imports'].iloc[-1] - df['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Imports - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['imports'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['imports'].iloc[-1] - df_without_tariff['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - FRB/US",
                    f"${df_base_simulation['imports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['imports'].iloc[-1] - df_base_simulation['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['imports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['imports'].iloc[-1] - df_base_simulation_with_tariff['imports'].iloc[-2]):,.2f}B"
                )
            with col4:
                st.metric(
                    "Net Foreign Investment Income - RL-FRB/US Agent",
                    f"${df['net_foreign_investment_income'].iloc[-1]:,.2f}B",
                    f"{(df['net_foreign_investment_income'].iloc[-1] - df['net_foreign_investment_income'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    f"Net Foreign Investment Income - {historical_label_or_hypothetical_label}",
                    f"${df_without_tariff['net_foreign_investment_income'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['net_foreign_investment_income'].iloc[-1] - df_without_tariff['net_foreign_investment_income'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Net Foreign Investment Income - FRB/US",
                    f"${df_base_simulation['net_foreign_investment_income'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['net_foreign_investment_income'].iloc[-1] - df_base_simulation['net_foreign_investment_income'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Net Foreign Investment Income - FRB/US - With Tariff",
                    f"${df_base_simulation_with_tariff['net_foreign_investment_income'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['net_foreign_investment_income'].iloc[-1] - df_base_simulation_with_tariff['net_foreign_investment_income'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying trade metrics: {str(e)}")
