import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
import base64
from datetime import datetime
import zipfile
from io import BytesIO

# Color scheme for charts
MUTED_REDS = {
    'dark': '#B22222',    # Firebrick
    'light': '#CD5C5C'    # Indian Red
}

# Global list to store all figures for batch download
all_figures = []

def export_plot_as_png(fig, filename_prefix="plot"):
    """
    Export a Plotly figure as PNG bytes.
    
    Args:
        fig: Plotly figure object
        filename_prefix: Prefix for the filename
    
    Returns:
        PNG bytes
    """
    # Convert plot to PNG bytes
    img_bytes = fig.to_image(format="png", width=2400, height=800, scale=2)
    return img_bytes

def add_figure_to_collection(fig, filename):
    """
    Add a figure to the global collection for batch download.
    
    Args:
        fig: Plotly figure object
        filename: Name for the file (without extension)
    """
    global all_figures
    all_figures.append({
        'figure': fig,
        'filename': f"{filename}.png"
    })

def create_zip_of_all_figures():
    """
    Create a ZIP file containing all collected figures as PNG files.
    
    Returns:
        BytesIO object containing the ZIP file
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for item in all_figures:
            png_bytes = export_plot_as_png(item['figure'], item['filename'])
            zip_file.writestr(item['filename'], png_bytes)
    
    zip_buffer.seek(0)
    return zip_buffer

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
        'Personal Income Tax Revenue', 'Corporate Income Tax Revenue', 'Exports', 'Imports', 'Debt to GDP',
        'Interest Rate', 'PCPI', 'Transfer Payments Ratio', 'Federal Expenditures',
        'Personal Income Tax Rates', 'Corporate Income Tax Rates', 'Government Transfer Payments', 
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

def mean_absolute_error_with_export(df, df_without_tariff, df_without_rl, title, year_range=None, small_value=False, dark_mode=False, show_export_button=True):
    """
    Enhanced version of mean_absolute_error with PNG export functionality.
    """
    fig = mean_absolute_error(df, df_without_tariff, df_without_rl, title, year_range, small_value, dark_mode)
    
    if show_export_button:
        # Add to collection for batch download
        year_suffix = f"_{year_range[0]}_{year_range[1]}" if year_range else ""
        add_figure_to_collection(fig, f"mae_comparison{year_suffix}")
    
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

def root_mean_square_deviation_with_export(df, df_without_tariff, df_without_rl, title, year_range=None, small_value=False, dark_mode=False, show_export_button=True):
    """
    Enhanced version of root_mean_square_deviation with PNG export functionality.
    """
    fig = root_mean_square_deviation(df, df_without_tariff, df_without_rl, title, year_range, small_value, dark_mode)
    
    if show_export_button:
        # Add to collection for batch download
        year_suffix = f"_{year_range[0]}_{year_range[1]}" if year_range else ""
        add_figure_to_collection(fig, f"rmse_comparison{year_suffix}")
    
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

def symmetric_mean_absolute_percentage_error_with_export(df, df_without_tariff, df_without_rl, title, year_range=None, small_value=False, dark_mode=False, show_export_button=True):
    """
    Enhanced version of symmetric_mean_absolute_percentage_error with PNG export functionality.
    """
    fig = symmetric_mean_absolute_percentage_error(df, df_without_tariff, df_without_rl, title, year_range, small_value, dark_mode)
    
    if show_export_button:
        # Add to collection for batch download
        year_suffix = f"_{year_range[0]}_{year_range[1]}" if year_range else ""
        add_figure_to_collection(fig, f"smape_comparison{year_suffix}")
    
    return fig

# ---------------------------
# Sidebar: Select Data Source
# ---------------------------
# Choose between Historical Data and Trump Tariff plan.
data_source = st.sidebar.radio(
    "Select Data Source", 
    options=[
        "FRB/US FTPL", 
        "Historical Data FTPL Reserve 2000-2024", 
        "Historical Data FTPL Reserve 1985-2024",
        "Historical Data", 
        "Historical Data 2022-2024", 
        "Stephen Miran Tariff plan 50%", 
        "Trump Tariff plan 10%", 
        "Trump Tariff plan 20%", 
        "Trump Tariff plan 50%", 
        "Trump Tariff plan 100%"
    ],
    index=0
)

# ---------------------------
# Sidebar: PNG Export Options
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download All Figures")
enable_png_export = st.sidebar.checkbox(
    "Collect Figures for Download", 
    value=True, 
    help="Automatically collect all displayed figures for batch download as PNG files"
)

st.sidebar.markdown("""
**Export Settings:**
- **Format**: PNG (High Resolution)
- **Resolution**: 2400x800 pixels @ 2x scale
- **Packaging**: All figures in a single ZIP file

**Auto-Generation Feature:**
When enabled, the app automatically generates charts for **ALL metrics** 
(not just the selected one), so you get every possible chart in one download!
""")

# Add download button at the end of the page (will be shown in sidebar)
# We'll add a placeholder here and populate it later
download_placeholder = st.sidebar.empty()

# Let the user choose which metric to compare.
metric_options = [
    "GDP Growth (%)",
    "CPI (Inflation Index)",
    "Unemployment Rate (%)",
    "Real GDP (Billion)",
    "Nominal GDP (Billion)",
    "Federal Expenditures (in Billions)",
    "Personal Income Tax Revenue (Billion)",
    "Corporate Income Tax Revenue (Billion)",
    "Exports (Billion)",
    "Imports (Billion)",
    "Debt",
    "Federal Fund Rate (%)",
    "Government Transfer Payments (Billion)",
    "Federal Surplus (Billion)",
    "Trade Balance (Billion)",
    "Net Foreign Investment Income (Billion)",
]

# Set file path based on the data source selection.
data_file = "combined_simulation_data_1975_2024.csv" 
if data_source != "Historical Data":
    if data_source == "FRB/US FTPL":
        data_file = "combined_sim_data_ftpl_vs_non_ftpl.csv"
    if data_source == "Historical Data FTPL Reserve 2000-2024":
        data_file = "combined_simulation_data_2000_2024_Federal-Reserve-FTPL.csv" 
    if data_source == "Historical Data FTPL Reserve 1985-2024":
        data_file = "combined_simulation_data_1985_2024_Federal-Reserve-FTPL.csv" 
    if data_source == "Historical Data 2022-2024":
        data_file = "combined_simulation_data_2000_2024.csv" 
    if data_source == "Stephen Miran Tariff plan 50%":
        data_file = "combined_simulation_data_with_tariff_2025_2030.csv" 
    if data_source == "Trump Tariff plan 10%":
        data_file = "combined_simulation_data-10.csv" 
    if data_source == "Trump Tariff plan 20%":
        data_file = "combined_simulation_data-20.csv" 
    if data_source == "Trump Tariff plan 50%":
        data_file = "combined_simulation_data-50.csv" 
    if data_source == "Trump Tariff plan 100%":
        data_file = "combined_simulation_data-100.csv" 

# ---------------------------
# Load Data Function (with caching)
# ---------------------------

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Create a numeric representation of the quarter to facilitate ordering in charts.
    # Assumes the 'quarter' column is in the format "YYYYQx" (e.g., "2024Q1")
    def quarter_to_numeric(q_str):
        year = int(q_str[:4])
        quarter = int(q_str[5])
        return year + (quarter - 1) / 4
    
    data["quarter_numeric"] = data["quarter"].apply(quarter_to_numeric)
    
    # Translate (rename) the column names into more user-friendly English labels.
    rename_dict = {
        "quarter": "Quarter",
        "gdp_growth": "GDP Growth (%)",
        "inflation": "CPI (Inflation Index)",
        "unemployment": "Unemployment Rate (%)",
        "real_gdp": "Real GDP (Billion)",
        "nominal_gdp": "Nominal GDP (Billion)",
        "personal_tax": "Personal Income Tax Revenue (Billion)",
        "corporate_tax": "Corporate Income Tax Revenue (Billion)",
        "exports": "Exports (Billion)",
        "imports": "Imports (Billion)",
        "debt_to_gdp": "Debt",
        "interest_rate": "Federal Fund Rate (%)",
        "pcpi": "PCPI",
        "transfer_payments_ratio": "Transfer Payments Ratio",
        "federal_expenditures": "Federal Expenditures (in Billions)",
        "personal_tax_rates": "Personal Income Tax Rates (%)",
        "corporate_tax_rates": "Corporate Income Tax Rates (%)",
        "government_transfer_payments": "Government Transfer Payments (Billion)",
        "federal_surplus": "Federal Surplus (Billion)",
        "simulation_type": "Simulation Type",
        "trade_balance": "Trade Balance (Billion)",
        "net_foreign_investment_income": "Net Foreign Investment Income (Billion)"
    }
    data = data.rename(columns=rename_dict)
    
    # Create a new column "Year" by extracting the first 4 characters from the Quarter column.
    data["Year"] = data["Quarter"].str[:4]
    
    return data

# Load the selected dataset.
data = load_data(data_file)

# Set up page title and description
st.title("Combined Simulation Data Dashboard")
st.markdown(
    f"""
    This dashboard displays simulation data from the selected data source.
    
    **Data Source:** {data_source}   
    
    **Dashboard Features:**  
    - Sidebar filtering by Simulation Type  
    - Option to select Quarter Range  
    - Interactive charts comparing simulation metrics  
    """
)

# ---------------------------
# Sidebar: Filter by Simulation Type
# ---------------------------
simulation_types = data["Simulation Type"].unique().tolist()
selected_types = st.sidebar.multiselect(
    "Select Simulation Type",
    options=simulation_types,
    default=simulation_types  # Show all by default.
)

# Filter the dataset based on simulation type selection.
filtered_data = data[data["Simulation Type"].isin(selected_types)]

# Display the filtered data
st.subheader("Filtered Simulation Data")
if not filtered_data.empty:
    st.dataframe(filtered_data)
else:
    st.write("No data available for the selected simulation type(s).")

st.markdown("---")
st.subheader("Complete Data (for reference)")
st.dataframe(data)

# ---------------------------
# Chart: Comparison Across Simulation Types
# ---------------------------
st.markdown("---")
st.subheader("Comparison Chart Across Simulation Types")

selected_metric = st.selectbox("Select Metric for Comparison", metric_options)
st.write(f"Selected Metric: {selected_metric}")

def create_comparison_chart(data, metric):
    """Create a comparison chart for a specific metric."""
    fig = go.Figure()
    
    # Get unique simulation types
    unique_sim_types = data["Simulation Type"].unique()
    
    # Add a line trace for each simulation type
    for sim_type in unique_sim_types:
        sim_data = data[data["Simulation Type"] == sim_type]
        fig.add_trace(go.Scatter(
            x=sim_data["Quarter"],
            y=sim_data[metric],
            mode='lines+markers',
            name=sim_type,
            hovertemplate='<b>%{fullData.name}</b><br>Quarter: %{x}<br>' + metric + ': %{y}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{metric} Comparison Across Simulation Types",
        xaxis_title="Quarter",
        yaxis_title=metric,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(color='#000000'),
        yaxis_title_font=dict(size=20, color='black'),
        xaxis_title_font=dict(size=20, color='black'),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=18)
        )
    )
    
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=16, color='black'),
        gridcolor='#cccccc',
        zerolinecolor='#cccccc'
    )
    
    fig.update_yaxes(
        tickfont=dict(size=16, color='black'),
        gridcolor='#cccccc',
        zerolinecolor='#cccccc'
    )
    
    return fig

# Create and display the selected metric chart
comparison_fig = create_comparison_chart(filtered_data, selected_metric)
st.plotly_chart(comparison_fig, use_container_width=True)

# Generate charts for ALL metrics if export is enabled
if enable_png_export:
    progress_text = f"Generating {len(metric_options)} comparison charts for all metrics..."
    progress_bar = st.progress(0, text=progress_text)
    for idx, metric in enumerate(metric_options):
        fig = create_comparison_chart(filtered_data, metric)
        clean_metric_name = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')
        add_figure_to_collection(fig, f"comparison_{clean_metric_name}")
        progress_bar.progress((idx + 1) / len(metric_options), text=f"Generated {idx + 1}/{len(metric_options)} comparison charts")
    progress_bar.empty()  # Remove progress bar when done

# Add bar chart for comparison of key metrics across simulation types
st.subheader("Bar Chart Comparison of Key Metrics Across Simulation Types")


# Let the user choose which metric to use for the components comparison.
# (The default index 3 selects "Real GDP (Billion)" from the list below.)
component_metric = st.selectbox("Select Metric for Component Comparison", 
                                metric_options, index=3)

# --- Quarter Range Selection ---
# Use the filtered_data (from the sidebar selection) and further filter by quarter.
if not filtered_data.empty:
    # Create a sorted list of unique "Quarter" values based on the numeric representation.
    quarter_data = filtered_data[["Quarter", "quarter_numeric"]].drop_duplicates().sort_values("quarter_numeric")
    quarter_options = quarter_data["Quarter"].tolist()
    
    # Allow the user to select a quarter range.
    selected_quarter_range = st.select_slider(
        "Select Quarter Range",
        options=quarter_options,
        value=(quarter_options[0], quarter_options[-1])
    )
    
    # Retrieve the numeric values corresponding to the selected quarter range.
    start_quarter_str, end_quarter_str = selected_quarter_range
    start_numeric = quarter_data.loc[quarter_data["Quarter"] == start_quarter_str, "quarter_numeric"].iloc[0]
    end_numeric = quarter_data.loc[quarter_data["Quarter"] == end_quarter_str, "quarter_numeric"].iloc[0]
    
    # Further filter the data to only include rows within the selected quarter range.
    final_filtered_data = filtered_data[
        (filtered_data["quarter_numeric"] >= start_numeric) & 
        (filtered_data["quarter_numeric"] <= end_numeric)
    ]
else:
    final_filtered_data = filtered_data

# Use a color palette to support many simulation types.
# Light Orange, Orange, Red, Burgundian Red
color_palette = ['#FFD700', '#FFA500', '#FF4500', '#8B0000']

custom_pallets = ['#702963', '#FF0000', '#F39C7F', '#0000FF']

def render_component_comparison(dataframe, sim_types, metric):
    """Render a bar chart comparing the selected metric across multiple simulation types."""
    fig = go.Figure()
    
    # Loop over each simulation type provided.
    for i, sim in enumerate(sim_types):
        df_sim = dataframe[dataframe["Simulation Type"] == sim]
        fig.add_trace(go.Bar(
            x = df_sim['Quarter'],
            y = df_sim[metric],
            name = f'{metric} ({sim})',
            marker_color = custom_pallets[i % len(custom_pallets)]
        ))
    
    # Update chart layout for a light mode style
    fig.update_layout(
        title = f'{metric} Components Comparison',
        xaxis_title = 'Quarter',
        yaxis_title = metric,
        barmode = 'group',
        plot_bgcolor = 'rgba(255,255,255,1)',  # White background
        paper_bgcolor = 'rgba(255,255,255,1)',   # White paper
        font = dict(color='#000000'),            # Black text
        # Font for y axis title
        yaxis_title_font=dict(size=20, color='black'),
        # Font for x axis title
        xaxis_title_font=dict(size=20, color='black'),
        legend = dict(
            orientation = "h",
            yanchor = "bottom",
            y = 1.02,
            xanchor = "right",
            x = 1,
            font = dict(size=20)
        )
    )
    # fig.update_xaxes(gridcolor='#cccccc', zerolinecolor='#cccccc')
    # fig.update_yaxes(gridcolor='#cccccc', zerolinecolor='#cccccc')
    
    
    
    # Improve readability of x-axis labels
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=18, color='black'),  # Increased from 10 to 18
        gridcolor='black',
        zerolinecolor='#303030'
    )
    
    fig.update_yaxes(
        tickfont=dict(size=18, color='black'),  # Added explicit font size for y-axis
        gridcolor='black',
        zerolinecolor='#303030'
    )
    return fig


# Use the simulation types that the user selected in the sidebar.
sim_types = selected_types

if not final_filtered_data.empty and len(sim_types) > 0:
    st.markdown("---")
    st.subheader(f"{component_metric} Components Comparison")
    
    # Display the selected component metric chart
    comp_fig = render_component_comparison(final_filtered_data, sim_types, component_metric)
    st.plotly_chart(comp_fig, use_container_width=True)
    
    # Generate component comparison charts for ALL metrics if export is enabled
    if enable_png_export:
        progress_text_comp = f"Generating {len(metric_options)} component comparison charts for all metrics..."
        progress_bar_comp = st.progress(0, text=progress_text_comp)
        for idx, metric in enumerate(metric_options):
            fig = render_component_comparison(final_filtered_data, sim_types, metric)
            clean_metric_name = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')
            add_figure_to_collection(fig, f"component_comparison_{clean_metric_name}")
            progress_bar_comp.progress((idx + 1) / len(metric_options), text=f"Generated {idx + 1}/{len(metric_options)} component comparison charts")
        progress_bar_comp.empty()  # Remove progress bar when done
else:
    st.markdown("---")
    st.write("Some required datasets for the Component Comparison chart are missing.")

# ---------------------------
# Download All Figures Button
# ---------------------------
if enable_png_export and len(all_figures) > 0:
    st.markdown("---")
    st.subheader("📥 Download All Figures")
    
    # Show count of collected figures
    st.info(f"✅ Collected {len(all_figures)} figure(s) for download")
    
    # Categorize figures
    comparison_figs = [f for f in all_figures if f['filename'].startswith('comparison_') and not f['filename'].startswith('component_comparison_')]
    component_figs = [f for f in all_figures if f['filename'].startswith('component_comparison_')]
    other_figs = [f for f in all_figures if not f['filename'].startswith('comparison_') and not f['filename'].startswith('component_comparison_')]
    
    # List all collected figures  
    with st.expander("📋 View collected figures"):
        st.markdown("**Comparison Charts (Line):**")
        for item in comparison_figs:
            st.write(f"  • {item['filename']}")
        
        st.markdown("**Component Comparison Charts (Bar):**")
        for item in component_figs:
            st.write(f"  • {item['filename']}")
        
        if other_figs:
            st.markdown("**Other Charts:**")
            for item in other_figs:
                st.write(f"  • {item['filename']}")
    
    # Generate timestamp for the ZIP file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"all_figures_{timestamp}.zip"
    
    # Create the ZIP file
    zip_buffer = create_zip_of_all_figures()
    
    # Add download button in both main area and sidebar
    st.download_button(
        label="⬇️ Download All Figures as ZIP",
        data=zip_buffer,
        file_name=zip_filename,
        mime="application/zip",
        help=f"Download all {len(all_figures)} figures as high-resolution PNG files in a ZIP archive"
    )
    
    # Also add to sidebar using the placeholder
    download_placeholder.download_button(
        label="⬇️ Download All Figures",
        data=zip_buffer,
        file_name=zip_filename,
        mime="application/zip",
        help=f"Download {len(all_figures)} PNG files"
    )
    
    st.markdown(f"""
    **What's included:**
    - **{len(comparison_figs)} Comparison Line Charts** - One for each metric
    - **{len(component_figs)} Component Comparison Bar Charts** - One for each metric
    {f"- **{len(other_figs)} Other Charts** - Additional analysis charts" if other_figs else ""}
    
    **Technical Details:**
    - Resolution: 2400x800 pixels at 2x scale
    - Format: PNG (suitable for publications and presentations)
    - All metrics automatically generated (no need to select each one!)
    - Packaged in a single ZIP file for easy download
    """)
elif enable_png_export:
    st.markdown("---")
    st.info("📊 No figures collected yet. View charts above to collect them for download.")
 
