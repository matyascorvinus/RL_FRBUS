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
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
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
    
    # Calculate trade balance
    df['trade_balance'] = df['exports'] - df['imports']
    
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
    
    fig.update_layout(
        title=title,
        barmode='overlay',
        xaxis_title='Quarter',
        yaxis_title='Value',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
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
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
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
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
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
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
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
    traces = [
        {
            'x': df_rl_tariff['quarter'],
            'y': df_rl_tariff[metric],
            'name': f'{title} (AI Decision Makers/FRBUS)',
            'marker_color': MUTED_REDS['dark']
        },
        {
            'x': df_without_tariff['quarter'],
            'y': df_without_tariff[metric],
            'name': f'{title} (AI Decision Makers/FRBUS without Tariff)',
            'marker_color': MUTED_REDS['bright']
        },
        {
            'x': df_base_simulation['quarter'],
            'y': df_base_simulation[metric],
            'name': f'{title} (FRBUS-Based Simulation - Without Tariff)',
            'marker_color': MUTED_REDS['light']
        },
        {
            'x': df_base_simulation_with_tariff['quarter'],
            'y': df_base_simulation_with_tariff[metric],
            'name': f'{title} (FRBUS-Based Simulation - With Tariff)',
            'marker_color': MUTED_REDS['lightest']
        }
    ]
    for trace in traces:
        if chart_type == 'bar':
            # Add bar-specific parameters
            trace.update({
                'width': 0.25,
                'offset': -0.25 if 'FRBUS)' in trace['name'] else (0 if 'without Tariff)' in trace['name'] else 0.25)
            })
            fig.add_trace(go.Bar(**trace))
        else:  # scatter
            fig.add_trace(go.Scatter(**trace))
    
    # Update layout
    fig.update_layout(
        title=f'{title} Comparison',
        xaxis_title='Quarter',
        yaxis_title=y_axis_title,
        barmode='overlay' if chart_type == 'bar' else None,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
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
        name='Inflation - AI Decision Makers/FRBUS', 
        line=dict(color=MUTED_REDS['dark'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_without_tariff['quarter'], 
        y=inflation_without, 
        name='Inflation - AI Decision Makers/FRBUS without Tariff', 
        line=dict(color=MUTED_REDS['bright'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation['quarter'], 
        y=inflation_base, 
        name='Inflation - FRBUS-Based Simulation - Without Tariff', 
        line=dict(color=MUTED_REDS['light'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation_with_tariff['quarter'], 
        y=inflation_base_with_tariff, 
        name='Inflation - FRBUS-Based Simulation - With Tariff', 
        line=dict(color=MUTED_REDS['lightest'])
    ))

    fig.update_layout(
        title='Inflation Comparison',
        xaxis_title='Quarter',
        yaxis_title='Percentage (%)',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

def render_inflation_comparison_charts(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render inflation comparison charts"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['pcpi'], name='Inflation - AI Decision Makers/FRBUS', marker_color=MUTED_REDS['dark']))
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['pcpi'], name='Inflation - AI Decision Makers/FRBUS without Tariff', marker_color=MUTED_REDS['bright']))
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['pcpi'], name='Inflation - FRBUS-Based Simulation - Without Tariff', marker_color=MUTED_REDS['light']))

    fig.update_layout(
        title='Inflation Comparison',
        xaxis_title='Quarter',
        yaxis_title='Price Index',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
    )
    return fig

def render_unemployment_comparison_charts(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render unemployment comparison charts"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['unemployment'], name='Unemployment - AI Decision Makers/FRBUS', marker_color=MUTED_REDS['dark']))
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['unemployment'], name='Unemployment - AI Decision Makers/FRBUS without Tariff', marker_color=MUTED_REDS['bright']))
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['unemployment'], name='Unemployment - FRBUS-Based Simulation - Without Tariff', marker_color=MUTED_REDS['light']))

    fig.update_layout(
        title='Unemployment Comparison',
        xaxis_title='Quarter',
        yaxis_title='Percentage (%)',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
    )
    return fig

def render_gdp_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    # Add GDP components for AI Decision Makers/FRBUS
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['real_gdp'], 
                        name='Real GDP (AI Decision Makers/FRBUS)', 
                        marker_color=MUTED_REDS['dark'],  
                        width=bar_width,
                        offset=-bar_width)) 
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['real_gdp'], 
                        name='Real GDP (AI Decision Makers/FRBUS without Tariff)', 
                        marker_color=MUTED_REDS['bright'],  
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRBUS-Based Simulation - Without Tariff
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['real_gdp'], 
                        name='Real GDP (FRBUS-Based Simulation - Without Tariff)', 
                        marker_color=MUTED_REDS['light'],  
                        width=bar_width,
                        offset=bar_width))
    
    # Update layout with dark mode colors
    fig.update_layout(
        title='Real GDP Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
        font=dict(color='#E0E0E0'),  # Light gray text
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'  # Semi-transparent black background
        )
    )
    
    # Update axes for dark mode
    fig.update_xaxes(gridcolor='#303030', zerolinecolor='#303030')
    fig.update_yaxes(gridcolor='#303030', zerolinecolor='#303030')
    
    return fig

def render_gdp_charts_comparison_nominal(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for AI Decision Makers/FRBUS
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['nominal_gdp'], 
                        name='Nominal GDP (AI Decision Makers/FRBUS)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['nominal_gdp'], 
                        name='Nominal GDP (AI Decision Makers/FRBUS without Tariff)', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRBUS-Based Simulation - Without Tariff
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['nominal_gdp'], 
                        name='Nominal GDP (FRBUS-Based Simulation - Without Tariff)', marker_color=MUTED_REDS['light'],
                        width=bar_width,
                        offset=bar_width)) 
    
    fig.update_layout(
        title='Nominal GDP Components Comparison',
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
        font=dict(color='#E0E0E0'),  # Light gray text
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'  # Semi-transparent black background
        )
    )
    
    return fig

def render_personal_tax_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for AI Decision Makers/FRBUS
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['personal_tax'], 
                        name='Personal Tax (AI Decision Makers/FRBUS)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['personal_tax'], 
                        name='Personal Tax (AI Decision Makers/FRBUS without Tariff)', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRBUS-Based Simulation - Without Tariff
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['personal_tax'], 
                        name='Personal Tax (FRBUS-Based Simulation - Without Tariff)', marker_color=MUTED_REDS['light'],
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
    
    # Add GDP components for AI Decision Makers/FRBUS
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['corporate_tax'], 
                        name='Corporate Tax (AI Decision Makers/FRBUS)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['corporate_tax'], 
                        name='Corporate Tax (AI Decision Makers/FRBUS without Tariff)', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRBUS-Based Simulation - Without Tariff
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['corporate_tax'], 
                        name='Corporate Tax (FRBUS-Based Simulation - Without Tariff)', marker_color=MUTED_REDS['light'],
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
    
    # Add GDP components for AI Decision Makers/FRBUS
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['government_transfer_payments'], 
                        name='Government Transfer Payments (AI Decision Makers/FRBUS)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['government_transfer_payments'], 
                        name='Government Transfer Payments (AI Decision Makers/FRBUS without Tariff)', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRBUS-Based Simulation - Without Tariff
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['government_transfer_payments'], 
                        name='Government Transfer Payments (FRBUS-Based Simulation - Without Tariff)', marker_color=MUTED_REDS['light'],
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
    
    # Add GDP components for AI Decision Makers/FRBUS
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['debt_to_gdp'], 
                        name='Government Debt to GDP (AI Decision Makers/FRBUS)', marker_color=MUTED_REDS['dark'],
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['debt_to_gdp'], 
                        name='Government Debt to GDP (AI Decision Makers/FRBUS without Tariff)', marker_color=MUTED_REDS['bright'],
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for FRBUS-Based Simulation - Without Tariff
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['debt_to_gdp'], 
                        name='Government Debt to GDP (FRBUS-Based Simulation - Without Tariff)', marker_color=MUTED_REDS['light'],
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
        name='Real GDP Growth - AI Decision Makers/FRBUS', 
        line=dict(color=MUTED_REDS['dark'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_without_tariff['quarter'], 
        y=gdp_growth_without, 
        name='Real GDP Growth - AI Decision Makers/FRBUS without Tariff', 
        line=dict(color=MUTED_REDS['bright'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation['quarter'], 
        y=gdp_growth_base, 
        name='Real GDP Growth - FRBUS-Based Simulation - Without Tariff', 
        line=dict(color=MUTED_REDS['light'])
    ))

    fig.add_trace(go.Scatter(
        x=df_base_simulation_with_tariff['quarter'], 
        y=gdp_growth_base_with_tariff, 
        name='Real GDP Growth - FRBUS-Based Simulation - With Tariff', 
        line=dict(color=MUTED_REDS['lightest'])
    ))

    fig.update_layout(
        title='Real GDP Growth Rate Comparison (Annualized %)',
        xaxis_title='Quarter',
        yaxis_title='Growth Rate (%)',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
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
        name='Nominal GDP Growth - AI Decision Makers/FRBUS', 
        line=dict(color=MUTED_REDS['dark'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_without_tariff['quarter'], 
        y=gdp_growth_without, 
        name='Nominal GDP Growth - AI Decision Makers/FRBUS without Tariff', 
        line=dict(color=MUTED_REDS['bright'])
    ))
    
    fig.add_trace(go.Scatter(
        x=df_base_simulation['quarter'], 
        y=gdp_growth_base, 
        name='Nominal GDP Growth - FRBUS-Based Simulation - Without Tariff', 
        line=dict(color=MUTED_REDS['light'])
    ))

    fig.add_trace(go.Scatter(
        x=df_base_simulation_with_tariff['quarter'], 
        y=gdp_growth_base_with_tariff, 
        name='Nominal GDP Growth - FRBUS-Based Simulation - With Tariff', 
        line=dict(color=MUTED_REDS['lightest'])
    ))

    fig.update_layout(
        title='Nominal GDP Growth Rate Comparison (Annualized %)',
        xaxis_title='Quarter',
        yaxis_title='Growth Rate (%)',
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
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
                'federal_surplus': metrics['metrics']['gfsrpn']
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
                'federal_surplus': metrics_without_tariff['metrics']['gfsrpn']
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
                'federal_surplus': metrics_base_simulation['metrics']['gfsrpn']
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
                'federal_surplus': metrics_base_simulation_with_tariff['metrics']['gfsrpn']
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

# Connection controls
with col0: 
    if st.button('Resume', use_container_width=True):
        # Run simulation by calling the API endpoint
        response = requests.get('http://localhost:8000/run_simulation_resume')
        if response.status_code == 200:
            simulation_status.success("Training resumed")
        else:
            simulation_status.error("Failed to resume training")
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

if st.sidebar.button('Start Simulation After Training', use_container_width=True):
    thread = threading.Thread(target=st.session_state.stream.connect)
    thread.daemon = True
    add_script_run_ctx(thread)
    thread.start()
    # Run simulation by calling the API endpoint
    response = requests.get('http://localhost:8000/run_simulation')
    if response.status_code == 200 or response.status_code == 204:
        simulation_status.success("Simulation started")     
    else:
        simulation_status.error("Failed to start simulation")

if st.sidebar.button('Save Simulation', use_container_width=True, disabled=(not (hasattr(st.session_state.stream, 'metrics_history_rl_tariff') and not st.session_state.stream.metrics_history_rl_tariff.empty))):
    # Save the simulation data to a CSV file
    df_rl_tariff = st.session_state.stream.metrics_history_rl_tariff
    df_rl_tariff['simulation_type'] = 'AI Decision Makers/FRBUS'
    df_without_tariff = st.session_state.stream.metrics_history_without_tariff
    df_without_tariff['simulation_type'] = 'AI Decision Makers/FRBUS without Tariff'
    df_base_simulation = st.session_state.stream.metrics_history_base_simulation
    df_base_simulation['simulation_type'] = 'FRBUS-Based Simulation - Without Tariff'
    df_base_simulation_with_tariff = st.session_state.stream.metrics_history_base_simulation_with_tariff
    df_base_simulation_with_tariff['simulation_type'] = 'FRBUS-Based Simulation - With Tariff'

    df_rl_tariff.to_csv('simulation_data_rl_tariff.csv', index=False)
    df_without_tariff.to_csv('simulation_data_without_tariff.csv', index=False)
    df_base_simulation.to_csv('simulation_data_base_simulation.csv', index=False)
    df_base_simulation_with_tariff.to_csv('simulation_data_base_simulation_with_tariff.csv', index=False)
    # Combine all dataframes
    combined_df = pd.concat([df_rl_tariff, df_without_tariff, df_base_simulation, df_base_simulation_with_tariff], 
                          ignore_index=True)
    
    # Save combined data to a single CSV
    combined_df.to_csv('combined_simulation_data.csv', index=False)
    simulation_status.success("Simulation data saved to CSV files") 

    st.sidebar.download_button(
        label="Download Combined Simulation Data",
        data=combined_df.to_csv(index=False),
        file_name="combined_simulation_data.csv",
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
        st.plotly_chart(render_overview_charts(df, "Key Economic Indicators Over Time - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_without_tariff, "Key Economic Indicators Over Time - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_base_simulation, "Key Economic Indicators Over Time - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_base_simulation_with_tariff, "Key Economic Indicators Over Time - FRBUS-Based Simulation - With Tariff"), use_container_width=True)

        st.plotly_chart(render_overview_tax_rates_charts(df, "Tax Rates - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_without_tariff, "Tax Rates - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_base_simulation, "Tax Rates - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_base_simulation_with_tariff, "Tax Rates - FRBUS-Based Simulation - With Tariff"), use_container_width=True)

        st.plotly_chart(render_overview_inflation_charts(df, "Inflation - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_overview_inflation_charts(df_without_tariff, "Inflation - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_inflation_charts(df_base_simulation, "Inflation - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_inflation_charts(df_base_simulation_with_tariff, "Inflation - FRBUS-Based Simulation - With Tariff"), use_container_width=True)

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
        
        st.plotly_chart(render_actions_charts(df, "Actions - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_actions_charts(df_without_tariff, "Actions - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_actions_charts(df_base_simulation, "Actions - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_actions_charts(df_base_simulation_with_tariff, "Actions - FRBUS-Based Simulation - With Tariff"), use_container_width=True)
    elif selected_view == "GDP Metrics":
        st.plotly_chart(render_gdp_charts(df, "GDP Components - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_without_tariff, "GDP Components - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_base_simulation, "GDP Components - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_base_simulation_with_tariff, "GDP Components - FRBUS-Based Simulation - With Tariff"), use_container_width=True)
        
        st.plotly_chart(render_gdp_charts_comparison(df, df_without_tariff, df_base_simulation), use_container_width=True)
        st.plotly_chart(render_gdp_charts_comparison_nominal(df, df_without_tariff, df_base_simulation), use_container_width=True)
        # Additional GDP metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Real GDP - AI Decision Makers/FRBUS",
                    f"${df['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['real_gdp'].iloc[-1] - df['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - AI Decision Makers/FRBUS without Tariff",
                    f"${df_without_tariff['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['real_gdp'].iloc[-1] - df_without_tariff['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - FRBUS-Based Simulation - Without Tariff",
                    f"${df_base_simulation['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['real_gdp'].iloc[-1] - df_base_simulation['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - FRBUS-Based Simulation - With Tariff",
                    f"${df_base_simulation_with_tariff['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['real_gdp'].iloc[-1] - df_base_simulation_with_tariff['real_gdp'].iloc[-2]):,.2f}B"
                )
            with col2:
                st.metric(
                    "Nominal GDP - AI Decision Makers/FRBUS",
                    f"${df['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['nominal_gdp'].iloc[-1] - df['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - AI Decision Makers/FRBUS without Tariff",
                    f"${df_without_tariff['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['nominal_gdp'].iloc[-1] - df_without_tariff['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - FRBUS-Based Simulation - Without Tariff",
                    f"${df_base_simulation['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['nominal_gdp'].iloc[-1] - df_base_simulation['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - FRBUS-Based Simulation - With Tariff",
                    f"${df_base_simulation_with_tariff['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['nominal_gdp'].iloc[-1] - df_base_simulation_with_tariff['nominal_gdp'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying GDP metrics: {str(e)}")
    elif selected_view == "Revenue and Expenditure Metrics":
        st.plotly_chart(render_tax_charts(df, "Tax Revenue Components - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_without_tariff, "Tax Revenue Components - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_base_simulation, "Tax Revenue Components - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_base_simulation_with_tariff, "Tax Revenue Components - FRBUS-Based Simulation - With Tariff"), use_container_width=True)
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
                    "Personal Tax Revenue - AI Decision Makers/FRBUS",
                    f"${df['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df['personal_tax'].iloc[-1] - df['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - AI Decision Makers/FRBUS without Tariff",
                    f"${df_without_tariff['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['personal_tax'].iloc[-1] - df_without_tariff['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - FRBUS-Based Simulation - Without Tariff",
                    f"${df_base_simulation['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['personal_tax'].iloc[-1] - df_base_simulation['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - FRBUS-Based Simulation - With Tariff",
                    f"${df_base_simulation_with_tariff['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['personal_tax'].iloc[-1] - df_base_simulation_with_tariff['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - AI Decision Makers/FRBUS",
                    f"${df['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df['corporate_tax'].iloc[-1] - df['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - AI Decision Makers/FRBUS without Tariff",
                    f"${df_without_tariff['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['corporate_tax'].iloc[-1] - df_without_tariff['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - FRBUS-Based Simulation - Without Tariff",
                    f"${df_base_simulation['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['corporate_tax'].iloc[-1] - df_base_simulation['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - FRBUS-Based Simulation - With Tariff",
                    f"${df_base_simulation_with_tariff['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['corporate_tax'].iloc[-1] - df_base_simulation_with_tariff['corporate_tax'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying Revenue and Expenditure Metrics: {str(e)}")
    elif selected_view == "Trade Balance":
        st.plotly_chart(render_trade_charts(df, "Trade Balance and Components - AI Decision Makers/FRBUS"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_without_tariff, "Trade Balance and Components - AI Decision Makers/FRBUS without Tariff"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_base_simulation, "Trade Balance and Components - FRBUS-Based Simulation - Without Tariff"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_base_simulation_with_tariff, "Trade Balance and Components - FRBUS-Based Simulation - With Tariff"), use_container_width=True)
        # Additional trade metrics
        col1, col2, col3 = st.columns(3)
        try:
            with col1:
                trade_balance = df['exports'].iloc[-1] - df['imports'].iloc[-1]
                prev_trade_balance = df['exports'].iloc[-2] - df['imports'].iloc[-2]
                trade_balance_without_tariff = df_without_tariff['exports'].iloc[-1] - df_without_tariff['imports'].iloc[-1]
                prev_trade_balance_without_tariff = df_without_tariff['exports'].iloc[-2] - df_without_tariff['imports'].iloc[-2]
                trade_balance_base_simulation = df_base_simulation['exports'].iloc[-1] - df_base_simulation['imports'].iloc[-1]
                prev_trade_balance_base_simulation = df_base_simulation['exports'].iloc[-2] - df_base_simulation['imports'].iloc[-2]
                trade_balance_base_simulation_with_tariff = df_base_simulation_with_tariff['exports'].iloc[-1] - df_base_simulation_with_tariff['imports'].iloc[-1]
                prev_trade_balance_base_simulation_with_tariff = df_base_simulation_with_tariff['exports'].iloc[-2] - df_base_simulation_with_tariff['imports'].iloc[-2]
                st.metric(
                    "Trade Balance - AI Decision Makers/FRBUS",
                    f"${trade_balance:,.2f}B",
                    f"{(trade_balance - prev_trade_balance):,.2f}B"
                )
                st.metric(
                    "Trade Balance - AI Decision Makers/FRBUS without Tariff",
                    f"${trade_balance_without_tariff:,.2f}B",
                    f"{(trade_balance_without_tariff - prev_trade_balance_without_tariff):,.2f}B"
                )
                st.metric(
                    "Trade Balance - FRBUS-Based Simulation - Without Tariff",
                    f"${trade_balance_base_simulation:,.2f}B",
                    f"{(trade_balance_base_simulation - prev_trade_balance_base_simulation):,.2f}B"
                )
                st.metric(
                    "Trade Balance - FRBUS-Based Simulation - With Tariff",
                    f"${trade_balance_base_simulation_with_tariff:,.2f}B",
                    f"{(trade_balance_base_simulation_with_tariff - prev_trade_balance_base_simulation_with_tariff):,.2f}B"
                )
            with col2:
                st.metric(
                    "Exports - AI Decision Makers/FRBUS",
                    f"${df['exports'].iloc[-1]:,.2f}B",
                    f"{(df['exports'].iloc[-1] - df['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - AI Decision Makers/FRBUS without Tariff",
                    f"${df_without_tariff['exports'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['exports'].iloc[-1] - df_without_tariff['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - FRBUS-Based Simulation - Without Tariff",
                    f"${df_base_simulation['exports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['exports'].iloc[-1] - df_base_simulation['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - FRBUS-Based Simulation - With Tariff",
                    f"${df_base_simulation_with_tariff['exports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['exports'].iloc[-1] - df_base_simulation_with_tariff['exports'].iloc[-2]):,.2f}B"
                )

            with col3:
                st.metric(
                    "Imports - AI Decision Makers/FRBUS",
                    f"${df['imports'].iloc[-1]:,.2f}B",
                    f"{(df['imports'].iloc[-1] - df['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - AI Decision Makers/FRBUS without Tariff",
                    f"${df_without_tariff['imports'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['imports'].iloc[-1] - df_without_tariff['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - FRBUS-Based Simulation - Without Tariff",
                    f"${df_base_simulation['imports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['imports'].iloc[-1] - df_base_simulation['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - FRBUS-Based Simulation - With Tariff",
                    f"${df_base_simulation_with_tariff['imports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation_with_tariff['imports'].iloc[-1] - df_base_simulation_with_tariff['imports'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying trade metrics: {str(e)}")
