# from streamer import EconomicStream
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

metrics_history_rl_tariff = pd.DataFrame(columns=[
    'quarter', 'gdp_growth', 'inflation', 'unemployment',
    'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
    'exports', 'imports'
])

metrics_history_without_tariff = pd.DataFrame(columns=[
    'quarter', 'gdp_growth', 'inflation', 'unemployment',
    'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
    'exports', 'imports'
])

metrics_history_base_simulation = pd.DataFrame(columns=[
    'quarter', 'gdp_growth', 'inflation', 'unemployment',
    'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
    'exports', 'imports'
])
# Add visualization functions
def render_overview_charts(df, title):
    """Render overview charts"""
    fig = go.Figure()
    
    # Add traces for main economic indicators
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['gdp_growth'], name='GDP Growth'))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['unemployment'], name='Unemployment'))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['interest_rate'], name='Federal Funds Rate'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Percentage',
        hovermode='x unified'
    )
    
    return fig

# Add visualization functions
def render_overview_tax_rates_charts(df, title):
    """Render overview charts"""
    fig = go.Figure()
    
    # Add traces for main economic indicators
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['personal_tax_rates'], name='Personal Tax Rates'))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['corporate_tax_rates'], name='Corporate Tax Rates'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Percentage',
        hovermode='x unified'
    )
    
    return fig

def render_gdp_charts(df, title):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components
    fig.add_trace(go.Bar(x=df['quarter'], y=df['real_gdp'], name='Real GDP',
                        width=bar_width,
                        offset=-bar_width))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['nominal_gdp'], name='Nominal GDP',
                        width=bar_width,
                        offset=0))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Billions of $',
        barmode='overlay'
    )
    
    return fig

def render_gdp_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    # Add GDP components for AI Decision Makers with Tariff - 50%
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['real_gdp'], 
                        name='Real GDP (AI Decision Makers with Tariff - 50%)', marker_color='blue',
                        width=bar_width,
                        offset=-bar_width)) 
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['real_gdp'], 
                        name='Real GDP (Without Tariff)', marker_color='red',
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for Base Simulation
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['real_gdp'], 
                        name='Real GDP (Base)', marker_color='orange',
                        width=bar_width,
                        offset=bar_width)) 
    fig.update_layout(
        title='Real GDP Components Comparison',
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

def render_gdp_charts_comparison_nominal(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for AI Decision Makers with Tariff - 50%
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['nominal_gdp'], 
                        name='Nominal GDP (AI Decision Makers with Tariff - 50%)', marker_color='blue',
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['nominal_gdp'], 
                        name='Nominal GDP (Without Tariff)', marker_color='red',
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for Base Simulation
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['nominal_gdp'], 
                        name='Nominal GDP (Base)', marker_color='orange',
                        width=bar_width,
                        offset=bar_width)) 
    
    fig.update_layout(
        title='Nominal GDP Components Comparison',
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

def render_tax_charts(df, title):
    """Render tax-related charts"""
    fig = go.Figure()
    
    # Add tax components
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['personal_tax'], name='Personal Tax'))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['corporate_tax'], name='Corporate Tax'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Quarter',
        yaxis_title='Value',
        # barmode='stack'
    )
    
    return fig

def render_personal_tax_charts_comparison(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width
    
    # Add GDP components for AI Decision Makers with Tariff - 50%
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['personal_tax'], 
                        name='Personal Tax (AI Decision Makers with Tariff - 50%)', marker_color='blue',
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['personal_tax'], 
                        name='Personal Tax (Without Tariff)', marker_color='red',
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for Base Simulation
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['personal_tax'], 
                        name='Personal Tax (Base)', marker_color='orange',
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
    
    # Add GDP components for AI Decision Makers with Tariff - 50%
    fig.add_trace(go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['corporate_tax'], 
                        name='Corporate Tax (AI Decision Makers with Tariff - 50%)', marker_color='blue',
                        width=bar_width,
                        offset=-bar_width)) 
    
    # Add GDP components for Without Tariff
    fig.add_trace(go.Bar(x=df_without_tariff['quarter'], y=df_without_tariff['corporate_tax'], 
                        name='Corporate Tax (Without Tariff)', marker_color='red',
                        width=bar_width,
                        offset=0)) 
    
    # Add GDP components for Base Simulation
    fig.add_trace(go.Bar(x=df_base_simulation['quarter'], y=df_base_simulation['corporate_tax'], 
                        name='Corporate Tax (Base)', marker_color='orange',
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
def render_trade_charts(df, title):
    """Render trade balance charts"""
    fig = go.Figure()
    
    # Calculate bar positions
    bar_width = 0.25  # Adjust this value to control bar width  
    
    # Calculate trade balance
    df['trade_balance'] = df['exports'] - df['imports']
    
    # Add trade components
    fig.add_trace(go.Bar(x=df['quarter'], y=df['trade_balance'], name='Trade Balance',
                        width=bar_width,
                        offset=-bar_width))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['exports'], name='Exports',
                        width=bar_width,
                        offset=0))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['imports'], name='Imports',
                        width=bar_width,
                        offset=bar_width))
    
    fig.update_layout(
        title=title,
        barmode='overlay',
        xaxis_title='Quarter',
        yaxis_title='Value'
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
                    f"{metrics_data['rl_decision']['metrics']['hggdp']:.2f}%",
                    f"{metrics_data['rl_decision']['metrics']['hggdp'] - metrics_data['rl_decision']['previous_metrics']['hggdp']:.2f}%"
                )
                
                st.metric(
                    "Inflation Rate",
                    f"{metrics_data['rl_decision']['metrics']['pcpi']:.2f}%",
                    f"{metrics_data['rl_decision']['metrics']['pcpi'] - metrics_data['rl_decision']['previous_metrics']['pcpi']:.2f}%"
                )
                
                st.metric(
                    "Unemployment",
                    f"{metrics_data['rl_decision']['metrics']['lur']:.2f}%",
                    f"{metrics_data['rl_decision']['metrics']['lur'] - metrics_data['rl_decision']['previous_metrics']['lur']:.2f}%"
                )
            except Exception as e:
                st.error(f"Error displaying metrics: {str(e)}")

def render_tax_charts_for_all_simulations(df_rl_tariff, df_without_tariff, df_base_simulation):
    """Render tax-related charts with rates and amounts"""
    fig = go.Figure()
    
    # Create subplots: top for tax rates, bottom for tax amounts
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Tax Rates', 'Tax Revenue'),
                       vertical_spacing=0.2)
    
    # Plot Tax Rates (top subplot)
    fig.add_trace(
        go.Scatter(x=df_rl_tariff['quarter'], y=df_rl_tariff['personal_tax_rates'], 
                  name='Personal Tax Rate (RL)', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_rl_tariff['quarter'], y=df_rl_tariff['corporate_tax_rates'], 
                  name='Corporate Tax Rate (RL)', line=dict(color='blue', dash='dash')),
        row=1, col=1
    )
    
    # Add rates for other simulations...
    
    # Plot Tax Amounts (bottom subplot)
    fig.add_trace(
        go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['personal_tax'],
               name='Personal Tax Revenue (RL)', marker_color='blue'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df_rl_tariff['quarter'], y=df_rl_tariff['corporate_tax'],
               name='Corporate Tax Revenue (RL)', marker_color='lightblue'),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,  # Increased height for better visibility
        title='Tax Rates and Revenue Comparison',
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
    fig.update_yaxes(title_text="Revenue ($B)", row=2, col=1)
    
    return fig

# Create connection controls
class EconomicStream:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.lock = threading.Lock()
        self.url = "ws://localhost:8000/ws/metrics"

            
        # metrics: Dict[str, float] = {
        #     'hggdp': 0.0,  # GDP growth
        #     'xgdpn': 0.0,  # Nominal GDP
        #     'xgdp': 0.0,   # Real GDP
        #     'tpn': 0.0,    # Personal tax revenues
        #     'tcin': 0.0,   # Corporate tax revenues
        #     'trp': 0.0,    # Personal tax rates
        #     'trci': 0.0,   # Corporate tax rates
        #     'gtrt': 0.0,   # Transfer payments ratio
        #     'egfet': 0.0,  # Federal expenditures
        #     'frs10': 0.0,  # Interest rate
        #     'pcpi': 0.0,   # PCI value
        #     'lur': 0.0,    # Unemployment rate
        #     'gfdbtn': 0.0, # Debt-to-GDP ratio
        #     'emn': 0.0,    # Imports
        #     'exn': 0.0     # Exports
        # }
        self.metrics_history_rl_tariff = pd.DataFrame(columns=[
            'quarter', 'gdp_growth', 'inflation', 'unemployment',
            'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
            'exports', 'imports', 'debt_to_gdp', 'interest_rate', 'pcpi', 'transfer_payments_ratio', 'federal_expenditures', 'personal_tax_rates', 'corporate_tax_rates'
        ])

        self.metrics_history_without_tariff = pd.DataFrame(columns=[
            'quarter', 'gdp_growth', 'inflation', 'unemployment',
            'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
            'exports', 'imports', 'debt_to_gdp', 'interest_rate', 'pcpi', 'transfer_payments_ratio', 'federal_expenditures', 'personal_tax_rates', 'corporate_tax_rates'
        ])

        self.metrics_history_base_simulation = pd.DataFrame(columns=[
            'quarter', 'gdp_growth', 'inflation', 'unemployment',
            'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
            'exports', 'imports', 'debt_to_gdp', 'interest_rate', 'pcpi', 'transfer_payments_ratio', 'federal_expenditures', 'personal_tax_rates', 'corporate_tax_rates'
        ])

        
        # Initialize metrics storage locally
        self.last_message = None

    def handle_message(self, message_data):
        """Process incoming message and update metrics"""
        self.lock.acquire()
        try:
            # Extract metrics from message
            metrics = message_data['rl_decision']
            metrics_without_tariff = message_data['without_tariff']
            metrics_base_simulation = message_data['base_simulation']
            
            # Create new row for DataFrame
            new_data = {
                'quarter': metrics['quarter'],
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
                'interest_rate': metrics['metrics']['frs10'],
                'pcpi': metrics['metrics']['pcpi'],
                'transfer_payments_ratio': metrics['metrics']['gtrt'],
                'federal_expenditures': metrics['metrics']['egfet'],
                'personal_tax_rates': metrics['metrics']['trp'],
                'corporate_tax_rates': metrics['metrics']['trci']
            }
            
            new_data_without_tariff = {
                'quarter': metrics_without_tariff['quarter'],
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
                'interest_rate': metrics_without_tariff['metrics']['frs10'],
                'pcpi': metrics_without_tariff['metrics']['pcpi'],
                'transfer_payments_ratio': metrics_without_tariff['metrics']['gtrt'],
                'federal_expenditures': metrics_without_tariff['metrics']['egfet'],
                'personal_tax_rates': metrics_without_tariff['metrics']['trp'],
                'corporate_tax_rates': metrics_without_tariff['metrics']['trci']
            }
            
            new_data_base_simulation = {
                'quarter': metrics_base_simulation['quarter'],
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
                'interest_rate': metrics_base_simulation['metrics']['frs10'],
                'pcpi': metrics_base_simulation['metrics']['pcpi'],
                'transfer_payments_ratio': metrics_base_simulation['metrics']['gtrt'],
                'federal_expenditures': metrics_base_simulation['metrics']['egfet'],
                'personal_tax_rates': metrics_base_simulation['metrics']['trp'],
                'corporate_tax_rates': metrics_base_simulation['metrics']['trci']
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
            
            # Keep only last 50 records
            if len(self.metrics_history_rl_tariff) > 50:
                self.metrics_history_rl_tariff = self.metrics_history_rl_tariff.tail(50)
            
            if len(self.metrics_history_without_tariff) > 50:
                self.metrics_history_without_tariff = self.metrics_history_without_tariff.tail(50)
                
            if len(self.metrics_history_base_simulation) > 50:
                self.metrics_history_base_simulation = self.metrics_history_base_simulation.tail(50)

            # Reset index after all operations
            self.metrics_history_rl_tariff = self.metrics_history_rl_tariff.reset_index(drop=True)
            self.metrics_history_without_tariff = self.metrics_history_without_tariff.reset_index(drop=True)
            self.metrics_history_base_simulation = self.metrics_history_base_simulation.reset_index(drop=True)
                
            # Store last message
            self.last_message = message_data
            
            # Force Streamlit to rerun
            st.rerun()
        except Exception as e:
            print(f"Error handling message: {str(e)}")
            st.error(f"Error handling message: {str(e)}")
        finally:
            self.lock.release()

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            message_data = json.loads(message)
            print(f"Received message for quarter: {message_data['rl_decision']['quarter']}")
            
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
            print(f"Error processing message: {str(e)}")
            st.error(f"Error processing message: {str(e)}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {str(error)}")
        st.error(f"WebSocket error: {str(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        st.warning("WebSocket connection closed")

    def on_open(self, ws):
        print("WebSocket connected")
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
    page_icon="📈",
    layout="wide"
)

st.title("Economic Simulation Dashboard")

# Add view selector in sidebar
st.sidebar.header("Dashboard Views")
selected_view = st.sidebar.radio(
    "Select View",
    ["Overview", "GDP Metrics", "Tax Metrics", "Trade Balance"]
)

# Create placeholder for real-time updates
placeholder = st.empty()

# Create connection controls
connection_status = st.sidebar.empty()
col1, col2 = st.sidebar.columns(2)

# Initialize stream object if not exists
if 'stream' not in st.session_state:
    st.session_state.stream = EconomicStream(placeholder)

# Connection controls
with col1:
    if st.button('Connect', use_container_width=True):
        thread = threading.Thread(target=st.session_state.stream.connect)
        thread.daemon = True
        add_script_run_ctx(thread)
        thread.start()
        connection_status.success("Connecting...")

with col2:
    if st.button('Disconnect', use_container_width=True):
        if hasattr(st.session_state.stream, 'ws'):
            st.session_state.stream.ws.close()
            connection_status.warning("Disconnected")

# Main dashboard area
if hasattr(st.session_state.stream, 'metrics_history_rl_tariff') and not st.session_state.stream.metrics_history_rl_tariff.empty:
    df = st.session_state.stream.metrics_history_rl_tariff
    df_without_tariff = st.session_state.stream.metrics_history_without_tariff
    df_base_simulation = st.session_state.stream.metrics_history_base_simulation
    # Display charts based on selected view
    if selected_view == "Overview":
        st.plotly_chart(render_overview_charts(df, "Key Economic Indicators Over Time - AI Decision Makers with Tariff - 50%"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_without_tariff, "Key Economic Indicators Over Time - AI Decision Makers Without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_charts(df_base_simulation, "Key Economic Indicators Over Time - Base Simulation"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df, "Tax Rates - AI Decision Makers with Tariff - 50%"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_without_tariff, "Tax Rates - AI Decision Makers Without Tariff"), use_container_width=True)
        st.plotly_chart(render_overview_tax_rates_charts(df_base_simulation, "Tax Rates - Base Simulation"), use_container_width=True)
    elif selected_view == "GDP Metrics":
        st.plotly_chart(render_gdp_charts(df, "GDP Components - AI Decision Makers with Tariff - 50%"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_without_tariff, "GDP Components - AI Decision Makers Without Tariff"), use_container_width=True)
        st.plotly_chart(render_gdp_charts(df_base_simulation, "GDP Components - Base Simulation"), use_container_width=True)
        st.plotly_chart(render_gdp_charts_comparison(df, df_without_tariff, df_base_simulation), use_container_width=True)
        st.plotly_chart(render_gdp_charts_comparison_nominal(df, df_without_tariff, df_base_simulation), use_container_width=True)
        # Additional GDP metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Real GDP - AI Decision Makers with Tariff - 50%",
                    f"${df['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['real_gdp'].iloc[-1] - df['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - AI Decision Makers Without Tariff",
                    f"${df_without_tariff['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['real_gdp'].iloc[-1] - df_without_tariff['real_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Real GDP - Base Simulation",
                    f"${df_base_simulation['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['real_gdp'].iloc[-1] - df_base_simulation['real_gdp'].iloc[-2]):,.2f}B"
                )
            with col2:
                st.metric(
                    "Nominal GDP - AI Decision Makers with Tariff - 50%",
                    f"${df['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['nominal_gdp'].iloc[-1] - df['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - AI Decision Makers Without Tariff",
                    f"${df_without_tariff['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['nominal_gdp'].iloc[-1] - df_without_tariff['nominal_gdp'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Nominal GDP - Base Simulation",
                    f"${df_base_simulation['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['nominal_gdp'].iloc[-1] - df_base_simulation['nominal_gdp'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying GDP metrics: {str(e)}")
            
    elif selected_view == "Tax Metrics":
        st.plotly_chart(render_tax_charts(df, "Tax Revenue Components - AI Decision Makers with Tariff - 50%"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_without_tariff, "Tax Revenue Components - AI Decision Makers Without Tariff"), use_container_width=True)
        st.plotly_chart(render_tax_charts(df_base_simulation, "Tax Revenue Components - Base Simulation"), use_container_width=True)
        st.plotly_chart(render_personal_tax_charts_comparison(df, df_without_tariff, df_base_simulation), use_container_width=True)
        st.plotly_chart(render_corporate_tax_charts_comparison(df, df_without_tariff, df_base_simulation), use_container_width=True)
        st.plotly_chart(render_tax_charts_for_all_simulations(df, df_without_tariff, df_base_simulation), use_container_width=True)
        # Additional tax metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Personal Tax Revenue - AI Decision Makers with Tariff - 50%",
                    f"${df['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df['personal_tax'].iloc[-1] - df['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - AI Decision Makers Without Tariff",
                    f"${df_without_tariff['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['personal_tax'].iloc[-1] - df_without_tariff['personal_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Personal Tax Revenue - Base Simulation",
                    f"${df_base_simulation['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['personal_tax'].iloc[-1] - df_base_simulation['personal_tax'].iloc[-2]):,.2f}B"
                )
            with col2:
                st.metric(
                    "Corporate Tax Revenue - AI Decision Makers with Tariff - 50%",
                    f"${df['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df['corporate_tax'].iloc[-1] - df['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - AI Decision Makers Without Tariff",
                    f"${df_without_tariff['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['corporate_tax'].iloc[-1] - df_without_tariff['corporate_tax'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Corporate Tax Revenue - Base Simulation",
                    f"${df_base_simulation['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['corporate_tax'].iloc[-1] - df_base_simulation['corporate_tax'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying tax metrics: {str(e)}")
            
    elif selected_view == "Trade Balance":
        st.plotly_chart(render_trade_charts(df, "Trade Balance and Components - AI Decision Makers with Tariff - 50%"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_without_tariff, "Trade Balance and Components - AI Decision Makers Without Tariff"), use_container_width=True)
        st.plotly_chart(render_trade_charts(df_base_simulation, "Trade Balance and Components - Base Simulation"), use_container_width=True)
        
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
                st.metric(
                    "Trade Balance - AI Decision Makers with Tariff - 50%",
                    f"${trade_balance:,.2f}B",
                    f"{(trade_balance - prev_trade_balance):,.2f}B"
                )
                st.metric(
                    "Trade Balance - AI Decision Makers Without Tariff",
                    f"${trade_balance_without_tariff:,.2f}B",
                    f"{(trade_balance_without_tariff - prev_trade_balance_without_tariff):,.2f}B"
                )
                st.metric(
                    "Trade Balance - Base Simulation",
                    f"${trade_balance_base_simulation:,.2f}B",
                    f"{(trade_balance_base_simulation - prev_trade_balance_base_simulation):,.2f}B"
                )
            with col2:
                st.metric(
                    "Exports - AI Decision Makers with Tariff - 50%",
                    f"${df['exports'].iloc[-1]:,.2f}B",
                    f"{(df['exports'].iloc[-1] - df['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - AI Decision Makers Without Tariff",
                    f"${df_without_tariff['exports'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['exports'].iloc[-1] - df_without_tariff['exports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Exports - Base Simulation",
                    f"${df_base_simulation['exports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['exports'].iloc[-1] - df_base_simulation['exports'].iloc[-2]):,.2f}B"
                )
            with col3:
                st.metric(
                    "Imports - AI Decision Makers with Tariff - 50%",
                    f"${df['imports'].iloc[-1]:,.2f}B",
                    f"{(df['imports'].iloc[-1] - df['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - AI Decision Makers Without Tariff",
                    f"${df_without_tariff['imports'].iloc[-1]:,.2f}B",
                    f"{(df_without_tariff['imports'].iloc[-1] - df_without_tariff['imports'].iloc[-2]):,.2f}B"
                )
                st.metric(
                    "Imports - Base Simulation",
                    f"${df_base_simulation['imports'].iloc[-1]:,.2f}B",
                    f"{(df_base_simulation['imports'].iloc[-1] - df_base_simulation['imports'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying trade metrics: {str(e)}")
 