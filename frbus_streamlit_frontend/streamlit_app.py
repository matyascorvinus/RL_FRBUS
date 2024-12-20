from streamer import EconomicStream
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

metrics_history = pd.DataFrame(columns=[
    'quarter', 'gdp_growth', 'inflation', 'unemployment',
    'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
    'exports', 'imports'
])
# Add visualization functions
def render_overview_charts(df):
    """Render overview charts"""
    fig = go.Figure()
    
    # Add traces for main economic indicators
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['gdp_growth'], name='GDP Growth'))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['inflation'], name='Inflation'))
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['unemployment'], name='Unemployment'))
    
    fig.update_layout(
        title='Key Economic Indicators Over Time',
        xaxis_title='Quarter',
        yaxis_title='Percentage',
        hovermode='x unified'
    )
    
    return fig

def render_gdp_charts(df):
    """Render GDP-related charts"""
    fig = go.Figure()
    
    # Add GDP components
    fig.add_trace(go.Bar(x=df['quarter'], y=df['real_gdp'], name='Real GDP'))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['nominal_gdp'], name='Nominal GDP'))
    
    fig.update_layout(
        title='GDP Components',
        xaxis_title='Quarter',
        yaxis_title='Value',
        barmode='group'
    )
    
    return fig

def render_tax_charts(df):
    """Render tax-related charts"""
    fig = go.Figure()
    
    # Add tax components
    fig.add_trace(go.Bar(x=df['quarter'], y=df['personal_tax'], name='Personal Tax'))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['corporate_tax'], name='Corporate Tax'))
    
    fig.update_layout(
        title='Tax Revenue Components',
        xaxis_title='Quarter',
        yaxis_title='Value',
        barmode='stack'
    )
    
    return fig

def render_trade_charts(df):
    """Render trade balance charts"""
    fig = go.Figure()
    
    # Calculate trade balance
    df['trade_balance'] = df['exports'] - df['imports']
    
    # Add trade components
    fig.add_trace(go.Scatter(x=df['quarter'], y=df['trade_balance'], name='Trade Balance'))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['exports'], name='Exports'))
    fig.add_trace(go.Bar(x=df['quarter'], y=df['imports'], name='Imports'))
    
    fig.update_layout(
        title='Trade Balance and Components',
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
# Create connection controls

class EconomicStream:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.lock = threading.Lock()
        self.url = "ws://localhost:8000/ws/metrics"
        self.metrics_history = pd.DataFrame(columns=[
            'quarter', 'gdp_growth', 'inflation', 'unemployment',
            'real_gdp', 'nominal_gdp', 'personal_tax', 'corporate_tax',
            'exports', 'imports'
        ])
        # Initialize metrics storage locally
        self.last_message = None

    def handle_message(self, message_data):
        """Process incoming message and update metrics"""
        self.lock.acquire()
        try:
            # Extract metrics from message
            metrics = message_data['rl_decision']
            
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
                'imports': metrics['metrics']['emn']
            }
            
            # Update metrics history
            self.metrics_history = pd.concat([
                self.metrics_history,
                pd.DataFrame([new_data])
            ]).reset_index(drop=True)
            
            # Add sorting key
            def quarter_to_sortable(q):
                year, quarter = q.split('q')
                return int(year) * 10 + int(quarter)
            
            # Sort by quarter using the custom sorting key
            self.metrics_history['sort_key'] = self.metrics_history['quarter'].apply(quarter_to_sortable)
            self.metrics_history = self.metrics_history.sort_values('sort_key')
            self.metrics_history = self.metrics_history.drop('sort_key', axis=1)
            
            # Keep only last 50 records
            if len(self.metrics_history) > 50:
                self.metrics_history = self.metrics_history.tail(50)
            
            # Reset index after all operations
            self.metrics_history = self.metrics_history.reset_index(drop=True)
                
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
if hasattr(st.session_state.stream, 'metrics_history') and not st.session_state.stream.metrics_history.empty:
    df = st.session_state.stream.metrics_history
    
    # Display charts based on selected view
    if selected_view == "Overview":
        st.plotly_chart(render_overview_charts(df), use_container_width=True)
        
    elif selected_view == "GDP Metrics":
        st.plotly_chart(render_gdp_charts(df), use_container_width=True)
        
        # Additional GDP metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Real GDP",
                    f"${df['real_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['real_gdp'].iloc[-1] - df['real_gdp'].iloc[-2]):,.2f}B"
                )
            with col2:
                st.metric(
                    "Nominal GDP",
                    f"${df['nominal_gdp'].iloc[-1]:,.2f}B",
                    f"{(df['nominal_gdp'].iloc[-1] - df['nominal_gdp'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying GDP metrics: {str(e)}")
            
    elif selected_view == "Tax Metrics":
        st.plotly_chart(render_tax_charts(df), use_container_width=True)
        
        # Additional tax metrics
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric(
                    "Personal Tax Revenue",
                    f"${df['personal_tax'].iloc[-1]:,.2f}B",
                    f"{(df['personal_tax'].iloc[-1] - df['personal_tax'].iloc[-2]):,.2f}B"
                )
            with col2:
                st.metric(
                    "Corporate Tax Revenue",
                    f"${df['corporate_tax'].iloc[-1]:,.2f}B",
                    f"{(df['corporate_tax'].iloc[-1] - df['corporate_tax'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying tax metrics: {str(e)}")
            
    elif selected_view == "Trade Balance":
        st.plotly_chart(render_trade_charts(df), use_container_width=True)
        
        # Additional trade metrics
        col1, col2, col3 = st.columns(3)
        try:
            with col1:
                trade_balance = df['exports'].iloc[-1] - df['imports'].iloc[-1]
                prev_trade_balance = df['exports'].iloc[-2] - df['imports'].iloc[-2]
                st.metric(
                    "Trade Balance",
                    f"${trade_balance:,.2f}B",
                    f"{(trade_balance - prev_trade_balance):,.2f}B"
                )
            with col2:
                st.metric(
                    "Exports",
                    f"${df['exports'].iloc[-1]:,.2f}B",
                    f"{(df['exports'].iloc[-1] - df['exports'].iloc[-2]):,.2f}B"
                )
            with col3:
                st.metric(
                    "Imports",
                    f"${df['imports'].iloc[-1]:,.2f}B",
                    f"{(df['imports'].iloc[-1] - df['imports'].iloc[-2]):,.2f}B"
                )
        except Exception as e:
            st.error(f"Error displaying trade metrics: {str(e)}")
 