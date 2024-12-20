import streamlit as st
import pandas as pd 
import json
from websocket import create_connection
from websocket._app import WebSocketApp
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

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
    def __init__(self, placeholder, metrics_history):
        self.placeholder = placeholder
        self.lock = threading.Lock()
        self.url = "ws://localhost:8000/ws/metrics"
        
        # Initialize metrics storage 
        self.metrics_history = metrics_history

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
            
            # Keep only last 50 records for performance
            if len(self.metrics_history) > 50:
                self.metrics_history = self.metrics_history.tail(50)
                
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

