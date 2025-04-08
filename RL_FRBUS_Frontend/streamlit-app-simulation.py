import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Sidebar: Select Data Source
# ---------------------------
# Choose between Historical Data and Trump Tariff plan.
data_source = st.sidebar.radio(
    "Select Data Source", 
    options=["Historical Data", "Historical Data 2022-2024", "Trump Tariff plan 10%", "Trump Tariff plan 20%", "Trump Tariff plan 50%", "Trump Tariff plan 100%"],
    index=1
)
# Set file path based on the data source selection.
data_file = "combined_simulation_data_1975_2024.csv" 
if data_source != "Historical Data":
    if data_source == "Historical Data 2022-2024":
        data_file = "combined_simulation_data_2000_2024.csv" 
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
# @st.cache_data
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
        "personal_tax": "Personal Tax (Billion)",
        "corporate_tax": "Corporate Tax (Billion)",
        "exports": "Exports (Billion)",
        "imports": "Imports (Billion)",
        "debt_to_gdp": "Debt",
        "interest_rate": "Federal Fund Rate (%)",
        "pcpi": "PCPI",
        "transfer_payments_ratio": "Transfer Payments Ratio",
        "federal_expenditures": "Federal Expenditures (in Billions)",
        "personal_tax_rates": "Personal Tax Rates",
        "corporate_tax_rates": "Corporate Tax Rates",
        "government_transfer_payments": "Government Transfer Payments (Billion)",
        "federal_surplus": "Federal Surplus (Billion)",
        "simulation_type": "Simulation Type"
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

# Let the user choose which metric to compare.
metric_options = [
    "GDP Growth (%)",
    "CPI (Inflation Index)",
    "Unemployment Rate (%)",
    "Real GDP (Billion)",
    "Nominal GDP (Billion)",
    "Federal Expenditures (in Billions)",
    "Personal Tax (Billion)",
    "Corporate Tax (Billion)",
    "Exports (Billion)",
    "Imports (Billion)",
    "Debt",
    "Federal Fund Rate (%)",
    "Government Transfer Payments (Billion)"
]
selected_metric = st.selectbox("Select Metric for Comparison", metric_options)
st.write(f"Selected Metric: {selected_metric}")
# Create an Altair line chart using the filtered data.
# In this chart, the x-axis uses the quarter_numeric value (for proper ordering)
# and the tooltip shows the original 'Quarter' string.
chart = alt.Chart(filtered_data).mark_line(point=True).encode(
    x=alt.X("quarter_numeric:Q", title="Quarter", axis=alt.Axis(titleFontSize=18, labelFontSize=18, titleColor='black', labelColor='black')),
    y=alt.Y(f"{selected_metric}:Q", title=selected_metric, axis=alt.Axis(titleFontSize=18, labelFontSize=18, titleColor ='black', labelColor='black')),
    color=alt.Color("Simulation Type:N", title="Simulation Type", legend=alt.Legend(titleFontSize=18, labelFontSize=18, titleColor='black', labelColor='black')),
    tooltip=["Quarter", f"{selected_metric}", "Simulation Type"]
).properties(
    width=700,
    height=400,
    title=f"{selected_metric} Comparison Across Simulation Types"
).interactive()

st.altair_chart(chart, use_container_width=True)

# Add bar chart for comparison of key metrics across simulation types
st.subheader("Bar Chart Comparison of Key Metrics Across Simulation Types")

# ---------------------------
# Real GDP Components Comparison Chart (from streamlit-app.py)
# ---------------------------

# Define the muted red color scheme if not defined already
MUTED_REDS = {
    'dark': '#8B0000',    # Dark red (for tariff simulation)
    'bright': '#FF4500',  # Bright red/orange (for original data)
    'light': '#FFA07A'    # Light salmon (for FRBUS-based simulation)
}

PURPLE_RED_LIGHT_BLUE = {
    'purple': '#702963',    # Dark red (for tariff simulation)
    'red': '#FF0000',  # Bright red/orange (for original data)
    'light_blue': '#add8e6'    # Light salmon (for FRBUS-based simulation)
}
# ---------------------------
# Component Comparison Chart using selected metric from metrics_options
# ---------------------------

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

custom_pallets = ['#702963', '#FF0000', '#add8e6', '#0000FF']

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
    comp_fig = render_component_comparison(final_filtered_data, sim_types, component_metric)
    st.plotly_chart(comp_fig, use_container_width=True)
else:
    st.markdown("---")
    st.write("Some required datasets for the Component Comparison chart are missing.")
 
