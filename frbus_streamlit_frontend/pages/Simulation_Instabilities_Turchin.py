import streamlit as st
import pandas as pd

# Page title and description
with st.container():
    st.title('Dominus Simulation Guidelines')
    st.write('This interface presents the key metrics and indicators for analyzing societal stability based on structural-demographic theory.')


simulation_guidelines = """
| **Category**                 | **Subcategory**             | **Data to Collect**                                                                 |
|------------------------------|----------------------------|-------------------------------------------------------------------------------------|
| **Population Dynamics**      | Numbers and Growth Rates   | - Total population over time | Annual population growth rate | Rural vs. urban population proportions |
|                              | Age Structure             | - Age distribution (focus on 20–29 age group) | Trends in youth bulges |
|                              | Urbanization              | - Urbanization rate (percentage of population in urban areas) | Migration flows (rural to urban) |
|                              | Relative Wages            | - Median wages | GDP per capita | Trends in real wages over time | Urban vs. rural cost of living |
| **Elite Dynamics**           | Elite Numbers             | - Total number of elites | Breakdown of elites by category (political, economic, cultural) | Established vs. new aspirant elites |
|                              | Composition of Elites     | - Sectoral distribution of elites (government, business, academia, etc.) | Rising sectors contributing to elite status (e.g., IT, real estate) |
|                              | Economic Attributes       | - Average elite incomes compared to GDP per capita | Wealth distribution among elites | Ownership data of key companies or assets |
|                              | Intraelite Competition    | - Number of elite positions vs. elite numbers | Trends in conspicuous consumption (luxury goods, real estate, vehicles) | Incidents of intraelite conflict or rivalry |
|                              | Mobility                  | - Upward social mobility pathways (e.g., education, entrepreneurship) | Downward mobility indicators (e.g., economic downturns, purges) |
| **State Attributes**         | Size of the State         | - Total state employees | Percentage of GDP spent by the government | Key state-controlled sectors |
|                              | Economic Health           | - Government revenues and expenditures | Debt levels as a percentage of GDP | Budget allocation by sector |
|                              | Legitimacy                | - Public trust in government institutions | Incidents of corruption or governance failures | Legitimacy mechanisms (e.g., welfare programs) |
| **Indicators of Instability**| Mass Mobilization Potential (MMP) | - Inverse relative wages (median wages/GDP per capita) | Urbanization rates | Proportion of 20–29 age cohort |
|                              | Elite Mobilization Potential (EMP) | - Inverse elite income (average elite income/GDP per capita) | Elite numbers vs. elite positions | Metrics of elite rivalry or competition |
|                              | State Fiscal Distress (SFD) | - National debt as a percentage of GDP | Public trust/distrust metrics | Historical and current fiscal crises |
| **Feedback Dynamics**        | Economic Links            | - Relationship between population growth and wages | Impact of urbanization on employment and wages |
|                              | Elite Dynamics            | - Effects of elite overproduction on conflict | Wealth and income concentration among elites |
|                              | Political Feedback        | - Links between economic downturns, elite dissatisfaction, and state fiscal stress |

"""

st.markdown(simulation_guidelines)
import pandas as pd
import requests

# Fetch the data. Age distribution (focus on 20–29 age group)
df = pd.read_csv("https://ourworldindata.org/grapher/population-by-age-group.csv?v=1&csvType=full&useColumnShortNames=false", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


# Fetch the data. Urbanization rate (percentage of population in urban areas)
df = pd.read_csv("https://ourworldindata.org/grapher/population-of-cities-towns-and-villages.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

# Fetch the data. Daily median income
df = pd.read_csv("https://ourworldindata.org/grapher/daily-median-income.csv?v=1&csvType=full&useColumnShortNames=false", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

# Fetch the data. GDP per capita
df = pd.read_csv("https://ourworldindata.org/grapher/gdp-per-capita-worldbank.csv?v=1&csvType=full&useColumnShortNames=false", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

# Convert the guidelines into a more structured format
categories = {
    'Population Dynamics': {
        'Numbers and Growth Rates': [
            'Total population over time',
            'Annual population growth rate',
            'Rural vs. urban population proportions'
        ],
        'Age Structure': [
            'Age distribution (focus on 20–29 age group)',
            'Trends in youth bulges'
        ],
        'Urbanization': [
            'Urbanization rate (percentage of population in urban areas)',
            'Migration flows (rural to urban)'
        ],
        'Relative Wages': [
            'Median wages',
            'GDP per capita',
            'Trends in real wages over time',
            'Urban vs. rural cost of living'
        ]
    },
    'Elite Dynamics': {
        'Elite Numbers': [
            'Total number of elites',
            'Breakdown of elites by category',
            'Established vs. new aspirant elites'
        ],
        'Composition of Elites': [
            'Sectoral distribution of elites',
            'Rising sectors contributing to elite status'
        ],
        'Economic Attributes': [
            'Average elite incomes compared to GDP per capita',
            'Wealth distribution among elites',
            'Ownership data of key companies or assets'
        ],
        'Intraelite Competition': [
            'Number of elite positions vs. elite numbers',
            'Trends in conspicuous consumption',
            'Incidents of intraelite conflict or rivalry'
        ],
        'Mobility': [
            'Upward social mobility pathways',
            'Downward mobility indicators'
        ]
    },
    'State Attributes': {
        'Size of the State': [
            'Total state employees',
            'Percentage of GDP spent by the government',
            'Key state-controlled sectors'
        ],
        'Economic Health': [
            'Government revenues and expenditures',
            'Debt levels as a percentage of GDP',
            'Budget allocation by sector'
        ],
        'Legitimacy': [
            'Public trust in government institutions',
            'Incidents of corruption or governance failures',
            'Legitimacy mechanisms'
        ]
    },
    'Indicators of Instability': {
        'Mass Mobilization Potential (MMP)': [
            'Inverse relative wages',
            'Urbanization rates',
            'Proportion of 20–29 age cohort'
        ],
        'Elite Mobilization Potential (EMP)': [
            'Inverse elite income',
            'Elite numbers vs. elite positions',
            'Metrics of elite rivalry or competition'
        ],
        'State Fiscal Distress (SFD)': [
            'National debt as a percentage of GDP',
            'Public trust/distrust metrics',
            'Historical and current fiscal crises'
        ]
    },
    'Feedback Dynamics': {
        'Economic Links': [
            'Relationship between population growth and wages',
            'Impact of urbanization on employment and wages'
        ],
        'Elite Dynamics': [
            'Effects of elite overproduction on conflict',
            'Wealth and income concentration among elites'
        ],
        'Political Feedback': [
            'Links between economic downturns, elite dissatisfaction, and state fiscal stress'
        ]
    }
}

# Add data sources if available for each category
data_sources = {
    'Population Dynamics': {
        'Official Sources': [
            {'name': 'General Statistics Office of Vietnam', 'url': 'https://www.gso.gov.vn/en/population/'},
            {'name': 'World Bank Vietnam Data', 'url': 'https://data.worldbank.org/country/vietnam'},
            {'name': 'UN Population Division - Vietnam', 'url': 'https://population.un.org/wpp/Download/Standard/Population/'}
        ]
    },
    'Elite Dynamics': {
        'Official Sources': [
            {'name': 'Ministry of Planning and Investment', 'url': 'http://www.mpi.gov.vn/en'},
            {'name': 'Forbes Vietnam', 'url': 'https://forbes.vn/'},
            {'name': 'Vietnam Chamber of Commerce and Industry', 'url': 'https://vcci.com.vn/'}
        ]
    },
    'State Attributes': {
        'Official Sources': [
            {'name': 'State Bank of Vietnam', 'url': 'https://www.sbv.gov.vn/'},
            {'name': 'Ministry of Finance', 'url': 'https://www.mof.gov.vn/'},
            {'name': 'Vietnam Government Portal', 'url': 'http://chinhphu.vn/'}
        ]
    },
    'Indicators of Instability': {
        'International Sources': [
            {'name': 'Asian Development Bank - Vietnam', 'url': 'https://www.adb.org/countries/viet-nam/'},
            {'name': 'IMF Vietnam Reports', 'url': 'https://www.imf.org/en/Countries/VNM'},
            {'name': 'World Bank Governance Indicators', 'url': 'https://databank.worldbank.org/source/worldwide-governance-indicators'}
        ]
    }
}

# Create tabs for main categories
tabs = st.tabs(list(categories.keys()))

# Populate each tab with its subcategories
for tab_idx, (category, subcategories) in enumerate(categories.items()):
    with tabs[tab_idx]:
        st.header(category)
        
        # Add data sources if available for this category
        if category in data_sources:
            with st.expander("📚 Data Sources"):
                for source_type, sources in data_sources[category].items():
                    st.subheader(source_type)
                    for source in sources:
                        st.markdown(f"[{source['name']}]({source['url']})")
        
        # Create expanders for each subcategory
        for subcategory, metrics in subcategories.items():
            with st.expander(f"📊 {subcategory}"):
                # Create a form for each subcategory
                with st.form(f"form_{category}_{subcategory}".replace(" ", "_")):
                    st.subheader("Data Collection Metrics")
                    
                    # Create input fields for each metric
                    values = {}
                    for metric in metrics:
                        key = f"{category}_{subcategory}_{metric}".replace(" ", "_")
                        values[metric] = st.number_input(
                            f"{metric}",
                            min_value=0.0,
                            help=f"Enter value for {metric}"
                        )
                    
                    # Add notes field
                    notes = st.text_area(
                        "Notes",
                        placeholder="Add any relevant notes or observations..."
                    )
                    
                    # Submit button
                    submitted = st.form_submit_button("Save Data")
                    if submitted:
                        st.success("Data saved successfully!")
                        # Here you would typically save the data to a database
                        # For now, we'll just display it
                        st.write("Recorded values:", values)
                        if notes:
                            st.write("Notes:", notes)

# Add a section for visualization (placeholder)
with st.expander("📈 Visualize Data"):
    st.write("Visualization tools will be added here...")
    # Here you would add plots and charts based on the collected data