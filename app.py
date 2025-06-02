import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
from datetime import datetime

# --- Page Setup ---
st.set_page_config(
    page_title="Example Parcel Intelligence (Cornelius, NC)",
    layout="wide",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style='padding: 15px; background-color: #e3f2fd; border-left: 5px solid #2196F3; border-radius: 6px; margin-bottom: 10px; color: #000000;'>
    <strong>🔧 Demo Mode:</strong> This dashboard uses public property data to showcase the kinds of tools I can build for developers and real estate teams. 
    <br>This is <em>not</em> a live product feed — but all features are customizable for your needs.
</div>
""", unsafe_allow_html=True)


with st.expander("📘 About this tool"):
    st.markdown("""
    This dashboard helps developers, investors, and urban planners explore underutilized parcels in Mecklenburg County.
    Use filters on the left to tailor your search based on zoning, land use, teardown potential, and more.
    """)

# Custom CSS for styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stMetricLabel {font-size: 14px !important;}
    .stMetricValue {font-size: 18px !important;}
    .stSelectbox, .stMultiselect {padding: 8px 12px;}
    .stDataFrame {border-radius: 8px;}
    .stDownloadButton button {background-color: #4CAF50 !important; color: white !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px !important;}
    .stTabs [aria-selected="true"] {background-color: #e8f5e9 !important;}
    .css-1aumxhk {background-color: #ffffff; background-image: none;}
    .property-card {background-color: white; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .highlight-row {background-color: #fffde7 !important;}
    .debug-info {background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- Load Data with Enhanced Error Handling ---
@st.cache_data
def load_data():
    try:
        # Load data with comprehensive cleaning
        df = pd.read_csv("charlotte_cornelius_redacted.csv")
        
        # Clean numeric columns - handle various formats
        def clean_numeric(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                x = x.replace('$', '').replace(',', '').strip()
                if x == '':
                    return np.nan
            try:
                return float(x)
            except:
                return np.nan

        numeric_cols = ['LandValue', 'ImprovementValue', 'TotalValue', 'Acres']
        for col in numeric_cols:
            df[col] = df[col].apply(clean_numeric)
        
        df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')
        
        # Calculate derived metrics with robust null handling
        df['ValuePerAcre'] = np.where(
            (df['Acres'] > 0) & (df['TotalValue'] > 0),
            df['TotalValue']/df['Acres'],
            np.nan
        )
        df['LandToTotalRatio'] = np.where(
            df['TotalValue'] > 0,
            df['LandValue']/df['TotalValue'],
            np.nan
        )

        import re
        from pyproj import Transformer
        def convert_to_latlon(x, y):
            """Convert projected coordinates to WGS84 (lat/long)"""
            try:
                transformer = Transformer.from_crs("EPSG:2264", "EPSG:4326", always_xy=True)  # State Plane NAD83 to WGS84
                lon, lat = transformer.transform(x, y)
                return (lat, lon)
            except:
                return (np.nan, np.nan)

        def extract_coords(geom):
            if pd.isna(geom) or not isinstance(geom, str):
                return (np.nan, np.nan)

            # Extract first coordinate pair from MULTIPOLYGON
            match = re.search(r"MULTIPOLYGON \(\(\(([0-9.]+) ([0-9.]+)", geom)
            if match:
                try:
                    x, y = float(match.group(1)), float(match.group(2))
                    return convert_to_latlon(x, y)
                except:
                    return (np.nan, np.nan)
            return (np.nan, np.nan)


        df[['Latitude', 'Longitude']] = df['Geometry'].apply(lambda g: pd.Series(extract_coords(g)))
        
        # Create age column with null handling
        current_year = datetime.now().year
        df['PropertyAge'] = current_year - df['YearBuilt']
        df['PropertyAge'] = df['PropertyAge'].apply(lambda x: x if x > 0 else np.nan)
        
        # Create teardown flag with null handling
        df['IsTeardown'] = (
            (df['LandToTotalRatio'].fillna(0) > 0.7) & 
            (df['PropertyAge'].fillna(0) > 50) & 
            (df['ImprovementValue'].fillna(0) < df['LandValue'].fillna(0)
        ))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# --- Sidebar Filters with Flexible Defaults ---
st.sidebar.header("🔎 Filter Options")

# Location Filters
st.sidebar.subheader("Location Filters")
subdivisions = sorted(df['Subdivision'].dropna().unique())
selected_subdivisions = st.sidebar.multiselect(
    "Subdivisions", 
    subdivisions,
    help="Filter by specific subdivisions (optional)"
)

# Property Characteristics
st.sidebar.subheader("Property Characteristics")

# Year built filter with dynamic range
valid_years = df['YearBuilt'].dropna()
if len(valid_years) > 0:
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())
else:
    min_year = 1900
    max_year = datetime.now().year

year_built_range = st.sidebar.slider(
    "Year Built Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)  # Default to full range
)

# Value range filter with dynamic percentiles
if not df.empty:
    value_min = int(df['TotalValue'].quantile(0.05))
    value_max = int(df['TotalValue'].quantile(0.95))
else:
    value_min = 0
    value_max = 1000000

value_range = st.sidebar.slider(
    "Total Value Range ($)",
    min_value=value_min,
    max_value=value_max,
    value=(value_min, value_max)  # Default to middle 90%
)

# Land Use Filter - Smart Defaults
use_options = sorted(df['LandUse'].dropna().unique())
top_uses = df['LandUse'].value_counts().nlargest(5).index.tolist()
default_uses = [use for use in top_uses if use in use_options]

selected_uses = st.sidebar.multiselect(
    "Land Use Type (optional)", 
    options=use_options,
    default=default_uses,
    help="Filter by current land use"
)

# Owner Type Filter
owner_types = sorted(df['OwnerType'].dropna().unique())
selected_owner_types = st.sidebar.multiselect(
    "Owner Types (optional)", 
    owner_types,
    help="Filter by owner type"
)

# Investment Potential Filters
st.sidebar.subheader("Investment Potential")
teardown_filter = st.sidebar.radio(
    "Teardown Candidates", 
    ["All", "Teardown Only", "Non-Teardown"],
    index=0
)

high_ratio = st.sidebar.checkbox(
    "High Land-to-Total Value Ratio (>0.7)", 
    value=False
)

recent_sales = st.sidebar.checkbox(
    "Recently Sold (Last 3 Years)", 
    value=False
)

# --- Apply Filters with Debugging ---
filtered = df.copy()

# Apply filters sequentially with null handling
if selected_subdivisions:
    filtered = filtered[filtered['Subdivision'].isin(selected_subdivisions)]

if selected_uses:
    filtered = filtered[filtered['LandUse'].isin(selected_uses)]

if selected_owner_types:
    filtered = filtered[filtered['OwnerType'].isin(selected_owner_types)]

# Year built filter - keep null values
filtered = filtered[
    (filtered['YearBuilt'].isna()) | 
    (
        (filtered['YearBuilt'] >= year_built_range[0]) & 
        (filtered['YearBuilt'] <= year_built_range[1])
    )
]

# Value range filter
filtered = filtered[
    (filtered['TotalValue'] >= value_range[0]) & 
    (filtered['TotalValue'] <= value_range[1])
]

# Teardown filter
if teardown_filter == "Teardown Only":
    filtered = filtered[filtered['IsTeardown'] == True]
elif teardown_filter == "Non-Teardown":
    filtered = filtered[filtered['IsTeardown'] == False]

# High ratio filter
if high_ratio:
    filtered = filtered[
        (filtered['LandToTotalRatio'].notna()) & 
        (filtered['LandToTotalRatio'] > 0.7)
    ]

# Recent sales filter
if recent_sales and 'LastSaleDate' in filtered.columns:
    three_years_ago = datetime.now().year - 3
    filtered = filtered[
        pd.to_datetime(filtered['LastSaleDate'], errors='coerce').dt.year >= three_years_ago
    ]

if filtered.empty:
    st.warning("⚠️ No properties match your current filters. Try relaxing your selections.")
    st.stop()

# --- Main Dashboard Content ---
# [Rest of your dashboard tabs and visualizations can remain the same]
# Include all your tab1, tab2, tab3, tab4 content here exactly as before
# Only the filtering logic above has been modified

# --- Main Dashboard ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview  ", "📍 Map View  ", "📈 Market Trends  ", "🔍 Property Explorer  "])

with tab1:
    # Key Metrics
    
    st.subheader("Key Market Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Parcels", f"{len(filtered):,}")
    col2.metric("Teardown Candidates", f"{filtered['IsTeardown'].sum():,}", 
               f"{filtered['IsTeardown'].mean()*100:.1f}%")
    col3.metric("Avg. Value/Acre", f"${filtered['ValuePerAcre'].mean():,.0f}" if not filtered['ValuePerAcre'].isna().all() else "N/A")
    col4.metric("Avg. Land Ratio", f"{filtered['LandToTotalRatio'].mean():.2f}" if not filtered['LandToTotalRatio'].isna().all() else "N/A")
    col5.metric("Median Acres", f"{filtered['Acres'].median():.2f}" if not filtered['Acres'].isna().all() else "N/A")
    
    # Top Opportunities Section
    st.subheader("Top Redevelopment Opportunities")
    with st.expander("📈 How is the Opportunity Score calculated?"):
        st.markdown("""
        The **Opportunity Score** helps highlight parcels with strong redevelopment potential. It's calculated using:
        
        - **Land-to-Total Value Ratio (50%)**: Higher values suggest the structure adds little value.
        - **Property Age (30%)**: Older buildings are more likely to be outdated or code-deficient.
        - **Lot Size / Acres (20%)**: Larger parcels provide more flexibility for redevelopment or subdivision.

        **Formula**:
        ```
        0.5 × LandToTotalRatio + 0.3 × (PropertyAge / 100) + 0.2 × (Acres / MaxAcres)
        ```

        Only parcels with enough data are scored. Scores are relative and should be interpreted as comparative signals — not guarantees.
        """)

    if not filtered.empty:
        # Calculate opportunity score
        filtered['OpportunityScore'] = (
            (filtered['LandToTotalRatio'].fillna(0) * 0.5) +
            ((filtered['PropertyAge'].fillna(0) / 100) * 0.3 +
            ((filtered['Acres'].fillna(0) / filtered['Acres'].max()) * 0.2)
            ))
        
        top_opportunities = filtered.nlargest(10, 'OpportunityScore')[[
            'ParcelID', 'PropertyAddress', 'Subdivision', 'LandUse', 
            'YearBuilt', 'TotalValue', 'Acres', 'LandToTotalRatio', 'IsTeardown'
        ]]
        
        st.dataframe(
            top_opportunities.style.format({
                'TotalValue': '${:,.0f}',
                'LandToTotalRatio': '{:.2f}',
                'Acres': '{:.2f}'
            }).apply(
                lambda row: [
                    'background-color: #333333; color: white' if row['IsTeardown'] else ''
                    for _ in row
                ],
                axis=1
            ),
            use_container_width=True
        )

    
    # Value Distribution Charts
    st.subheader("Value Distribution Analysis")
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        fig = px.histogram(
            filtered, 
            x='TotalValue', 
            nbins=50, 
            title='Distribution of Total Property Values',
            labels={'TotalValue': 'Total Value ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with fig_col2:
        fig = px.scatter(
            filtered,
            x='Acres',
            y='TotalValue',
            color='IsTeardown',
            title='Lot Size vs. Total Value',
            hover_data=['PropertyAddress'],
            labels={
                'Acres': 'Lot Size (Acres)',
                'TotalValue': 'Total Value ($)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Map Visualization
    st.subheader("Geospatial Distribution of Properties")
    
    if not filtered.empty and 'Latitude' in filtered.columns and 'Longitude' in filtered.columns:
        # Create base map
        m = folium.Map(
            location=[35.4866, -80.8601], 
            zoom_start=13,
            tiles='cartodbpositron'
        )
        
        # Add heatmap
        heat_data = filtered[['Latitude', 'Longitude', 'TotalValue']].dropna()
        heat_data = [[row['Latitude'], row['Longitude'], row['TotalValue']/100000] 
                    for _, row in heat_data.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m)
        
        # Add markers for teardowns
        teardowns = filtered[filtered['IsTeardown'] == True]
        for _, row in teardowns.iterrows():
            if not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']):
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    color='red',
                    fill=True,
                    fill_color='red',
                    popup=f"""
                    <b>Address:</b> {row['PropertyAddress']}<br>
                    <b>Value:</b> ${row['TotalValue']:,.0f}<br>
                    <b>Acres:</b> {row['Acres']:.2f}<br>
                    <b>Land Ratio:</b> {row['LandToTotalRatio']:.2f}
                    """
                ).add_to(m)
        
        folium_static(m, width=1200, height=600)
        
        # Subdivision Summary
        st.subheader("Subdivision Analysis")
        subdivision_stats = filtered.groupby('Subdivision').agg({
            'ParcelID': 'count',
            'TotalValue': 'mean',
            'Acres': 'mean',
            'LandToTotalRatio': 'mean',
            'IsTeardown': 'sum'
        }).sort_values('TotalValue', ascending=False)
        
        subdivision_stats.columns = [
            'Property Count', 
            'Avg. Value', 
            'Avg. Acres',
            'Avg. Land Ratio',
            'Teardown Count'
        ]
        
        st.dataframe(
            subdivision_stats.style.format({
                'Avg. Value': '${:,.0f}',
                'Avg. Acres': '{:.2f}',
                'Avg. Land Ratio': '{:.2f}'
            }).background_gradient(cmap='YlGn'),
            use_container_width=True
        )
    else:
        st.warning("Map data not available. Missing latitude/longitude coordinates.")

with tab3:
    # Market Trends Analysis
    st.subheader("Market Trends by Property Characteristics")
    
    if not filtered.empty:
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            fig = px.box(
                filtered,
                x='LandUse',
                y='TotalValue',
                color='IsTeardown',
                title='Value Distribution by Land Use',
                labels={'TotalValue': 'Total Value ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with fig_col2:
            fig = px.scatter(
                filtered,
                x='PropertyAge',
                y='LandToTotalRatio',
                color='OwnerType',
                title='Age vs. Land Value Ratio',
                labels={
                    'PropertyAge': 'Property Age (years)',
                    'LandToTotalRatio': 'Land-to-Total Ratio'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Value per acre analysis
        st.subheader("Value per Acre Analysis")
        fig = px.scatter(
            filtered,
            x='Acres',
            y='ValuePerAcre',
            color='Subdivision',
            size='TotalValue',
            hover_data=['PropertyAddress'],
            title='Lot Size vs. Value per Acre',
            labels={
                'Acres': 'Lot Size (Acres)',
                'ValuePerAcre': 'Value per Acre ($)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Ownership Analysis
        st.subheader("Ownership Insights")
        owner_stats = filtered.groupby('OwnerType').agg({
            'ParcelID': 'count',
            'TotalValue': 'mean',
            'Acres': 'mean',
            'LandToTotalRatio': 'mean'
        }).sort_values('ParcelID', ascending=False)
        
        owner_stats.columns = [
            'Property Count', 
            'Avg. Value', 
            'Avg. Acres',
            'Avg. Land Ratio'
        ]
        
        st.dataframe(
            owner_stats.style.format({
                'Avg. Value': '${:,.0f}',
                'Avg. Acres': '{:.2f}',
                'Avg. Land Ratio': '{:.2f}'
            }).background_gradient(cmap='Blues'),
            use_container_width=True
        )

with tab4:
    # Detailed Property Explorer
    st.subheader("Property Detail Explorer")
    with st.expander("🏚️ How is 'Teardown Candidate' determined?"):
        st.markdown("""
        A property is marked as a **Teardown Candidate** if:
        
        - The **Land-to-Total Value Ratio** is **greater than 0.7**, AND
        - The **Property Age** is **over 50 years**, AND
        - The **Improvement Value** is **less than the Land Value**

        These rules suggest the structure adds little value and may be better suited for redevelopment.
        """)

    if not filtered.empty:
        selected_property = st.selectbox(
            "Select a property to explore:",
            options=filtered['PropertyAddress'] + " (" + filtered['ParcelID'].astype(str) + ")",
            index=0
        )
        
        selected_id = selected_property.split("(")[1].replace(")", "").strip()
        property_data = filtered[filtered['ParcelID'] == selected_id].iloc[0]
        
        # Property Card
        st.markdown(f"""
        <div class='property-card' style='background-color: #2e2e2e;'>
            <h4 style='margin-top:0;'>{property_data['PropertyAddress']}</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                <div>
                    <p><b>Parcel ID:</b> {property_data['ParcelID']}</p>
                    <p><b>Subdivision:</b> {property_data.get('Subdivision', 'N/A')}</p>
                    <p><b>Land Use:</b> {property_data.get('LandUse', 'N/A')}</p>
                    <p><b>Year Built:</b> {property_data.get('YearBuilt', 'N/A')}</p>
                    <p><b>Property Age:</b> {property_data.get('PropertyAge', 'N/A')} years</p>
                </div>
                <div>
                    <p><b>Total Value:</b> ${property_data.get('TotalValue', 0):,.0f}</p>
                    <p><b>Land Value:</b> ${property_data.get('LandValue', 0):,.0f}</p>
                    <p><b>Improvement Value:</b> ${property_data.get('ImprovementValue', 0):,.0f}</p>
                    <p><b>Acres:</b> {property_data.get('Acres', 0):.2f}</p>
                    <p><b>Value/Acre:</b> ${property_data.get('ValuePerAcre', 0):,.0f}</p>
                </div>
            </div>
            <div style='margin-top: 15px;'>
                <p><b>Owner:</b> 🔒 Redacted in public demo</p>
                <p><b>Mailing Address:</b> 🔒 Available in client dashboards</p>
                <p><b>Last Sale Date:</b> {property_data.get('LastSaleDate', 'N/A')}</p>
                <p><b>Teardown Candidate:</b> {'✅ Yes' if property_data.get('IsTeardown', False) else '❌ No'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show comparable properties
        st.subheader("Comparable Properties")
        comparable = filtered[
            (filtered['Subdivision'] == property_data.get('Subdivision')) &
            (filtered['LandUse'] == property_data.get('LandUse')) &
            (filtered['ParcelID'] != property_data['ParcelID'])
        ].sort_values('TotalValue')
        
        if not comparable.empty:
            styled_comparables = comparable[[
                'ParcelID', 'PropertyAddress', 'YearBuilt', 
                'TotalValue', 'Acres', 'LandToTotalRatio', 'IsTeardown'
            ]].style.format({
                'TotalValue': '${:,.0f}',
                'Acres': '{:.2f}',
                'LandToTotalRatio': '{:.2f}'
            }).applymap(
                lambda _: 'background-color: #2e2e2e'  # Light yellow for all cells
            )

            st.dataframe(styled_comparables, use_container_width=True)

        else:
            st.info("No comparable properties found with current filters.")
    st.markdown("---")
    st.markdown("""
    <div style='background-color:#2e2e2e; padding: 15px; border-radius: 8px; margin-top: 20px;'>
        <h4 style='color:#fff;'>🚀 Let's Build Yours</h4>
        <p style='color:#ddd;'>Interested in a dashboard like this for your firm or market?</p>
        <p style='color:#ddd;'>📧 <a href="mailto:schenkotto1@gmail.com">schenkotto1@gmail.com</a><br>
        🔗 <a href="https://www.linkedin.com/in/otto-schenk/">linkedin.com/in/otto-schenk</a></p>
    </div>
    """, unsafe_allow_html=True)


# --- Download Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Export Data")
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered.to_csv(index=False),
    file_name="cornelius_properties.csv",
    mime="text/csv"
)
st.sidebar.caption("🚫 Owner names and mailing addresses have been redacted in this demo for privacy. Full data available in client-specific dashboards.")


with st.sidebar.expander("📊 Want more insights like crime, zoning, or permits?"):
    st.markdown("""
    This demo only scratches the surface. I can integrate:
    - 🧭 **Crime heatmaps** (e.g., violent crimes, property crimes)
    - 🏗️ **Permit history and open construction**
    - 🏫 **School scores, walkability, and transit**
    - 📐 **Zoning overlays and compliance alerts**
    - 🏘️ **Custom comparables (BR/BA, GLA, year built)**
    - 📄 **One-click PDF reports or email alerts**
    
    Let’s tailor this for your neighborhood, city, or investment strategy.
    """)


st.sidebar.markdown("---")
st.sidebar.markdown("""
📩 **Want a dashboard like this for your business?**

Let's build one together — reach out at  
📧 [schenkotto1@gmail.com](mailto:schenkotto1@gmail.com)  
🔗 [linkedin.com/in/otto-schenk](https://www.linkedin.com/in/otto-schenk/)
""")

# --- Footer ---
st.markdown("""
---
<div style='text-align:center; padding:15px; background-color:#333333; border-radius:8px;'>
    <p style='margin:0;'>🔧 Built by the <b>REDC Team</b> | Data updated: {}</p>
</div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)