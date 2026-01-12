import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# ----------------------
# Load Data
# ----------------------
with zipfile.ZipFile("DataCoSupplyChainDataset.zip") as z:
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f, encoding='latin1')

# Preprocessing
df.columns = df.columns.str.lower().str.replace(" ", "_")
df['order_date'] = pd.to_datetime(df['order_date_(dateorders)'])
df['shipping_date'] = pd.to_datetime(df['shipping_date_(dateorders)'])
df['shipping_delay'] = (df['shipping_date'] - df['order_date']).dt.days
df['order_month'] = df['order_date'].dt.to_period('M').astype(str)
df['customer_lname'] = df['customer_lname'].fillna("Unknown")
df['customer_zipcode'] = df['customer_zipcode'].fillna(df['customer_zipcode'].mode()[0])
df.drop(columns=['product_description'], inplace=True, errors='ignore')
df['late_delivery_risk'] = df['late_delivery_risk'].astype('category')
df['order_status'] = df['order_status'].astype('category')
for col in ['sales','order_item_quantity']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# K-Means Clustering
cluster_data = df[['sales','order_item_quantity']]
scaled = StandardScaler().fit_transform(cluster_data)
df['cluster_kmeans'] = KMeans(n_clusters=3, random_state=42).fit_predict(scaled)
df['cluster_label'] = df['cluster_kmeans'].map({0:'Small Customer',1:'Medium Customer',2:'Large Customer'})

# ----------------------
# Custom CSS untuk Background Gambar
# ----------------------
st.markdown("""
    <style>
    /* Background Gambar Dashboard */
    .stApp {
        background: url("https://img.freepik.com/free-photo/global-logistics-transportation-network_23-2152005448.jpg?semt=ais_hybrid&w=740&q=80") no-repeat center center fixed;
        background-size: cover;
    }
    .title-center {
    text-align: center;
    color: white;  /* judul putih */
    font-size: 72px;
    font-weight: bold;
    margin-bottom: 20px;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
    }
    .card {
        background-color: rgba(255,255,255,0.85);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .title-center {
        text-align: center;
        color: white;
        font-size: 72px;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }
    .section {
        background-color: rgba(0,0,0,0);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    div[data-baseweb="select"] > div {
        font-size: 28px;   /* ukuran font lebih besar */
        height: 50px;      /* tinggi selectbox */
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# Title
# ----------------------
st.markdown("<h1 style='text-align:center; color:white; font-size:72px; font-weight:bold; text-shadow:2px 2px 6px rgba(0,0,0,0.7);'>Supply Chain Dashboard</h1>", unsafe_allow_html=True)


# ----------------------
# Market Filter
# ----------------------
selected_markets = st.selectbox("Filter Market", options=["Filter Market"] + sorted(df['market'].unique()))
if selected_markets != "Filter Market":
    dff = df[df['market'] == selected_markets]
else:
    dff = df.copy()

# ----------------------
# KPI Section
# ----------------------
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    total_sales = dff['sales'].sum()
    avg_delay = round(dff['shipping_delay'].mean(),1)
    total_orders = len(dff)

    kpi1, kpi2, kpi3 = st.columns([1,1,1])
    kpi1.markdown(f"<div class='card'>Total Sales<br>${total_sales:,.0f}</div>", unsafe_allow_html=True)
    kpi2.markdown(f"<div class='card'>Avg Shipping Delay<br>{avg_delay} days</div>", unsafe_allow_html=True)
    kpi3.markdown(f"<div class='card'>Total Orders<br>{total_orders}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Charts Section
# ----------------------
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_pie = px.pie(dff, names='order_status', title='Order Status')
        fig_pie.update_layout(margin=dict(l=10,r=10,t=50,b=10), plot_bgcolor='rgba(255,255,255,0.85)', paper_bgcolor='rgba(255,255,255,0.85)', title_font=dict(size=24, color='black', family="Arial"))
        st.plotly_chart(fig_pie, use_container_width=True, height=300)

        fig_bar = px.bar(dff.groupby('market',as_index=False)['sales'].sum(),
                         x='market', y='sales', title='Sales by Market')
        fig_bar.update_layout(margin=dict(l=10,r=10,t=50,b=10), plot_bgcolor='rgba(255,255,255,0.85)', paper_bgcolor='rgba(255,255,255,0.85)', title_font=dict(size=24, color='black', family="Arial"))
        st.plotly_chart(fig_bar, use_container_width=True, height=300)

    with chart_col2:
        fig_line = px.line(dff.groupby('order_month',as_index=False)['sales'].sum(),
                           x='order_month', y='sales', title='Monthly Sales')
        fig_line.update_layout(margin=dict(l=10,r=10,t=50,b=10), plot_bgcolor='rgba(255,255,255,0.85)', paper_bgcolor='rgba(255,255,255,0.85)', title_font=dict(size=24, color='black', family="Arial"))
        st.plotly_chart(fig_line, use_container_width=True, height=300)

        fig_scatter = px.scatter(dff, x='sales', y='order_item_quantity',
                                 color='cluster_label', title='Customer Segmentation')
        fig_scatter.update_layout(margin=dict(l=10,r=10,t=50,b=10), plot_bgcolor='rgba(255,255,255,0.85)', paper_bgcolor='rgba(255,255,255,0.85)', title_font=dict(size=24, color='black', family="Arial"))
        st.plotly_chart(fig_scatter, use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)