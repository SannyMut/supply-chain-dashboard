import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================
# LOAD & PREPROCESS DATA
# ======================
with zipfile.ZipFile("DataCoSupplyChainDataset.zip") as z:
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f, encoding='latin1')

df.columns = df.columns.str.lower().str.replace(" ", "_")
df['order_date_(dateorders)'] = pd.to_datetime(df['order_date_(dateorders)'])
df['shipping_date_(dateorders)'] = pd.to_datetime(df['shipping_date_(dateorders)'])
df['shipping_delay'] = (df['shipping_date_(dateorders)'] - df['order_date_(dateorders)']).dt.days
df['order_month'] = df['order_date_(dateorders)'].dt.to_period('M').astype(str)

df['customer_lname'] = df['customer_lname'].fillna("Unknown")
df['customer_zipcode'] = df['customer_zipcode'].fillna(df['customer_zipcode'].mode()[0])
df.drop(columns=['product_description'], inplace=True, errors='ignore')

df['late_delivery_risk'] = df['late_delivery_risk'].astype('category')
df['order_status'] = df['order_status'].astype('category')

for col in ['sales','order_item_quantity']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ======================
# K-MEANS CLUSTERING
# ======================
cluster_data = df[['sales','order_item_quantity']]
scaled = StandardScaler().fit_transform(cluster_data)

df['cluster_kmeans'] = KMeans(n_clusters=3, random_state=42).fit_predict(scaled)
df['cluster_label'] = df['cluster_kmeans'].map({
    0:'Small Customer',
    1:'Medium Customer',
    2:'Large Customer'
})

# ======================
# STREAMLIT LAYOUT
# ======================
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")
st.markdown("<h2 style='text-align: center; color: white; background: linear-gradient(135deg,#74ebd5,#ACB6E5); padding:10px'>Supply Chain Dashboard</h2>", unsafe_allow_html=True)

# Dropdown filter
selected_markets = st.multiselect(
    "Filter Market",
    options=sorted(df['market'].unique())
)

dff = df.copy()
if selected_markets:
    dff = dff[dff['market'].isin(selected_markets)]

# KPI Cards
total_sales = dff['sales'].sum()
avg_delay = round(dff['shipping_delay'].mean(),1)
total_orders = len(dff)

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{total_sales:,.0f}")
col2.metric("Avg Shipping Delay", f"{avg_delay} days")
col3.metric("Total Orders", f"{total_orders}")

# 2x2 Grid Charts
fig_pie = px.pie(dff, names='order_status', title='Order Status')
fig_line = px.line(dff.groupby('order_month', as_index=False)['sales'].sum(),
                   x='order_month', y='sales', title='Monthly Sales')
fig_bar = px.bar(dff.groupby('market', as_index=False)['sales'].sum(),
                 x='market', y='sales', title='Sales by Market')
fig_scatter = px.scatter(dff, x='sales', y='order_item_quantity',
                         color='cluster_label', title='Customer Segmentation')

row1_col1, row1_col2 = st.columns(2)
row1_col1.plotly_chart(fig_pie, use_container_width=True)
row1_col2.plotly_chart(fig_line, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
row2_col1.plotly_chart(fig_bar, use_container_width=True)
row2_col2.plotly_chart(fig_scatter, use_container_width=True)