from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================
# LOAD & PREPROCESS DATA
# ======================

import pandas as pd
import zipfile

# Baca CSV dari ZIP
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
# DASH APP
# ======================
app = Dash(__name__)

def calculate_kpi(data):
    return (
        data['sales'].sum(),
        round(data['shipping_delay'].mean(),1),
        len(data)
    )

def beautify_fig(fig):
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=35, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=13
    )
    return fig

# ======================
# LAYOUT
# ======================
app.layout = html.Div(
    style={
        'background':'linear-gradient(135deg,#74ebd5,#ACB6E5)',
        'minHeight':'100vh',
        'padding':'12px',
        'fontFamily':'Arial'
    },
    children=[

        html.H2("Supply Chain Dashboard",
                style={'textAlign':'center','color':'white','marginBottom':'10px'}),

        # Dropdown
        html.Div([
            dcc.Dropdown(
                id='dropdown-market',
                options=[{'label':m,'value':m} for m in sorted(df['market'].unique())],
                multi=True,
                placeholder='Filter Market'
            )
        ], style={'width':'45%','margin':'auto','marginBottom':'12px'}),

        # KPI Cards
        html.Div(id='kpi-cards',
            style={
                'display':'grid',
                'gridTemplateColumns':'repeat(3,1fr)',
                'gap':'10px',
                'marginBottom':'12px'
            }
        ),

        # 2x2 GRID GRAPH
        html.Div(
            style={
                'display':'grid',
                'gridTemplateColumns':'repeat(2,1fr)',
                'gap':'12px'
            },
            children=[
                html.Div(dcc.Graph(id='pie-order-status', style={'height':'240px'}, config={'displayModeBar':False}),
                         style={'background':'rgba(255,255,255,0.9)','borderRadius':'10px','padding':'5px'}),

                html.Div(dcc.Graph(id='line-monthly-sales', style={'height':'240px'}, config={'displayModeBar':False}),
                         style={'background':'rgba(255,255,255,0.9)','borderRadius':'10px','padding':'5px'}),

                html.Div(dcc.Graph(id='bar-market-sales', style={'height':'240px'}, config={'displayModeBar':False}),
                         style={'background':'rgba(255,255,255,0.9)','borderRadius':'10px','padding':'5px'}),

                html.Div(dcc.Graph(id='scatter-cluster', style={'height':'240px'}, config={'displayModeBar':False}),
                         style={'background':'rgba(255,255,255,0.9)','borderRadius':'10px','padding':'5px'}),
            ]
        )
    ]
)

# ======================
# CALLBACK
# ======================
@app.callback(
    Output('pie-order-status','figure'),
    Output('line-monthly-sales','figure'),
    Output('bar-market-sales','figure'),
    Output('scatter-cluster','figure'),
    Output('kpi-cards','children'),
    Input('dropdown-market','value')
)
def update_dashboard(selected_markets):

    dff = df.copy()
    if selected_markets:
        dff = dff[dff['market'].isin(selected_markets)]

    total_sales, avg_delay, total_orders = calculate_kpi(dff)

    kpis = [
        html.Div(f"Total Sales\n{total_sales:,.0f}",
                 style={'background':'rgba(255,255,255,0.85)','borderRadius':'10px','padding':'10px','textAlign':'center'}),
        html.Div(f"Avg Shipping Delay\n{avg_delay} days",
                 style={'background':'rgba(255,255,255,0.85)','borderRadius':'10px','padding':'10px','textAlign':'center'}),
        html.Div(f"Total Orders\n{total_orders}",
                 style={'background':'rgba(255,255,255,0.85)','borderRadius':'10px','padding':'10px','textAlign':'center'})
    ]

    fig_pie = beautify_fig(px.pie(dff, names='order_status', title='Order Status'))
    fig_line = beautify_fig(px.line(dff.groupby('order_month',as_index=False)['sales'].sum(),
                                    x='order_month', y='sales', title='Monthly Sales'))
    fig_bar = beautify_fig(px.bar(dff.groupby('market',as_index=False)['sales'].sum(),
                                  x='market', y='sales', title='Sales by Market'))
    fig_scatter = beautify_fig(px.scatter(dff, x='sales', y='order_item_quantity',
                                          color='cluster_label', title='Customer Segmentation'))

    return fig_pie, fig_line, fig_bar, fig_scatter, kpis

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)