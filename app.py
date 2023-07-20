import pandas as pd
import requests
import json
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Streamlit Dashboard",
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: black;'>Price Optimization in retail energy markets</h1>",
    unsafe_allow_html=True,
)


with st.sidebar:
    states_of_interest = pd.read_csv("data/states.csv", index_col=0)
    state = st.selectbox("US State", list(states_of_interest["state"].values), index=4)
#    st.write("You selected:", state)


#############################  Modeling #######################

df = pd.read_csv(
    "data/electricity_sales_2000_to_2023_monthly_processed.csv", index_col=0
)
df["period"] = pd.to_datetime(df["period"])


df_state = df[df["stateDescription"] == state].reset_index(drop=True)


Q1 = df_state["price"].quantile(0.25)
Q3 = df_state["price"].quantile(0.75)
IQR = Q3 - Q1
a_min = Q1 - (1.5 * IQR)
a_max = Q3 + (1.5 * IQR)
df_state = df_state[
    (df_state["price"] <= a_max) & (df_state["price"] >= a_min)
].reset_index(drop=True)


X = df_state[["month", "price"]]
y = df_state["sales"]


model = LinearRegression()
model.fit(X, y)


Price = np.linspace(0.0, df_state["price"].max() + 10.0, 1000)

with st.sidebar:
    cost = st.slider("Fixed cost", 0.0, 1.0, 0.3)
#    st.write("FC ", cost)


month = datetime.now().month + 1

df_price_optimize = pd.DataFrame({"month": month, "price": Price})
df_price_optimize["demand"] = model.predict(df_price_optimize)
df_price_optimize["revenue"] = (
    (df_price_optimize["price"] - cost) * df_price_optimize["demand"] * 0.01
)

#############################  Modeling #######################


with st.sidebar:
    price = st.slider(
        "Current price", 0.0, df_state["price"].max() + 10.0, float(df_state["price"].min())
    )
#    st.write("Price ", price)


col1, col2, col3 = st.columns(3)


def block(value, metric, rgb_code):
    wch_colour_box = rgb_code
    wch_colour_font = (0, 0, 0)
    fontsize = 18
    valign = "left"
    iconname = "fa-solid fa-square-check"
    sline = metric
    lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'
    i = value

    htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                {wch_colour_box[1]}, 
                                                {wch_colour_box[2]}, 0.75); 
                            color: rgb({wch_colour_font[0]}, 
                                    {wch_colour_font[1]}, 
                                    {wch_colour_font[2]}, 0.75); 
                            font-size: {fontsize}px; 
                            border-radius: 7px; 
                            padding-left: 12px; 
                            padding-top: 1px; 
                            padding-bottom: 1px; 
                            line-height:25px;'>
                            <i class='{iconname} fa-xs'></i> <b>{i}</b>
                            </style><BR><span style='font-size: 14px; 
                            margin-top: 0;'>{sline}</style></span></p>"""

    st.markdown(lnk + htmlstr, unsafe_allow_html=True)


with col1:
    block(price, metric="Current Price (cents per kWh)", rgb_code=(135, 206, 235))
with col2:
    input_data = pd.DataFrame([{"month": month, "price": price}])
    output_data = int(model.predict(input_data)[0])
    block(output_data, metric="Expected demand on next month (M kWh units)", rgb_code=(255, 204, 203))
with col3:
    profit_data = "{:.2f}".format((price - cost) * output_data * 0.01)
    block(
        profit_data + " Million USD",
        metric="Next month revenue based on current price",
        rgb_code=(144, 238, 144),
    )


col4, col5, col6 = st.columns(3)

with col4:
    index_max = df_price_optimize[["revenue"]].idxmax().values[0]
    block(
        np.round(df_price_optimize["price"][index_max], 2),
        metric="Optimum price (cents per kWh)",
        rgb_code=(135, 206, 235),
    )
with col5:
    output_data = int(df_price_optimize["demand"][index_max])
    block(output_data, metric="Optimum demand on next month (M kWh units)", rgb_code=(255, 204, 203))
with col6:
    profit_data = "{:.2f}".format(df_price_optimize["revenue"][index_max])
    block(
        profit_data + " Million USD",
        metric="Next month revenue based on optimum price",
        rgb_code=(144, 238, 144),
    )


############################# Plotting ########################

col11, col12 = st.columns(2)

with col11:
    fig_PriceVsQuantity = px.line(
        df_price_optimize,
        x="price",
        y="revenue",
        width=400,
        height=400,
        labels={"price": "Price (cents per kWh)", "revenue": "Revenue (Million USD)"},
    )
    index_max = df_price_optimize[["revenue"]].idxmax().values[0]
    fig_PriceVsQuantity.add_annotation(
        x=df_price_optimize["price"][index_max],
        y=df_price_optimize["revenue"][index_max],
        text="<b> Optimal point </b>",
        showarrow=True,
        arrowhead=1,
    )

    fig_PriceVsQuantity.add_vline(
        x=df_price_optimize["price"][index_max],
        line_width=3,
        line_dash="dash",
        line_color="black",
        opacity=0.3,
    )
    fig_PriceVsQuantity.add_hline(
        y=df_price_optimize["revenue"][index_max],
        line_width=3,
        line_dash="dash",
        line_color="black",
        opacity=0.3,
    )

    fig_PriceVsQuantity.update_xaxes(
        title_font=dict(size=16, family="Courier", color="black")
    )
    fig_PriceVsQuantity.update_yaxes(
        title_font=dict(size=16, family="Courier", color="black")
    )
    fig_PriceVsQuantity.update_layout(title="<b>Revenue vs Price</b>")

    # fig_PriceVsQuantity.update_xaxes(
    #    range=[df_price_optimize["price"].min(), df_state["price"].max()]
    # )
    # fig_PriceVsQuantity.update_yaxes(range=[3, 9])
    st.plotly_chart(fig_PriceVsQuantity)


with col12:
    fig_PriceVsQuantity = px.scatter(
        df_state,
        x="price",
        y="sales",
        color="year",
        trendline="ols",
        labels={"price": "Price (cents per kWh)", "sales": "Demand (Million kWh)"},
    )

    fig_PriceVsQuantity.update_xaxes(
        title_font=dict(size=16, family="Courier", color="black")
    )
    fig_PriceVsQuantity.update_yaxes(
        title_font=dict(size=16, family="Courier", color="black")
    )
    fig_PriceVsQuantity.update_layout(title="<b>Demand vs Price</b>")

    st.plotly_chart(fig_PriceVsQuantity, use_container_width=True, theme=None)

############################# Plotting ########################


col13, col14 = st.columns(2)

with col13:
    fig = px.line(
        df_state,
        x="period",
        y="price",
        width=400,
        height=400,
        labels={"period": "Date", "price": "Price (cents per kWh)"},
    )
    fig.update_xaxes(title_font=dict(size=16, family="Courier", color="black"))
    fig.update_yaxes(title_font=dict(size=16, family="Courier", color="black"))
    fig.update_layout(title="<b>Historical Data</b>")

    st.plotly_chart(fig)


with col14:
    monthly_sales = df_state.groupby("month")["sales"].mean().reset_index()

    fig = px.line(
        monthly_sales,
        x="month",
        y="sales",
        width=300,
        height=400,
        labels={"month": "Month", "sales": "Demand (Million kWh)"},
        markers=True,
    )
    fig.update_traces(marker_size=10)

    fig.update_layout(xaxis=dict(tickmode="linear", tick0=1, dtick=1))
    fig.update_xaxes(title_font=dict(size=16, family="Courier", color="black"))
    fig.update_yaxes(title_font=dict(size=16, family="Courier", color="black"))
    fig.update_layout(title="<b>Average Monthly Sales</b>")

    st.plotly_chart(fig, use_container_width=True)


############################# Plotting ########################


with st.sidebar:
    st.markdown("##")

 #   index_max = df_price_optimize[["revenue"]].idxmax().values[0]
 #   demand_data = int(df_price_optimize["demand"][index_max])
 #   profit_data = "{:.2f}".format(df_price_optimize["revenue"][index_max])



    st.write(
        f"**The maximum revenue is achieved at the optimal point shown in the figure. Fixed cost is kept variable in a narrow range since it is dependent on a particular state**"
    )

    url = "https://www.eia.gov/opendata/"
    st.write("**:red[Data Source]** : [eia.gov](%s)" % url)



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
