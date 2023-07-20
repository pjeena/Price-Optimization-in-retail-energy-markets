import pandas as pd
import requests
import json
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


def train_model(state: str):
    df = pd.read_csv(
        "data/electricity_sales_2000_to_2023_monthly_processed.csv", index_col=0
    )
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

    return model, df_state


def find_optimum_price(model, df_state):
    Price = np.linspace(0.0, df_state["price"].max() + 10.0, 1000)
    cost = 0.3
    month = 2

    df_price_optimize = pd.DataFrame({"month": month, "price": Price})
    df_price_optimize["demand"] = model.predict(df_price_optimize)
    df_price_optimize["revenue"] = (
        df_price_optimize["price"] - cost
    ) * df_price_optimize["demand"]

    return df_price_optimize


def plot_revenue_price(df_price_optimize):
    fig_PriceVsQuantity = px.line(
        df_price_optimize, x="price", y="revenue", width=300, height=300
    )
    fig_PriceVsQuantity.show()


if __name__ == "__main__":
    state = "Texas"
    model, df_state = train_model(state=state)
    df_price_optmize = find_optimum_price(model=model, df_state=df_state)
    plot_revenue_price(df_price_optimize=df_price_optmize)
