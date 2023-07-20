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


def preprocess_data():
    '''
    This function preprocesses data by converting data formats and saves in a csv format
    '''
    df = pd.read_csv("data/electricity_sales_2000_to_2023_monthly.csv", index_col=0)

    states_of_interest = pd.read_csv("data/states.csv", index_col=0)
    states_of_interest = list(states_of_interest["state"].values)

    df = df[df["stateDescription"].isin(states_of_interest)].reset_index(drop=True)

    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month

    df = df[["period", "month", "year", "stateDescription", "price", "sales"]]
    df = df.sort_values(by=["stateDescription", "year", "month"]).reset_index(drop=True)

    df.to_csv("../data/electricity_sales_2000_to_2023_monthly_processed.csv")
