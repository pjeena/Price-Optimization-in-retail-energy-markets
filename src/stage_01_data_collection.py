import pandas as pd
import requests
import json
from decouple import config


def collect_data():
    """
    This function downloads data from https://www.eia.gov/opendata/ API and saves as a csv file
    """

    API_KEY = config("API_KEY")
    df = pd.DataFrame()
    for offset in range(0, 100000, 5000):
        url = "https://api.eia.gov/v2/electricity/retail-sales/data/?frequency=monthly&data[0]=customers&data[1]=price&data[2]=revenue&data[3]=sales&facets[sectorid][]=IND&sort[0][column]=period&sort[0][direction]=asc&offset={}&length=5000&api_key={}".format(
            offset, API_KEY
        )
        response_API = requests.get(url)
        print(response_API.status_code, offset)
        data = response_API.text
        data = json.loads(data)
        data = pd.DataFrame(data["response"]["data"])
        df = pd.concat([df, data])

    return df


df_eia = collect_data()


#df_eia.to_csv("data/electricity_sales_2000_to_2023_monthly.csv")
