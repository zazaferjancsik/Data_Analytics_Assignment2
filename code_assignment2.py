import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def normalize(dataColumn):
    min_value = min(dataColumn)
    max_value = max(dataColumn)
    
    normalized_data = [(x-min_value)/(max_value - min_value) for x in dataColumn]

    return normalized_data

shoppers_data = pd.read_csv("online_shoppers_intention.csv")

float_shoppers_data = shoppers_data.select_dtypes("float")

for i in float_shoppers_data.columns:
    shoppers_data[i] = normalize(float_shoppers_data[i])

int_shoppers_data = shoppers_data.select_dtypes("int")

for i in int_shoppers_data.columns:
    shoppers_data[i] = normalize(int_shoppers_data[i])

print(shoppers_data.describe())

b13_shoppers = shoppers_data.query('Browser == 1')

other_shoppers = shoppers_data.query('Browser != 13')

df = {
    "Administrative Mean": [b13_shoppers["Administrative"].mean(), other_shoppers["Administrative"].mean()],
    "Administrative_Duration Mean": [b13_shoppers["Administrative_Duration"].mean(), other_shoppers["Administrative_Duration"].mean()],
    "Informational Mean": [b13_shoppers["Informational"].mean(), other_shoppers["Informational"].mean()],
    "Informational_Duration Mean": [b13_shoppers["Informational_Duration"].mean(), other_shoppers["Informational_Duration"].mean()],
    "ProductRelated Mean": [b13_shoppers["ProductRelated"].mean(), other_shoppers["ProductRelated"].mean()],
    "ProductRelated_Duration Mean": [b13_shoppers["ProductRelated_Duration"].mean(), other_shoppers["ProductRelated_Duration"].mean()],
    "BounceRates Mean": [b13_shoppers["BounceRates"].mean(), other_shoppers["BounceRates"].mean()],
    "ExitRates Mean": [b13_shoppers["ExitRates"].mean(), other_shoppers["ExitRates"].mean()],
    "PageValues Mean": [b13_shoppers["PageValues"].mean(), other_shoppers["PageValues"].mean()],
    "SpecialDay Mean": [b13_shoppers["SpecialDay"].mean(), other_shoppers["SpecialDay"].mean()],
    # "OperatingSystems Mean": [b13_shoppers["OperatingSystems"].mean(), other_shoppers["OperatingSystems"].mean()],
    # "Region Mean": [b13_shoppers["Region"].mean(), other_shoppers["Region"].mean()],
    # "TrafficType Mean": [b13_shoppers["TrafficType"].mean(), other_shoppers["TrafficType"].mean()],
    "Weekend Mean": [b13_shoppers["Weekend"].mean(), other_shoppers["Weekend"].mean()],
    "Revenue Mean": [b13_shoppers["Revenue"].mean(), other_shoppers["Revenue"].mean()],
    
}

comparison_shoppers = pd.DataFrame(data=df)

print(comparison_shoppers.T)

comparison_shoppers_melted = comparison_shoppers.melt(var_name='Variable', value_name='Value', ignore_index=False)
comparison_shoppers_melted = comparison_shoppers_melted.reset_index()

comparison_fig = px.bar(
    data_frame=comparison_shoppers_melted,
    x="Variable",
    y="Value",
    color="index",
    barmode="group",
)

comparison_fig.show()

