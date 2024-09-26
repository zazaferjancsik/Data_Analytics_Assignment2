import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

columns_to_drop = ['Month', 'VisitorType']
numerical_shoppers_data = shoppers_data.drop(columns=columns_to_drop, errors='ignore')

numerical_shoppers_data['Weekend'] = numerical_shoppers_data['Weekend'].map({True: 1, False: 0})
numerical_shoppers_data['Revenue'] = numerical_shoppers_data['Revenue'].map({True: 1, False: 0})

print(numerical_shoppers_data.describe())

b13_shoppers = shoppers_data.query('Browser == 1')

other_shoppers = shoppers_data.query('Browser != 1')

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

comparison_shoppers_melted = comparison_shoppers.melt(var_name='Variable', value_name='Value', ignore_index=False)
comparison_shoppers_melted = comparison_shoppers_melted.reset_index()

comparison_fig = px.bar(
    data_frame=comparison_shoppers_melted,
    x="Variable",
    y="Value",
    color="index",
    barmode="group",
)

comparison_fig.update_layout(
    xaxis=dict(tickfont=dict(family='Arial Black', size=14)),
    yaxis=dict(tickfont=dict(family='Arial Black', size=14))

)

comparison_fig.show()


# Same as 4.1
# Severity is the target variable, what we want to predict
# seperate the Severity column
X = numerical_shoppers_data.drop(columns=['Revenue'])
y = numerical_shoppers_data['Revenue']

# apply PCA with n_components=5
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)

# create a DataFrame with PCA components
pca_df = pd.DataFrame(data=X_pca, columns=[f'Principal Component {i}' for i in range(1, 16)])
# print(pca_df)

# print the explained variance ratio
## the sum will be 1.0, meaning all the component explain total variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# barplot the explained variance
plt.bar([f'PC {i}' for i in range(1, 16)],
        explained_variance, color='lightblue')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

# visualize the PCA result in the first two principal components
# because the first two components explain 89.95%  of the variance
# c stand for color, since there are only two condition of Severity(y) so set two color to easier visualize
colors = ['red' if label == 0 else 'purple' for label in y]
plt.scatter(x=pca_df['Principal Component 1'], y=pca_df['Principal Component 2'],
            c=colors)
plt.title('Principal Components 1 vs 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# Create legend
Viewer = mpatches.Patch(color='red', label='Revenue=0 (Viewer)')
Buyer = mpatches.Patch(color='purple', label='Revenue=1(Buyer)')
# Add the legend to the plot
plt.legend(handles=[Viewer, Buyer])
plt.show()


####### 3.1 Affinaty Propagation Clustering ######

# data_scaled_df is preprocessed dataset
clustering = AffinityPropagation(random_state=5).fit(numerical_shoppers_data)

# Labels
labels = clustering.labels_

# visualzation
plt.figure(figsize=(10, 6))
scatter = plt.scatter(['ExitRates'], numerical_shoppers_data['ProductRelated'], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title("Affinity Propagation Clustering (ExitRates vs ProductRelated)")
plt.xlabel('ExitRates')
plt.ylabel('ProductRelated')

plt.show()