import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import AffinityPropagation, DBSCAN, Birch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


#normalization function
def normalize(dataColumn):
    min_value = min(dataColumn)
    max_value = max(dataColumn)
    
    normalized_data = [(x-min_value)/(max_value - min_value) for x in dataColumn]

    return normalized_data

#importing data
shoppers_data = pd.read_csv("online_shoppers_intention.csv")

#normalizing when applicable
float_shoppers_data = shoppers_data.select_dtypes("float")

for i in float_shoppers_data.columns:
    shoppers_data[i] = normalize(float_shoppers_data[i])
    

int_shoppers_data = shoppers_data.select_dtypes("int")

for i in int_shoppers_data.columns:
    shoppers_data[i] = normalize(int_shoppers_data[i])

print(shoppers_data.describe())

#taking the relevant values for machine learning
columns_to_drop = ['Month', 'VisitorType']
numerical_shoppers_data = shoppers_data.drop(columns=columns_to_drop, errors='ignore')

#recoding boolean values
numerical_shoppers_data['Weekend'] = numerical_shoppers_data['Weekend'].map({True: 1, False: 0})
numerical_shoppers_data['Revenue'] = numerical_shoppers_data['Revenue'].map({True: 1, False: 0})

print(numerical_shoppers_data.describe())

########    Part 1: Comparison between browser 13 and other browsers    ######################
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

# comparison_fig.show()

#Apply PCA on our data                                                          #Could take a look at SVD as well
X = numerical_shoppers_data.drop(columns=['Revenue'])
y = numerical_shoppers_data['Revenue']
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=[f'Principal Component {i}' for i in range(1, 3)])

explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# barplot the explained variance
plt.bar([f'PC {i}' for i in range(1, 3)],
        explained_variance, color='lightblue')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
# plt.show()

# c stand for color, since there are only two condition of Severity(y) so set two color to easier visualize
colors = ['red' if label == 0 else 'purple' for label in y]
plt.scatter(x=pca_df['Principal Component 1'], y=pca_df['Principal Component 2'],
            c=colors)
plt.title('Principal Components 1 vs 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

Viewer = mpatches.Patch(color='red', label='Revenue=0 (Viewer)')
Buyer = mpatches.Patch(color='purple', label='Revenue=1(Buyer)')

plt.legend(handles=[Viewer, Buyer])
# plt.show()


####### 3.1 Affinaty Propagation Clustering ######

# data_scaled_df is preprocessed dataset
# clustering = AffinityPropagation(random_state=5).fit(pca_df)

# # Labels
# labels = clustering.labels_

# # visualzation
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=labels, cmap='viridis')
# plt.colorbar(scatter)
# plt.title("Affinity Propagation Clustering (PC1 vs PC2)")
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')

# plt.show()

# print("End")


#######     3.2 DBSCAN Clustering   ########

X = pca_df[['Principal Component 1', 'Principal Component 2']]
dbscan = DBSCAN(eps=0.4, min_samples=2).fit(X)
# Labels
labels_dbscan = dbscan.labels_

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X['Principal Component 1'], X['Principal Component 2'], c=labels_dbscan, cmap='viridis')
plt.colorbar(scatter)
plt.title("DBSCAN clustering (Principal Component 1 vs principal Component 2)")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()

#######     3.3 Birch Clustering    #########

birch = Birch(n_clusters=2, threshold=0.5)
birch.fit(X)

# Get cluster labels
labels_birch = birch.labels_

# Visualize 
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X['Principal Component 1'], X['Principal Component 2'], c=labels_birch, cmap='viridis')
plt.colorbar(scatter)
plt.title("BIRCH Clustering (Principal Component 1 vs Principal Componenet 2)")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()

print(X.iloc[0])

########## 4.1 Silhouette Score ################

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def silhouette_samples(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        current_label = labels[i]
        current_point = X[i]
        
        same_cluster_points = X[labels == current_label]
        if len(same_cluster_points) > 1:  
            a = np.mean([euclidean_distance(current_point, other) 
                        for other in same_cluster_points if not np.array_equal(current_point, other)])
        else:
            a = 0
        
        b = np.inf
        for other_label in unique_labels:
            if other_label == current_label:
                continue
            other_cluster_points = X[labels == other_label]
            mean_distance = np.mean([euclidean_distance(current_point, other) for other in other_cluster_points])
            if mean_distance < b:
                b = mean_distance
        
        silhouette_scores[i] = (b - a) / max(a, b)
    
    return silhouette_scores

def silhouette_score(X, labels):
    silhouette_scores = silhouette_samples(X, labels)
    return np.mean(silhouette_scores)

dbscan_score = silhouette_score(X, labels_dbscan)
print(f"Silhouette Score for DBSCAN: {dbscan_score}")

birch_score = silhouette_score(X, labels_birch)
print(f"Silhouette Score for DBSCAN: {birch_score}")

##########  4.2 Davies Bouldin Score    ##################
def evaluate_davies_bouldin(X, labels_dbscan, labels_birch):
    score_dbscan = davies_bouldin_score(X, labels_dbscan)
    score_birch = davies_bouldin_score(X, labels_birch)

    print(f"Davies-Bouldin Score for DBSCAN: {score_dbscan}")
    print(f"Davies-Bouldin Score for Birch: {score_birch}")

X = pca_df[['Principal Component 1', 'Principal Component 2']]
evaluate_davies_bouldin(X, labels_dbscan, labels_birch)

############ 4.3 Calsinki-Harabasz Index    ############

def evaluate_davies_bouldin(X, labels_dbscan, labels_birch):
    score_dbscan = davies_bouldin_score(X, labels_dbscan)
    score_birch = davies_bouldin_score(X, labels_birch)

    print(f"Davies-Bouldin Score for DBSCAN: {score_dbscan}")
    print(f"Davies-Bouldin Score for Birch: {score_birch}")

## chage the X to the clustring variable
X = pca[['ExitRates', 'ProductRelated']]
evaluate_davies_bouldin(X, labels_affinity, labels_dbscan, labels_birch)