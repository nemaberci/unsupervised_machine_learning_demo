import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS


columns = [
    'Marital_Status','Income','Kidhome','Teenhome',
    'Dt_Customer',
    'MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts',
    'MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases',
    'NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',
    'AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response'
]
dataf = pd.read_csv('marketing_campaign.csv', delimiter="\t", converters={
    'Marital_Status': lambda x: 1 if x == 'Married' or x == 'Together' else 0,
    'Education': lambda x: 1 if x == 'Master' or x == 'PhD' or x == 'Graduation' else 0
}, header=0)
# df = dataf.drop(['Dt_Customer','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',
#     'AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response'], axis=1).dropna()
df = dataf[['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts','NumWebPurchases','NumStorePurchases', 'NumDealsPurchases', 'NumCatalogPurchases','Kidhome','Teenhome']].dropna()
df[['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts']] = StandardScaler().fit_transform(df[['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts']])
# print(df.head(5))
features = df.columns
X = df.loc[:, features].values

nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto')
bnrs = nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)   
sort_dist = []
for i in range(len(distances)):
    if max(distances[i]) < 6_000:
        sort_dist.append(max(distances[i]))
sort_dist.sort()
# print(sort_dist)
n_clusters = 3
clustering = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
clustering.fit(X)
labels = clustering.labels_

# plt.plot(sort_dist)
# plt.show()


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
# print(principalComponents)
# plt.plot(sort_dist)
# plt.show()
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1) 
ax.set_title("ward_" + str(n_clusters), fontsize = 20)
colors = ['red', 'green', 'blue', 'gray', 'purple', 'cyan']

for i in range(len(X)):
    # print(embeddedXtsne[i])
    # print(df.loc[[i]])
    # color = '#' + (str(hex((df.loc[[i]]['Year_Birth'].values[0] - 1800)))[2:]).zfill(2) + (str(hex((df.loc[[i]]['Marital_Status'].values[0]) * 255))[2:]).zfill(2) + (str(hex((df.loc[[i]]['Education'].values[0]) * 255))[2:]).zfill(2)
    ax.scatter(principalComponents[i, 0]
                , principalComponents[i, 1]
                , s = 50
                , c = colors[labels[i]])
ax.grid()

print("Silhouette Coefficient (Should be close to 1): {:.5f}".format( metrics.silhouette_score(X, labels)))
# [-1, 1] -1: rossz kalszterezési eredmény; 0: átfedő klaszterek; 1 jó klaszterezési eredmény

print("Davies-Bouldin Index (Should be low): {:.5f}".format( metrics.davies_bouldin_score(X, labels)))
# az alacsonyabb értékek jobban szeparált klaszetereket jelentenek

print("Calinski-Harabasz Index (Should be high): {:.5f}".format(metrics.calinski_harabasz_score(X, labels)))
#minél magasabb az érték, annál sűrűbbek és jól szeparáltak a klaszterek

clusters = [
    [], [], []
]

for i in range(len(X)):
    clusters[labels[i]].append(X[i])

# output_mtx = [np.array(['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts','NumWebPurchases','NumStorePurchases', 'NumDealsPurchases', 'NumCatalogPurchases'])]
output_mtx = []
for cluster in clusters:
    output_mtx.append(np.mean(cluster, axis=0))
print(np.array(output_mtx))
np.savetxt('output.csv', np.array(output_mtx), delimiter=",")

plt.savefig("ward_pca" + str(n_clusters) + ".png")
# plt.show()
plt.close(fig)