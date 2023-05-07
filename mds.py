import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


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
df = dataf[['Income','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases']].dropna()
df = pd.DataFrame(StandardScaler().fit_transform(df))
print(df.head(5))
features = df.columns
X = df.loc[:, features].values

nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto')
bnrs = nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)   
sort_dist = []
to_delete = []
for i in range(len(distances)):
    if max(distances[i]) < 6_000:
        sort_dist.append(max(distances[i]))
    else:
        to_delete.append(i)
X = np.delete(X, to_delete, axis=0)
sort_dist.sort()

mds = MDS(n_components=2, n_init=10, max_iter=25)
embeddedX = mds.fit_transform(X)

for method in ['average', 'complete', 'ward', 'single']:

    for n in range(2, 5):

        Z = linkage(X, method)
        fig = plt.figure(figsize = (5,5))
        ax = fig.add_subplot(1,1,1) 
        ax.set_title('Test', fontsize = 20)
        # dn = dendrogram(Z)
        clustering = AgglomerativeClustering(n_clusters = n, linkage = method)
        colors = ['red', 'green', 'blue', 'gray', 'purple', 'cyan']
        clustering.fit(X)
        labels = clustering.labels_
        # print(labels)

        for i in range(len(X)):
            # print(embeddedXtsne[i])
            # print(df.loc[[i]])
            # color = '#' + (str(hex((df.loc[[i]]['Year_Birth'].values[0] - 1800)))[2:]).zfill(2) + (str(hex((df.loc[[i]]['Marital_Status'].values[0]) * 255))[2:]).zfill(2) + (str(hex((df.loc[[i]]['Education'].values[0]) * 255))[2:]).zfill(2)
            ax.scatter(embeddedX[i, 0]
                        , embeddedX[i, 1]
                        , s = 50
                        , c = colors[labels[i]])
        ax.grid()

        print("Silhouette Coefficient (Method: {}, num of clusters: {:n}) (Should be close to 1): {:.5f}".format(method, n, metrics.silhouette_score(X, labels)))
        # [-1, 1] -1: rossz kalszterezési eredmény; 0: átfedő klaszterek; 1 jó klaszterezési eredmény

        print("Davies-Bouldin Index (Method: {}, num of clusters: {:n})  (Should be low): {:.5f}".format(method, n, metrics.davies_bouldin_score(X, labels)))
        # az alacsonyabb értékek jobban szeparált klaszetereket jelentenek

        print("Calinski-Harabasz Index (Method: {}, num of clusters: {:n})  (Should be high): {:.5f}".format(method, n, metrics.calinski_harabasz_score(X, labels)))
        #minél magasabb az érték, annál sűrűbbek és jól szeparáltak a klaszterek

        # plt.show()
        plt.savefig("mds_" + method + "_" + str(n) + ".png")
        plt.close(fig)