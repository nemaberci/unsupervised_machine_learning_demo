import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, dbscan
from sklearn.cluster import KMeans

for eps in [5,10,15,20]:

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
    
    # df = dataf[['Income','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases']].dropna()
    df = dataf[['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts','NumWebPurchases','NumStorePurchases', 'NumDealsPurchases', 'NumCatalogPurchases','Kidhome','Teenhome']].dropna()
    df[['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts']] = StandardScaler().fit_transform(df[['Income','MntWines','MntSweetProducts','MntFruits','MntMeatProducts','MntFishProducts']])
    print(df.head(5))
    features = df.columns
    X = df.loc[:, features].values

    tsne = TSNE(n_components = 3, perplexity=eps, n_iter=1000)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto')
    bnrs = nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)   
    sort_dist = []
    for i in range(len(distances)):
        if max(distances[i]) < 6000:
            sort_dist.append(max(distances[i]))
    sort_dist.sort()
    # print(sort_dist)

    embeddedXtsne = tsne.fit_transform(X)
    # plt.plot(sort_dist)
    # plt.show()
    # Z = linkage(X, 'ward')
    # plt.figure()
    # dn = dendrogram(Z)
    # plt.show()
    n_clusters = 3
    # for dbscaneps in [0.09, 0.12, 0.15, 0.2, 0.25]:

    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title("ward_" + str(n_clusters) + "_" + str(eps), fontsize = 20)
    # kmeans = KMeans(n_clusters=y, init='k-means++', n_init=30, max_iter=1000) 
    # kmeans = kmeans.fit(X)
    # labels = kmeans.predict(X)
    ax.set_title("ward_" + str(n_clusters) + "_" + str(eps), fontsize = 20)
    clustering = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
    clustering.fit(X)
    labels = clustering.labels_
    # ax.set_title("dbscan_" + str(dbscaneps) + "_" + str(eps), fontsize = 20)
    # dbs = DBSCAN(eps=dbscaneps, min_samples=5)
    # dbs = dbs.fit(X)
    # labels = dbs.labels_
    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise = list(labels).count(-1)
    # print(n_clusters, n_noise)

    # kmeans.cluster_centers_ # a centroidok
    colors = ['red', 'green', 'blue', 'gray', 'purple', 'cyan', 'black', 'yellow', 'magenta', 'indigo', 'pink', 'brown', 'lightblue', 'lavender']
    print(labels)

    for i in range(len(X)):
        # print(embeddedXtsne[i])
        # print(df.loc[[i]])
        # color = '#' + (str(hex((df.loc[[i]]['Year_Birth'].values[0] - 1800)))[2:]).zfill(2) + (str(hex((df.loc[[i]]['Marital_Status'].values[0]) * 255))[2:]).zfill(2) + (str(hex((df.loc[[i]]['Education'].values[0]) * 255))[2:]).zfill(2)
        ax.scatter(embeddedXtsne[i, 0]
                    , embeddedXtsne[i, 1]
                    , s = 25
                    , c = colors[labels[i]])
    ax.grid()

    print("Silhouette Coefficient (Should be close to 1): {:.5f}".format( metrics.silhouette_score(X, labels)))
    # [-1, 1] -1: rossz kalszterezési eredmény; 0: átfedő klaszterek; 1 jó klaszterezési eredmény
    
    print("Davies-Bouldin Index (Should be low): {:.5f}".format( metrics.davies_bouldin_score(X, labels)))
    # az alacsonyabb értékek jobban szeparált klaszetereket jelentenek
    
    print("Calinski-Harabasz Index  (Should be high): {:.5f}".format( metrics.calinski_harabasz_score(X, labels)))
    #minél magasabb az érték, annál sűrűbbek és jól szeparáltak a klasztere    
    # plt.show()
    plt.savefig("ward_tsne_" + str(n_clusters) + "_" + str(eps) + ".png")
    plt.close(fig)