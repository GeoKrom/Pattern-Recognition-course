import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from numpy import linalg as LA
from scipy.sparse import csgraph


            
def readDataSpam(filename):
    infile = open(filename)
    data = infile.readlines()
    for i in range(len(data)):
        data[i] = data[i].split(",")
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
            
    return data

def readDataOcc(filename):
    infile = open(filename)
    data = infile.readlines()
    data = data[1:]
    for i in range(len(data)):
        data[i] = data[i].split(",")
        del data[i][0]
        del data[i][0]
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
            
    return data



spamData = readDataSpam("spambase.data")
occData = readDataOcc("datatest.txt")
rightSpamData = []
rightOccData = []
for i in range(len(spamData)):
    rightSpamData.append(spamData[i][len(spamData[i])-1])
    spamData[i] = spamData[i][:len(spamData[i])-1]
    
for i in range(len(occData)):
    rightOccData.append(occData[i][len(occData[i])-1])
    occData[i] = occData[i][:len(occData[i])-1]
    

def normalize(dataset):
    array = dataset
    temp = np.amax(dataset, axis =0)
    temp = temp.tolist()
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            array[i][j] = np.around(array[i][j] / temp[j], decimals =5)
    return array

spamDataNorm = normalize(spamData)
occDataNorm = normalize(occData)

def eigenDecomposition(AffMatrix, plot = False):
    # Compute Laplacian matrix, L = D - A
    L = csgraph.laplacian(AffMatrix)
    # Compute eigenvalues from |L- uI| = 0 and eigenvectors from Lx = ux
    eigenvalues, eigenvectors = LA.eigh(L)

    # A plot for the eigenvalues of the affinity matrix
    if plot:
        plt.figure(figsize=(14, 6))
        plt.title("Largest eigenvalues of input matrix")
        plt.scatter(np.range(np.round(eigenvalues, decimals = 2)), np.round((eigenvalues), decimals = 2))
        plt.grid()
        plt.show()
    print(np.round((eigenvalues), decimals = 2))
    # Compute the largest eigengap
    index_largest_gap = np.argmax(np.diff(np.round((eigenvalues), decimals = 2)))
    print(index_largest_gap)
    # Find the optimal number of clusters from the maximum eigengap
    nb_clusters = index_largest_gap + 2
    return nb_clusters, eigenvalues, eigenvectors

def SpectralClusteringFunc(K, dataset, rightdataset):
    cluster = SpectralClustering(n_clusters=K, affinity = 'cosine')
    cluster.fit(dataset)
    #print(cluster.labels_)
    
    
    affinity_matrix = cluster.affinity_matrix_
    k, _, _ = eigenDecomposition(affinity_matrix)
    print(f'Optimal number of clusters are: {k}')
    
    
    contingency_matrix = metrics.cluster.contingency_matrix(rightdataset, cluster.labels_)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / len(dataset)
    print("Purity for %d Clusters is: %f" % (K, purity))

    # Gia thn pleiopsifia se kathe cluster
    clustersCategories = []
    for i in range(K):

        if contingency_matrix[0][i] > contingency_matrix[1][i]:
            clustersCategories.append(0)
        else:
            clustersCategories.append(1)

    # Gia to F-Measure
    TotalFMeasure = 0
    for i in range(K):  # Gia kathe K
        TruePositive = 0
        TrueNegative = 0
        FalsePositive = 0
        FalseNegative = 0
        for j in range(len(dataset)):  # Gia kathe paradeigma
            label = cluster.labels_[j]  # Krata to label tou paradeigmatos sumfwna me ton kmeans
            if (label != i):  # an den einai idio me to cluster pou eksetazoume
                continue
            else:  # an einai idio
                if rightdataset[j] == clustersCategories[label] and clustersCategories[label] == 1:
                    TruePositive = TruePositive + 1
                elif rightdataset[j] == clustersCategories[label] and clustersCategories[label] == 0:
                    TrueNegative = TrueNegative + 1
                elif rightdataset[j] != clustersCategories[label] and clustersCategories[label] == 1:
                    FalsePositive = FalsePositive + 1
                elif rightdataset[j] != clustersCategories[label] and clustersCategories[label] == 0:
                    FalseNegative = FalseNegative + 1

        if TruePositive != 0 and FalsePositive != 0:
            precision = TruePositive / (TruePositive + FalsePositive)
            recall = TruePositive / (TruePositive + FalseNegative)
            F1 = 2 / ((1 / precision) + (1 / recall))
        else:
            precision = 0
            recall = 0
            F1 = 0

        TotalFMeasure = TotalFMeasure + F1
    print("Total F-Measure for %d Clusters is: %f" % (K, TotalFMeasure))

print("Spectral Clustering for Spambase Data:")
SpectralClusteringFunc(4600, spamDataNorm, rightSpamData)
print("\n")
print("Spectral Clustering for Occupancy Data:")
#SpectralClusteringFunc(31, occDataNorm, rightOccData)

