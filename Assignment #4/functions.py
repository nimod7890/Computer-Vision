from pathlib import Path
import numpy as np
import random

N_IMG=1000
N_DIM=128
N_CLUSTERS=8
ITER_K_MEANS=60
N_10e4=100_000
MAX_DIM=N_DIM*N_CLUSTERS #1024
N_SEED=500


def get_rand_array(array,num):
    randIdx=np.random.choice(len(array),num,replace=False)
    return array[randIdx]

def L2_distance(feature,cluster):
    return np.sqrt(np.sum((feature-cluster)**2))

def predict_cluster(feature,clusters):
    d=[L2_distance(feature,cluster) for cluster in clusters]
    return np.argmin(d)

def init_kmeans_plus(data):
    clusters=[random.choice(data)]
    for _ in range(N_CLUSTERS-1):
        distance=np.array([min([L2_distance(feature,cluster) for cluster in clusters]) for feature in data])
        cluster=data[np.argmax(distance)]
        clusters.append(cluster)
    return clusters

def cluster_centers_kmeans(data,clusters):
    
    for i in range(ITER_K_MEANS):
        print(f"[{i}/{ITER_K_MEANS}] clusters: {np.sum(clusters)}")
        tmpClusters=np.zeros((N_CLUSTERS,N_DIM))
        count=[0]*N_CLUSTERS
        for feature in data:
            mini=predict_cluster(feature,clusters)
            count[mini]+=1
            tmpClusters[mini]+=feature
        for cluster,cnt in zip(tmpClusters,count):
            cluster/=cnt
        clusters=tmpClusters
    print(f"[result] clusters:{np.sum(clusters)}")

    return clusters

def get_SIFT_features(IMG_PATH):
    features=[item for item in Path(f'./feats/{IMG_PATH}.sift').read_bytes()]
    return np.reshape(features,(-1,N_DIM))
    
def get_descriptor(data,clusters):
    vlad=np.zeros((MAX_DIM))
    for feature in data:
        label=predict_cluster(feature,clusters)
        start=label*N_DIM
        end=start+N_DIM
        vlad[start:end]+=feature-clusters[label]
    vlad=np.sign(vlad)*np.sqrt(np.abs(vlad))
    vlad/= np.sqrt(np.sum(vlad**2)) 
    return vlad