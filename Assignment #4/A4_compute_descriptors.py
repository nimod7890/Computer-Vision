from functions import *
import struct

random.seed(N_SEED)

data=[]
for num in range(N_IMG):
    features=get_SIFT_features(str(num).zfill(5))
    data.extend(features)
data=np.array(data)

rand10e4=get_rand_array(data,N_10e4)
clustersInit=np.array(init_kmeans_plus(rand10e4)) #k-means++
# clusters=get_rand_array(rand10e4,N_CLUSTERS) #k-means

clustersFinal=cluster_centers_kmeans(rand10e4,clustersInit)

descriptor = np.zeros((N_IMG, MAX_DIM))
for num in range(N_IMG):
    features=get_SIFT_features(str(num).zfill(5))
    descriptor[num]=get_descriptor(features,clustersFinal)

with open(f'kmeansP_{N_SEED}({ITER_K_MEANS}).des', 'wb') as f:
    f.write(struct.pack('ii', N_IMG, MAX_DIM))
    f.write(descriptor.astype('float32').tobytes())