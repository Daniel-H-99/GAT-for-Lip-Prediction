import numpy as np
import pickle as pkl
import os

NUM_NODES = 6000
TRAIN_PORTION = 0.8
NUM_TEST = 2000

num_train = int(NUM_NODES * TRAIN_PORTION)
num_eval = NUM_NODES - num_train
indices = []
features = []
labels = []
train_idx = []
eval_idx = []
test_idx = []

with open('/home/ubuntu/workspace/datasets/train/config.txt', 'r') as f:
    nodes = f.readlines()
with open('/home/ubuntu/workspace/datasets/test/config.txt', 'r') as f:
    test_nodes = f.readlines()
with open('/home/ubuntu/workspace/datasets/train/PCA_reducedKp.pickle', 'rb') as f:
    _labels = pkl.load(f)
with open('/home/ubuntu/workspace/datasets/test/PCA_reducedKp.pickle', 'rb') as f:
    _test_labels = pkl.load(f)
    
nodes = nodes[:NUM_NODES] + test_nodes[:NUM_TEST]

for (idx, node) in enumerate(nodes):
    indices.append([idx])
    is_test = False
    if idx < num_train:
        train_idx.append(idx)
    elif idx < NUM_NODES:
        eval_idx.append(idx)
    else:
        is_test = True
        test_idx.append(idx)
    tmp = node.strip().split('/')
    vid, id = tmp[0], tmp[1]
    if not is_test:
        with open(os.path.join('/home/ubuntu/workspace/datasets/train', vid, 'audio', '{:05d}.pickle'.format(int(id) - 1)), 'rb') as f:
            feature = pkl.load(f)
        features.append(feature.reshape(-1))
        labels.append(_labels[vid][id].reshape(-1))
    else:
        with open(os.path.join('/home/ubuntu/workspace/datasets/test', vid, 'audio', '{:05d}.pickle'.format(int(id) - 1)), 'rb') as f:
            feature = pkl.load(f)
        features.append(feature.reshape(-1))
        labels.append(_test_labels[vid][id].reshape(-1))        

indices = np.array(indices)
features = np.array(features)
labels = np.array(labels)
train_idx = np.array(train_idx)
eval_idx = np.array(eval_idx)
test_idx = np.array(test_idx)
print("indices: {}".format(indices.shape))
print("features: {}".format(features.shape))
print("labels: {}".format(labels.shape))

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features)
distances, edges = nbrs.kneighbors(features)
indices_features_labels = np.concatenate([indices, features, labels], axis=1)


with open('/home/ubuntu/workspace/GAT/data/face/nodes.pickle', 'wb') as f:
    pkl.dump(indices_features_labels, f)
with open('/home/ubuntu/workspace/GAT/data/face/edges.pickle', 'wb') as f:
    pkl.dump(edges, f)
with open('/home/ubuntu/workspace/GAT/data/face/train_indice.pickle', 'wb') as f:
    pkl.dump(train_idx, f)
with open('/home/ubuntu/workspace/GAT/data/face/eval_indice.pickle', 'wb') as f:
    pkl.dump(eval_idx, f)
with open('/home/ubuntu/workspace/GAT/data/face/test_indice.pickle', 'wb') as f:
    pkl.dump(test_idx, f)


