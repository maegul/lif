# Don't think this approach will work
# the hierarchical clustering removes important information for my purposes

# ===========
import numpy as np
import scipy as sp
from scipy.cluster import hierarchy
from scipy.spatial import distance
from matplotlib import pyplot as plt
# -----------
# ===========
distance.pdist(
    np.array([
        [0, 1],
        [1, 1],
        [3, 4],
        [-3, -45]
    ])
    )
# -----------
# ===========
hierarchy.linkage(
    np.array([
    [0 ,1],
    [1, 1],
    [3, 4],
    [-3, -45]
    ])
    )
# -----------
# ===========
# same as above
z = hierarchy.linkage(
    distance.pdist(np.array([
    [0 ,1],
    [1, 1],
    [3, 4],
    [-3, -45]
    ]))
    )
z
# -----------
# first two cols: indices of "trees"
# for 4 initial vectors: trees 0,1,2,3 are the original vectors
# 4 and 5 here are the new clusters that have been derived
# third col is distance and fourth is number of original vectors in the cluster

# array([[ 0.        ,  1.        ,  1.        ,  2.        ],
#        [ 2.        ,  4.        ,  3.60555128,  3.        ],
#        [ 3.        ,  5.        , 46.09772229,  4.        ]])
# ===========
i = 0
z[i, 0], z[i, 1], z[i, 2]
# -----------

# ===========
fig = plt.figure()
dn = hierarchy.dendrogram(z)
plt.show
# -----------
# So ... use RF distances as proxy for RF overlap
# use pdist to get distances
# and linkage to get clusters of shared connectivity
# Can calculate ahead of time overlapf of RF relative to distance
# given the spatial parameters.
# If using multiple RFs and multiple orientation biases, can
# calculate for each pairing
# ===========
z_root, nodes = hierarchy.to_tree(z, rd=True)
# -----------
# ===========
nodes[0]
# -----------

# ===========
rfs = np.array([
    [0.1, 1],
    [-0.1, 1],
    [1, 0.1],
    [1, -0.1],
    [0, 0],
    [0.5, 0.5],
    [-0.5, -0.5]
    ])
z = hierarchy.linkage(rfs, method='complete', optimal_ordering=True)
# -----------
# ===========
z
# -----------
# ===========
fig = plt.figure()
dn = hierarchy.dendrogram(z)
plt.show
# -----------
