import networkx as nx
import pandas as pd
import numpy as np
import pickle

DATA_DIR = './facebook/'

# 文件前缀
# [int(filename.split('.')[0]) for filename in os.listdir(DATA_DIR) if '.allfeat' in filename]
FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]

# Store all ego graphs in pickle files as (adj, features) tuples
for ego_user in FB_EGO_USERS:
    edges_dir = DATA_DIR + str(ego_user) + '.edges'
    feats_dir = DATA_DIR + str(ego_user) + '.allfeat'
    
    # Read edge-list
    with open(edges_dir, 'r') as edges_f:
        g = nx.read_edgelist(edges_f, nodetype=int)

    # Add ego user (directly connected to all other nodes)
    g.add_node(ego_user)
    for node in g.nodes():
        if node != ego_user:
            g.add_edge(ego_user, node)

    # read features into dataframe
    df = pd.read_table(feats_dir, sep=' ', header=None, index_col=0)

    # Add features from dataframe to networkx nodes
    for node_index, features_series in df.iterrows():
        # Haven't yet seen node (not in edgelist) --> add it now
        if not g.has_node(node_index):
            g.add_node(node_index)
            g.add_edge(node_index, ego_user)

        ### g.node[node_index]['features'] = features_series.as_matrix()
        g.nodes[node_index]['features'] = features_series.values

    assert nx.is_connected(g) # [断言：在表达式条件为 false 的时候触发异常](https://www.runoob.com/python3/python3-assert.html)
    
    # Get adjacency matrix in sparse format (sorted by g.nodes())
    adj = nx.adjacency_matrix(g) # 稀疏矩阵存储，可以使用adj.todense()转换为邻接矩阵

    # Get features matrix (also sorted by g.nodes())
    features = np.zeros((df.shape[0], df.shape[1])) # num nodes, num features
    for i, node in enumerate(g.nodes()):
        features[i,:] = g.nodes[node]['features']

    # Save adj, features in pickle file
    network_tuple = (adj, features)
    
    with open("fb-processed/{0}-adj-feat.pkl".format(ego_user), "wb") as f:
        pickle.dump(network_tuple, f)