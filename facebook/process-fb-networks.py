# 读取边表.edges和和特征.allfeat，并将邻接矩阵和对应特征存储为pkl
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
    # 边表转化为networkx图
    with open(edges_dir, 'r') as edges_f:
        g = nx.read_edgelist(edges_f, nodetype=int)

    # Add ego user (directly connected to all other nodes)
    # 添加ego user(与其它节点直接相连)
    g.add_node(ego_user)
    for node in g.nodes():
        if node != ego_user:
            g.add_edge(ego_user, node)

    # read features into dataframe
    # 索引为节点ID
    df = pd.read_table(feats_dir, sep=' ', header=None, index_col=0)

    # Add features from dataframe to networkx nodes
    for node_index, features_series in df.iterrows():
        # Haven't yet seen node (not in edgelist) --> add it now
        # 解决孤立节点不在边表中的问题
        if not g.has_node(node_index):
            g.add_node(node_index)
            g.add_edge(node_index, ego_user)

        ### g.node[node_index]['features'] = features_series.as_matrix()
        # 节点添加特征
        g.nodes[node_index]['features'] = features_series.values

    # 确保构造了连通图
    assert nx.is_connected(g) # [断言：在表达式条件为 false 的时候触发异常](https://www.runoob.com/python3/python3-assert.html)
    
    # Get adjacency matrix in sparse format (sorted by g.nodes())
    adj = nx.adjacency_matrix(g) # 稀疏矩阵存储，可以使用adj.todense()转换为邻接矩阵

    # Get features matrix (also sorted by g.nodes())
    # 特征矩阵
    features = np.zeros((df.shape[0], df.shape[1])) # num nodes, num features
    for i, node in enumerate(g.nodes()):
        features[i,:] = g.nodes[node]['features']

    # Save adj, features in pickle file
    network_tuple = (adj, features)
    
    with open("fb-processed/{0}-adj-feat.pkl".format(ego_user), "wb") as f:
        pickle.dump(network_tuple, f)