import networkx as nx
import os

DATA_DIR = './twitter/'
# 获取Twitter数据文件列表
twitter_files = os.listdir(DATA_DIR)
# 寻找所有的.edges文件
edgelists = [filename for filename in twitter_files if '.edges' in filename]

# 创建并打开结果文件
# [Python中打开文件的讲解——open](https://zhuanlan.zhihu.com/p/95989702)
with open('./twitter-ego-summary.txt', 'w') as out_f:
    
    # 分别记录最多节点数和最多边的ego network
    max_nodes = 0
    max_nodes_file = None
    max_edges = 0
    max_edges_file = None

    # 遍历edgelist文件
    for edgelist_file in edgelists:
        	# Read edgelist --> graph
        with open(DATA_DIR+edgelist_file, 'r') as edges_f:
            
            # 创建有向图
            	g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph())
        
        print('Current file: ', edgelist_file)
        print('Number of nodes: ', g.number_of_nodes())
        print('Number of edges: ', g.number_of_edges())
        print('')
        
        out_f.write('Current file: ' + edgelist_file + '\n')
        out_f.write('Number of nodes: ' + str(g.number_of_nodes()) + '\n')
        out_f.write('Number of edges: ' + str(g.number_of_edges()) + '\n')
        out_f.write('\n')
    
        	# 分别更新节点数和边数最大的结果
       	if g.number_of_nodes() > max_nodes:
       		max_nodes_file = edgelist_file
       		max_nodes = g.number_of_nodes()
       
       	if g.number_of_edges() > max_edges:
       		max_edges_file = edgelist_file
       		max_edges = g.number_of_edges()
   
    # 打印最终结果
    print('Most nodes: ', max_nodes_file, ' (', max_nodes, ') nodes')
    print('Most edges: ', max_edges_file, ' (', max_edges, ') edges')
    
    out_f.write('Most nodes: ' + max_nodes_file + ' (' + str(max_nodes) + ') nodes' + '\n')
    out_f.write('Most edges: ' + max_edges_file + ' (' + str(max_edges) + ') edges' + '\n')