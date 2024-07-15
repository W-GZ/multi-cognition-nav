import dgl
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
from category_bgr.category import scene_name_list, category_name_list, target_name_list

SemanticGraph = dgl.DGLGraph()
# nodes and nodes features
SemanticGraph.add_nodes(26, {'x': torch.ones(26, 3)})

# edges
# a_raw = torch.load("adjmat.dat")
a_raw = torch.load("adjmat_manual_gibson.dat")
for i in range(len(a_raw)):
    a_raw[i][i] = 0
x, y = np.where(a_raw == 1)
SemanticGraph.add_edges(x, y)

# labels
category_name_list = category_name_list()

labels = {n: category_name_list[n] for n in range(len(category_name_list))}

nx_G = SemanticGraph.to_networkx().to_undirected()
# pos = nx.random_layout(nx_G)
pos = nx.circular_layout(nx_G)
# pos = nx.shell_layout(nx_G)
# pos = nx.bipartite_layout(nx_G, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# # top = nx.bipartite.sets(nx_G)[0]
# # pos = nx.bipartite_layout(nx_G, top)
# pos = nx.spring_layout(nx_G)
# pos = nx.kamada_kawai_layout(nx_G)
# pos = nx.spectral_layout(nx_G)
# # pos = nx.planar_layout(nx_G)
# pos = nx.spiral_layout(nx_G)
# # pos = nx.multipartite_layout(nx_G)

# random_layout circular_layout shell_layout bipartite_layout spring_layout
# kamada_kawai_layout spectral_layout planar_layout spiral_layout multipartite_layout

# nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

nx.draw_networkx_nodes(nx_G, pos, node_size=300)  # 100,1800
nx.draw_networkx_edges(nx_G, pos, width=1)  # 1,5
nx.draw_networkx_labels(nx_G, pos, labels=labels, font_size=10)  # 40
plt.show()
