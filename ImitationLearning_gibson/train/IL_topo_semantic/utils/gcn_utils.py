import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
# https://blog.csdn.net/CY19980216/article/details/110629996


class GCN_3_layers(nn.Module):
    def __init__(self, in_feats, hidden_size1, hidden_size2, out):
        super(GCN_3_layers, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size1)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = GraphConv(hidden_size1, hidden_size2)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = GraphConv(hidden_size2, out)

    def forward(self, g, inputs):
        # h = self.conv1(g, inputs)
        # h = torch.relu(h)
        # h = self.conv2(g, h)
        # h = torch.relu(h)
        # h = self.conv3(g, h)
        h = self.conv1(g, inputs)
        h = self.leaky_relu1(h)
        h = self.conv2(g, h)
        h = self.leaky_relu2(h)
        h = self.conv3(g, h)
        return torch.softmax(h, dim=1)


class GraphPooling(nn.Module):
    """
    Z = AXWS
    """
    def __init__(self, in_feats, hidden_size1, hidden_size2, out):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size1)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = GraphConv(hidden_size1, hidden_size2)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = GraphConv(hidden_size2, out)

    def forward(self, g, X, S):
        """
        计算三层gcn
        """
        h = self.conv1(g, X)
        h = self.leaky_relu1(h)
        h = self.conv2(g, h)
        h = self.leaky_relu2(h)
        h = self.conv3(g, h)
        h = h.mm(S.transpose(1, 0))
        return torch.softmax(h, dim=1)



# import dgl
# import networkx as nx
#
# G = dgl.DGLGraph()
# # add edges
# G.add_nodes(10, {'x': torch.ones(10, 6)})
# G.add_edges(0, 1)
# G.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5
# G.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9
#
# GCN_ = GCN_3_layers(6, 4, 2, 2)
# S = GCN_(G, G.ndata['x'])
# # print(S.shape)
# # print(S.transpose(1, 0).shape)
#
# G1 = dgl.DGLGraph()
# # add edges
# G1.add_nodes(12, {'y': torch.ones(12, 8)})
# G1.add_edges(0, 1)
# G1.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5
# G1.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9
# # print(G.number_of_edges())
#
# GraphPooling_ = GraphPooling(8, 4, 2, 2)
#
# C = GraphPooling_(G1, G1.ndata['y'], S)
# print(C.shape)

