import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# https://blog.csdn.net/qq_36643449/article/details/123529791


def normalize(A, symmetric=True):
    A = A + torch.eye(A.size(0))
    # d = torch.abs(A.sum(1))
    d = A.sum(1)
    # print(d)
    # x = np.where(d <= 0.0)
    # print(len(x[0]))

    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))

        return D.mm(A).mm(D)  # D^(-1/2)AD^(-1/2)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN_3_layers_(nn.Module):
    """
    Z = AXW
    """
    def __init__(self, dim_in, hidden_size1, hidden_size2, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_size1)  # , bias=False)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # , bias=False)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc3 = nn.Linear(hidden_size2, dim_out)  # , bias=False)

    def forward(self, A, X):
        """
        计算三层gcn
        """

        X = self.leaky_relu1(self.fc1(A.mm(X)))  # X就是公式中的H
        X = self.leaky_relu2(self.fc2(A.mm(X)))
        X = self.fc3(A.mm(X))

        return torch.softmax(X, dim=1)


class GraphPooling_(nn.Module):
    """
    Z = AXWS
    """
    def __init__(self, dim_in, hidden_size1, hidden_size2, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_size1)  # , bias=False)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # , bias=False)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc3 = nn.Linear(hidden_size2, dim_out)  # , bias=False)

    def forward(self, A, X, S=None):
        """
        计算三层gcn
        """
        X = self.leaky_relu1(self.fc1(A.mm(X)))  # X就是公式中的H
        X = self.leaky_relu2(self.fc2(A.mm(X)))
        X = self.fc3(A.mm(X)).mm(S.transpose(1, 0))
        return torch.softmax(X, dim=1)

# # # # #
# import dgl
# import networkx as nx
#
# G = dgl.DGLGraph()
# # add edges
# G.add_nodes(10, {'x': torch.ones(10, 6)})
# G.add_edges(0, 0)
# G.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5

# # G.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9
# # print(G.number_of_edges())
# print(G.all_edges())
# print(G.all_edges()[0])
# print(G.all_edges()[1])
# all_edges = [G.all_edges()[0].cpu().detach(), G.all_edges()[1].cpu().detach()]
# print(all_edges[0])
# print(all_edges[1])
# nx_G = G.to_networkx()
# A = nx.adjacency_matrix(nx_G).todense()
# A_normed = normalize(torch.FloatTensor(A), True)
# # print(A_normed.shape)
# # print(G.ndata['x'].shape)
#
# GCN_ = GCN_3_layers(6, 4, 2, 2)
# S = GCN_(A_normed, G.ndata['x'])
# # print(S.shape)
# # print(S.transpose(1, 0).shape)
#
# G1 = dgl.DGLGraph()
# # add edges
# G1.add_nodes(12, {'y': torch.ones(12, 8), 'x': torch.ones(12, 6)})
# G1.add_edges(0, 1)
# G1.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5
# G1.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9
# # print(G.number_of_edges())
# nx_G1 = G1.to_networkx()
# B = nx.adjacency_matrix(nx_G1).todense()
# B_normed = normalize(torch.FloatTensor(B), True)
#
# GraphPooling_ = GraphPooling(6, 4, 2, 2)
#
# C = GraphPooling_(B_normed, G1.ndata['x'], S)
# print(C.shape)

# print(G1.ndata['y'][2].shape)
