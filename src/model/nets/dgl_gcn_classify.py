import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from src.model.nets.base_net import BaseNet
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(fn.copy_src(src='h', out='m'), reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class):
        super(GCNClassifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_class)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        #print(g.device)
        h = g.ndata['x']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        hg = self.classify(hg)
        return F.softmax(hg, dim=1)



