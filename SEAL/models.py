# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from types import FunctionType
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv, 
                                global_sort_pool, global_add_pool, global_mean_pool)
import pdb


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset, 
                 use_feature=False, node_embedding=None, dropout=0.5, fuse='minus'):
        super(GCN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        self.fuse = fuse
        if self.fuse=='att':
            self.att = SemanticAttention(hidden_channels)
        else:
            self.att = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None, 
                    edge_mask=None, edge_mask_original=None, return_hidden=False):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        x = fake_edge(x, edge_index, edge_mask, edge_mask_original,edge_weight,batch, self)
        hidden = x
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if return_hidden:
            return x, hidden
        else:
            return x
    
    def gnn_forward(self, x,edge_index,batch,edge_weight):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
        else:  # sum pooling
            x = global_add_pool(x, batch)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset=None, 
                 use_feature=False, node_embedding=None, dropout=0.5, fuse='minus'):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        self.fuse = fuse
        if self.fuse=='att':
            self.att = SemanticAttention(hidden_channels)
        else:
            self.att = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def gnn_forward(self, x,edge_index,batch,edge_weight):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
        else:  # sum pooling
            x = global_add_pool(x, batch)
        return x

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None, 
                    edge_mask=None, edge_mask_original=None, return_hidden=False):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        x = fake_edge(x, edge_index, edge_mask, edge_mask_original,edge_weight,batch, self)
        hidden = x
        # classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if return_hidden:
            return x, hidden
        else:
            return x

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        """
            N x M x dim
        """
        w = self.project(z)  # (N, M, 1)
        beta = torch.softmax(w, dim=1)  # (N, M, 1)
        # beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out = (beta * z).sum(1)  # (N, M)

        return out
    
    def reset_parameters(self):
        for lin in self.project:
            if isinstance(lin, nn.Linear):
                lin.reset_parameters()
        return self

# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 dynamic_train=False, GNN=GCNConv, use_feature=False, 
                 node_embedding=None, fuse='minus'):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

        self.fuse = fuse
        if self.fuse=='att':
            self.att = SemanticAttention(dense_dim)
        else:
            self.att = None

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None, 
                    edge_mask=None, edge_mask_original=None, return_hidden=False):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        
        x = fake_edge(x, edge_index, edge_mask, edge_mask_original,edge_weight,batch, self)
        hidden = x
        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if return_hidden:
            return x, hidden
        else:
            return x

    def gnn_forward(self, x, edge_index, batch, edge_weight):
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        return x



class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, train_eps=False, fuse='minus'):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        if self.jk:
            in_channels = num_layers * hidden_channels
        else:
            in_channels = hidden_channels
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        self.fuse = fuse
        if self.fuse=='att':
            self.att = SemanticAttention(in_channels)
        else:
            self.att = None


    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None, 
                    edge_mask=None, edge_mask_original=None, return_hidden=False):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        x = fake_edge(x, edge_index, edge_mask, edge_mask_original,edge_weight,batch, self)
        hidden = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if return_hidden:
            return x, hidden
        else:
            return x
    
    def gnn_forward(self, x,edge_index,batch,edge_weight):
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = global_mean_pool(xs[-1], batch)
        return x



def fake_edge(x, edge_index, edge_mask ,edge_mask_original, edge_weight, batch, self):
    if edge_weight:
        edge_weight_minus = edge_weight[edge_mask]
        edge_weight_original = edge_weight[edge_mask_original]
    else:
        edge_weight_minus = None
        edge_weight_original = None
    if self.fuse=='att':
        x_plus = self.gnn_forward(x,edge_index,batch,edge_weight)
        x_minus = self.gnn_forward(x,edge_index[:,edge_mask],batch,edge_weight_minus)
        x_stack = torch.stack((x_plus,x_minus),dim=1)
        x = self.att(x_stack)
    elif self.fuse=='plus':
        x = self.gnn_forward(x,edge_index,batch,edge_weight)
    elif self.fuse=='mean':
        x_plus = self.gnn_forward(x,edge_index,batch,edge_weight)
        x_minus = self.gnn_forward(x,edge_index[:,edge_mask],batch,edge_weight_minus)
        x_stack = torch.stack((x_plus,x_minus),dim=1)
        x = x_stack.mean(dim=1)
    elif self.fuse=='original':
        x = self.gnn_forward(x,edge_index[:,edge_mask_original],batch,edge_weight_original)
    elif self.fuse=='minus': # default 'minus'
        x = self.gnn_forward(x,edge_index[:,edge_mask],batch,edge_weight_minus)
    else:
        raise NotImplementedError
    return x