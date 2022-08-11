# -*- coding: utf-8 -*-
from argparse import Namespace
from pathlib import Path
import random
from typing import List, Any

import numpy as np
import scipy.sparse as ssp
import torch
import torch_geometric
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset
from scipy.sparse.csgraph import shortest_path
from sklearn import metrics
from torch_geometric.datasets import Planetoid, FakeDataset
from torch_geometric.utils import (from_scipy_sparse_matrix, is_undirected,
                                   to_undirected)
from torch_sparse import SparseTensor, coalesce
from tqdm import tqdm

from fakeedge.negative_sample import (global_neg_sample, global_perm_neg_sample,
                                   local_neg_sample)
root_dir= Path.home()/'files'

class Data(torch_geometric.data.Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face' or key == 'mapping':
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key or key == 'mapping':
            return 1
        else:
            return 0

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
        else:
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge


def evaluate_hits(evaluator, pos_val_pred, neg_val_pred,
                  pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred,
                 pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results

def evaluate_auc_pr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    result = {}
    val_label = torch.concat([torch.tensor([1]).repeat(pos_val_pred.shape[0]),
                              torch.tensor([0]).repeat(neg_val_pred.shape[0])])
    val_pred = torch.concat([pos_val_pred, neg_val_pred])

    test_label = torch.concat([torch.tensor([1]).repeat(pos_test_pred.shape[0]),
                              torch.tensor([0]).repeat(neg_test_pred.shape[0])])
    test_pred = torch.concat([pos_test_pred, neg_test_pred])

    fpr, tpr, thresholds = metrics.roc_curve(val_label, val_pred, pos_label=1) # (y,pred)
    val_aucroc = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(val_label, val_pred, pos_label=1) # y, pred
    # Use AUC function to calculate the area under the curve of precision recall curve
    val_aucpr = metrics.auc(recall, precision)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, test_pred, pos_label=1) # (y,pred)
    test_aucroc = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(test_label, test_pred, pos_label=1) # y, pred
    # Use AUC function to calculate the area under the curve of precision recall curve
    test_aucpr = metrics.auc(recall, precision)
    
    result["aucroc"] = (val_aucroc, test_aucroc)
    result["aucpr"] = (val_aucpr, test_aucpr)
    return result

def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def adj_normalization(adj_t):
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t

def get_default_args():
    default_args = {
        "lr": 0.001,
        "dropout":0.0,
        "grad_clip_norm":2.0,
        "gnn_num_layers":2,
        "mlp_num_layers":2,
        "emb_hidden_channels":256,
        "gnn_hidden_channels":256,
        "mlp_hidden_channels":256,
        "gnn_encoder_name":"SAGE",
        "predictor_name":"MLP",
        "loss_func":"AUC",
        "optimizer_name":"Adam",
        "use_node_feats":False,
        "train_node_emb":True,
        "pretrain_emb":''
    }
    args = Namespace(
        **default_args
    )
    return args

def data_process(args):
    if args.data_name.startswith("ogbl"):
        dataset = PygLinkPropPredDataset(name=args.data_name, root=root_dir)
        data = dataset[0]
        split_edge = dataset.get_edge_split()
    else: # will do a data split and return split_edge
        if args.data_name in ["cora",\
                            "citeseer",\
                            "pubmed",]:\
                            # "facebook"]:

            dataset = Planetoid(root=root_dir, name=args.data_name)
            data = dataset[0]
        
        elif args.data_name in ["BlogCatalog",
                                "Celegans",
                                "Ecoli",
                                "NS",
                                "PB",
                                "Power",
                                "Router",
                                "USAir",
                                "Yeast"]:
            data = load_unsplitted_data(args)
        elif args.data_name == 'random': # Too good to be true. Generate a random graph here
            data = FakeDataset(avg_num_nodes=3000)[0]
        else:
            raise NotImplementedError(f"Can't read data {args.data_name}")
        num_nodes = torch.max(data.edge_index)+1
        assert is_undirected(data.edge_index)
        if args.val_frac==0:
            # force to generate some valid edge
            val_frac=0.05
        else:
            val_frac=args.val_frac
        split = T.RandomLinkSplit(num_val=val_frac,
                                    num_test=args.test_frac,
                                    is_undirected=True,
                                    split_labels=True,
                                    add_negative_train_samples=False)
        train,val,test = split(data)
        # train.edge_index only train true edge
        # val.edge_index only train true edge
        # test.edge_index train+val true edge

        if args.val_frac==0:
            train.edge_index = test.edge_index.clone()
            val.edge_index = test.edge_index.clone()
            train.pos_edge_label_index = torch.cat([train.pos_edge_label_index, val.pos_edge_label_index.clone()],axis=1)
            train.pos_edge_label = torch.cat([train.pos_edge_label, val.pos_edge_label.clone()])

        # split_edge has shape num_edges x 2
        split_edge = {"train":{"edge":train.pos_edge_label_index.t()},
                    "valid":{"edge":val.pos_edge_label_index.t(),
                            "edge_neg":val.neg_edge_label_index.t()},
                    "test":{"edge":test.pos_edge_label_index.t(),
                            "edge_neg":test.neg_edge_label_index.t()}}
        data = train

    
    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    
    # data.edge_index used here
    # edge_index is bidirectional if it's undirected graph
    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)
    
    if hasattr(data, 'num_features'):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    if args.data_name == 'ogbl-citation2':
        data.adj_t = data.adj_t.to_symmetric()

    if args.data_name == 'ogbl-collab':
        # only train edges after specific year
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape(
                (split_edge['train']['year'] >= args.year).nonzero(as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            # create adjacency matrix
            new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

        # Use training + validation edges
        if args.use_valedges_as_input:
            full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
            full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
            # create adjacency matrix
            new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

            if args.use_coalesce:
                full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, num_nodes, num_nodes)

            # edge weight normalization
            split_edge['train']['edge'] = full_edge_index.t()
            deg = data.adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            split_edge['train']['weight'] = deg_inv_sqrt[full_edge_index[0]] * full_edge_weight * deg_inv_sqrt[
                full_edge_index[1]]

    if args.train_percent<100:
        split_edge["train"] = sample_edges("train",split_edge, args.train_percent)
    
    if args.val_percent<100:
        split_edge["valid"] = sample_edges("valid",split_edge, args.val_percent)
    
    if args.test_percent<100:
        split_edge["test"] = sample_edges("test",split_edge, args.test_percent)

    if args.encoder.upper() == 'GCN':
        # Pre-compute GCN normalization.
        data.adj_t = gcn_normalization(data.adj_t)

    return data, split_edge, num_nodes, num_node_feats

def sample_edges(split, split_edge, ratio):
    """
        inplace modify `split_edge`
    """
    ratio = ratio/100
    edges = split_edge[split]
    for edge_type in edges:
        num = edges[edge_type].shape[0]
        perm = torch.randperm(num)
        edges[edge_type] = edges[edge_type][perm[:int(num * ratio)]]
        print(f"Sample {num} --> {int(num * ratio)} {edge_type} edges for {split}")
    return edges

# =========== Code adpated from WalkPooling =============

def load_unsplitted_data(args):
    # read .mat format files
    data_dir = 'data/{}.mat'.format(args.data_name)
    print('Load data from: '+ data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index,num_nodes = torch.max(edge_index).item()+1)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    return data


def k_hop_subgraph(node_idx, num_hops, edge_index, max_nodes_per_hop = None,num_nodes = None):
    """
    return:
        subset: nodes in k-hop subgraph. use idx in the original graph
        edge_index: edges only in k-hop subgraph. use new idx in the subgraph
            The number starts from 0 to len(subset), which map subset[i]: in original graph --> i: in subgraph
        mapping: give the location index of the node pair within the `subsets`
            node_idx == subset[mapping].
            It is also the node pair's new idx in the subgraph.
        edge_mask: the mask for edges in subgraph `edge_index`. False means it is either
            plus edge (negative example) or minus edge (positive example).
    """
    if num_nodes == None:
        num_nodes = torch.max(edge_index)+1
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    if max_nodes_per_hop == None:
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out = edge_mask )
            subsets.append(col[edge_mask])
    else:
        not_visited = row.new_empty(num_nodes, dtype=torch.bool)
        not_visited.fill_(True)
        for _ in range(num_hops):
            node_mask.fill_(False)# the source node mask in this hop
            node_mask[subsets[-1]] = True #mark the sources
            not_visited[subsets[-1]] = False # mark visited nodes
            torch.index_select(node_mask, 0, row, out = edge_mask) # indices of all neighbors
            neighbors = col[edge_mask].unique() #remove repeats
            neighbor_mask = row.new_empty(num_nodes, dtype=torch.bool) # mask of all neighbor nodes
            edge_mask_hop = row.new_empty(row.size(0), dtype=torch.bool) # selected neighbor mask in this hop
            neighbor_mask.fill_(False)
            neighbor_mask[neighbors] = True
            neighbor_mask = torch.logical_and(neighbor_mask, not_visited) # all neighbors that are not visited
            ind = torch.where(neighbor_mask==True) #indicies of all the unvisited neighbors
            if ind[0].size(0) > max_nodes_per_hop:
                perm = torch.randperm(ind[0].size(0))
                ind = ind[0][perm]
                neighbor_mask[ind[max_nodes_per_hop:]] = False # randomly select max_nodes_per_hop nodes
                torch.index_select(neighbor_mask, 0, col, out = edge_mask_hop)# find the indicies of selected nodes
                edge_mask = torch.logical_and(edge_mask,edge_mask_hop) # change edge_mask
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    inv = inv.view(-1, 1)
    return subset, edge_index, inv, edge_mask

def plus_edge(data_observed, p_edge, num_hops, drnl=False, max_nodes_per_hop=None):
    """
        p_edge : shape : (2,)
    """
    nodes, edge_index_m, mapping, _ = k_hop_subgraph(node_idx= p_edge, num_hops=num_hops,\
 edge_index = data_observed.edge_index, max_nodes_per_hop=max_nodes_per_hop ,num_nodes=data_observed.num_nodes)
    if 'x' in data_observed:
        x_sub = data_observed.x[nodes,:]
    else:
        x_sub = None
    edge_index_p = edge_index_m
    edge_index_p = torch.cat((edge_index_p, mapping),dim=1)
    edge_index_p = torch.cat((edge_index_p, mapping[[1,0]]),dim=1)

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1),dtype=torch.bool)
    edge_mask[-1] = False
    edge_mask[-2] = False
    edge_mask_original = edge_mask.clone()

    if drnl:
        num_nodes = nodes.shape[0]
        z = drnl_node_labeling(edge_index_m, mapping[0,0],mapping[1,0],num_nodes)
    else:
        z = 0
    data = Data(edge_index = edge_index_p, x = x_sub, 
                edge_mask = edge_mask,
                edge_mask_original = edge_mask_original,
                n_id = nodes,
                mapping = mapping,
                z = z,
                num_nodes = nodes.shape[0])

    #label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    # data.label = float(label)

    return data

def minus_edge(data_observed, p_edge, num_hops, drnl=False, max_nodes_per_hop=None):
    """
        p_edge : shape : (2,)
    """
    nodes, edge_index_p, mapping,_ = k_hop_subgraph(node_idx= p_edge, num_hops=num_hops,\
 edge_index = data_observed.edge_index,max_nodes_per_hop=max_nodes_per_hop, num_nodes = data_observed.num_nodes)
    if 'x' in data_observed:
        x_sub = data_observed.x[nodes,:]
    else:
        x_sub = None

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1), dtype = torch.bool)
    edge_mask_original = torch.ones(edge_index_p.size(1), dtype = torch.bool)
    ind = torch.where((edge_index_p == mapping).all(dim=0))
    assert ind[0].numel() > 0, "Not Found the adding edge in the graph"
    edge_mask[ind[0]] = False
    
    ind = torch.where((edge_index_p == mapping[[1,0]]).all(dim=0))
    assert ind[0].numel() > 0, "Not Found the adding edge in the graph"
    edge_mask[ind[0]] = False
    if drnl:
        num_nodes = nodes.shape[0]
        z = drnl_node_labeling(edge_index_p[:,edge_mask], mapping[0,0],mapping[1,0],num_nodes)
    else:
        z = 0
    data = Data(edge_index = edge_index_p, x = x_sub, 
                edge_mask = edge_mask,
                edge_mask_original = edge_mask_original,
                n_id = nodes,
                mapping = mapping,
                z = z,
                num_nodes = nodes.shape[0])

    #label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    # data.label = float(label)
    return data


"Code adopted and implemented from https://github.com/muhanzhang/SEAL"

def drnl_node_labeling(edge_index, src, dst, num_nodes):

    edge_weight = torch.ones(edge_index.size(1), dtype=int)
    adj = ssp.csr_matrix(
            (edge_weight, (edge_index[0], edge_index[1])), 
            shape=(num_nodes, num_nodes))
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='trunc'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.
    return z.to(torch.int)


def fake_edge_process(num_hops, data, node_pairs, plus: bool, drnl:bool, max_nodes_per_hop:int = None):
    """
        edge_index : (2,num_edges)
        node_pairs : (num_pairs,2)
    """
    graphs = []
    if plus:
        f = plus_edge
    else:
        f = minus_edge
    tqdm_iterate = tqdm(node_pairs)
    for pair in tqdm_iterate:
        one_graph = f(data, pair, num_hops, drnl, max_nodes_per_hop) 
        graphs.append(one_graph)
    return graphs

def process_graph(split, data, edges, positive:bool, num_hops, drnl:bool, max_nodes_per_hop=None):
    if (split == 'train') and (positive):
        plus = False # only remove edges when it's training set and positive edges
    else:
        plus = True # for negative edges, or val/test set, always plus edges
    processed = fake_edge_process(num_hops, data, edges, plus, drnl, max_nodes_per_hop)
    return processed