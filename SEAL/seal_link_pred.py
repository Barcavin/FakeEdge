# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import time
import os, sys
import os.path as osp
from pathlib import Path
from shutil import copy
import copy as cp
from sklearn import metrics
from tqdm import tqdm
import pdb

import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected, degree
from torch_geometric.nn import GCNConv, GATConv, SGConv, ChebConv, GINConv, GATConv, SAGEConv, GATv2Conv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
warnings.simplefilter('ignore', UserWarning)
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import *
from models import *


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None
        
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, self.split!='train')
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, True)

        collate = self.collate(pos_list + neg_list)
        torch.save(collate, self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.split = split
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None
        
    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        # except train and pos, others always plus
        plus = False if self.split == 'train' and y == 1 else True
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop, 
                             self.max_nodes_per_hop, node_features=self.data.x, 
                             y=y, directed=self.directed, A_csc=self.A_csc, plus = plus)
        data = construct_pyg_graph(*tmp, self.node_label, plus=plus)

        return data


def train_heuristic():
    if model.tree:
        # Test link prediction heuristics.
        pos_edge, neg_edge = get_pos_neg_edges("train", split_edge, 
                                               data.edge_index, 
                                               data.num_nodes, 
                                               100)
        pos_logits = model(A,pos_edge,1)
        neg_logits = model(A,neg_edge,0)
        logits = torch.cat([pos_logits,neg_logits])
        y = torch.cat([torch.ones(pos_logits.size(0)),torch.zeros(neg_logits.size(0))])
        model.clf.fit(logits.reshape(-1,1), y)
    return 0

def test_heuristic(write_out_file=None):
    pos_edge, neg_edge = get_pos_neg_edges("valid", split_edge, 
                                               data.edge_index, 
                                               data.num_nodes, 
                                               100)
    pos_val_pred = model(A,pos_edge,0)
    neg_val_pred = model(A,neg_edge,0)
    val_pred = torch.cat([pos_val_pred,neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0)),torch.zeros(neg_val_pred.size(0))])

    pos_edge, neg_edge = get_pos_neg_edges("test", split_edge, 
                                               data.edge_index, 
                                               data.num_nodes, 
                                               100)
    pos_test_pred = model(A,pos_edge,0)
    neg_test_pred = model(A,neg_edge,0)
    test_pred = torch.cat([pos_test_pred,neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0)),torch.zeros(neg_test_pred.size(0))])

    y_true = []
    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

def train():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.edge_mask, data.edge_mask_original)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(write_out_file=None):
    model.eval()

    y_pred, y_true, val_h = [], [], []
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits, hidden_val = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.edge_mask, data.edge_mask_original, True)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
        if write_out_file is not None:
            val_h.append(hidden_val.cpu())
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true, test_h = [], [], []
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits, hidden_test = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.edge_mask, data.edge_mask_original, True)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
        if write_out_file is not None:
            test_h.append(hidden_test.cpu())
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]

    y_true = []
    if write_out_file is not None:
        train_h = []
        for data in tqdm(train_loader, ncols=70):
            data = data.to(device)
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            logits, hidden_train = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.edge_mask, data.edge_mask_original, True)
            train_h.append(hidden_train.cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))
        train_true = torch.cat(y_true)
        train_h = torch.cat(train_h)
        val_h = torch.cat(val_h)
        test_h = torch.cat(test_h)
        for name,label in zip(['pos','neg'],[1,0]):
            torch.save(train_h[train_true==label], write_out_file + f'_train_hidden_{name}.pt')
            torch.save(val_h[val_true==label], write_out_file + f'_val_hidden_{name}.pt')
            torch.save(test_h[test_true==label], write_out_file + f'_test_hidden_{name}.pt')
    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results


@torch.no_grad()
def test_multiple_models(models):
    for m in models:
        m.eval()

    y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, m in enumerate(models):
            logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred[i].append(logits.view(-1).cpu())
            y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    val_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    val_true = [torch.cat(y_true[i]) for i in range(len(models))]
    pos_val_pred = [val_pred[i][val_true[i]==1] for i in range(len(models))]
    neg_val_pred = [val_pred[i][val_true[i]==0] for i in range(len(models))]

    y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, m in enumerate(models):
            logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred[i].append(logits.view(-1).cpu())
            y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    test_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    test_true = [torch.cat(y_true[i]) for i in range(len(models))]
    pos_test_pred = [test_pred[i][test_true[i]==1] for i in range(len(models))]
    neg_test_pred = [test_pred[i][test_true[i]==0] for i in range(len(models))]
    
    Results = []
    for i in range(len(models)):
        if args.eval_metric == 'hits':
            Results.append(evaluate_hits(pos_val_pred[i], neg_val_pred[i], 
                                         pos_test_pred[i], neg_test_pred[i]))
        elif args.eval_metric == 'mrr':
            Results.append(evaluate_mrr(pos_val_pred[i], neg_val_pred[i], 
                                        pos_test_pred[i], neg_test_pred[i]))
        elif args.eval_metric == 'auc':
            Results.append(evaluate_auc(val_pred[i], val_true[i], 
                                        test_pred[i], test_pred[i]))
    return Results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
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

    results.update(evaluate_auc_pr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred))
    return results
        

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
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


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

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
        

# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument('--fast_split', action='store_true', 
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--GNN', type=str, default='GCNConv')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--fuse', type=str, default='minus')
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--val-ratio', type=float, default=0.05)
parser.add_argument('--test-ratio', type=float, default=0.1)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--num_workers', type=int, default=0, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true', 
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', action='store_true', 
                    help="only test without training")
parser.add_argument('--test_multiple_models', action='store_true', 
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None, 
                    help="test a link prediction heuristic (CN or AA)")
parser.add_argument('--write_out', action='store_true')
parser.add_argument('--pooling', type=str, default="hadamard")
parser.add_argument('--csv', type=str, default="")
args = parser.parse_args()

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'

args.res_dir = os.path.join('SEAL/results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
# if not args.keep_old:
#     # Backup python files.
#     copy('seal_link_pred.py', args.res_dir)
#     copy('utils.py', args.res_dir)
log_file = os.path.join("SEAL",args.res_dir, 'log.txt')
Path(log_file).parents[0].mkdir(parents=True, exist_ok=True)
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
# with open(os.path.join("SEAL",args.res_dir, 'cmd_input.txt'), 'a') as f:
#     f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

if args.csv:
    csv = Path(args.csv)
    csv.mkdir(exist_ok=True)
    csv_file_name = csv / f"{'SEAL' if args.model=='DGCNN' else args.model}_{args.dataset}_{args.fuse}_hops_{args.num_hops}_layers_{args.num_layers}_pooling_{args.pooling}_jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}.csv"
set_random_seed(args.seed)



if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
else:
    args.eval_metric = 'hits'
    directed = False
evaluator = Evaluator(name="ogbl-ddi")
if args.eval_metric == 'hits':
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
        'aucroc': Logger(args.runs, args),
        'aucpr': Logger(args.runs, args),
    }
elif args.eval_metric == 'mrr':
    loggers = {
        'MRR': Logger(args.runs, args),
    }
elif args.eval_metric == 'auc':
    loggers = {
        'AUC': Logger(args.runs, args),
    }

for run in range(args.runs):
    if args.dataset.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args.dataset,\
                                        root=Path.home() / 'files',\
                                        )
        split_edge = dataset.get_edge_split()
        data = dataset[0]
    elif args.dataset in ["cora",\
                        "citeseer",\
                        "pubmed",]:
        path = Path.home() / 'files'
        dataset = Planetoid(path, args.dataset)
        # split_edge = do_edge_split(dataset, args.fast_split)
        # data = dataset[0]
        # data.edge_index = split_edge['train']['edge'].t()
        data, split_edge = my_edge_split(dataset[0],args.val_ratio,args.test_ratio)
    elif args.dataset in ["BlogCatalog",
                                    "Celegans",
                                    "Ecoli",
                                    "NS",
                                    "PB",
                                    "Power",
                                    "Router",
                                    "USAir",
                                    "Yeast"]:
        data = load_unsplitted_data(args)
        data, split_edge = my_edge_split(data,args.val_ratio,args.test_ratio)
    else:
        raise NotImplementedError

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        if not directed:
            val_edge_index = to_undirected(val_edge_index)
        data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

    # if args.dataset.startswith('ogbl'):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = data.num_nodes
    edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                    shape=(num_nodes, num_nodes))
    if args.use_heuristic:
        # Test link prediction heuristics.
        num_nodes = data.num_nodes
        if 'edge_weight' in data:
            edge_weight = data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

        A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                        shape=(num_nodes, num_nodes))

        pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                    data.edge_index, 
                                                    data.num_nodes)
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                        data.edge_index, 
                                                        data.num_nodes)
        pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
        neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
        pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
        neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

        if args.eval_metric == 'hits':
            results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
        elif args.eval_metric == 'mrr':
            results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
        elif args.eval_metric == 'auc':
            val_pred = torch.cat([pos_val_pred, neg_val_pred])
            val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                                torch.zeros(neg_val_pred.size(0), dtype=int)])
            test_pred = torch.cat([pos_test_pred, neg_test_pred])
            test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                                torch.zeros(neg_test_pred.size(0), dtype=int)])
            results = evaluate_auc(val_pred, val_true, test_pred, test_true)

        for key, result in results.items():
            loggers[key].add_result(0, result)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(f=f)
        csv={}
        for key in loggers.keys():
            print(key)
            csv[key] = loggers[key].print_statistics(0)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(0, f=f)
        if args.csv:
            with open(csv_file_name, 'a') as f:
                for i in range(10):
                    f.write(json.dumps(csv) + '\n')
        exit()


    # SEAL.
    path = 'dataset/' + args.dataset + '_seal{}'.format(args.data_appendix)
    use_coalesce = True if args.dataset == 'ogbl-collab' else False
    if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
        args.num_workers = 0

    dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
    train_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.train_percent, 
        split='train', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    ) 
    if False:  # visualize some graphs
        import networkx as nx
        from torch_geometric.utils import to_networkx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        for g in loader:
            f = plt.figure(figsize=(20, 20))
            limits = plt.axis('off')
            g = g.to(device)
            node_size = 100
            with_labels = True
            G = to_networkx(g, node_attrs=['z'])
            labels = {i: G.nodes[i]['z'] for i in range(len(G))}
            nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                    labels=labels)
            f.savefig('tmp_vis.png')
            pdb.set_trace()

    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
    val_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.val_percent, 
        split='valid', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
    test_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.test_percent, 
        split='test', 
        use_coalesce=use_coalesce, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    )

    max_z = 1000  # set a large max_z so that every z has embeddings to look up


    if args.train_node_embedding:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad=False
    else:
        emb = None

    heuristics = False
    if args.model == 'DGCNN':
        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      train_dataset, args.dynamic_train, GNN=eval(args.GNN),use_feature=args.use_feature, 
                      node_embedding=emb, fuse=args.fuse).to(device)
    elif args.model == 'SAGE':
        model = SAGE(args.hidden_channels, args.num_layers, max_z, train_dataset,  
                     args.use_feature, node_embedding=emb, fuse=args.fuse, pooling=args.pooling).to(device)
    elif args.model == 'GCN':
        model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb, fuse=args.fuse, pooling=args.pooling).to(device)
    elif args.model == 'GIN':
        model = GIN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature, node_embedding=emb, fuse=args.fuse).to(device)
    elif args.model == 'gMPNN':
        model = gMPNN(args.hidden_channels, fuse=args.fuse, max_z=max_z, use_feature=args.use_feature).to(device)
    elif args.model == "PA":
        model = PA(args.fuse)
        heuristics = True
        args.epochs = 1
    elif args.model == "Jac":
        model = Jac(args.fuse)
        heuristics = True
        args.epochs = 1
    elif args.model == "PA+DT":
        model = PA(args.fuse, True)
        heuristics = True
        args.epochs = 1
    elif args.model == "Jac+DT":
        model = Jac(args.fuse, True)
        heuristics = True
        args.epochs = 1
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}', file=f)

    start_epoch = 1
    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_optimizer_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        start_epoch = args.continue_from + 1
        args.epochs -= args.continue_from
    
    if args.only_test:
        results = test()
        for key, result in results.items():
            loggers[key].add_result(run, result)
        for key, result in results.items():
            valid_res, test_res = result
            print(key)
            print(f'Run: {run + 1:02d}, '
                  f'Valid: {100 * valid_res:.2f}%, '
                  f'Test: {100 * test_res:.2f}%')
        pdb.set_trace()
        exit()

    if args.test_multiple_models:
        model_paths = [
        ] # enter all your pretrained .pth model paths here
        models = []
        for path in model_paths:
            m = cp.deepcopy(model)
            m.load_state_dict(torch.load(path))
            models.append(m)
        Results = test_multiple_models(models)
        for i, path in enumerate(model_paths):
            print(path)
            with open(log_file, 'a') as f:
                print(path, file=f)
            results = Results[i]
            for key, result in results.items():
                loggers[key].add_result(run, result)
            for key, result in results.items():
                valid_res, test_res = result
                to_print = (f'Run: {run + 1:02d}, ' +
                            f'Valid: {100 * valid_res:.2f}%, ' +
                            f'Test: {100 * test_res:.2f}%')
                print(key)
                print(to_print)
                with open(log_file, 'a') as f:
                    print(key, file=f)
                    print(to_print, file=f)
        pdb.set_trace()
        exit()

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if heuristics:
            loss = train_heuristic()
        else:
            loss = train()

        if epoch % args.eval_steps == 0:
            if heuristics:
                results = test_heuristic()
            else:
                results = test()
            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                # torch.save(model.state_dict(), model_name)
                # torch.save(optimizer.state_dict(), optimizer_name)

                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print)
                    with open(log_file, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
    csv={}
    for key in loggers.keys():
        print(key)
        csv[key] = loggers[key].print_statistics(run)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(run, f=f)
    if args.write_out:
        test(f"SEAL/{args.dataset}_Run_{run}_{args.fuse}_")
    if args.csv:
        with open(csv_file_name, 'a') as f:
            f.write(json.dumps(csv) + '\n')

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)
print(f'Total number of parameters is {total_params}')
print(f'Results are saved in {args.res_dir}')


