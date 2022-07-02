import torch
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce

from fakeedge.utils import get_pos_neg_edges, edge_injection, process_graph



class FakeEdgeDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, split='train', 
                 node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, num_neg=1,neg_sampler_name='global'):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.split = split
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.num_neg = num_neg
        self.neg_sampler_name = neg_sampler_name
        super(FakeEdgeDataset, self).__init__(root)
        # data = torch.load(self.processed_paths[0])
        self.pos = data["pos"]
        self.neg = data["neg"]

    @property
    def processed_file_names(self):
        if self.split=="train":
            name = 'SEAL_{}_data_neg_{}'.format(self.split, self.num_neg)
        else:
            name = 'SEAL_{}_data'.format(self.split)
        name += '.pt'
        return [name]

    def process(self):
        pos_graphs_minus,neg_graphs_plus = process_graph(self.split, self.data,self.split_edge,
                    self.num_hops,self.neg_sampler_name,
                    self.num_neg)
        dump = {
            "pos":pos_graphs_minus,
            "neg":neg_graphs_plus
        }
        torch.save(dump, self.processed_paths[0])
        del pos_graphs_minus, neg_graphs_plus