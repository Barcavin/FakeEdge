import torch
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce

from fakeedge.utils import get_pos_neg_edges

class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, split='train', 
                 node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.split = split
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        super(SEALDataset, self).__init__(root)
        data = torch.load(self.processed_paths[0])
        self.pos = data["pos"]
        self.neg = data["neg"]

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

        # if self.use_coalesce:  # compress mutli-edge into edge with weight
        #     self.data.edge_index, self.data.edge_weight = coalesce(
        #         self.data.edge_index, self.data.edge_weight, 
        #         self.data.num_nodes, self.data.num_nodes)
        
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        collate = self.collate(pos_list + neg_list)
        torch.save(collate, self.processed_paths[0])
        del pos_list, neg_list