import torch
from torch_geometric.data import InMemoryDataset
from plnlp.utils import get_pos_neg_edges, process_graph, root_dir


class FakeEdgeDataset(InMemoryDataset):
    def __init__(self, root, data, edges, positive, num_hops, split='train', #ratio_per_hop=1.0,
                 max_nodes_per_hop=None,dynamic=False):
        self.data = data
        self.edges = edges
        assert self.edges.dim()==2
        assert self.edges.shape[1] == 2, "Edges should be of shape (N,2)"
        self.positive = positive
        self.num_hops = num_hops
        self.split = split
        # self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.dynamic = dynamic
        super(FakeEdgeDataset, self).__init__(root)
        if self.dynamic:
            self.data = self._data
            self.slices = self._slices
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        edge_class = 'pos' if self.positive else 'neg'
        dymanic_str = '.tmp' if self.dynamic else ''
        name = f'FakeEdge_{self.split}_data'
        return [f"{name}_{edge_class}.pt{dymanic_str}"]

    def process(self):
        processed = process_graph(split=self.split, 
                                data=self.data,
                                edges=self.edges,
                                positive=self.positive,
                                num_hops=self.num_hops,
                                drnl=True, # always calculate drnl
                                max_nodes_per_hop=self.max_nodes_per_hop,
                                )
        if self.dynamic:
            self._data, self._slices = self.collate(processed)
        else:
            torch.save(self.collate(processed), self.processed_paths[0])
            del processed


def get_dataset(split, data, split_edge, num_hops, neg_sampler_name=None,num_neg=None,max_nodes_per_hop=None,dynamic=False,data_name_append=None):
    pos_edge, neg_edge = get_pos_neg_edges(split, split_edge,
                                        edge_index=data.edge_index,
                                        num_nodes=data.num_nodes,
                                        neg_sampler_name=neg_sampler_name,
                                        num_neg=num_neg)
    if data_name_append:
        root = root_dir/data_name_append
    else:
        root = root_dir
    # If Datasets is pre-processed, then we won't use neg_edge sampled above for training phase
    pos = FakeEdgeDataset(root, data, pos_edge, positive=True, num_hops=num_hops,split=split,max_nodes_per_hop=max_nodes_per_hop,dynamic=dynamic)
    neg = FakeEdgeDataset(root, data, neg_edge.reshape(-1,2), positive=False, num_hops=num_hops,split=split,max_nodes_per_hop=max_nodes_per_hop,dynamic=dynamic)
    return pos,neg