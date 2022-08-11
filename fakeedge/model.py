# -*- coding: utf-8 -*-
from torch_geometric.loader import DataLoader
from fakeedge.layer import *
from fakeedge.loss import *
from fakeedge.utils import *

MAX_Z = 1000

class BaseModel(object):
    """
        Parameters
        ----------
        lr : double
            Learning rate
        dropout : double
            dropout probability for gnn and mlp layers
        gnn_num_layers : int
            number of gnn layers
        mlp_num_layers : int
            number of gnn layers
        *_hidden_channels : int
            dimension of hidden
        num_nodes : int
            number of graph nodes
        num_node_feats : int
            dimension of raw node features
        gnn_encoder_name : str
            gnn encoder name
        predictor_name: str
            link predictor name
        loss_func: str
            loss function name
        optimizer_name: str
            optimization method name
        device: str
            device name: gpu or cpu
        use_node_feats: bool
            whether to use raw node features as input
        train_node_emb: bool
            whether to train node embeddings based on node id
        pretrain_emb: str
            whether to load pretrained node embeddings
    """

    def __init__(self, lr, dropout, grad_clip_norm, gnn_num_layers, mlp_num_layers, emb_hidden_channels,
                 gnn_hidden_channels, mlp_hidden_channels, num_nodes, num_node_feats, gnn_encoder_name,
                 predictor_name, loss_func, optimizer_name, device, use_node_feats, train_node_emb,
                 pretrain_emb=None, drnl=False, weight_decay=0, fusion_type='att'):
        # track args
        self.meta = locals().copy()
        self.meta.pop("self")
        self.meta.pop("device")

        self.init_lr = lr
        self.loss_func_name = loss_func
        self.optimizer_name = optimizer_name
        self.num_nodes = num_nodes
        self.num_node_feats = num_node_feats
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.clip_norm = grad_clip_norm
        self.device = device
        self.weight_decay = weight_decay
        self.drnl = drnl
        self.fusion_type = fusion_type
        assert fusion_type in ['att','plus','minus','mean', 'original', 'concat']
        # Input Layer
        self.input_channels, self.emb, self.drnl_emb = create_input_layer(num_nodes=num_nodes,
                                                           num_node_feats=num_node_feats,
                                                           hidden_channels=emb_hidden_channels,
                                                           use_node_feats=use_node_feats,
                                                           drnl=drnl,
                                                           train_node_emb=train_node_emb,
                                                           pretrain_emb=pretrain_emb,
                                                           )
        if self.emb is not None:
            self.emb = self.emb.to(device)

        if self.drnl_emb is not None:
            self.drnl_emb = self.drnl_emb.to(device)
        
        # GNN Layer
        self.encoder = create_gnn_layer(input_channels=self.input_channels,
                                        hidden_channels=gnn_hidden_channels,
                                        num_layers=gnn_num_layers,
                                        dropout=dropout,
                                        encoder_name=gnn_encoder_name).to(device)

        # Predict Layer
        self.predictor_name = predictor_name.upper()
        self.predictor = create_predictor_layer(hidden_channels=mlp_hidden_channels,
                                                num_layers=mlp_num_layers,
                                                dropout=dropout,
                                                predictor_name=predictor_name).to(device)

        # semantic attention
        if self.fusion_type == 'concat':
            self.semantic_att = ConcatFuse(in_channels=gnn_hidden_channels).to(device)
        else:# if self.fusion_type == 'att':
            self.semantic_att = SemanticAttention(in_size=gnn_hidden_channels).to(device)

        # Parameters and Optimizer
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters()) + list(self.semantic_att.parameters())

        if self.emb:
            self.para_list += list(self.emb.parameters())
        if self.drnl_emb:
            self.para_list += list(self.drnl_emb.parameters())
        
        self.setup_optimizer()



    def param_init(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        self.semantic_att.reset_parameters()
        self.setup_optimizer()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)
        if self.drnl_emb is not None:
            torch.nn.init.xavier_uniform_(self.drnl_emb.weight)

    def create_input_feat(self, data):
        # nodes = torch.arange(0,self.num_nodes).to(self.device)
        nodes = data.n_id.to(self.device)
        input_feat = []
        if self.use_node_feats:
            input_feat.append(data.x.to(self.device))
        
        if self.train_node_emb:
            input_feat.append(self.emb(nodes))
        
        if self.drnl:
            input_feat.append(self.drnl_emb(data.z))

        input_feat = torch.cat(input_feat,axis=1)
        return input_feat
    
    def setup_optimizer(self):
        if self.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.para_list, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True) # 1e-5
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=self.init_lr, weight_decay=self.weight_decay)
        return self.optimizer

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_func_name == 'CE':
            loss = ce_loss(pos_out, neg_out)
        elif self.loss_func_name == 'MSE':
            loss = mse_loss(pos_out, neg_out)
        elif self.loss_func_name == 'InfoNCE':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'LogRank':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'HingeAUC':
            loss = hinge_auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'AdaAUC' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedAUC' and margin is not None:
            loss = weighted_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'AdaHingeAUC' and margin is not None:
            loss = adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedHingeAUC' and margin is not None:
            loss = weighted_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def train(self, batch_size, num_neg, train_list):
        self.encoder.train()
        self.predictor.train()
        self.semantic_att.train()

        total_loss = total_examples = 0 
        pos_train_edge, neg_train_edge = train_list
        
        for pos_g,neg_g in zip(DataLoader(pos_train_edge, batch_size, shuffle=True),
                        DataLoader(neg_train_edge, batch_size*num_neg, shuffle=True)):
            self.optimizer.zero_grad()
            pos_g,neg_g = pos_g.to(self.device),neg_g.to(self.device)

            pos_out = self.forward(pos_g)
            neg_out = self.forward(neg_g)


            loss = self.calculate_loss(pos_out, neg_out, num_neg)#, margin=weight_margin)
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def test(self, batch_size, evaluator, eval_metric, val_list, test_list, write_out_file=None, train_list=None):
        self.encoder.eval()
        self.predictor.eval()
        self.semantic_att.eval()
        
        if write_out_file:
            pos_train_edge, neg_train_edge = train_list
            self.batch_forward(pos_train_edge, batch_size,write_out_file+'_train')
            write_out_file_val = write_out_file+'_valid'
            write_out_file_test = write_out_file+'_test'
        else:
            write_out_file_val=None
            write_out_file_test=None

        
        pos_valid_edge,neg_valid_edge = val_list
        pos_test_edge,neg_test_edge = test_list

        pos_valid_pred = self.batch_forward(pos_valid_edge, batch_size,write_out_file_val)
        neg_valid_pred = self.batch_forward(neg_valid_edge, batch_size)
        pos_test_pred = self.batch_forward(pos_test_edge, batch_size,write_out_file_test)
        neg_test_pred = self.batch_forward(neg_test_edge, batch_size)


        if eval_metric == 'hits':
            results = evaluate_hits(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)
        else:
            results = evaluate_mrr(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)

        results.update(evaluate_auc_pr(
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred
        ))
        return results

    @classmethod
    def load(cls, model_path,device,num_nodes=None,num_node_feats=None):
        state = torch.load(model_path)
        args_dict = vars(get_default_args())
        args_dict.update(state["args"])

        if num_nodes:
            args_dict["num_nodes"] = num_nodes
        if num_node_feats:
            args_dict["num_node_feats"] = num_node_feats
        args_dict["device"] = device

        model = cls(**args_dict)
        model.emb.load_state_dict(state["emb"])
        model.encoder.load_state_dict(state["model"])
        model.predictor.load_state_dict(state["predictor"])
        model.semantic_att.load_state_dict(state["semantic_att"])

        return model

    def save(self, model_path):
        state = {
            "args":self.meta,
            "emb":self.emb.state_dict(),
            "model":self.encoder.state_dict(),
            "predictor":self.predictor.state_dict(),
            "semantic_att":self.semantic_att.state_dict(),
        }
        torch.save(state, model_path)

    def forward(self, graph: Data, return_hidden=False):
        x = self.create_input_feat(graph)
        if self.fusion_type=='att': #['att','plus','minus','mean']
            plus = self.encoder(x, graph.edge_index) # all nodes x D
            plus = plus[graph.mapping] # 2(src and dst) x batch_size x hidden_size
            plus = plus[0] * plus[1] # batch_size x hidden_size

            minus = self.encoder(x, graph.edge_index[:,graph.edge_mask])
            minus = minus[graph.mapping] 
            minus = minus[0] * minus[1] # batch_size x hidden_size
            encoded = torch.stack([plus, minus],dim=1) # batch_size x plus_minus x hidden_size
            out = self.semantic_att(encoded) # batch_size x feat_dim
        elif self.fusion_type=='plus':
            out = self.encoder(x, graph.edge_index)[graph.mapping]
            out = out[0] * out[1]
        elif self.fusion_type=='minus':
            out = self.encoder(x, graph.edge_index[:,graph.edge_mask])[graph.mapping]
            out = out[0] * out[1]
        elif self.fusion_type=='mean':
            plus = self.encoder(x, graph.edge_index) # all nodes x D
            plus = plus[graph.mapping] # 2(src and dst) x batch_size x hidden_size
            plus = plus[0] * plus[1] # batch_size x hidden_size

            minus = self.encoder(x, graph.edge_index[:,graph.edge_mask])
            minus = minus[graph.mapping] 
            minus = minus[0] * minus[1] # batch_size x hidden_size
            out = torch.stack([plus, minus],dim=1).mean(axis=1)
        # elif self.fusion_type=='concat':
        #     plus = self.encoder(x, graph.edge_index)[graph.mapping] 
        #     minus = self.encoder(x, graph.edge_index[:,graph.edge_mask])[graph.mapping] 
        #     out = self.semantic_att(plus,minus) 
        elif self.fusion_type=="original":
            out = self.encoder(x, graph.edge_index[:,graph.edge_mask_original])[graph.mapping]
            out = out[0] * out[1]
        else:
            raise ValueError("fusion_type must be one of ['att','plus','minus','mean','original']")
        pred = self.predictor(out).squeeze()
        if return_hidden:
            return pred, out
        else:
            return pred

    def batch_forward(self,data_list,batch_size,write_out_file=None):
        pred = []
        embedding = []
        for g in DataLoader(data_list, batch_size):
            g = g.to(self.device)
            if write_out_file:
                out, embed = self.forward(g,True)
                embed = embed.cpu()
                embedding.append(embed)
            else:
                out = self.forward(g)
            out=out.cpu()
            pred.append(out)
        concat = torch.cat(pred, dim=0)
        if write_out_file:
            embedding = torch.cat(embedding)
            torch.save(embedding, write_out_file+'_hidden_pos.pt')
        return concat
    


def create_input_layer(num_nodes, num_node_feats, hidden_channels, use_node_feats=True, drnl=False,
                       train_node_emb=False, pretrain_emb=None):
    """
    return input_dim,emb. `emb` is of shape (MAX_Z, hidden_channels). The embedding for node labeling
    """
    emb = None
    drnl_emb = None
    input_dim = 0
    if use_node_feats:
        input_dim += num_node_feats
    
    if train_node_emb:
        emb = torch.nn.Embedding(num_nodes, hidden_channels)
        input_dim += hidden_channels
    elif pretrain_emb is not None and pretrain_emb != '':
        
        weight = torch.load(pretrain_emb)
        emb = torch.nn.Embedding.from_pretrained(weight)
        input_dim += emb.weight.size(1)

    if drnl:
        input_dim += hidden_channels # use the same number of hidden channels
        drnl_emb = torch.nn.Embedding(MAX_Z, hidden_channels)

    return input_dim, emb, drnl_emb


def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout=0, encoder_name='SAGE'):
    if encoder_name.upper() == 'GCN':
        return GCN(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'GAT':
        return GAT(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'GATv2':
        return GATv2(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    else:
        return SAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)


def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    # if predictor_name == 'DOT':
    #     return DotPredictor()
    # elif predictor_name == 'BIL':
    #     return BilinearPredictor(hidden_channels)
    if predictor_name == 'MLP':
        return MLPPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
    # elif predictor_name == 'MLPDOT':
    #     return MLPDotPredictor(hidden_channels, 1, num_layers, dropout)
    # elif predictor_name == 'MLPBIL':
    #     return MLPBilPredictor(hidden_channels, 1, num_layers, dropout)
    # elif predictor_name == 'MLPCAT':
    #     return MLPCatPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index