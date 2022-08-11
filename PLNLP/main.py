# -*- coding: utf-8 -*-
import argparse
import json
import time
from pathlib import Path

import torch
from ogb.linkproppred import Evaluator

import wandb
from plnlp.logger import Logger
from plnlp.model import BaseModel
from plnlp.utils import data_process, set_random_seed
from plnlp.dataset import get_dataset


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='GATv2')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')
    parser.add_argument('--data_name', type=str, default='ogbl-ddi')
    parser.add_argument('--val_frac', type=float, default=0.05)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--eval_metric', type=str, default='hits')
    parser.add_argument('--res_dir', type=str, default='log')
    parser.add_argument('--pretrain_emb', type=str, default='')
    parser.add_argument('--dynamic', type=str2bool, default=False)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--emb_hidden_channels', type=int, default=128)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--num_neg', type=int, default=3)
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--fusion', type=str, default='att')
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--year', type=int, default=-1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--scheduler_gamma', type=float, default=0.99)
    parser.add_argument('--use_node_feats', type=str2bool, default=True)
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=False)
    parser.add_argument('--drnl', type=str2bool, default=True)
    parser.add_argument('--use_valedges_as_input', type=str2bool, default=False)
    parser.add_argument('--eval_last_best', type=str2bool, default=True)
    parser.add_argument('--write_out', type=str2bool, default=False)
    parser.add_argument('--csv', type=str2bool, default=False)

    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    args = argument()
    print(args)
    wandb.init(project="FakeEdge", entity='kevindong',group=args.data_name)
    wandb.config.update(args)
    set_random_seed(42)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f"{args.data_name}.pth"

    # create log file and save args
    res_dir = Path(args.res_dir)
    res_dir.mkdir(exist_ok=True)
    log_file_name = 'log_' + args.data_name + '_' + str(int(time.time())) + '.txt'
    log_file = res_dir / log_file_name
    csv = Path("results/")
    csv.mkdir(exist_ok=True)
    csv_file_name = csv / f"{args.data_name}_{args.fusion}_{int(time.time())}_PLNLP.csv"

    wandb.run.summary["log_file"] = str(log_file)
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    try:
        evaluator = Evaluator(name=args.data_name)
    except ValueError:
        evaluator = Evaluator(name="ogbl-ddi")

    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args, name='Hits@20'),
            'Hits@50': Logger(args.runs, args, name='Hits@50'),
            'Hits@100': Logger(args.runs, args, name='Hits@100'),
            'aucroc': Logger(args.runs, args, name='aucroc'),
            'aucpr': Logger(args.runs, args, name='aucpr'),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
            'aucroc': Logger(args.runs, args),
            'aucpr': Logger(args.runs, args),
        }

    if args.data_name in ('Ecoli','PB','pubmed'):
        args.max_nodes_per_hop=100

    if args.data_name=='Power':
        args.num_hops=3

    if args.data_name=='Yeast':
        args.max_nodes_per_hop=100
    for run in range(args.runs): # If model_load, no training
        data, split_edge, num_nodes, num_node_feats = data_process(args)
        model = BaseModel(
            lr=args.lr,
            dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels,
            gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,
            num_nodes=num_nodes,
            num_node_feats=num_node_feats,
            gnn_encoder_name=args.encoder,
            predictor_name=args.predictor,
            loss_func=args.loss_func,
            optimizer_name=args.optimizer,
            device=device,
            use_node_feats=args.use_node_feats,
            train_node_emb=args.train_node_emb,
            pretrain_emb=args.pretrain_emb,
            drnl=args.drnl,
            weight_decay=args.weight_decay,
            fusion_type=args.fusion,
        )
        model.param_init()

        if run == 0:
            total_params = sum(p.numel() for param in model.para_list for p in param)
            total_params_print = f'Total number of model parameters is {total_params}'
            print(total_params_print)
            with open(log_file, 'a') as f:
                f.write(total_params_print + '\n')
        
        if args.scheduler_gamma < 1:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma=args.scheduler_gamma)
        else:
            scheduler = None
        train_list = get_dataset("train",data,split_edge, 
                                args.num_hops,
                               neg_sampler_name=args.neg_sampler,num_neg=args.num_neg,max_nodes_per_hop=args.max_nodes_per_hop,
                               dynamic=args.dynamic,data_name_append=args.data_name)
        val_list = get_dataset("valid",data, split_edge, 
                                args.num_hops,  max_nodes_per_hop=args.max_nodes_per_hop,dynamic=args.dynamic,data_name_append=args.data_name)
        test_list = get_dataset("test",data, split_edge, 
                                args.num_hops,  max_nodes_per_hop=args.max_nodes_per_hop,dynamic=args.dynamic,data_name_append=args.data_name)
        start_time = time.time()

        for epoch in range(1, 1 + args.epochs):
            if scheduler:
                cur_lr = scheduler.get_last_lr()[0]
            else:
                cur_lr = args.lr
            loss = model.train(args.batch_size, args.num_neg, train_list)

            if epoch % args.eval_steps == 0:
                results = model.test(batch_size=args.batch_size,
                                     evaluator=evaluator,
                                     eval_metric=args.eval_metric,
                                     val_list=val_list,
                                     test_list=test_list,
                                     )
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                spent_time = time.time() - start_time
                wandb.log({f"Loss :Run {run}":loss})
                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Learning Rate: {cur_lr:.4f}, '
                                f'Valid: {100 * valid_res:.2f}%, '
                                f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print)
                    wandb.log({f"{key}:Run {run} Valid":100 * valid_res,
                                f"{key}:Run {run} Test":100 * test_res})
                    with open(log_file, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
                print('---')
                print(
                    f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                print('---')
                start_time = time.time()

            if scheduler:
                scheduler.step()
            torch.cuda.empty_cache()
        
        if args.write_out:
            model.test(batch_size=args.batch_size,
                    evaluator=evaluator,
                    eval_metric=args.eval_metric,
                    val_list=val_list,
                    test_list=test_list,
                    write_out_file=f"{args.data_name}_Run_{run}_{args.fusion}",
                    train_list=train_list,
                    )
        # Run finish
        del train_list
        del val_list
        del test_list
        csv = {}
        for key in loggers.keys():
            print(key)
            csv[key] = loggers[key].print_statistics(run, last_best=args.eval_last_best)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f, last_best=args.eval_last_best)
            
        if args.csv:
            with open(csv_file_name, 'a') as f:
                f.write(json.dumps(csv) + '\n')

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(last_best=args.eval_last_best)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f, last_best=args.eval_last_best)


if __name__ == "__main__":
    main()
