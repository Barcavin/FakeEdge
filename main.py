import argparse
import subprocess
import time
import wandb

data_names = ["cora",
            "citeseer",
            "pubmed",
            "Celegans",
            "Ecoli",
            "NS",
            "PB",
            "Power",
            "Router",
            "USAir",
            "Yeast"]

fuse_names = ["original",
              "plus",
              "minus",
              "mean",
              "att"]        

def main():
    parser = argparse.ArgumentParser(description="Entrypoint for Fake Edge")
    parser.add_argument("--method", type=str, help="Method to run")
    parser.add_argument("--data", type=str, choices=data_names)
    parser.add_argument("--fuse", type=str, choices=fuse_names)
    wandb.init(project="FakeEdge", entity='kevindong')
    args = parser.parse_args()
    time_stamp = int(time.time())
    print("starting")
    if args.method in ["SEAL","GCN","SAGE","GIN"]:
        model = "DGCNN" if args.method == "SEAL" else args.method
        if args.data == "cora":
            process = subprocess.run(['python','SEAL/seal_link_pred.py',
                            '--dataset',args.data,
                            '--num_hops', '3',
                            '--use_feature','--hidden_channels','256', '--runs', '10',
                            '--fuse',args.fuse,
                            '--model',model,
                            '--dynamic_train','--dynamic_val','--dynamic_test','--csv'],
                            capture_output=True,check=True)
        elif args.data == "citeseer":
            process = subprocess.run(['python', 'SEAL/seal_link_pred.py', 
                            '--dataset', args.data, 
                            '--num_hops', '3', 
                            '--hidden_channels', '256', '--runs', '10', 
                            '--fuse',args.fuse, 
                            '--model',model, 
                            '--dynamic_train', '--dynamic_val', '--dynamic_test', '--csv'],
                            capture_output=True,check=True)
        elif args.data == "pubmed":
            process = subprocess.run(['python', 'SEAL/seal_link_pred.py', 
                            '--dataset', args.data, 
                            '--num_hops', '3', 
                            '--use_feature', '--runs', '10', 
                            '--fuse',args.fuse, 
                            '--model',model, 
                            '--dynamic_train', '--dynamic_val', '--dynamic_test', '--csv'],
                            capture_output=True,check=True)
        else:
            process = subprocess.run(['python', 'SEAL/seal_link_pred.py',
                                     '--dataset', args.data, 
                                     '--num_hops', '2', 
                                     '--hidden_channels', '128', '--runs', '10', 
                                     '--fuse',args.fuse, 
                                     '--model',model, 
                                     '--dynamic_train', '--dynamic_val', '--dynamic_test', '--csv'],
                            capture_output=True,check=True)
    elif args.method == "WalkPool":
        if args.data in ["cora","citeseer","pubmed"]:
            for seed in range(1,11):
                process = subprocess.run(['python', 'WalkPooling/main.py', 
                                        '--seed', seed, 
                                        '--data-name', args.data, 
                                        '--drnl', '0', '--init-attribute', 'none', '--init-representation', 'none', 
                                        '--embedding-dim', '32', '--practical-neg-sample', 'true', 
                                        '--fuse',args.fuse, 
                                        '--csv',time_stamp],
                                capture_output=True,check=True)
                print(process.stdout.decode("utf-8"))
        else:
            for seed in range(1,11):
                process = subprocess.run(['python', 'WalkPooling/main.py', 
                                        '--seed', seed,
                                        '--use-splitted' ,'false',
                                        '--data-name', args.data, 
                                        '--drnl', '0', '--init-attribute', 'ones', '--init-representation', 'none', 
                                        '--embedding-dim', '16', '--practical-neg-sample', 'true', 
                                        '--fuse',args.fuse, 
                                        '--csv',time_stamp],
                                capture_output=True,check=True)
                print(process.stdout.decode("utf-8"))
        return 0
    elif args.method == "PLNLP":
        if args.data in ["cora","citeseer","pubmed"]:
            process = subprocess.run(['python', 'PLNLP/main.py', 
                                    '--batch_size','128', 
                                    '--data_name',args.data, 
                                    '--drnl','true', '--dynamic','true', 
                                    '--encoder','GATv2', 
                                    '--fusion',args.fuse, 
                                    '--runs','10', 
                                    '--use_node_feats','true', '--csv','true'],
                                    capture_output=True,check=True)
        else:
            process = subprocess.run(['python', 'PLNLP/main.py', 
                                    '--batch_size','128', 
                                    '--data_name',args.data, 
                                    '--drnl','true', '--dynamic','true', 
                                    '--encoder','GATv2', 
                                    '--fusion',args.fuse, 
                                    '--runs','10', 
                                    '--use_node_feats','false', 
                                    '--train_node_emb','false', 
                                    '--csv','true'],
                                    capture_output=True,check=True)
    elif args.method == "Test":
        process = subprocess.run(['echo', 'Test_Message'],
                            capture_output=True,check=True)
    print(process.stdout.decode("utf-8"))
    return 0

if __name__ == "__main__":
    main()
    wandb.finish()

