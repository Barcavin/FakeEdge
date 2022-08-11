import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

def load_data(dataset, fuse, run=0):
    X_train = torch.load(f"{dataset}_Run_{run}_{fuse}__train_hidden_pos.pt",map_location='cpu')
    X_test = torch.cat([torch.load(f"{dataset}_Run_{run}_{fuse}__val_hidden_pos.pt",map_location='cpu'),
                        torch.load(f"{dataset}_Run_{run}_{fuse}__test_hidden_pos.pt",map_location='cpu')])

    X = torch.concat([X_train,X_test])
    y = torch.concat([torch.zeros(X_train.shape[0]),torch.ones(X_test.shape[0])])
    return X,y

def train(model, X, y, criterion, optimizer, batch_size):
    model.train()
    for perm in DataLoader(range(len(y)), batch_size, shuffle=True):
        X1, y1 = X[perm], y[perm]
        optimizer.zero_grad()
        output = model(X1)
        loss = criterion(output.view(-1), y1)
        loss.backward()
        optimizer.step()

def test(model, X, y, batch_size, out_file=None):
    model.eval()
    scores = torch.tensor([])
    labels = torch.tensor([])
    for perm in DataLoader(range(len(y)), batch_size):
        X1, y1 = X[perm], y[perm]
        output = model(X1)
        scores = torch.cat((scores,output.view(-1).cpu().clone().detach()),dim = 0)
        labels = torch.cat((labels,y1.view(-1).cpu().clone().detach()),dim = 0)
    if out_file:
        out = torch.stack([scores,labels],dim=1).numpy()
        np.savetxt(out_file,out)
    return roc_auc_score(labels, scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Router")
    parser.add_argument('--fuse', type=str, default='mean')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_test_split', type=float, default=0.8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X,y = load_data(args.dataset, args.fuse, args.run)
    X = X.to(device)
    y = y.to(device)
    perm = torch.randperm(X.shape[0])
    train_idx = perm[:int(X.shape[0]*args.train_test_split)]
    test_idx = perm[int(X.shape[0]*args.train_test_split):]

    X_train = X[train_idx]
    y_train = y[train_idx]
    print(f"'Test' samples during training: {y_train.sum()/y_train.shape[0]}")
    X_test = X[test_idx]
    y_test = y[test_idx]
    model = MLP(X.shape[1], args.hidden_size).to(device)
    print(f"'Test' samples during testing: {y_test.sum()/y_test.shape[0]}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        train(model, X_train, y_train, criterion, optimizer, args.batch_size)
        print(f"Epoch {epoch+1}/{args.epochs}: ROC AUC = {test(model, X_test, y_test, args.batch_size)}")
    test(model, X_test, y_test, args.batch_size,f"{args.dataset}_Run_{args.run}_{args.fuse}_discriminate.txt")
if __name__ == '__main__':
    main()