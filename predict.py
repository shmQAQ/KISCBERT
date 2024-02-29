import torch 
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import Predict_Dataset 
from model import Predict_Model
from rdkit import RDLogger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import argparse


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gpu_check() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def train(model, train_loader, optimizer, criterion, device, pre_train_path=None):
    #fine-tune the model
    if pre_train_path is not None:
        pretrain_model_dict = torch.load(pre_train_path)
        model_dict = model.state_dict()
        pretrain_model_dict = {k: v for k, v in pretrain_model_dict.items() if k in model_dict}
        model_dict.update(pretrain_model_dict)
        model.load_state_dict(model_dict)

    model.train()
    train_loss = 0
    for i, (x, y, adjoin_matrix) in enumerate(train_loader):
        x, y, adjoin_matrix = x.to(device), y.to(device), adjoin_matrix.to(device)
        seq = (x == 0).float()
        mask = seq.unsqueeze(1).unsqueeze(1)
        outputs = model(x=x, mask=mask, adjoin_matrix=adjoin_matrix)
        outputs = outputs.squeeze()
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate(model, val_loader, criterion,max_target, min_target, device):
    #evaluate
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (x, y, adjoin_matrix) in enumerate(val_loader):
            x, y, adjoin_matrix = x.to(device), y.to(device), adjoin_matrix.to(device)
            outputs = model(x, adjoin_matrix)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    y_true = y_true * (max_target - min_target) + min_target
    y_pred = y_pred * (max_target - min_target) + min_target
    return val_loss / len(val_loader), y_true, y_pred

def main(args):
    set_random_seed(args.random_seed)
    device = gpu_check()
    model = Predict_Model(num_layers=args.num_layers, d_model=args.d_model, d_ff=args.d_model*2, num_heads=args.num_heads, vocab_size=args.vocab_size, dropout_rate=args.hidden_dropout_prob)
    model.to(device)
    pre_train_path = os.path.join(args.pretrain_path, 'model_weights.pth') if args.pretrain_path is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    dataset = Predict_Dataset(args.file_path, args.smiles_field, args.label_field, args.add_H)
    max_target = max(dataset.df[args.label_field])
    min_target = min(dataset.df[args.label_field])
    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.random_seed)
    for fold, (train_index, val_index) in enumerate(kfold.split(dataset)):
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True)
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device, pre_train_path)
            val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, max_target, min_target, device)
            r2 = r2_score(y_true, y_pred)
            print(f'Fold: {fold+1}, Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {r2:.4f}')
    

if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')

    arg = argparse.ArgumentParser()
    arg.add_argument('--batch_size', type=int, default=32)
    arg.add_argument('--epochs', type=int, default=100)
    arg.add_argument('--hidden_dropout_prob', type=float, default=0.10)
    arg.add_argument('--add_H', type=bool, default=True)
    arg.add_argument('--file_path', type=str, default='bert/data/weights.csv')
    arg.add_argument('--smiles_field', type=str, default='smiles')
    arg.add_argument('--label_field', type=str, default='weights')
    arg.add_argument('--vocab_size', type=int, default=18)
    arg.add_argument('--d_model', type=int, default=256)
    arg.add_argument('--num_layers', type=int, default=6)
    arg.add_argument('--num_heads', type=int, default=4)
    arg.add_argument('--random_seed', type=int, default=42)
    arg.add_argument('--kfold', type=int, default=5)
    arg.add_argument('--split_ratio', type=float, default=0.8)
    arg.add_argument('--pretrain_path', type=str, default=None)
    args = arg.parse_args()
    main(args)