import torch 
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import Predict_Dataset 
from model import Predict_Model
from rdkit import RDLogger
from sklearn.metrics import mean_squared_error, r2_score
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

def data_split(dataset, split_ratio):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def train(model, train_loader, optimizer, criterion, device, pre_train_path=None):
    #fine-tune the model
    if pre_train_path is not None:
        model.load_state_dict(torch.load(pre_train_path))
    model.train()
    train_loss = 0
    for i, (x, y, adjoin_matrix) in enumerate(train_loader):
        x, y, adjoin_matrix = x.to(device), y.to(device), adjoin_matrix.to(device)
        outputs = model(x, adjoin_matrix)
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
    set_random_seed(42)
    device = gpu_check()
    model = Predict_Model(num_layers=args.num_layers, d_model=args.d_model, dff=args.d_model*2, num_heads=args.num_heads, vocab_size=args.vocab_size, dropout_rate=args.hidden_dropout_prob)
    model.to(device)
    pre_train_path = os.path.join(args.pretrain_path, 'model_weights.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    dataset = Predict_Dataset(args.file_path, args.smiles_field, args.label_field, args.add_H)
    max_target = max(dataset.data[args.label_field])
    min_target = min(dataset.data[args.label_field])
    train_dataset, val_dataset  = data_split(dataset, args.split_ratio)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, pre_train_path)    
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, max_target, min_target, device)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')
    

if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--hidden_dropout_prob', type=float, default=0.10)
    args.add_argument('--add_H', type=bool, default=True)
    args.add_argument('--file_path', type=str, default='/data')
    args.add_argument('--smiles_field', type=str, default='smiles')
    args.add_argument('--label_field', type=str, default='kisc')
    args.add_argument('--vocab_size', type=int, default=18)
    args.add_argument('--d_model', type=int, default=256)
    args.add_argument('--num_layers', type=int, default=6)
    args.add_argument('--num_heads', type=int, default=4)
    args.add_argument('--split_ratio', type=float, default=0.8)
    args.add_argument('--pretrain_path', type=str, default='model_weights')

    main(args)