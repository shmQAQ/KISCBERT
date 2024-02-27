import torch 
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import Predict_Dataset, scaffold_split
from model import PredictionModel
from rdkit import RDLogger
from sklearn.metrics import mean_squared_error, r2_score
import argparse

RDLogger.DisableLog('rdApp.*')

arg = argparse.ArgumentParser()
arg.add_argument('--batch_size', type=int, default=32)
arg.add_argument('--epochs', type=int, default=100)
arg.add_argument('--max_length', type=int, default=300)
arg.add_argument('--hidden_dropout_prob', type=float, default=0.15)
arg.add_argument('--add_H', type=bool, default=True)
arg.add_argument('--file_path', type=str, default='/data')
arg.add_argument('--smiles_field', type=str, default='smiles')
arg.add_argument('--label_field', type=str, default='kisc')
arg.add_argument('--vocab_size', type=int, default=18)
arg.add_argument('--d_model', type=int, default=256)
arg.add_argument('--num_layers', type=int, default=6)
arg.add_argument('--num_heads', type=int, default=4)
arg.add_argument('--pretrain_path', type=str, default='model_weights')



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
    model.train()
    train_loss = 0
    if pre_train_path is not None:
        model.load_state_dict(torch.load(pre_train_path))
    for i, (x, y, adjoin_matrix) in enumerate(train_loader):
        x, y, adjoin_matrix = x.to(device), y.to(device), adjoin_matrix.to(device)
        optimizer.zero_grad()
        outputs = model(x, adjoin_matrix)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)
    

def evaluate(model, val_loader, criterion, device):
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
    return val_loss / len(val_loader), y_true, y_pred

def main(args):
    set_random_seed(42)
    device = gpu_check()
    model = PredictionModel(num_layers=args.num_layers, d_model=args.d_model, dff=args.d_model*2, num_heads=args.num_heads, vocab_size=args.vocab_size, a=1)
    model.to(device)
    pre_train_path = os.path.join(args.pretrain_path, 'model_weights.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    dataset = Predict_Dataset(args.file_path, args.smiles_field, args.label_field, args.max_length, args.add_H)
    train_dataset, val_dataset, test_dataset = scaffold_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    train_loss = []
    val_loss = []
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion, device, pre_train_path)
        train_loss.append(loss)
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {loss:.4f}')
    print('Finished Training')
    val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'R2 Score: {r2_score(y_true, y_pred)}')
    print(f'MSE: {mean_squared_error(y_true, y_pred)}')
    

if __name__ == "__main__":
    args = arg.parse_args()
    main(args)