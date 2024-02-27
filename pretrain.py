import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from dataset import Pretrain_Dataset
from model import BERT
from rdkit import RDLogger
import argparse 

RDLogger.DisableLog('rdApp.*')

arg = argparse.ArgumentParser()
arg.add_argument('--batch_size', type=int, default=32)
arg.add_argument('--epochs', type=int, default=100)
arg.add_argument('--max_length', type=int, default=300)
arg.add_argument('--hidden_dropout_prob', type=float, default=0.15)
arg.add_argument('--noise_feq', type=float, default=0.15)
arg.add_argument('--add_H', type=bool, default=True)
arg.add_argument('--file_path', type=str, default='/data')
arg.add_argument('--smiles_field', type=str, default='smiles')
arg.add_argument('--vocab_size', type=int, default=18)
arg.add_argument('--d_model', type=int, default=256)
arg.add_argument('--num_layers', type=int, default=6)
arg.add_argument('--num_heads', type=int, default=4)
arg.add_argument('--save_path', type=str, default='model_weights')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_check():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def train_step(x, adjoin_matrix, y, char_weight, optimizer, model, train_loss, train_accuracy):
    seq = torch.eq(x, 0).float()
    mask = seq.unsqueeze(1).unsqueeze(2)
    predictions = model(y, adjoin_matrix=adjoin_matrix, mask=mask)
    loss = CrossEntropyLoss(predictions.view(-1, predictions.size(-1)), x.view(-1), char_weight.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    train_accuracy.append((torch.argmax(predictions, dim=-1) == y).float().mean().item())
    return train_loss, train_accuracy

def main(args):
    set_random_seed(42)
    device = gpu_check()
    model = BERT(args.num_layers, args.d_model, args.d_model*2, args.num_heads, args.vocab_size, args.max_length, args.hidden_dropout_prob)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataset = Pretrain_Dataset(args.file_path, args.smiles_field, args.max_length, args.noise_feq, args.add_H)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True).to(device)
    train_loss = []
    train_accuracy = []
    for epoch in range(args.epochs):
        for x, y, adjoin_matrix, weights in train_loader:
            x, y, adjoin_matrix, weights = x, y, adjoin_matrix, weights
            train_loss, train_accuracy = train_step(x, adjoin_matrix, y, weights, optimizer, model, train_loss, train_accuracy)
        print(f'Epoch: {epoch+1}, Loss: {np.mean(train_loss)}, Accuracy: {np.mean(train_accuracy)}')
    model._save_to_state_dict(os.path.join(args.save_path, 'model_weights.pth'))

if __name__ == '__main__':
    args = arg.parse_args()
    main(args)