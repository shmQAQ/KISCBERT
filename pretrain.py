import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from dataset import Pretrain_Dataset
from model import BertModel
from rdkit import RDLogger
import argparse 

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
    predictions = model(x, adjoin_matrix=adjoin_matrix, mask=mask)
    criterion = CrossEntropyLoss(reduction='none')
    loss = criterion(predictions.view(-1, predictions.size(-1)), y.view(-1))
    weighted_loss = loss * char_weight.view(-1)
    loss = weighted_loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    train_accuracy.append((torch.argmax(predictions, dim=-1) == y).float().mean().item())
    return train_loss, train_accuracy

def main(args):
    set_random_seed(args.random_seed)
    device = gpu_check()
    model = BertModel(args.num_layers, args.d_model, args.num_heads, 2*args.d_model, args.vocab_size, args.hidden_dropout_prob)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataset = Pretrain_Dataset(args.file_path, args.smiles_field,args.add_H)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=train_dataset.collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    train_loss = []
    train_accuracy = []
    for epoch in range(args.epochs):
        start_time = time.time()
        for x, y,adjoin_matrix,weights in train_loader:
            x, y, adjoin_matrix, weights = x.to(device), y.to(device),adjoin_matrix.to(device), weights.to(device)
            train_loss, train_accuracy = train_step(x, adjoin_matrix, y, weights, optimizer, model, train_loss, train_accuracy)
        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch+1}, Loss: {np.mean(train_loss):.4f}, Accuracy: {np.mean(train_accuracy):.4f}, Time_per_epoch: {epoch_time:.4f}')

    model._save_to_state_dict(os.path.join(args.save_path, 'model_weights.pth'))

if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--max_length', type=int, default=300)
    args.add_argument('--hidden_dropout_prob', type=float, default=0.15)
    args.add_argument('--add_H', type=bool, default=True)
    args.add_argument('--file_path', type=str, default='bert/data/weights.csv')
    args.add_argument('--smiles_field', type=str, default='smiles')
    args.add_argument('--vocab_size', type=int, default=18)
    args.add_argument('--d_model', type=int, default=256)
    args.add_argument('--num_layers', type=int, default=6)
    args.add_argument('--num_heads', type=int, default=4)
    args.add_argument('--save_path', type=str, default='model_weights')
    args.add_argument('--random_seed', type=int, default=42)

    main(args)