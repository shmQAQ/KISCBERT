# rewrite pytorch Dataset class to handle the data
# input: a csv file with columns: smiles, kisc(10e7), krisc(10e5)
#output tokenized smiles, target and adjacency matrix
from torch.utils.data import Dataset
#from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import pandas as pd
import os
from utils import smiles2adjoin

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
         
num2str =  {i:j for j,i in str2num.items()}

def pad_or_truncate_1d(tensor_list, max_size, pad_value=0):
    padded_tensors = []
    for tensor in tensor_list:
        if tensor.size(0) > max_size:
            tensor = tensor[:max_size]
        elif tensor.size(0) < max_size:
            pad_size = max_size - tensor.size(0)
            pad = torch.full((pad_size,), pad_value)
            tensor = torch.cat([tensor, pad], dim=0)
        padded_tensors.append(tensor)
    return torch.stack(padded_tensors)

def pad_or_truncate_2d(matrix_list, max_size, pad_value=0):
    padded_matrices = []
    for matrix in matrix_list:
        if max(matrix.size(0), matrix.size(1)) > max_size:
            matrix = matrix[:max_size, :max_size]
        padded_matrix = torch.full((max_size, max_size), pad_value)
        rows = min(matrix.size(0), max_size)
        cols = min(matrix.size(1), max_size)
        padded_matrix[:rows, :cols] = matrix[:rows, :cols]
        padded_matrices.append(padded_matrix)
    return torch.stack(padded_matrices)

class Pretrain_Dataset(Dataset):
    '''
    pytorch dataset class for pretraining
    read the csv file and tokenize the smiles, and add noise to the tokenized smiles
    return tokenized smiles
    '''

    def __init__(self, path, smiles_field='smiles', addH=True):
        self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH
        self.padding_length = self.get_padding_length()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx][self.smiles_field]
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        adjoin_matrix = torch.from_numpy(adjoin_matrix)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = torch.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = torch.tensor(nums_list, dtype=torch.int64)
        weight = torch.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)
        x = torch.tensor(nums_list, dtype=torch.int64)

        return x, y, adjoin_matrix, weight

    def collate_fn(self, batch):
        x, y, adjoin_matrix, weight = zip(*batch)
        x = pad_or_truncate_1d(x, max_size=self.padding_length, pad_value=self.vocab['<pad>'])
        y = pad_or_truncate_1d(y, max_size=self.padding_length, pad_value=self.vocab['<pad>'])
        adjoin_matrix = pad_or_truncate_2d(adjoin_matrix, max_size=self.padding_length, pad_value=-1e9)
        weight = pad_or_truncate_1d(weight, max_size=self.padding_length, pad_value=0)
        return x, y, adjoin_matrix, weight
    
    def get_padding_length(self):
        padding_length = 0
        for i in range(len(self.df)):
            smiles = self.df.iloc[i][self.smiles_field]
            atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
            padding_length = max(padding_length, len(atoms_list))
        return padding_length

    

class Predict_Dataset(Dataset):
    '''
    pytorch dataset class for prediction
    read the csv file and tokenize the smiles
    return tokenized smiles, target and adjacency matrix    
    '''
    def __init__(self, file_path, smiles_field, target_field, add_H=True):
        try:
            self.df = pd.read_csv(os.path.join(file_path))
        except:
            self.df = pd.read_excel(os.path.join(file_path))
        self.smiles_field = smiles_field
        self.target_field = target_field
        self.add_H = add_H
        self.str2num = str2num
        self.num2str = num2str
        self.padding_length = self.get_padding_length()

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        smiles = self.df.iloc[idx][self.smiles_field]
        target = self.df.iloc[idx][self.target_field]
        max_target = max(self.df[self.target_field])
        min_target = min(self.df[self.target_field])
        target = (target - min_target) / (max_target - min_target) #normalize the target
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.add_H)
        adjoin_matrix = torch.from_numpy(adjoin_matrix)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = torch.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
        x = torch.tensor(nums_list, dtype=torch.int64)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y, adjoin_matrix
    
     

    def collate_fn(self, batch):
        x, y, adjoin_matrix = zip(*batch)
        x = pad_or_truncate_1d(x, max_size=self.padding_length, pad_value=self.str2num['<pad>'])
        adjoin_matrix = pad_or_truncate_2d(adjoin_matrix, max_size=self.padding_length, pad_value=-1e9)
        y = torch.stack(y)
        return x, y, adjoin_matrix
    
    def get_padding_length(self):
        padding_length = 0
        for i in range(len(self.df)):
            smiles = self.df.iloc[i][self.smiles_field]
            atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.add_H)
            padding_length = max(padding_length, len(atoms_list))
        return padding_length

