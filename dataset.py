# rewrite pytorch Dataset class to handle the data
# input: a csv file with columns: smiles, kisc(10e7), krisc(10e5)
#output tokenized smiles, target and adjacency matrix
from torch.utils.data import Dataset
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import pandas as pd
import os
from itertools import compress
from rdkit import Chem
from utils import smiles2adjoin

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
         
num2str =  {i:j for j,i in str2num.items()}


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset.iloc[train_idx]
    valid_dataset = dataset.iloc[valid_idx]
    test_dataset = dataset.iloc[test_idx]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)
    
class Pretrain_Dataset(Dataset):
    '''
    pytorch dataset class for pretraining
    read the csv file and tokenize the smiles, and add noise to the tokenized smiles
    return tokenized smiles
    '''

    def __init__(self, file_path, smiles_field, max_length=300, noise_feq=0.15, add_H=True):
        self.data = pd.read_csv(os.path.join(file_path))
        self.smiles_field = smiles_field
        self.max_length = max_length
        self.noise_feq = noise_feq
        self.add_H = add_H
        self.str2num = str2num
        self.num2str = num2str


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx,self.smiles_field]
        x, y, adjoin_matrix, weights = self.numericalize(smiles)
        return x, y, adjoin_matrix, weights

        
    def numericalize(self, smiles):
        '''
        convert smiles to numerical representation
        '''
        print('handling smiles')
        smiles = smiles.np().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.add_H)
        atoms_list = ['<global>'] + atoms_list
        num_list = [str2num.get(atom, str2num['<unk>']) for atom in atoms_list]
        
        temp = np.ones((len(num_list), len(num_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp) * -1e9

        choice = np.random.permutation(len(num_list)-1)[:max(1, int(len(num_list)*self.noise_feq))]+1

        y = np.array(num_list).astype('int64')
        weights = np.zeros(len(num_list))
        for i in choice:
            rand = np.random.rand()
            if rand < 0.8:
                y[i] = str2num['<mask>']
                weights[i] = 1
            elif rand < 0.9:
                y[i] = np.random.randint(1, 18)
                weights[i] = 1
        
        x = np.array(num_list).astype('int64')
        weights = np.array(weights).astype('float32')
        print('smiles handled')
        return x, y, adjoin_matrix, weights
    

class Predict_Dataset(Dataset):
    '''
    pytorch dataset class for prediction
    read the csv file and tokenize the smiles
    return tokenized smiles, target and adjacency matrix    
    '''
    def __init__(self, file_path, smiles_field, target_field, max_length=300, add_H=True):
        self.data = pd.read_csv(os.path.join(file_path))
        self.smiles_field = smiles_field
        self.target_field = target_field
        self.max_length = max_length
        self.add_H = add_H
        self.str2num = str2num
        self.num2str = num2str

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx,self.smiles_field]
        target = self.data.iloc[idx,self.target_field]
        x, y, adjoin_matrix = self.numericalize(smiles, target)
        return x, y, adjoin_matrix
        
    def numericalize(self, smiles, target):
            '''
            convert smiles to numerical representation
            '''
            smiles = smiles.np().decode()
            fg_list = fg_list(smiles)
            atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.add_H)
            atoms_list = ['<global>'] + atoms_list
            num_list = [str2num.get(atom, str2num['<unk>']) for atom in atoms_list]
            temp = np.ones((len(num_list), len(num_list)))
            temp[1:, 1:] = adjoin_matrix
            adjoin_matrix = (1-temp) * -1e9
            x = np.array(num_list).astype('int64')
            y = np.array(target).astype('float32')

            return x, y, adjoin_matrix
    
