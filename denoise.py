'''
carry out noise detection and removal by cleanlab

'''
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader
from dataset import Predict_Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from model import Predict_Model
from cleanlab.regression.learn import CleanLearning


# wrapping the Predict_Model class to be compatible with cleanlab sklearn API
class CleanlabPredictModel(BaseEstimator, RegressorMixin):
    '''
    need to be compatible with sklearn API, do as fit(), 
    accomplish the noise detection and removal by cleanlab
    
    
    
    
    '''
    