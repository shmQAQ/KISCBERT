#pytorch implementation of BERT model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def gelu(x):
    """
      Implementation of the OpenAI's gelu activation function.
      Also see https://arxiv.org/abs/1606.08415
    """
    return  0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def scaled_dot_product_attention(q, k, v, mask=None, adjoin_matrix=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    dk = k.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk).float())

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights



class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask, adjoin_matrix):

        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, adjoin_matrix)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        scaled_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(scaled_attention)

        return output, attention_weights
    

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedforward, self).__init__()
        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)
        self.activation = gelu

    def forward(self, x):
        x = self.activation(self.dense1(x))
        x = self.dense2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiheadAttention(d_model, num_heads)
        self.positionwise_feedforward = PositionwiseFeedforward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask, adjoin_matrix):
        attention_output, _ = self.multi_head_attention(x, x, x, mask, adjoin_matrix)
        attention_output = self.dropout1(attention_output)
        x = self.layernorm1(x + attention_output)

        feedforward_output = self.positionwise_feedforward(x)
        feedforward_output = self.dropout2(feedforward_output)
        x = self.layernorm2(x + feedforward_output)

        return x, attention_output
    

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask, adjoin_matrix):
        seq_len = x.shape[1]
        adjoin_matrix = adjoin_matrix.unsqueeze(1)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.dropout(x)
        attention_weights = []
        xs = []
        for i in range(self.num_layers):
            x, attention_weight = self.encoder_layers[i](x, mask, adjoin_matrix)
            xs.append(x)
            attention_weights.append(attention_weight)
        return x
        

class BertModel(nn.Module):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, d_ff=512, vocab_size=18, dropout_rate=0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, vocab_size)
        self.activation = gelu
        self.ln = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask, adjoin_matrix):
        x = self.encoder(x, mask, adjoin_matrix)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x
    

class Predict_Model(nn.Module):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, d_ff=512, vocab_size=17, dropout_rate=0.1):
        super(Predict_Model, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 1)
        self.activation = gelu
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask, adjoin_matrix):
        x = self.encoder(x, mask, adjoin_matrix)
        x = x[:, 0, :] 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

'''
class Predict_Model2(nn.Module):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, d_ff=512, vocab_size=17, dropout_rate=0.1):
        super(Predict_Model, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate)
        self.fc1 = nn.Linear(d_model * 2, d_model)  
        self.fc2 = nn.Linear(d_model, 1)
        self.activation = gelu
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, mask1, adjoin_matrix1, x2, mask2, adjoin_matrix2):
        x1 = self.encoder(x1, mask1, adjoin_matrix1)
        x1 = x1[:, 0, :] 
        x2 = self.encoder(x2, mask2, adjoin_matrix2)
        x2 = x2[:, 0, :] 
        x = torch.cat([x1, x2], dim=-1) 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


'''