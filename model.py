#pytorch implementation of BERT model

import torch
import torch.nn as nn
import math
import numpy as np


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return  0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def get_atten_pad_mask(seq_q, seq_k):
    len_q = seq_q.size(1)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(-1, len_q, -1)  # b x lq x lk


class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)


    def forward(self, x, token_type_ids):
        words_embeddings = self.word_embeddings(x)
        position_embeddings = self.position_embeddings(x)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, adjoin_matrix):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1))
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        if adjoin_matrix is not None:
            scores = scores + adjoin_matrix
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, attn_mask, adjoin_matrix=None):
        residual, batch_size = Q, Q.size(0) 
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = self.attention(q_s, k_s, v_s, attn_mask, adjoin_matrix)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn
    

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        return self.fc(inputs)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, enc_inputs, enc_self_attn_mask, adjoin_matrix=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, adjoin_matrix)  # enc_inputs to same Q,K,V
        enc_outputs = self.dropout1(enc_outputs)
        out1 = self.layer_norm1(enc_inputs + enc_outputs)
        enc_outputs = self.pos_ffn(out1)
        enc_outputs = self.dropout2(enc_outputs)
        enc_outputs = self.layer_norm2(out1 + enc_outputs)
        return enc_outputs, attn
    

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_k, d_v, d_ff, dropout, max_seq_len):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff, dropout) for _ in range(n_layers)])
        self.max_seq_len = max_seq_len

    def forward(self, enc_inputs, enc_self_attn_mask, adjoin_matrix=None):
        enc_outputs = enc_inputs
        attns = []
        for enc_layer in self.enc_layers:
            enc_outputs, enc_self_attn = enc_layer(enc_outputs, enc_self_attn_mask, adjoin_matrix)
            attns.append(enc_self_attn)
        return enc_outputs, attns
    

class BERT(nn.Module):
    def __init__(self, num_layer=6, d_model=256, d_ff=1024, n_heads=8, d_k=32, d_v=32, vocab_size=18, max_position_embeddings=512,  hidden_dropout_prob=0.1, max_seq_len=512):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, max_position_embeddings,type_vocab_size=vocab_size)
        self.encoder = Encoder(num_layer, d_model, n_heads, d_k, d_v, d_ff, hidden_dropout_prob, max_seq_len)
        self.activation = gelu
        self.fc1 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc2 = nn.Linear(d_model, vocab_size)   

    def forward(self, x, mask, token_type_ids, adjoin_matrix=None):
        embedding = self.embedding(x, token_type_ids)
        enc_self_attn_mask = get_atten_pad_mask(x, x)
        enc_outputs, attns = self.encoder(embedding, enc_self_attn_mask, adjoin_matrix)
        enc_outputs = self.activation(self.fc1(enc_outputs))
        enc_outputs = self.layer_norm(enc_outputs)
        logits = self.fc2(enc_outputs)
        return logits
    

class PredictionModel(nn.Module):
    def __init__(self, num_layer=6, d_model=256, d_ff=1024, n_heads=8, d_k=32, d_v=32, vocab_size=18, max_position_embeddings=512,  hidden_dropout_prob=0.1, max_seq_len=512):
        super(PredictionModel, self).__init__()
        self.enbedding = Embedding(vocab_size, d_model, max_position_embeddings, type_vocab_size=vocab_size)
        self.encoder = Encoder(num_layer, d_model, n_heads, d_k, d_v, d_ff, hidden_dropout_prob, max_seq_len)
        self.activation = gelu
        self.fc1 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x, mask, adjoin_matrix=None):
        enc_outputs, attns = self.encoder(x, mask, adjoin_matrix)
        x = enc_outputs[:,0,:]
        #avg_pool = torch.mean(enc_outputs, 1)
        #x = torch.cat((x, avg_pool), 1)
        x = self.activation(self.fc1(x))
        x = self.layer_norm(x)
        x = self.fc2(x)
        return x
    

class PredictionModel2(nn.Module):
    def __init__(self, num_layer=6, d_model=256, d_ff=1024, n_heads=8, d_k=32, d_v=32, hidden_dropout_prob=0.1, max_seq_len=512):
        super(PredictionModel, self).__init__()
        self.encoder = Encoder(num_layer, d_model, n_heads, d_k, d_v, d_ff, hidden_dropout_prob, max_seq_len)
        self.query_fc = nn.Linear(d_model, d_model)  
        self.key_fc = nn.Linear(d_model, d_model)     
        self.value_fc = nn.Linear(d_model, d_model)  
        self.fc1 = nn.Linear(d_model * 2 , d_model)  
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, x1, mask1, x2, mask2, adjoin_matrix1=None, adjoin_matrix2=None):
        enc_outputs1, _ = self.encoder(x1, mask1, adjoin_matrix1)

        enc_outputs2, _ = self.encoder(x2, mask2, adjoin_matrix2)

        query = self.query_fc(enc_outputs1[:, 0, :])
        key = self.key_fc(enc_outputs2[:, 0, :])
        value = self.value_fc(enc_outputs2[:, 0, :])
        attention_scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1))
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attended_output = torch.matmul(attention_probs, value).squeeze(1)

        # Concatenate the two encoded CLS tokens, attended_output and distance
        merged_output = torch.cat((enc_outputs1[:, 0, :], attended_output), dim=-1)

        x = self.activation(self.fc1(merged_output))
        x = self.layer_norm(x)
        x = self.fc2(x)
        
        return x

