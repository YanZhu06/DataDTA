import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics
from einops.layers.torch import Rearrange, Reduce

from dataset import PT_FEATURE_SIZE
CHAR_SMI_SET_LEN = 64


class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=256, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.ReLU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                     )
        self.name = name
        
    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return   x +  self.layers(x)


class ResDilaCNNBlocks(nn.Module):
    #def __init__(self, feaSize, filterSize, blockNum=5, dropout=0.35, name='ResDilaCNNBlocks'):
    def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1,2,4,8,16], dropout=0.5, name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()#
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize,filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(dilaSizeList[i%len(dilaSizeList)],filterSize,dropout=dropout))
            #self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(filterSize,dropout=dropout))
        self.name = name
        self.act = nn.ReLU()
        

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x) # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1,2)) # => batchSize × seqLen × filterSize
        x = self.act(x) # => batchSize × seqLen × filterSize
        
        # x = self.pool(x.transpose(1, 2))
        x = Reduce('b c t -> b c', 'max')(x)
        return x


class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """
    
    def __init__(self, embed_size, head_num, dropout, residual = True ):
        """
        """
        super(MultiHeadAttentionInteract,self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))
        
        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.act = nn.ReLU()

        # for weight in self.parameters():
        #     nn.init.xavier_uniform_(weight)
        
    
    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))
        

        Query = torch.stack(torch.split(Query, self.attention_head_size, dim = 2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim = 2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim = 2))
        

        inner = torch.matmul(Query, Key.transpose(-2,-1))
        inner = inner / self.attention_head_size ** 0.5
        

        attn_w = F.softmax(inner, dim=-1)
 

        attn_w = F.dropout(attn_w, p = self.dropout)
        

        results = torch.matmul(attn_w, Value)
        

        results = torch.cat(torch.split(results, 1, ), dim = -1)
        results = torch.squeeze(results, dim = 0) # (bs, fields, D)
 
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))
        
        results = self.act(results)
        
        return results
    

class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.act = nn.ReLU()
         
    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = self.act(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear

        return x


class DualInteract(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout = 0.0):
        super(DualInteract, self).__init__()
        
        self.bit_wise_net = Highway(input_size = field_dim * embed_size,
                                       num_highway_layers = 1)


        hidden_dim = 1024
 
        self.vec_wise_net = MultiHeadAttentionInteract(embed_size = embed_size, 
                                                       head_num = head_num, 
                                                       dropout = dropout)

        # self.vec_wise_net = DisentangledSelfAttention(embedding_dim = embed_size)                           

        self.trans_bit_nn =  nn.Sequential(
            nn.LayerNorm(field_dim * embed_size ),
            nn.Linear(field_dim * embed_size , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim , field_dim * embed_size),
            # nn.PReLU(),
            ) 
        self.trans_vec_nn =  nn.Sequential(
            nn.LayerNorm(field_dim * embed_size ),
            nn.Linear(field_dim * embed_size , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim , field_dim * embed_size),  
            # nn.PReLU(),
            )
    
    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
       
        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)
        
        m_bit = self.trans_bit_nn(bit_wise_x)
        m_vec = self.trans_vec_nn(vec_wise_x)

        m_x =   m_vec + m_bit #  +   x.reshape(b, f * e)   
        return  m_x

class DisentangledSelfAttention(nn.Module):
    

    def __init__(self, embedding_dim, attention_dim=256, num_heads=2, dropout_rate=0.2,
                 use_residual=True, use_scale=True, relu_before_att=True):
        super(DisentangledSelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.relu_before_att = relu_before_att

        self.W_Q = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_K = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_V = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))


        self.W_q = nn.Linear(embedding_dim, self.attention_dim)
        self.W_k = nn.Linear(embedding_dim, self.attention_dim)
        self.W_v = nn.Linear(embedding_dim, self.attention_dim)
        self.W_unary = nn.Linear(embedding_dim, num_heads)

        if use_residual:
            self.W_res = nn.Linear(embedding_dim, self.attention_dim)
        else:
            self.W_res = None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        
        
    def forward(self, x):

        query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        residual = query
        unary = self.W_unary(key) # [batch, num_fields, num_heads]
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        if self.relu_before_att:
            query = query.relu()
            key = key.relu()
            value = value.relu()

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.head_dim)
        key = key.view(batch_size * self.num_heads, -1, self.head_dim)
        value = value.view(batch_size * self.num_heads, -1, self.head_dim)

        # whiten
        mu_query = query - query.mean(dim=1, keepdim=True)
        mu_key = key - key.mean(dim=1, keepdim=True)
        pair_weights = torch.bmm(mu_query, mu_key.transpose(1, 2))
        if self.use_scale:
            pair_weights /= self.head_dim ** 0.5
        pair_weights = F.softmax(pair_weights, dim=2) # [num_heads * batch, num_fields, num_fields]

        unary_weights = F.softmax(unary, dim=1)
        unary_weights = unary_weights.view(batch_size * self.num_heads, -1, 1)
        unary_weights = unary_weights.transpose(1, 2) # [num_heads * batch, 1, num_fields]
        
        attn_weights = pair_weights + unary_weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, value)
        output = output.view(batch_size, -1, self.attention_dim)

        return output






class MultiViewNet(nn.Module):
    
    def __init__(self, embed_dim=256):
        
        super(MultiViewNet, self).__init__()

        PT_FEATURE_SIZE = 40
        # onehot smiles
        self.embed_smile = nn.Embedding(65, embed_dim)
        self.embed_prot = nn.Embedding(26, embed_dim)
        self.onehot_smi_net = ResDilaCNNBlocks( embed_dim, embed_dim, name='res_compound')
        self.onehot_prot_net = ResDilaCNNBlocks(embed_dim, embed_dim,name='res_prot')

        
        hidden_dim = 1024
        proj_dim = 256
        field_dim = 4
        self.feature_interact =  DualInteract(field_dim=field_dim, embed_size=proj_dim, head_num=8)

        self.projection_prot_f =  nn.Sequential(
             nn.LayerNorm(126),#42
             nn.Linear(126 , proj_dim),
             #nn.ReLU(),
             #nn.Linear(hidden_dim , proj_dim),
             )

        
        self.projection_smi_3d =  nn.Sequential(
             nn.LayerNorm(900),
             nn.Linear(900 , proj_dim),
             #nn.ReLU(),
             #nn.Linear(hidden_dim , proj_dim),
             )
        
        self.transform = nn.Sequential(
             nn.LayerNorm(proj_dim * 3),
             nn.Linear(proj_dim * 3, 512),
             nn.Linear(512,1),
         )

        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, pkt, comp900, smi, seq):#42,900,65,26

        #CKSAAP = CKSAAP.to(torch.float32)
        smile_vectors_onehot = self.embed_smile(smi) 
        proteinFeature_onehot = self.embed_prot(seq)
        #protein_group_cksaap = self.projection_prot_cksaap(CKSAAP)

        proteinFeature_onehot = self.onehot_prot_net( proteinFeature_onehot)       
        compoundFeature_onehot = self.onehot_smi_net( smile_vectors_onehot )              

        graph_embedding = self.projection_smi_3d(comp900)
        protein_group_f = self.projection_prot_f(pkt)
        

        all_features = torch.stack([graph_embedding,  compoundFeature_onehot,proteinFeature_onehot], dim=2)
        all_features =  self.norm(all_features.permute(0,2,1))     #protein_group_f,
        all_features = self.feature_interact(all_features)
        out = self.transform(all_features)
        return out 
from datetime import datetime
from pathlib import Path
import os
seed = np.random.randint(33927, 33928)
def test(model: nn.Module, test_loader, loss_function, device, show, _p, record):
    path = '/home/zhuyan/2022y7m/run-run/'
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    np.savetxt(path + _p + record + 'targets.csv', targets)#, fmt ='%d'
    np.savetxt(path + _p + record + 'outputs.csv', outputs)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
        'pearson': metrics.get_pearson(targets, outputs),
    }

    return evaluation

#     return evaluation
