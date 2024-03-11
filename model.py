import torch
from torch import nn
from data2 import MRIdata
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
import scipy.sparse as sp
import random

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, 3, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, fmri):
        fmri = fmri.float()
        data = self.conv2d(fmri.unsqueeze(1))
        data = self.sig(data)
        data = data.squeeze(1)
        return data

class Get_sMRI_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = torch.tensor(MRIdata().G)
        self.attention = AttentionModule()

    def forward(self,x):
        x = self.G  
        x = self.attention(x.unsqueeze(0)).squeeze(0)
        corrcoef = torch.corrcoef(x.T)
        return corrcoef

class Get_fMRI_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(16)

    def forward(self, weighted_fmri):
        FMatrix = self.norm(weighted_fmri)
        return FMatrix


class DRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(DRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # Swap the second and third dimensions
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        x = x.squeeze(0)
        # print('x.shape',x.shape)
        # print('hidden.shape', hidden.shape)
        out, hidden = self.gru(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)
        return out


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, 140, self.hidden_dim)
        return hidden


class DGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(DGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        print(input_size, hidden_size, num_layers,)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(2,  100)
        # print(x.shape,h0.shape)
        # print('x.shape',x.shape)
        # print('h0.shape', h0.shape)
        out, _ = self.gru(x, h0)  
        out = self.dropout(out)
        return out

class Combined(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.attentionmodule = AttentionModule()
        self.get_sMRI_feature = Get_sMRI_feature()
        self.get_fMRI_feature = Get_fMRI_feature()
        # print(nfeat, nhid, nclass, 2, dropout)
        self.gcn1 = DRNN(116, nhid, nclass, 2, dropout)
        self.gcn2 = DGRU(116, nhid, 2, dropout)
        self.ln2 = nn.LayerNorm(116)
        self.ln3 = nn.LayerNorm(116)
        self.I = torch.eye(116)

    def forward(self, F, S):
        print(F.shape)
        adj_F = self.get_fMRI_feature(F.unsqueeze(0))
        adj_S = self.get_sMRI_feature(S.unsqueeze(0))
        a = self.gcn1(adj_F)
        b = self.gcn2(adj_S)
        a = a.view(-1)
        a = a.repeat((13456 // 4480) + 1)
        a = a[:13456]
        a = a.view(116, 116)

        b = b.view(-1)
        b = b.repeat((13456 // 4480) + 1)
        b = b[:13456]
        b = b.view(116, 116)

        print('11111',a.shape)
        F = self.ln2(a)
        S = self.ln3(b)    
        F_np = F.detach().cpu().numpy()
        S_np = S.detach().cpu().numpy()
        G_np = np.random.randn(116, 116)
        P_np = np.random.randn(116, 116)
        # 对数据应用高斯核矩阵
        result_F_gaussian_np = np.matmul(F_np, G_np)
        result_S_gaussian_np = np.matmul(S_np, G_np)
        # 对数据应用多项式核矩阵
        result_F_polynomial_np = np.matmul(F_np, P_np)
        result_S_polynomial_np = np.matmul(S_np, P_np)
        # 合并结果
        combined_result_gaussian_np = np.concatenate((result_F_gaussian_np, result_S_gaussian_np), axis=1)
        combined_result_polynomial_np = np.concatenate((result_F_polynomial_np, result_S_polynomial_np), axis=1)
        final_combined_result_np = np.concatenate((combined_result_gaussian_np, combined_result_polynomial_np), axis=1)
        result_np = final_combined_result_np[:, 0:2]
        # result_np = result_np[:, :, 0]
        result = torch.tensor(result_np).float()
        return result

