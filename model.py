import torch
from torch import nn
from data2 import MRIdata
# from audtorch.metrics.functional import pearsonr
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
import scipy.sparse as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#    # adj = torch.coo_matrix(adj)
#     rowsum = adj.sum(1)
#     d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
#     d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
#     adj=torch.matmul(d_mat_inv_sqrt,adj)
#     adj=torch.matmul(adj,d_mat_inv_sqrt)
#     return adj

class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        # self.pool=nn.AvgPool2d(3,1)
        self.conv1d = nn.Conv1d(1, 1, 3, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, fmri):
        # fmir: [batch,channel=1,length,#patients]
        data = self.conv2d(fmri.unsqueeze_(1))

        data = torch.mean(data, 2)
        data = self.sig(self.conv1d(data))
        data = self.conv1d(data)
        data = data.unsqueeze_(2).repeat(1, 1, fmri.shape[2], 1)

        return (fmri * data).squeeze()


class Get_sMRI_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = torch.tensor(MRIdata().G)
        print()

    # def forward(self):
    #     x=self.G
    #     f = (x.shape[0] - 1) / x.shape[0]
    #     x_reducemean = x - torch.mean(x,dim=0)
    #     numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
    #     var_ = x.var(axis=0).reshape(x.shape[1], 1)
    #     denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
    #     corrcoef = numerator / denominator
    #     return  self.select(corrcoef,threshold=0.5)
    def forward(self):
        x = self.G  # .detach().numpy() #698 116
        corrcoef = torch.corrcoef(x.T)
        # corrcoef = torch.corrcoef(x, rowvar=False)
        return self.select(corrcoef, threshold=0.9)

    def select(self, Fmatrix, threshold):
        # Fmatrix[Fmatrix<=threshold]=0
        Fmatrix[torch.abs(Fmatrix) <= threshold] = 0
        return Fmatrix.float()
        # return torch.FloatTensor(Fmatrix)


class Get_fMRI_feature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weighted_fmri):
        # weighted_fmri  (batch, 140,116)
        # mu=torch.mean(weighted_fmri,0)
        # cv=self.cov(weighted_fmri)
        # delta=weighted_fmri
        # FMatrix=torch.mean(torch.einsum('bki,bij->bkj',weighted_fmri.transpose(2,1),weighted_fmri),0).squeeze()
        # a=torch.mean(weighted_fmri,0).detach().numpy() #116, 140
        # FMatrix = np.corrcoef(a,rowvar=False) 
        # b=weighted_fmri.detach().numpy() # 140, 116
        b = weighted_fmri.detach()
        adjs = []
        # for i in range(weighted_fmri.shape[0]):
        #     FMatrix = np.corrcoef(b[i],rowvar=False)
        #     adjs.append(FMatrix)
        # FMatrix=np.stack(adjs,axis=0)  #16, 116, 116
        for i in range(weighted_fmri.shape[0]):
            FMatrix = torch.corrcoef(b[i].T)
            # FMatrix = self.select(FMatrix,0.5)
            # FMatrix=normalize_adj(FMatrix+torch.eye(116).to(device))
            adjs.append(FMatrix.unsqueeze_(0))

        FMatrix = torch.cat(adjs, dim=0)  # 16, 116, 116

        return self.select(FMatrix, 0.9)

    def select(self, Fmatrix, threshold):
        # Fmatrix[Fmatrix<=threshold]=0
        Fmatrix[torch.abs(Fmatrix) <= threshold] = 0
        return Fmatrix.float()

    # def cov(self,input):
    #     # (batch, 140, 116)
    #     b,  h, w = input.size()
    #     x = input- torch.mean(input)
    #     x = x.view(b , h * w)
    #     cov_matrix = torch.matmul(x.T, x) / x.shape[0]
    #     return cov_matrix


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None),
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        support = torch.einsum('bij,ik->bjk', input, self.weight)
        support = support.transpose(1, 2)
        # output = torch.spmm(adj, support)
        output = torch.einsum('ki,bji->bkj', adj, support)
        if self.bias is not None:
            return (output + self.bias).transpose_(1, 2)
        else:
            return output.transpose_(1, 2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionf(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionf, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None),
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        support = torch.einsum('bij,ik->bjk', input, self.weight)
        support = support.transpose(1, 2)
        # output = torch.spmm(adj, support)
        output = torch.einsum('bki,bji->bkj', adj, support)
        if self.bias is not None:
            return (output + self.bias).transpose_(1, 2)
        else:
            return output.transpose_(1, 2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (batch, lenght, num_of_brain)
        x = nn.ReLU()(self.gc1(x, adj))
        # x = nn.Dropout(self.dropout)(x)
        # x = nn.ReLU()(self.gc2(x, adj))
        x = self.dropout(x)
        x = nn.ReLU()(self.gc3(x, adj))
        # x = nn.Dropout(self.dropout)(x)

        return x


class GCNf(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNf, self).__init__()

        self.gc1 = GraphConvolutionf(nfeat, nhid)
        # self.gc2 = GraphConvolutionf(nhid, nhid)
        self.gc3 = GraphConvolutionf(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (batch, lenght, num_of_brain)
        x = nn.ReLU()(self.gc1(x, adj))
        # x = nn.Dropout(self.dropout)(x)
        # x = nn.ReLU()(self.gc2(x, adj))
        x = self.dropout(x)
        x = nn.ReLU()(self.gc3(x, adj))

        return x


class MRICls(Module):
    def __init__(self, c_in=4 * 116, c_out=64) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(c_in, c_out),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 2)
                                    )

    def forward(self, F, S):
        # F: batch, 2, 116
        # S: batch, 2 ,116
        b = torch.einsum('bkj,bij->bjki', F, S)
        b = torch.sigmoid_(torch.mean(b.reshape(b.shape[0], b.shape[1], -1), 2))
        # b: batch,116
        b = b.unsqueeze(1).repeat(1, 2, 1)
        F = F * b
        S = S * b
        # print()
        feature = torch.cat((F, S), 1)
        feature = feature.reshape(feature.shape[0], -1)
        output = self.linear(feature)

        # output = (output - output.min()) / (output.max() - output.min())

        output = nn.Softmax(1)(output)
        return output


class Combined(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.attentionmodule = AttentionModule()

        self.get_sMRI_feature = Get_sMRI_feature()
        self.get_fMRI_feature = Get_fMRI_feature()

        self.gcn1 = GCNf(nfeat, nhid, nclass, dropout)  # FRMI
        self.gcn2 = GCN(3, 20, nclass, dropout)         # SRMI
        self.MRIcls = MRICls()
        # self.ln1= nn.LayerNorm(116)

        self.ln2 = nn.LayerNorm(116)  # 归一化
        self.ln3 = nn.LayerNorm(116)  # 归一化
        self.I = torch.eye(116).to(device)

    def forward(self, F, S):
        print('___________________________')
        # I = self.I
        # F=self.attentionmodule(self.ln1(F))
        print('传入F.shape', F.shape)
        print('传入S.shape', S.shape)

        FF = self.attentionmodule(F)
        # F=torch.rand([16,140,116])
        adj_F = self.get_fMRI_feature(F.squeeze())
        # adj_F = torch.add(adj_F, I)
        F = self.ln2(self.gcn1(FF, adj_F))  # 归一化
        print('FF.shape', FF.shape)
        print('adj_F.shape', adj_F.shape)
        print('特征F.shape', F.shape)
        
        adj_S = self.get_sMRI_feature().to(device)
        # adj_S = torch.add(adj_S, I)
        # adj_S=normalize_adj(adj_S)
        S = self.ln3(self.gcn2(S, adj_S))  # 归一化
        print('adj_S.shape', adj_S.shape)
        print('特征S.shape', S.shape)
        
        Output = self.MRIcls(F, S)
        print('Output.shape', Output.shape)
        print('___________________________')
        print()
        print()

        return Output


if __name__ == '__main__':
    nfeat = 140
    nhid = 100
    nclass = 2
    dropout = 0.5
    F = torch.rand(16, 140, 116)
    S = torch.rand(16, 3, 116)
    combined = Combined(nfeat, nhid, nclass, dropout)
    out = combined(F, S)
    print('1', F.data.cpu().numpy())
    pass
