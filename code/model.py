import numpy as np
import torch.nn as nn
import torch
import args
import random as rand
import heapq


import math
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Prop_esNet(Module):
    def __init__(self, adj):  
        super(Prop_esNet, self).__init__()
        self.adj_cpu = adj
        self.adj = torch.tensor(adj).to(args.device)

        self.weight = Parameter(torch.FloatTensor(adj.shape[0], adj.shape[1]))
        self.reset_parameters()

    def prob_map(self, snap_pred, snapshot):
        snapshot_seed = snapshot[range(snapshot.shape[0] - snap_pred.shape[0], snapshot.shape[0]), :]
        snapshot_seed = snapshot_seed - snapshot[range(snapshot.shape[0] - snap_pred.shape[0] - 1, snapshot.shape[0] - 1), :] 
        act_num = torch.sum(torch.from_numpy(snapshot_seed).float().to(args.device), dim=1)
        assert len(act_num) == snap_pred.shape[0], 'sum执行方式错误'

        for i in range(len(act_num)):
            tmp = zip(range(len(snap_pred[i].tolist())), snap_pred[i].tolist())
            largeN = heapq.nlargest(int(act_num[i]), tmp, key=lambda x: x[1])
            for YZ in largeN:
                snap_pred[i][YZ[0]] = 1
            snap_pred[i][torch.where(snap_pred[i] != 1)] = 0
            assert sum(snap_pred[i]) % 1 <= 0.001, '计算错了'
        return snap_pred


    def reset_parameters(self):
        for row in range(self.adj.shape[0]):
            for col in range(self.adj.shape[0]):
                if self.adj[row][col] < 0.5:
                    self.weight.data[row][col] = 0
                else:
                    self.weight.data[row][col] = rand.choice([0.03, 0.05, 0.07, 0.1, 0.12])

    
    def forward(self, snapshot_vec_series, time_pick):
        
        assert len(snapshot_vec_series.shape) == 2, '维度不对'
        assert snapshot_vec_series.shape[0] == args.pick_T_num == len(time_pick), '传入参数不对'
        assert snapshot_vec_series.shape[1] == args.network_verNum, '节点数量不对'

        snapshot_vec_series = torch.from_numpy(snapshot_vec_series).float().to(args.device)
        snap_next_pred = torch.zeros(args.pick_T_num-2, args.network_verNum, dtype=float).to(args.device)  
        for T_index in range(1, args.pick_T_num-1):  
            
            row_vec = snapshot_vec_series[T_index] - snapshot_vec_series[T_index-1]
            assert torch.sum(snapshot_vec_series[T_index]) == torch.sum(snapshot_vec_series[T_index-1]) \
                   + torch.sum(row_vec), '错误'
            
            
            for time in range(time_pick[T_index+1]-time_pick[T_index]):  
                
                row_vec = row_vec.view((1, -1)).expand(self.weight.shape)
                row_vec_temp = row_vec.detach().cpu().numpy()
                P2 = self.weight.T * row_vec  
                
                P3 = torch.ones(self.weight.shape).to(args.device) - P2
                P3_temp = P3.detach().cpu().numpy()
                neg_mul = torch.prod(P3, dim=1)
                neg_mul_temp = neg_mul.detach().cpu().numpy()
                row_vec = torch.ones((args.network_verNum,)).to(args.device) - neg_mul  

                
                
            snap_next_pred[T_index - 1] = row_vec
        
        return snap_next_pred



    def loss(self, snap_pred, snapshot):
        snapshot_seed = snapshot[range(snapshot.shape[0] - snap_pred.shape[0], snapshot.shape[0]), :]
        snapshot_seed = snapshot_seed - snapshot[range(snapshot.shape[0] - snap_pred.shape[0] - 1, snapshot.shape[0] - 1), :]
        assert snapshot_seed.shape[0] == snap_pred.shape[0], '维度错误'
        assert snapshot_seed.shape[1] == snap_pred.shape[1], '维度错误'
        return F.mse_loss(snap_pred, torch.from_numpy(snapshot_seed).float().to(args.device))
        
        

    def parameters_correction(self):
        for row in range(self.adj_cpu.shape[0]):
            for col in range(self.adj_cpu.shape[0]):
                if self.adj_cpu[row][col] < 0.5:
                    self.weight.data[row][col] = 0
                else:
                    if self.weight.data[row][col] < 0:
                        
                        self.weight.data[row][col] = Parameter( abs(self.weight.data[row][col]) )
                        


class GraphConv(Module):

    def __init__(self, in_features, out_features, bias=True, scale_identity=False, dropout=0, activation=None,
                 Dice_M2=None):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = dropout
        self.activation = activation
        self.Dice_M2 = Dice_M2
        self.scale_identity=scale_identity

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def Dice_similar(self, A):

        
        return self.Dice_M2

    def laplacian_matrix(self, A):
        D = torch.diag(A.sum(1))
        D_hat = D ** (-0.5)
        D_hat[torch.isinf(D_hat)] = 0

        I = torch.eye(A.shape[0]).to(args.device)
        if self.scale_identity is True:
            
            S = self.Dice_similar(A).to(args.device)
            return D_hat * (A + I + S) * D_hat
        else:
            return D_hat * (A + I) * D_hat

    def forward(self, myCollection):  
        input_onestamp, adj = myCollection[0], myCollection[1]
        assert len(input_onestamp.shape) == 3, '输入的数据不对劲'
        
        support = torch.bmm(input_onestamp,
                            self.weight.tile(input_onestamp.shape[0], 1, 1))  
        

        L = self.laplacian_matrix(adj).to(args.device)
        output = torch.bmm(L.tile(input_onestamp.shape[0], 1, 1), support)
        
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        if self.dropout > 0.01:
            return (F.dropout(output, self.dropout, training=self.training), myCollection[1])
        else:
            return (output, myCollection[1])

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 n_hidden=0,
                 dropout=0.2,
                 behavior_identity=False,
                 FC=True,
                 Dice_M1=None):
        super(GCN, self).__init__()

        
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                activation=nn.ReLU(inplace=True),
                                                scale_identity=behavior_identity,
                                                dropout=0.1 if layer != len(filters) - 1 else 0,
                                                Dice_M2=Dice_M1) for layer, f in enumerate(filters)]))

        self.FC = FC
        if FC is True:
            
            fc = []
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            if n_hidden > 0:
                fc.append(nn.Linear(filters[-1], n_hidden))
                fc.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    fc.append(nn.Dropout(p=dropout))
                n_last = n_hidden
            else:
                n_last = filters[-1]
            fc.append(nn.Linear(n_last, out_features))
            self.fc = nn.Sequential(*fc)

    def forward(self, data, T_index):
        if self.FC is True:
            data_features = data[3][:, :, :, range(0, 2)]  
            deta_features_onestamp = data_features[:, T_index, :, :]  
        else:
            deta_features_onestamp = torch.eye(args.network_verNum, args.network_verNum).tile(data[3].shape[0], 1, 1).to(args.device)  

        data_adj = data[0]
        if len(data[0].shape) == 3:
            data_adj = data_adj[0]  


        x = self.gconv((deta_features_onestamp, data_adj))  
        if self.FC is True:
            x = self.fc(x[0])
            return x
        else:
            return x[0]

    
    def loss(self, pred, label_sourceNum, label_hot=None, prob_pred=None, γ=None, GCN_models=None, RU_model=None):  
        B = pred.shape[0]
        loss_total = torch.tensor([0.], ).to(args.device)
        for i in range(B):
            pred_S_one_B = pred[i, 0, :]  
            pred_I_one_B = pred[i, 1, :]  
            labels_one_B = label_sourceNum[i, :]  

            begin = 0
            for source_index in labels_one_B:
                for idx_current in range(begin, source_index):
                    
                    loss_total = loss_total + ((1 - pred_S_one_B[idx_current]) + (pred_I_one_B[idx_current])) * (
                                labels_one_B.shape[0] / pred_S_one_B.shape[0])
                
                loss_total = loss_total + ((pred_S_one_B[source_index]) + (1 - pred_I_one_B[source_index])) * (
                            1 - (labels_one_B.shape[0] / pred_S_one_B.shape[0]))
                begin = source_index
        assert len(prob_pred.squeeze(-1).shape) == len(label_hot.shape), '不对劲1'
        assert prob_pred.squeeze(-1).shape[0] == label_hot.shape[0], '不对劲1'
        assert prob_pred.squeeze(-1).shape[1] == label_hot.shape[1], '不对劲1'

        cross = F.cross_entropy(pred, label_hot)
        mse = F.mse_loss(prob_pred.squeeze(-1), label_hot.float())
        loss_one_batch = loss_total / B

        loss_total = loss_total / B + args.network_verNum * (F.cross_entropy(pred, label_hot) + F.mse_loss(prob_pred.squeeze(-1), label_hot.float()))
        return loss_total





def gru_forward( input, initial_states, w_ih, w_hh,b_ih, b_hh ):
    prev_h = initial_states
    bs, T,i_size = input.shape
    h_size =w_ih.shape[0] //3
    
    batch_w_ih = w_ih.unsqueeze( 0 ).tile( bs, 1, 1)
    batch_w_hh = w_hh.unsqueeze(0).tile ( bs, 1, 1)
    output = torch.zeros(bs,T,h_size)
    for t in range(T):
        x = input[ :, t,:] 
        w_times_x = torch . bmm ( batch_w_ih,x.unsqueeze(-1)) 
        w_times_x = w_times_x.squeeze(-1)  
        w_times_h_prev = torch. bmm ( batch_w_hh, prev_h.unsqueeze(-1)) 
        w_times_h_prev = w_times_h_prev.squeeze(-1) 
        r_t = torch.sigmoid(w_times_x[:, :h_size]+w_times_h_prev[:,:h_size]+b_ih[ :h_size]+b_hh[ :h_size]) 
        z_t = torch.sigmoid(w_times_x[ :, h_size:2*h_size]+w_times_h_prev[:, h_size:2*h_size]+\
                            b_ih[h_size:2*h_size]+b_hh[h_size:2*h_size]) 
        n_t = torch.tanh(w_times_x[ :,2*h_size:3*h_size]+b_ih[2*h_size:3*h_size]+\
                         r_t*(w_times_h_prev[ :, 2*h_size:3*h_size]+b_hh[2*h_size:3*h_size])) 
        prev_h = (1 - z_t)*n_t + z_t*prev_h
        output[ :, t,:]= prev_h
    return output, prev_h


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nclass = nclass

        if nhid > 0:
            
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):  
        x = F.dropout(x, self.dropout, training=self.training)
        if self.nhid > 0:
            
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=1)
        else:
            
            
            out_ = torch.zeros(args.network_verNum, args.pick_T_num, self.nclass).to(args.device)
            for att in self.attentions:
                out_ = out_ + F.elu(att(x, adj))
            out_ = out_/len(self.attentions)
            out_ = F.dropout(out_, self.dropout, training=self.training)
            return F.softmax(out_[:, 0, :], dim=1)



class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        
        
        
        
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.FloatTensor(2 * out_features, 1))
        self.reset_parameters()

        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.a is not None:
            self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj, using_loop =False):  
        h_p = torch.transpose(h, 0, 1)
        Wh = torch.matmul(h_p, self.W) 
        alpha_ = torch.zeros(Wh.shape[0], Wh.shape[1], Wh.shape[1])
        if using_loop is True:
            for node_idx in range(Wh.shape[0]):
                Wh_single = Wh[node_idx]   
                
                alpha_[node_idx, 0, 0] = torch.cat((Wh_single[0], Wh_single[0])) @ self.a
                alpha_[node_idx, 0, 1] = torch.cat((Wh_single[0], Wh_single[1])) @ self.a

                alpha_[node_idx, 1, 0] = torch.cat((Wh_single[1], Wh_single[0])) @ self.a
                alpha_[node_idx, 1, 1] = torch.cat((Wh_single[1], Wh_single[1])) @ self.a
                alpha_[node_idx, 1, 2] = torch.cat((Wh_single[1], Wh_single[2])) @ self.a

                alpha_[node_idx, 2, 1] = torch.cat((Wh_single[2], Wh_single[1])) @ self.a
                alpha_[node_idx, 2, 2] = torch.cat((Wh_single[2], Wh_single[2])) @ self.a
                alpha_[node_idx, 2, 3] = torch.cat((Wh_single[2], Wh_single[3])) @ self.a

                alpha_[node_idx, 3, 2] = torch.cat((Wh_single[3], Wh_single[2])) @ self.a
                alpha_[node_idx, 3, 3] = torch.cat((Wh_single[3], Wh_single[3])) @ self.a
                alpha_[node_idx, 3, 4] = torch.cat((Wh_single[3], Wh_single[4])) @ self.a

                alpha_[node_idx, 4, 3] = torch.cat((Wh_single[4], Wh_single[3])) @ self.a
                alpha_[node_idx, 4, 4] = torch.cat((Wh_single[4], Wh_single[4])) @ self.a
        else:
            qq = torch.cat((Wh[:, 0, :], Wh[:, 0, :]), dim=1)
            qqq = qq @ self.a
            alpha_[:, 0, 0] = (torch.cat((Wh[:, 0, :], Wh[:, 0, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 0, 1] = (torch.cat((Wh[:, 0, :], Wh[:, 1, :]), dim=1) @ self.a).squeeze(1)

            alpha_[:, 1, 0] = (torch.cat((Wh[:, 1, :], Wh[:, 0, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 1, 1] = (torch.cat((Wh[:, 1, :], Wh[:, 1, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 1, 2] = (torch.cat((Wh[:, 1, :], Wh[:, 2, :]), dim=1) @ self.a).squeeze(1)

            alpha_[:, 2, 1] = (torch.cat((Wh[:, 2, :], Wh[:, 1, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 2, 2] = (torch.cat((Wh[:, 2, :], Wh[:, 2, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 2, 3] = (torch.cat((Wh[:, 2, :], Wh[:, 3, :]), dim=1) @ self.a).squeeze(1)

            alpha_[:, 3, 2] = (torch.cat((Wh[:, 3, :], Wh[:, 2, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 3, 3] = (torch.cat((Wh[:, 3, :], Wh[:, 3, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 3, 4] = (torch.cat((Wh[:, 3, :], Wh[:, 4, :]), dim=1) @ self.a).squeeze(1)

            alpha_[:, 4, 3] = (torch.cat((Wh[:, 4, :], Wh[:, 3, :]), dim=1) @ self.a).squeeze(1)
            alpha_[:, 4, 4] = (torch.cat((Wh[:, 4, :], Wh[:, 4, :]), dim=1) @ self.a).squeeze(1)
        

        
        
        attention = F.softmax(alpha_, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training).to(args.device)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] 
        

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)


        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
    bs, T, i_size, h_size = 2, 3, 4, 5
    input = torch.randn(bs, T, i_size)
    h0 = torch.randn(bs, h_size)

    gru_layer = nn.GRU(i_size, h_size, batch_first=True)
    output, h_final = gru_layer(input, h0.unsqueeze(0))
    print(output)
    for k, v in gru_layer.named_parameters():
        print(k, v.shape)

    output_custom, h_final_custom = gru_forward(input, h0, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, \
                                                gru_layer.bias_ih_l0, gru_layer.bias_hh_l0)
    print(torch.allclose(output, output_custom))
    print(torch.allclose(h_final, h_final_custom))
