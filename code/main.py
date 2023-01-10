import math
import sys
import time
import torch
import heapq
import args as args
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F

from dataReader import DataReader
from dataLodaer import GraphData
from model import GCN
from model import Prop_esNet
from model import GAT

from scipy.spatial import distance
from torch.utils.data import DataLoader

import os


torch.cuda.set_device(1)

print()

args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))
loss_fn = F.cross_entropy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)



def prop_estimation(datareader_all):
    prob_es_net = Prop_esNet(datareader_all.data['adj_list'][0]).to(args.device)
    optimizer_prob = optim.Adam(prob_es_net.parameters(), lr=0.005)
    prob_es_net.train()
    loss_min = 10000
    for epoch in range(10):
        for i, snapshot_5 in enumerate(datareader_all.data['snapshots']):
            optimizer_prob.zero_grad()
            snapshot_pred = prob_es_net(snapshot_5, [1, 2, 3, 4, 5])
            loss = prob_es_net.loss(snap_pred=snapshot_pred.float(), snapshot=snapshot_5)
            loss.backward()
            optimizer_prob.step()
            prob_es_net.parameters_correction()

            if i % 50 == 0:
                print('eopch', (epoch+1), '训练到第', i, '张样本对应的损失: ', loss)

            if epoch >= 8 or True:
                if loss_min > loss.item():
                    loss_min = loss.item()
                    state = {'net': prob_es_net, 'net_params': prob_es_net.state_dict(),
                             'optimizer': optimizer_prob.state_dict(), 'epoch': epoch}
                    torch.save(state, './data/modelpara_trainingAvailable_abs_facebook.pth')
                        

    print('传播模型训练完成.......')
    return prob_es_net.weight.detach().cpu().numpy()

def dynamicFea(datareader_all):
    
    
    
    avg_infl = torch.mean(network_W)
    neighbor_sum = datareader_all.data['adj_list'].sum(1)
    shpae_GTV = datareader_all.data['snapshots'].shape
    influnce_data = np.zeros((shpae_GTV[0], shpae_GTV[1], shpae_GTV[2], 4))
    for graph_index in range(shpae_GTV[0]):
        print(graph_index)
        for T_index in range(shpae_GTV[1]):
            snap_one = datareader_all.data['snapshots'][graph_index, T_index, :]
            influnce_data[graph_index, T_index, :, 0] = 1 - snap_one
            influnce_data[graph_index, T_index, :, 1] = snap_one

            
            for node_index in range(shpae_GTV[2]):
                snap_one_node = snap_one.copy()
                node_influence = network_W[node_index, :].clone()  
                node_influence_uninfe = network_W[node_index, :].clone()

                snap_one_node[torch.where(node_influence == 0)[0].cpu()] = 0
                infe_num = snap_one_node.sum()
                total_num = neighbor_sum[0, node_index]  
                
                

                node_influence[np.where(snap_one_node == 0)[0]] = 0
                
                
                node_influence_uninfe[np.where(snap_one_node == 1)[0]] = 0
                
                
                
                influnce_data[graph_index, T_index, node_index, 2] = torch.sum(node_influence).cpu() / total_num
                
                
                influnce_data[graph_index, T_index, node_index, 3] = avg_infl - (torch.sum(node_influence_uninfe).cpu() / total_num)
                

    np.save('./data/dynamic_Feas_facebook.npy', influnce_data)
    return influnce_data


def staticFea(datareader_all):
    
    
    
    avg_infl = torch.mean(network_W)
    neighbor_sum = datareader_all.data['adj_list'].sum(1)
    shpae_GTV = datareader_all.data['snapshots'].shape
    influnce_data = np.zeros((shpae_GTV[0], shpae_GTV[1], shpae_GTV[2], 2))
    for graph_index in range(shpae_GTV[0]):
        print(graph_index)
        for T_index in range(shpae_GTV[1]):
            snap_one = datareader_all.data['snapshots'][graph_index, T_index, :]
            
            
            
            for node_index in range(shpae_GTV[2]):
                
                snap_one_node = snap_one.copy()
                snap_one_node[np.where(datareader_all.data['adj_list'][0, node_index, :] == 0)] = 0
                infe_num = snap_one_node.sum()
                total_num = neighbor_sum[0, node_index]  
                inf_rate = infe_num / total_num
                uninf_rate = 1 - inf_rate

                influnce_data[graph_index, T_index, node_index, 0] = inf_rate
                influnce_data[graph_index, T_index, node_index, 1] = uninf_rate

    np.save('./data/static_Feas_facebook.npy', influnce_data)
    return influnce_data

def collate_batch(batch):

    B = len(batch)  
    
    
    
    Chanels = batch[0][-1].shape[1]  
    N_nodes_max = args.network_verNum

    A = torch.zeros(1, N_nodes_max, N_nodes_max)
    A[0, :, :] = batch[0][0]

    x = torch.zeros(B, args.pick_T_num, N_nodes_max, Chanels)
    P = torch.zeros(B, args.pick_T_num, N_nodes_max)
    labels = torch.zeros(B, batch[0][1].shape[0])
    for b in range(B):
        x[b, 0, :] = batch[b][3]
        x[b, 1, :] = batch[b][4]
        x[b, 2, :] = batch[b][5]
        x[b, 3, :] = batch[b][6]
        x[b, 4, :] = batch[b][7]
        P[b, :, :] = batch[b][2]
        labels[b, :] = batch[b][1]

    N_nodes = torch.from_numpy(np.array(N_nodes_max)).long()
    
    return [A, labels, P, x, N_nodes]






def Dice_similar(adj):
    S = torch.zeros(adj.shape[0], adj.shape[0])
    for row in range(adj.shape[0]):
        print(row)
        for col in range(adj.shape[0]):
            if row == col:
                S[row][col] = 1
            if row < col:
                s_i_j = 1 - distance.dice(adj[row], adj[col])
                S[row][col] = s_i_j
                S[col][row] = s_i_j
    np.save('./data/Dice_facebook.npy', S.numpy())
    return S


def Attetion_M(pred, bidirectional=False):  
    mid = int(pred.shape[3]/2)
    
    if bidirectional is True:  
        pred_forward = pred[:, :, :, :mid]                 
        pred_backward = pred[:, :, :, mid:2*mid]
        pred_2d = (pred_forward+pred_backward)/2
    else:
        pred_2d = pred
    return pred_2d[:, 4, :, :]

def F_score_computation(pred, labels_mul_hot, source_num):  
    B = pred.shape[0]
    F_score_total = 0
    pred = torch.transpose(pred, 1, 2)

    for i in range(B):
        F_score_one_B = 0
        
        pred_one_B = pred[i, :, 1]  
        labels_mul_hot_one_B = labels_mul_hot[i, :] 
        tmp = zip(range(len(pred_one_B.tolist())), pred_one_B.tolist())
        largeN = heapq.nlargest(source_num, tmp, key=lambda x: x[1])
        for YZ in largeN:
            if labels_mul_hot_one_B[YZ[0]] == 1:
                F_score_one_B = F_score_one_B + 1 / (source_num)
        F_score_total += F_score_one_B
    return F_score_total/B

def train(train_loader):

    start = time.time()
    train_loss, n_samples, F_score_total = 0, 0, 0
    for batch_idx, data in enumerate(train_loader):
        opt.zero_grad()
        B = data[3].shape[0]
        for i in range(len(data)):
            data[i] = data[i].to(args.device)

        output = torch.zeros(data[3].shape[0], data[3].shape[1], data[3].shape[2], args.output_GCN).to(args.device)
        for T_index in range(args.pick_T_num):
            
            feature_encode_onestamp = GCN_models_CU[T_index](data, T_index)

            output[:, T_index, :, :] = feature_encode_onestamp  
        

        GCN_Fea_Out = GCN_model_Fea(data, -1)
        GCN_Fea_Out = GCN_Fea_Out.unsqueeze(1).tile(1, args.pick_T_num, 1, 1)

        labels = data[1]  
        labels_mul_hot = torch.zeros(B, args.network_verNum).to(args.device)
        for i in range(B):
            seed_vec = labels[i].long()
            labels_mul_hot[i, :][seed_vec] = 1  

        
        if output.shape[3] == 1:
            earlystamp = data[3][:, 0, :, :]  
            earlystamp = earlystamp[:, :, 1]  
            output = torch.relu(output).squeeze(-1)  
            output = (output + earlystamp.unsqueeze(1).repeat(1, args.pick_T_num, 1))/2
            output = output.unsqueeze(3)  


        
        
        
        if args.bi_GRU is True:
            pred = torch.zeros(B, args.pick_T_num, args.network_verNum, 4).to(args.device)
        else:
            pred = torch.zeros(B, args.pick_T_num, args.network_verNum, 2).to(args.device)

        for v_index in range(args.network_verNum):    





            output_one_node, _ = GRU_model( torch.concat(
                (torch.concat(
                    (output[:, :, v_index, :], data[3][:, :, v_index, range(2, 4)]),
                    dim=2), GCN_Fea_Out[:, :, v_index, :]),
                dim=2)
                            )    
                                                                        
            pred[:, :, v_index, :] = output_one_node   
        

        
        pred_att = torch.zeros(data[3].shape[0], args.network_verNum, 2).to(args.device)
        if args.GAT is True:  
            for B in range(data[3].shape[0]):
                
                ADG = torch.tensor([[1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 1, 1]]).to(args.device)
                pred_att[B, :, :] = GAT_model(pred[B, :, :, :], ADG)
                    
            pred = pred_att

        else:
            pred = Attetion_M(pred, bidirectional=True)   

        
        pred = torch.transpose(pred, 1, 2)  
        
        loss = GCN_models_CU[0].loss(pred=pred, label_sourceNum=labels.long(), label_hot=labels_mul_hot.long(), prob_pred=output[:, 0, :, :])
        
        loss.backward()
        opt.step()

        time_iter = time.time() - start
        train_loss += loss.item() * len(output)
        n_samples += len(output)
        
        F_score_total += len(output) * F_score_computation(pred, labels_mul_hot,
                                    source_num=args.source_num)  
        if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1 or 0 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f})\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, F_score_total / n_samples,
                time_iter / (batch_idx + 1)))

def mytest(test_loader):
    GCN_model_Fea.eval()
    GRU_model.eval()
    GAT_model.eval()

    start = time.time()
    test_loss, n_samples, F_score_total = 0, 0, 0
    for batch_idx, data in enumerate(test_loader):
        B = data[3].shape[0]
        for i in range(len(data)):
            data[i] = data[i].to(args.device)
        output = torch.zeros(data[3].shape[0], data[3].shape[1], data[3].shape[2], args.output_GCN).to(args.device)
        for T_index in range(args.pick_T_num):
            feature_encode_onestamp = GCN_models_CU[T_index](data, T_index)

            output[:, T_index, :, :] = feature_encode_onestamp  
        

        GCN_Fea_Out = GCN_model_Fea(data, -1)
        GCN_Fea_Out = GCN_Fea_Out.unsqueeze(1).tile(1,args.pick_T_num,1,1)

        labels = data[1]  
        labels_mul_hot = torch.zeros(B, args.network_verNum).to(args.device)
        for i in range(B):
            seed_vec = labels[i].long()
            labels_mul_hot[i, :][seed_vec] = 1  

        
        if output.shape[3] == 1:
            earlystamp = data[3][:, 0, :, :]  
            earlystamp = earlystamp[:, :, 1]  
            output = torch.relu(output).squeeze(-1)  
            output = (output + earlystamp.unsqueeze(1).repeat(1, args.pick_T_num, 1))/2
            output = output.unsqueeze(3)  

        if args.bi_GRU is True:
            pred = torch.zeros(B, args.pick_T_num, args.network_verNum, 4).to(args.device)
        else:
            pred = torch.zeros(B, args.pick_T_num, args.network_verNum, 2).to(args.device)

        for v_index in range(args.network_verNum):    
            output_one_node, _ = GRU_model( torch.concat(
                (torch.concat(
                    (output[:, :, v_index, :], data[3][:, :, v_index, range(2, 4)]),
                    dim=2), GCN_Fea_Out[:, :, v_index, :]),
                dim=2)
                            )    
                                                                        
            pred[:, :, v_index, :] = output_one_node   
        

        
        pred_att = torch.zeros(data[3].shape[0], args.network_verNum, 2).to(args.device)
        if args.GAT is True:  
            for B in range(data[3].shape[0]):
                
                ADG = torch.tensor([[1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 1, 1]]).to(args.device)
                pred_att[B, :, :] = GAT_model(pred[B, :, :, :], ADG)
                
            pred = pred_att

        else:
            pred = Attetion_M(pred, bidirectional=True)  

        pred = torch.transpose(pred, 1, 2)  
        loss_ = GCN_models_CU[0].loss(pred=pred, label_sourceNum=labels.long(), label_hot=labels_mul_hot.long(), prob_pred=output[:, 0, :, :])
        time_iter = time.time() - start
        test_loss += loss_.item() * len(output)
        n_samples += len(output)
        
        F_score_total += len(output) * F_score_computation(pred, labels_mul_hot,
                                    source_num=args.source_num)  
        if batch_idx % args.log_interval == 0 or batch_idx == len(test_loader) - 1  or 0 == 0:
            
            print("\033[0;31;40m",
                  'Test acc: [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f})\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                n_samples, len(test_loader.dataset),
                100. * (batch_idx + 1) / len(test_loader), loss_.item(), test_loss / n_samples, F_score_total / n_samples,
                time_iter / (batch_idx + 1)),
                  "\033[0m")


test_flag = False
n_folds = args.folds

datareader = DataReader(data_dir='./data/%s/' % args.dataset,
                        rnd_state=rnd_state,
                        folds=n_folds,
                        use_cont_node_attr=args.use_cont_node_attr) if test_flag is False else None






network_W = torch.load('./data/modelpara_trainingAvailable_abs.pth')['net_params']['weight'].cuda()



dyn_feas = np.load('./data/static_Feas_facebook.npy')
datareader.data['features_T0'] = np.concatenate((datareader.data['features_T0'], dyn_feas[:, 0, :, :]), axis=2)
datareader.data['features_T1'] = np.concatenate((datareader.data['features_T1'], dyn_feas[:, 1, :, :]), axis=2)
datareader.data['features_T2'] = np.concatenate((datareader.data['features_T2'], dyn_feas[:, 2, :, :]), axis=2)
datareader.data['features_T3'] = np.concatenate((datareader.data['features_T3'], dyn_feas[:, 3, :, :]), axis=2)
datareader.data['features_T4'] = np.concatenate((datareader.data['features_T4'], dyn_feas[:, 4, :, :]), axis=2)




dice_M = torch.from_numpy(np.load('./data/Dice_facebook.npy')).to(args.device)

for fold_id in range(n_folds):
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id,
                          datareader=datareader,  
                          split=split)

        loader = DataLoader(gdata,  
                            batch_size=args.batch_size,  
                            shuffle=True,  
                            num_workers=args.threads,
                            collate_fn=collate_batch)  
        loaders.append(loader)  
        

    print('\nFOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders[0].dataset),
                                                   len(loaders[1].dataset)))

    
    
    GCN_models_CU = []

    optimizers = []
    schedulers = []
    for i in range(args.pick_T_num):
        GCN_model_CU = GCN(in_features=2,  
                        out_features=args.output_GCN,
                        n_hidden=args.n_hidden,  
                        filters=args.filters,
                        dropout=args.dropout,
                        behavior_identity=args.behavior_identity,
                        FC = True,
                        Dice_M1=dice_M).to(args.device)  
        GCN_models_CU.append(GCN_model_CU)




    GCN_model_Fea = GCN(in_features=args.network_verNum,  
                       out_features=-1,
                       n_hidden=0,  
                       filters=[int(math.sqrt(args.network_verNum))],
                       dropout=0.1,
                       behavior_identity=False,
                       FC=False,
                       Dice_M1=None).to(args.device)

    
    
    GRU_model = nn.GRU(input_size=args.output_GCN + 2 + int(math.sqrt(args.network_verNum)), hidden_size=2, num_layers=3, batch_first=True,
                       bidirectional=args.bi_GRU).to(args.device)
    
    

    GAT_model = GAT(nfeat=4, nhid=-1, nclass=2, dropout=0, alpha=0.01, nheads=args.nhead).to(args.device)

    print('\nInitialize model')
    print(GCN_models_CU[0])
    train_params = list(filter(lambda p: p.requires_grad, GCN_models_CU[0].parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    print('-------------------------------------------------------------------------------------------')
    print(GCN_model_Fea)
    train_params2 = list(filter(lambda p: p.requires_grad, GCN_model_Fea.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params2]))

    print('-------------------------------------------------------------------------------------------')
    print(GRU_model)
    train_params3 = list(filter(lambda p: p.requires_grad, GRU_model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params3]))

    print('-------------------------------------------------------------------------------------------')
    print(GAT_model)
    train_params4 = list(filter(lambda p: p.requires_grad, GAT_model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params4]))


    opt = torch.optim.Adam([
        {'params': GCN_models_CU[0].parameters(), 'lr': args.lr_GCN_CU},
        {'params': GCN_models_CU[1].parameters(), 'lr': args.lr_GCN_CU},
        {'params': GCN_models_CU[2].parameters(), 'lr': args.lr_GCN_CU},
        {'params': GCN_models_CU[3].parameters(), 'lr': args.lr_GCN_CU},
        {'params': GCN_models_CU[4].parameters(), 'lr': args.lr_GCN_CU},

        {'params': GCN_model_Fea.parameters(), 'lr': args.lr_GCN_Fea},

        {'params': GRU_model.parameters(), 'lr': args.lr_GRU},

        {'params': GAT_model.parameters(), 'lr': args.lr_GRU},
    ])
    for epoch in range(args.epochs):
        for i in range(len(GCN_models_CU)):
            GCN_models_CU[i].train()
        GCN_model_Fea.train()
        GRU_model.train()
        GAT_model.train()
        train(loaders[0])
    acc = mytest(loaders[1])