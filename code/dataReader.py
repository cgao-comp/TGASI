import numpy as np
import sklearn
import args as args
from os.path import join as pjoin
import os

class DataReader():

    def __init__(self,
                 data_dir,  
                 rnd_state=None,
                 use_cont_node_attr=True,
                 
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        data = {}

        
        

        
        
        nodes = {i: (i // args.network_verNum) for i in range(args.Nodes_total_NUM)}
        graphs = {
            row: np.array([col for col in range((row - 1) * args.network_verNum, row * args.network_verNum)])
            for row in range(1, args.graph_num + 1)}

        data['adj_list'] = np.array(self.read_graph_adj_regular(list(filter(lambda f: f.find('_A') >= 0, files))[0],
                                                           nodes, graphs))

        data['targets'] = np.array(
            self.read_source_index(
                list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0] ))

        data['snapshots'] = np.array(
            self.read_snapshots(
                list(filter(lambda f: f.find('snapshots') >= 0, files))[0] ))

        
        
        
        
        

        for T in range(args.pick_T_num):
            feature_name = 'features_T' + str(T)
            data[feature_name] = np.array(
                    self.read_features_from_sanpshot(
                    list(filter(lambda f: f.find('snapshots') >= 0, files))[0],
                    T
                )
            )

        assert args.graph_num == data['targets'].shape[0] \
               == data['snapshots'].shape[0] == data['features_T0'].shape[0], 'invalid data'
        assert data['adj_list'].shape[0] == 1, 'invalid data'

        
        train_ids, test_ids = split_ids(rnd_state.permutation(args.graph_num), folds=folds)  

        
        splits = []
        for fold in range(
                len(train_ids)):  
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
        data['splits'] = splits

        self.data = data

    
    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_source_index(self, fpath):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            index_oneSim = line.split(' ')
            index_oneSim.remove('\n')
            index_int = [int(s) for s in index_oneSim]
            data.append(index_int)
        return data


    def read_graph_adj_regular(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(' '))
        n = len(graphs[1])
        adj = np.zeros((n, n))
        for edge in edges:
            node1 = int(edge[0].strip())   
            node2 = int(edge[1].strip())
            adj[node1, node2] = 1
            adj[node2, node1] = 1

        adj_list = []
        
        
        adj_list.append(np.array(adj, copy=True))
        return adj_list

    def read_snapshots(self, fpath):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        snapshots = []
        snapshot_oneSim = []
        for index in range(len(lines)):
            snapshot = lines[index].split(' ')
            snapshot.remove('\n')
            snapshot_int = [int(float(s)) for s in snapshot]
            snapshot_oneSim.append(snapshot_int)
            if (index+1)%args.pick_T_num ==0:
                snapshots.append(snapshot_oneSim)
                snapshot_oneSim = []
        return snapshots

    def Min_Max_Norm(self, X):
        pass

    def read_features_from_sanpshot(self, fpath, T_index):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        fetures = []   
        feture_one_sim = []  
        for graphx5_index in range(len(lines)):  
            if graphx5_index % 5 ==T_index:
                feature_one_node = lines[graphx5_index].split(' ')
                feature_one_node.remove('\n')
                feature_one_node_int = [float(s) for s in feature_one_node]

                for node_index in range(len(feature_one_node_int)):
                    feature_one_comp = []
                    if 1 - feature_one_node_int[node_index] > 0.9:  
                        feature_one_comp.extend([1, 0])
                    else:
                        feature_one_comp.extend([0, 1])
                    feture_one_sim.append(feature_one_comp)
                fetures.append(feture_one_sim)
                feture_one_sim = []

        return fetures

    def read_features(self, fpath):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        fetures = []   
        feture_one_sim = [] 
        for index in range(len(lines)):
            feature_one_node = lines[index].split(' ')
            feature_one_node.remove('\n')
            feature_one_node_int = [float(s) for s in feature_one_node]

            feature_one_comp = []
            if 1 - feature_one_node_int[0] > 0.9:  
                feature_one_comp.extend([1, 0])
            else:
                feature_one_comp.extend([0, 1])

            

        
        
            feture_one_sim.append(feature_one_comp)
        
            if (index+1) % args.network_verNum == 0:
                fetures.append(feture_one_sim)
                feture_one_sim = []
        
        
        

        return fetures


    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])  
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

def split_ids(ids, folds=10):
    
    
    
    
    
    
    
    
    
    
    
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))  
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    train_ids = []
    for fold in range(folds):
        train_ids.append(np.array(
            [e for e in ids if e not in test_ids[fold]]))  
        assert len(train_ids[fold]) + len(test_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids