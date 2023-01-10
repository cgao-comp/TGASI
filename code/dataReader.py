import numpy as np
import sklearn
import args as args
from os.path import join as pjoin
import os

class DataReader():
    '''
    Class to read the txt files containing all data of the dataset.
    Should work for any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
    '''

    def __init__(self,
                 data_dir,  # folder with txt files
                 rnd_state=None,
                 use_cont_node_attr=True,
                 # use or not additional float valued node attributes available in some datasets
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        data = {}

        # 记录了node属于哪个图，以及图包含哪些节点
        # 对于规模不相同的图而言:

        # 对于规模相同的图而言: (传播源定位用这个即可)
        # 读出有多少张图, 然后再读出有多少个点
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

        # for T in range(args.pick_T_num):
        #     feature_name = 'features_T' + str(T)
        #     data[feature_name] = np.array(
        #         self.read_features(
        #         list(filter(lambda f: f.find(feature_name) >= 0, files))[0] ))

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

        # Create train/test sets first
        train_ids, test_ids = split_ids(rnd_state.permutation(args.graph_num), folds=folds)  # number of samples (graphs) in data

        # Create train sets
        splits = []
        for fold in range(
                len(train_ids)):  # [{train: array([1000+, 1]), test: array([100+, 1])}, {}, {}, ... , {}] 一共10个
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
        data['splits'] = splits

        self.data = data

    # 这里的local变量-data中存的是所有节点属于哪个图的信息
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
            node1 = int(edge[0].strip())   # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip())
            adj[node1, node2] = 1
            adj[node2, node1] = 1

        adj_list = []
        # for i in range(len(graphs)):
        #     adj_list.append(np.array(adj, copy=True))
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
        fetures = []   # row: 500
        feture_one_sim = []  # 198 row
        for graphx5_index in range(len(lines)):  #拿到一整个snapshot
            if graphx5_index % 5 ==T_index:
                feature_one_node = lines[graphx5_index].split(' ')
                feature_one_node.remove('\n')
                feature_one_node_int = [float(s) for s in feature_one_node]

                for node_index in range(len(feature_one_node_int)):
                    feature_one_comp = []
                    if 1 - feature_one_node_int[node_index] > 0.9:  # 说明该节点该时刻还未被感染     [第一个位置为S位, 第二个位置为I位]
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
        fetures = []   # row: 500
        feture_one_sim = [] # 198 row
        for index in range(len(lines)):
            feature_one_node = lines[index].split(' ')
            feature_one_node.remove('\n')
            feature_one_node_int = [float(s) for s in feature_one_node]

            feature_one_comp = []
            if 1 - feature_one_node_int[0] > 0.9:  # 说明该节点该时刻还未被感染     [第一个位置为S位, 第二个位置为I位]
                feature_one_comp.extend([1, 0])
            else:
                feature_one_comp.extend([0, 1])

            # del (feature_one_node_int[3])

        #     del (feature_one_node_int[0])
        #     feature_one_comp.extend(feature_one_node_int)
            feture_one_sim.append(feature_one_comp)
        #
            if (index+1) % args.network_verNum == 0:
                fetures.append(feture_one_sim)
                feture_one_sim = []
        #
        # if args.data_Max_Min_processing is True:
        #     self.Min_Max_Norm(fetures)

        return fetures


    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                # print('进来了....')
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])  # 把list转换为np的array
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            # print("graph_id: ", graph_id)  # graph_id:  3411
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

def split_ids(ids, folds=10):
    # if args.dataset == 'COLORS-3':
    #     assert folds == 1, 'this dataset has train, val and test splits'
    #     train_ids = [np.arange(500)]
    #     val_ids = [np.arange(500, 3000)]
    #     test_ids = [np.arange(3000, 10500)]
    # elif args.dataset == 'TRIANGLES':
    #     assert folds == 1, 'this dataset has train, val and test splits'
    #     train_ids = [np.arange(30000)]
    #     val_ids = [np.arange(30000, 35000)]
    #     test_ids = [np.arange(35000, 45000)]
    # else:
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))  # 一个fold中包含的数量为    len(样本数量)  /  几折
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    train_ids = []
    for fold in range(folds):
        train_ids.append(np.array(
            [e for e in ids if e not in test_ids[fold]]))  # 分成10份，然后对于每份，都是将其中9份设置为train，剩余的一份设置为test。实现了10倍的数据增广
        assert len(train_ids[fold]) + len(test_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids