import numpy as np
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
        if args.isRegular is not True:
            nodes, graphs = self.read_graph_nodes_relations(
                list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        else:
            # 对于规模相同的图而言: (传播源定位用这个即可)
            # 读出有多少张图, 然后再读出有多少个点
            nodes = {i: (i // args.network_verNum) + 1 for i in range(args.Nodes_total_NUM)}
            graphs = {
                row: np.array([col for col in range((row - 1) * args.network_verNum, row * args.network_verNum)])
                for row in range(1, args.graph_num + 1)}

        if args.isRegular is not True:
            data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes,
                                                   graphs)
        else:
            data['adj_list'] = self.read_graph_adj_regular(list(filter(lambda f: f.find('_A') >= 0, files))[0],
                                                           nodes, graphs)

        node_labels_file = list(filter(lambda f: f.find('node_labels') >= 0, files))
        if len(node_labels_file) == 1:
            data['features'] = self.read_node_features(node_labels_file[0], nodes, graphs,
                                                       fn=lambda s: int(s.strip()))
        else:
            data['features'] = None

        data['targets'] = np.array(
            self.parse_txt_file(
                list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0],
                line_parse_fn=lambda s: int(float(s.strip()))))

        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                   nodes, graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            if not np.allclose(adj, adj.T):  # 无向图都是对称矩阵
                print(sample_id, 'not symmetric')
            n = np.sum(adj)  # total sum of edges   这个计算的整个矩阵的值，对于无向无权图来说是边的两倍
            assert n % 2 == 0, n
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            degrees.extend(list(np.sum(adj, 1)))
            if data['features'] is not None:
                features.append(np.array(data['features'][sample_id]))

        # Create features over graphs as one-hot vectors for each node
        if data['features'] is not None:
            features_all = np.concatenate(features)
            features_min = features_all.min()
            num_features = int(features_all.max() - features_min + 1)  # number of possible values

        max_degree = np.max(degrees)
        features_onehot = []
        for sample_id, adj in enumerate(data['adj_list']):
            N = adj.shape[0]
            if data['features'] is not None:
                x = data['features'][sample_id]
                feature_onehot = np.zeros((len(x), num_features))
                for node, value in enumerate(x):
                    feature_onehot[node, value - features_min] = 1
            else:
                feature_onehot = np.empty((N, 0))
            if self.use_cont_node_attr:
                if args.dataset in ['COLORS-3', 'TRIANGLES']:
                    # first column corresponds to node attention and shouldn't be used as node features
                    feature_attr = np.array(data['attr'][sample_id])[:, 1:]
                else:
                    feature_attr = np.array(data['attr'][sample_id])  # shape: 42*29
            else:
                feature_attr = np.empty((N, 0))
            if args.degree:
                degree_onehot = np.zeros((N, max_degree + 1))
                degree_onehot[np.arange(N), np.sum(adj, 1).astype(np.int32)] = 1
            else:
                degree_onehot = np.empty((N, 0))  # 相当于没有加入degree的信息

                #  feature_onehot_shape: 42 * 3 (一个node对应feature)
                #  feature_attr_shape: 42 * 29 (一个node对应29个属性)
                #  degree_onehot_shape: 42 * 0 (没有)
            node_features = np.concatenate((feature_onehot, feature_attr, degree_onehot), axis=1)
            if node_features.shape[1] == 0:
                # dummy features for datasets without node labels/attributes
                # node degree features can be used instead
                node_features = np.ones((N, 1))
            features_onehot.append(node_features)

        num_features = features_onehot[0].shape[1]  # 32

        shapes = [len(adj) for adj in data['adj_list']]  # 记录每张图的节点数量
        labels = data['targets']  # graph class labels
        labels -= np.min(labels)  # to start from 0

        classes = np.unique(labels)  # 该函数会自动为序列排序，从小到大的顺序
        num_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(num_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == num_classes, np.unique(labels)

        def stats(x):
            return (np.mean(x), np.std(x), np.min(x), np.max(x))

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(degrees))
        print('Node features dim: \t\t%d' % num_features)
        print('N classes: \t\t\t%d' % num_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        # feature和attr的区别？ feature只有一个？需要处理为onehot，而attr却不需要做任何处理
        if data['features'] is not None:
            for u in np.unique(features_all):
                print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # Create train/test sets first
        train_ids, test_ids = split_ids(rnd_state.permutation(N_graphs), folds=folds)

        # Create train sets
        splits = []
        for fold in range(
                len(train_ids)):  # [{train: array([1000+, 1]), test: array([100+, 1])}, {}, {}, ... , {}] 一共10个
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes of a graph in dataset
        data['num_features'] = num_features
        data['num_classes'] = num_classes

        self.data = data

    # 这里的local变量-data中存的是所有节点属于哪个图的信息
    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list

    def read_graph_adj_regular(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        n = len(graphs[1])
        adj = np.zeros((n, n))
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            adj[node1, node2] = 1
            adj[node2, node1] = 1

        adj_list = []
        for i in range(len(graphs)):
            adj_list.append(np.array(adj, copy=True))
        return adj_list

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