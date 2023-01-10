import torch
import copy

class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 datareader,
                 fold_id,
                 split):
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id)

    def set_fold(self, data, fold_id):
        self.total = len(data['targets'])
        self.idx = data['splits'][fold_id][self.split]  # split分为train 和 test 两类    这就是每一折的数据集对应的图的id
        # use deepcopy to make sure we don't alter objects in folds
        self.snapshots = copy.deepcopy([data['snapshots'][i] for i in self.idx])  # 取出对应timestamp图的快照信息
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])  # 取出对应张图的y标签
        # self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])  # 取出对应的邻接矩阵
        self.adj_list = copy.deepcopy([data['adj_list'][0]])
        self.features_T0 = copy.deepcopy([data['features_T0'][i] for i in self.idx])  # 取出对应的特征
        self.features_T1 = copy.deepcopy([data['features_T0'][i] for i in self.idx])
        self.features_T2 = copy.deepcopy([data['features_T0'][i] for i in self.idx])
        self.features_T3 = copy.deepcopy([data['features_T0'][i] for i in self.idx])
        self.features_T4 = copy.deepcopy([data['features_T0'][i] for i in self.idx])

        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

    def __len__(self):
        return len(self.labels)

    # 一个batch_size下的数据集需要的内容都是来自于此函数.........然后如果需要进一步处理batch_size的内容，需要collate data
    def __getitem__(self, index):
        # convert to torch
        # 对于batch_size为16的，这里需要index16次，16次执行完以后，是一个包含对于batch_size为16的维的完整的大数据集，因此A会被扩充为Batch * N * N
        return [
                torch.from_numpy(self.adj_list[0]).float(),  # adjacency matrix
                torch.from_numpy(self.labels[index]),            # Y label
                torch.from_numpy(self.snapshots[index]),         # 快照snapshots

                torch.from_numpy(self.features_T0[index]).float(),  # node_features
                torch.from_numpy(self.features_T1[index]).float(),
                torch.from_numpy(self.features_T2[index]).float(),
                torch.from_numpy(self.features_T3[index]).float(),
                torch.from_numpy(self.features_T4[index]).float(),
        ]