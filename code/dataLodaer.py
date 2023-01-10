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
        self.idx = data['splits'][fold_id][self.split]  
        
        self.snapshots = copy.deepcopy([data['snapshots'][i] for i in self.idx])  
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])  
        
        self.adj_list = copy.deepcopy([data['adj_list'][0]])
        self.features_T0 = copy.deepcopy([data['features_T0'][i] for i in self.idx])  
        self.features_T1 = copy.deepcopy([data['features_T0'][i] for i in self.idx])
        self.features_T2 = copy.deepcopy([data['features_T0'][i] for i in self.idx])
        self.features_T3 = copy.deepcopy([data['features_T0'][i] for i in self.idx])
        self.features_T4 = copy.deepcopy([data['features_T0'][i] for i in self.idx])

        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, index):
        
        
        return [
                torch.from_numpy(self.adj_list[0]).float(),  
                torch.from_numpy(self.labels[index]),            
                torch.from_numpy(self.snapshots[index]),         

                torch.from_numpy(self.features_T0[index]).float(),  
                torch.from_numpy(self.features_T1[index]).float(),
                torch.from_numpy(self.features_T2[index]).float(),
                torch.from_numpy(self.features_T3[index]).float(),
                torch.from_numpy(self.features_T4[index]).float(),
        ]