dataset='Facebook'
isRegular=True


model='GAT'
lr_GCN_CU=float(0.01)
lr_GCN_Fea=float(0.03)
lr_GRU=float(0.01)
lr_correction=float(0.02)
lr_decay_steps='25,35'
wd=float(1e-4)
dropout=float(0.2)
filters='32'
output_GCN=1
output_GRU=2


nhead=8
filter_scale=1
n_hidden=0
n_hidden_edge=16
degree=False
epochs=10
batch_size=32
bn=False
folds=10
threads=0
log_interval=10
device='cuda'
seed=111
shuffle_nodes=False
torch_geom=False
adj_sq=False
behavior_identity=True
visualize=False
use_cont_node_attr=True
alpha=0.2
using_ATT=False
pick_T_num=5
ndim = 3
source_num = 404
data_Max_Min_processing = True

bi_GRU=True
GAT=True



network_verNum=4039
graph_num=1000
Nodes_total_NUM = 4039* 1000





