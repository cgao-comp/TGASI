import network.CreateGraph as CreateGraphPKG
import network.Graph as GraphPKG
import network.PropModel as PropPKG
import network.LPSI as LPSIPKG

import numpy as np
import random as rand
import sys
import copy
import args

def prob_mat_generator(net: GraphPKG.Graph_C,):
    prob_matrix = CreateGraphPKG.CreateGraph.toMatrix(net)
    # prob_data = copy.copy(graph.adj_matrix.data)
    # prob_indices = copy.copy(graph.adj_matrix.indices)
    # prob_indptr = copy.copy(graph.adj_matrix.indptr)
    # prob_shape = copy.copy(graph.adj_matrix.shape)
    avgD = CreateGraphPKG.CreateGraph.getAvgDegree(net)
    for i in range(len(prob_matrix)):
        for j in range(len(prob_matrix[0])):
            if prob_matrix[i][j] == 1:
                if net.vertexArray[i].degree > avgD:
                    # p = rand.choice([0.35, 0.4, 0.45, 0.5, 0.55, 0.6], )
                    p = rand.choice([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], )
                else:
                    # p = rand.choice([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3], )
                    p = rand.choice([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], )
                prob_matrix[i][j] = p
    return prob_matrix


def run_mc_(graph, seed_vec, diffusion_limit=25) -> np.ndarray:
    '''Run MC once
    args:
            graph: SparseGraph format
            seed_vec: multi-hot vector
    return:
            multi-hot vector of activated nodes
    '''
    activated_vec = copy.copy(seed_vec)
    influ_mat = [seed_vec, ]
    last_activated = np.argwhere(seed_vec == 1).flatten().tolist()
    next_activated = []
    diffusion_count = 0

    while len(last_activated) > 0:
        # 取出邻居
        for u in last_activated:
            u_neighs = np.array(CreateGraphPKG.CreateGraph.getNeighs(graph, u))

            for v in u_neighs:
                if (activated_vec[v] == 0) and rand.random() <= graph.prob_matrix[u][v]:  # activated
                    activated_vec[v] = 1
                    next_activated.append(v)

        current_activated = copy.copy(activated_vec)
        influ_mat.append(current_activated)

        last_activated = next_activated
        next_activated = []
        diffusion_count += 1
        if len(influ_mat) >= diffusion_limit:
            break

    if len(influ_mat) < diffusion_limit:
        for i in range(diffusion_limit - len(influ_mat)):
            influ_mat.append(copy.copy(influ_mat[-1]))

    influ_mat_V25 = np.array(influ_mat).T
    return influ_mat_V25

def seed_vec_generator(N, n):
    seed_vec = np.zeros((N,))
    # seeds = np.random.randint(0, N, size=n)
    seeds = rand.sample(range(0, N), n)
    seed_vec[seeds] = 1
    return seed_vec


def _fea_Cons(seed_vec, prob_matrix, ndim):
    seed_vec = seed_vec.reshape((-1, 1)) # 向量本身是1*198的，这里转换为[[],[],[], ... ,[]] 198*1的。这样就能够进行矩阵乘法了 (198*198) @ (198*1)
    import scipy.sparse as sp
    if sp.isspmatrix(prob_matrix):
        prob_matrix = prob_matrix.toarray()
    else:
        prob_matrix = np.array(prob_matrix)
    assert seed_vec.shape[0] == prob_matrix.shape[0], 'Seed vector is illegal'
    attr_mat = [seed_vec]
    for i in range(ndim-1):
        attr_mat.append(prob_matrix.T @ attr_mat[-1])

    attr_mat = np.concatenate(attr_mat, axis=(-1))
    return attr_mat # 其他节点对该节点的一个概率


# 1.读邻接表文件
graph = GraphPKG.Graph_C()
CreateGraphPKG.CreateGraph.initGraph(graph,
                                     "../data/social_net_data/facebook_edge_index0.txt",
                                     4039, 88234)
adjM = np.array(CreateGraphPKG.CreateGraph.toMatrix(graph))

# 2.构造影响力矩阵
graph.prob_matrix = prob_mat_generator(graph)

#3. 根据影响力矩阵生成K组数据，数据集的条目数量为W
K = args.graph_num
pick_T_num = 5
source_num = args.source_num
ndim = args.ndim  # 由两部分构成：前ndim-1行是由状态矩阵生成的，最后一行是LPSI算法生成的
# timestamp
# low determined and high determined
simu_time = 0
A_flag = False
while simu_time < K:
    simu_time = simu_time + 1
    # 3.1 生成seed向量数据
    seed_vec = seed_vec_generator(graph.verNum, source_num)
    seed_index = np.where(seed_vec == 1)[0]
    print("seed_vec num: ", seed_index.shape[0])
    if seed_index.shape[0] != source_num:
        simu_time = simu_time - 1
        continue
    # # 3.1.1 将label数据写入到文件中，即为标签数据。label.txt shape: [K, source_rate*V]/[K, source_rate*V]/[K, source_rate*V]/[K, source_rate*V]
    fp = open('../data/Facebook/Facebook_graph_labels.txt', 'a')
    for ver in seed_index:
        fp.write('{} '.format(ver))
    fp.write('\n')
    fp.close()

    # 3.2 根据向量数据产生一次传播数据
    influ_mat_V25 = run_mc_(graph, seed_vec)

    # 3.3 从0-25个时间步中随机选择pick_T_num对应数量的子图
    # 每一条模拟数据的shape: [timestamp=5, V] => 对应的snapshot.txt文件的格式为K模拟次数个[5, V] => snapshot.txt文件的shape: [5K, V]/[5K, V]/[5K, V]/[5K, V]/[5K, V]
    # 一次取timestamp行
    fp = open('../data/Facebook/Facebook_snapshots.txt', 'a')
    if rand.randint(1,5) == 1:
        index_5 = [0, 1, 2, 3, 4]
    else:
        index_5 = [1, 2, 3, 4, 5]
    for index in index_5:
        snap_one = influ_mat_V25[:, index]
        for node in snap_one:
            fp.write('{} '.format(node))
        fp.write('\n')
    fp.close()

    # for T in [1, 2, 3, 4, 5]:
    #     seed_vec_T = influ_mat_V25[:, T]
    #     file_name = '../data/Facebook/Facebook_features_T'+ str(T-1) +'.txt'
    #
    #     # 3.4 第T时刻生成节点的特征信息并将信息存储到Jazz_feature.txt文件夹中（以下代码是一个timestamp）
    #       #* 一个timestamp就会生成K*V, ndim对应维度的信息
    #     feature_col5 = _fea_Cons(seed_vec_T, graph.prob_matrix, ndim-1)
    #     feature_LPSI = LPSIPKG.LPSI_ALG(seed_vec_T, np.array(CreateGraphPKG.CreateGraph.toMatrix(graph)) )
    #     # # 3.4.1 生成特征数据
    #     ##### 每一条模拟数据的shape: [V, ndim=特征条数] => features.txt文件的格式为K模拟次数个[K*V, ndim]
    #     # 一次取V行
    #     # 其中第一列(dim0)是感染特征1或者0，因此在数据读入的时候需要处理为one-hot向量
    #     # 第二列到倒数第二行是传播特征，最后一列是LPSI的中心性特征
    #     fp = open(file_name, 'a')
    #     for index_node in range(graph.verNum):
    #         feature_one_node_onestamp = feature_col5[index_node, :]   #feature_col5: shape: V*ndim
    #         for feature in feature_one_node_onestamp:
    #             fp.write('{} '.format(feature))
    #         # 写入LPSI的特征
    #         fp.write('{} '.format(feature_LPSI[index_node]))
    #         fp.write('\n')
    #     fp.close()

    # ##### 每一条模拟数据的shape: [ndim=特征条数, V] => features.txt文件的格式为K模拟次数个[ndim, V]
    # # 一次取ndim行
    # fp = open('../data/Facebook/Facebook_features.txt', 'a')
    # for index in range(ndim-1):
    #     feature_one = feature_col5[:, index]
    #     for feature in feature_one:
    #         fp.write('{} '.format(feature))
    #     fp.write('\n')
    # # 写入LPSI的特征
    # for feature in feature_LPSI:
    #     fp.write('{} '.format(feature))
    # fp.write('\n')
    # fp.close()

    # 写入邻接矩阵
    if A_flag is False:
        A_flag = True
        adjM = np.array(CreateGraphPKG.CreateGraph.toMatrix(graph))
        fp = open('../data/Facebook/Facebook_A.txt', 'w')
        for i in range(len(adjM)):
            for j in range(len(adjM)):
                if i < j:
                    if adjM[i][j] == 1:
                        fp.write('{} {}\n'.format(i, j))
        fp.close()


#
# # 1. 构造邻接矩阵
# mynetwok = GraphPKG.Graph_C()
# CreateGraphPKG.CreateGraph.initGraph(mynetwok,
#                                      "../data/social_net_data/jazz_edge.txt",
#                                      198, 2742)
# # CreateGraphPKG.CreateGraph.outputGraph(mynetwok)
#
# for source_varName_typeInt in range(1, mynetwok.verNum+1):
#     rand_times=rand.randint(5, 7)
#     for times in range(rand_times):
#         # ### 1. 初始化图上的信息
#         CreateGraphPKG.CreateGraph.resetGraph(mynetwok)
#         # 2. 正向传播
#         mySI=PropPKG.SI_C()
#         rand_infection_scale = rand.uniform(0.1, 0.3)
#         rand_infection_rate = rand.uniform(0.3, 0.9)
#         # print("rand_infection_rate: ", rand_infection_rate, "rand_infection_scale: ", rand_infection_scale)
#         mySI.SI_Prop(mynetwok, rand_infection_rate, rand_infection_scale, source_varName_typeInt)
#         # print("source: ", mySI.source)
#
#         # 3. 提取并生成感染标签
#         Y = np.array(mySI.getStateLabel(mynetwok))
#
#         # 4. 将邻接表转换为矩阵
#         adjM = np.array(CreateGraphPKG.CreateGraph.toMatrix(mynetwok))
#
#         # 5. 执行LPSI算法提取标签
#         label_attrs = LPSIPKG.features_generation(Y, adjM, 0.5)
#         # print("attrs_shape: ", label_attrs.shape)
#
#         # 6. 写入特征数据
#         ### (6.1) 所有图的邻接表信息
#         if source_varName_typeInt == 1 and times == 0:
#             fp = open('../data/Facebook/Facebook_A.txt', 'w')
#             for i in range(len(adjM)):
#                 for j in range(len(adjM)):
#                     if i < j :
#                         if adjM[i][j] == 1:
#                             fp.write('{},{}\n'.format(i+1, j+1))
#             fp.close()
#
#         ### (6.2) 节点和图的映射关系
#           #由于我们的图和节点是固定的，这个数值是一个不变的值即可，不需要额外创建文件
#         print(adjM.shape[0])
#
#         ### 6.3 图的标签--target
#         fp = open('../data/Facebook/Facebook_graph_labels.txt', 'a')
#         fp.write('{}\n'.format(mySI.source))
#         fp.close()
#
#         ### (6.4) 节点arrribution (包括完成归一化)
#         fp = open('../data/Facebook/Facebook_node_attributes.txt', 'a')
#         trans = label_attrs.T
#         for index_i in range(trans.shape[0]):
#             for index_j in range(1, trans.shape[1]):
#                 if index_j==trans.shape[1]-1:
#                     fp.write('{}\n'.format(trans[index_i][index_j]))
#                 else:
#                     fp.write('{},'.format(trans[index_i][index_j]))
#         fp.close()
#
#         ### (6.5) 节点的label (Y=1和Y=-1属于这一类)
#         fp = open('../data/Facebook/Facebook_node_labels.txt', 'a')
#         for index_i in range(trans.shape[0]):
#             fp.write('{}\n'.format(0 if int(trans[index_i][0])==1 else -1))
#         fp.close()
#
#         # 7. 设置表示学习模型参数
#
#         # 8. 调用GCN完成训练
#
sys.exit(0)
# Test Dataset in AAAI-2017
# CreateGraphPKG.CreateGraph.initGraph(mynetwok,
#                                      "C:/Users/hdp/Desktop/Files/PY/myPY/pygcn-master/data/social_net_data/AAAI.txt",
#                                      16, 16)
# LPSIPKG.features_generation(np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1]), M, 0.5)
